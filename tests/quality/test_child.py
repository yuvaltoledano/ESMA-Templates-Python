"""The file part of the audit in a process of its own."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import structlog
from click.testing import CliRunner

from esma_milan import cli
from esma_milan.quality import assess_files, child
from esma_milan.quality.child import AssessmentProcessError, run_assessment
from esma_milan.quality.store import PREFIX, ResultStore
from tests.quality.builders import Pair

STARTED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
MARKER_VALUE = "SECRETCELLVALUE"


@pytest.fixture(autouse=True)
def _restore_logging() -> Any:
    yield
    structlog.reset_defaults()


class _Log:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, object]]] = []

    def error(self, event: str, **fields: object) -> None:
        self.events.append(("error", event, fields))

    def info(self, event: str, **fields: object) -> None:
        self.events.append(("info", event, fields))


def _poor_pair() -> Pair:
    return Pair().set("RREL43", "N/A").set("RREL30", "0", record=1).set("RREC13", "-5", record=2)


@pytest.mark.parametrize("pair", [Pair(), _poor_pair()], ids=["compliant", "poor"])
def test_the_child_finds_what_the_in_process_assessment_finds(tmp_path: Path, pair: Pair) -> None:
    loans, collaterals = pair.write(tmp_path)
    here = assess_files(loans, collaterals, started_at=STARTED)
    store = ResultStore.create(tmp_path / "out")
    there = run_assessment(loans, collaterals, store, STARTED)

    assert there.store_root == store.root
    assert there.loans.results.height == 0 and there.loans.records is None  # nothing came back in memory
    assert there.outcomes == here.outcomes
    assert there.loans.profiles == here.loans.profiles
    assert there.collateral.actual_columns == here.collateral.actual_columns
    assert (there.started_at, there.as_of, there.pool) == (here.started_at, here.as_of, here.pool)
    order = ["file", "record", "column_position", "rule_id", "reason"]
    text = pl.col(pl.Categorical).cast(pl.String)
    assert there.results.with_columns(text).sort(order, nulls_last=True).equals(
        here.results.with_columns(text).sort(order, nulls_last=True)
    )
    store.remove()


def test_a_fault_inside_the_child_is_reported_with_its_message(tmp_path: Path) -> None:
    _, collaterals = Pair().write(tmp_path)
    store = ResultStore.create(tmp_path / "out")
    with pytest.raises(AssessmentProcessError) as raised:
        run_assessment(tmp_path / f"{MARKER_VALUE}.csv", collaterals, store, STARTED)
    assert raised.value.exit_status == 1
    assert "FileNotFoundError" in str(raised.value) and MARKER_VALUE in str(raised.value)
    assert (store.root / child.FAULT_FILE).is_file()
    store.remove()


def _script(source: str) -> Any:
    return lambda *_: [sys.executable, "-c", source]


def test_a_killed_child_is_an_assessment_fault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(child, "_command", _script("import os, signal; os.kill(os.getpid(), signal.SIGKILL)"))
    loans, collaterals = Pair().write(tmp_path)
    store = ResultStore.create(tmp_path / "out")
    with pytest.raises(AssessmentProcessError) as raised:
        run_assessment(loans, collaterals, store, STARTED)
    assert raised.value.exit_status == -9
    assert str(raised.value) == "the assessment process ended with exit status -9"
    store.remove()


def test_a_child_that_dies_before_it_can_report_leaves_its_standard_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (
        "import sys\n"
        "for i in range(40): print('line', i, file=sys.stderr)\n"
        f"print('ModuleNotFoundError: {MARKER_VALUE}', file=sys.stderr); sys.exit(3)"
    )
    monkeypatch.setattr(child, "_command", _script(source))
    loans, collaterals = Pair().write(tmp_path)
    store = ResultStore.create(tmp_path / "out")
    with pytest.raises(AssessmentProcessError) as raised:
        run_assessment(loans, collaterals, store, STARTED)
    message = str(raised.value)
    assert raised.value.exit_status == 3
    assert message.endswith(f"ModuleNotFoundError: {MARKER_VALUE}")
    assert "line 39" in message and "line 5\n" not in message  # the last lines only
    store.remove()


def _run_cli(tmp_path: Path, pair: Pair, *extra: str) -> tuple[Any, Path]:
    loans, collaterals = pair.write(tmp_path)
    out = tmp_path / "out"
    result = CliRunner().invoke(
        cli.main,
        ["run", "--loans", str(loans), "--collaterals", str(collaterals), "--deal", "DEAL",
         "--output", str(out), "--quiet", "--aggregation", "by_loan", *extra],
    )
    return result, out / "DEAL"


def _summary(folder: Path) -> dict[str, Any]:
    path = next(p for p in folder.iterdir() if p.name.endswith("summary.json"))
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def test_a_dead_child_changes_nothing_about_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = f"import sys; print('{MARKER_VALUE}', file=sys.stderr); sys.exit(7)"
    monkeypatch.setattr(child, "_command", _script(source))
    log = _Log()
    monkeypatch.setattr(cli, "log", log)
    result, folder = _run_cli(tmp_path, Pair())
    assert result.exit_code == 0 and result.exception is None
    workbook = folder / "2024-06-30 DEAL Flattened loans and collaterals.xlsx"
    assert workbook.exists()
    assert result.output.strip().splitlines() == [str(workbook)]  # nothing else on standard output
    # One event, with the step, the type and the exit status; never the child's text.
    errors = [e for e in log.events if e[0] == "error"]
    assert errors == [("error", "data_quality_assessment_failed",
                       {"step": "assess_files", "error_type": "AssessmentProcessError", "exit_status": 7})]
    assert MARKER_VALUE not in json.dumps(log.events)
    # The summary, beside the workbook, says the assessment did not complete and why.
    summary = _summary(folder)
    assert summary["assessment"]["state"] == "interrupted"
    assert "exit status 7" in summary["assessment"]["reason"]
    assert MARKER_VALUE in summary["assessment"]["reason"]
    assert not [p for p in folder.iterdir() if p.name.startswith(PREFIX)]


def test_ctrl_c_during_the_assessment_writes_nothing_and_removes_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(*_: object, **__: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("esma_milan.quality.child.subprocess.run", interrupted)
    result, folder = _run_cli(tmp_path, Pair())
    assert result.exit_code == 1  # click's Abort, as for any Ctrl-C
    assert list(folder.iterdir()) == []


def test_the_assessment_is_skipped_when_no_working_directory_can_be_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_room(*_: object, **__: object) -> str:
        raise OSError(f"no space left for {MARKER_VALUE}")

    monkeypatch.setattr("esma_milan.quality.store.tempfile.mkdtemp", no_room)
    log = _Log()
    monkeypatch.setattr(cli, "log", log)
    result, folder = _run_cli(tmp_path, Pair())
    assert result.exit_code == 0 and result.exception is None
    assert [p.name for p in folder.iterdir()] == ["2024-06-30 DEAL Flattened loans and collaterals.xlsx"]
    assert log.events == [("error", "data_quality_assessment_failed",
                           {"step": "create_store", "error_type": "OSError"})]
    # And a failing pipeline still fails as it does without the report.
    result, _ = _run_cli(tmp_path, Pair().set("RREL30", "abc", record=1))
    assert type(result.exception).__name__ == "PipelineDataError" and result.exit_code == 1


@pytest.mark.parametrize(
    ("pair", "extra", "exit_code"),
    [(Pair(), (), 0), (Pair(), ("--dry-run",), 0), (Pair().set("RREL30", "abc", record=1), (), 1)],
    ids=["normal", "dry run", "pipeline raised"],
)
def test_each_kind_of_run_goes_through_the_child(
    tmp_path: Path, pair: Pair, extra: tuple[str, ...], exit_code: int
) -> None:
    result, folder = _run_cli(tmp_path, pair, *extra)
    assert result.exit_code == exit_code
    summary = _summary(folder)
    assert summary["assessment"] == {"state": "ran"}
    assert summary["files"][0]["records"] == 2
    assert not [p for p in folder.iterdir() if p.name.startswith(PREFIX)]  # the store is removed
