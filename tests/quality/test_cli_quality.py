"""The command line writes the summary and changes nothing else."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import structlog
from click.testing import CliRunner

from esma_milan import cli
from esma_milan.errors import PipelineDataError
from esma_milan.quality.export import ASSESSMENT_INTERRUPTED, write_json
from esma_milan.runner import run_pipeline
from tests.quality.builders import Pair

WORKBOOK = "2024-06-30 DEAL Flattened loans and collaterals.xlsx"
SUMMARY = "2024-06-30 DEAL Flattened loans and collaterals data quality summary.json"


@pytest.fixture(autouse=True)
def _restore_logging() -> Any:
    """The command configures structlog for the process; undo it."""
    yield
    structlog.reset_defaults()


def _invoke(tmp_path: Path, pair: Pair, *extra: str) -> tuple[Any, Path]:
    loans, collaterals = pair.write(tmp_path)
    out = tmp_path / "out"
    result = CliRunner().invoke(
        cli.main,
        ["run", "--loans", str(loans), "--collaterals", str(collaterals), "--deal", "DEAL",
         "--output", str(out), "--quiet", "--aggregation", "by_loan", *extra],
    )
    return result, out / "DEAL"


def _summary(folder: Path, name: str = SUMMARY) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((folder / name).read_text(encoding="utf-8"))
    return loaded


def test_summary_is_written_beside_the_workbook(tmp_path: Path) -> None:
    result, folder = _invoke(tmp_path, Pair())
    assert result.exit_code == 0
    assert sorted(p.name for p in folder.iterdir()) == sorted([WORKBOOK, SUMMARY])
    # Standard output is still one line: the workbook's path.
    assert result.output.strip().splitlines() == [str(folder / WORKBOOK)]
    summary = _summary(folder)
    assert summary["pipeline"] == {"completed": True}
    assert summary["artifacts"][0]["name"] == WORKBOOK
    assert summary["artifacts"][0]["state"] == "available"
    assert summary["policy_outcome"] == "not implemented in this release"
    assert summary["provenance"]["options"] == {
        "deal_name": "DEAL", "aggregation": "by_loan", "min_coverage": None, "dry_run": False,
    }
    assert summary["populations"]["labels"].startswith("observed")


def test_summary_on_a_dry_run(tmp_path: Path) -> None:
    result, folder = _invoke(tmp_path, Pair(), "--dry-run")
    assert result.exit_code == 0
    assert result.output.strip() == ""
    assert [p.name for p in folder.iterdir()] == [SUMMARY]  # no workbook, as before
    summary = _summary(folder)
    assert summary["artifacts"][0] == {"name": None, "kind": "MILAN workbook", "state": "not requested"}
    assert summary["requested_outputs"] == {"workbook": False, "summary": True}


def test_a_failed_run_ends_as_before_and_still_writes_the_summary(tmp_path: Path) -> None:
    pair = Pair().set("RREL30", "abc", record=1).set("RREC13", "-5", record=2)
    loans, collaterals = pair.write(tmp_path)
    with pytest.raises(PipelineDataError) as direct:
        run_pipeline(loans_file_path=loans, collaterals_file_path=collaterals, deal_name="DEAL",
                     output_dir=tmp_path / "direct", verbose=False)
    result, folder = _invoke(tmp_path, pair)
    # The same exception, message and exit status as calling the pipeline alone.
    assert type(result.exception) is PipelineDataError
    assert str(result.exception) == str(direct.value)
    assert result.exception.reason == direct.value.reason == "non_numeric_value"
    assert result.exit_code == 1
    assert result.output.strip() == ""
    assert [p.name for p in folder.iterdir()] == [SUMMARY]
    summary = _summary(folder)
    assert summary["execution"] == "pipeline_failed"
    assert summary["pipeline"]["reason_code"] == "non_numeric_value"
    ids = {f["finding_id"] for f in summary["findings"]}
    assert {"P5:violation:loans:RREL30", "P6(a):violation:collateral:RREC13"} <= ids


def test_unknown_cutoff_in_the_name_when_none_is_established(tmp_path: Path) -> None:
    pair = Pair().set("RREL6", "ND1").set("Pool_Cutoff_Date", "")
    result, folder = _invoke(tmp_path, pair)
    assert result.exit_code == 1  # the pipeline needs the cut-off, as before
    assert [p.name for p in folder.iterdir()] == [
        "unknown-cutoff DEAL Flattened loans and collaterals data quality summary.json"
    ]


def test_an_interrupted_run_writes_no_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted(**_: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_pipeline", interrupted)
    result, folder = _invoke(tmp_path, Pair())
    assert result.exit_code == 1  # click's Abort, as for any Ctrl-C
    assert not folder.exists()

    def exits(**_: object) -> None:
        raise SystemExit(3)

    monkeypatch.setattr(cli, "run_pipeline", exits)
    result, folder = _invoke(tmp_path, Pair())
    assert result.exit_code == 3
    assert not folder.exists()


class _Log:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, object]]] = []

    def error(self, event: str, **fields: object) -> None:
        self.events.append(("error", event, fields))

    def info(self, event: str, **fields: object) -> None:
        self.events.append(("info", event, fields))


def test_a_fault_in_the_assessment_is_logged_once_and_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_: object, **__: object) -> None:
        raise RuntimeError("a bug in a rule")

    log = _Log()
    monkeypatch.setattr(cli, "assess_files", broken)
    monkeypatch.setattr(cli, "log", log)
    result, folder = _invoke(tmp_path, Pair())
    assert result.exit_code == 0 and result.exception is None
    assert result.output.strip().splitlines() == [str(folder / WORKBOOK)]
    assert (folder / WORKBOOK).exists()
    errors = [e for e in log.events if e[0] == "error"]
    assert errors == [("error", "data_quality_assessment_failed",
                       {"step": "assess_files", "error": "RuntimeError('a bug in a rule')"})]
    summary = _summary(folder)
    assert summary["assessment"] == {
        "state": "interrupted", "statement": ASSESSMENT_INTERRUPTED,
        "reason": "RuntimeError: a bug in a rule",
    }
    assert summary["execution"] == "completed"


def test_the_summary_log_event_carries_the_path_and_no_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = _Log()
    monkeypatch.setattr(cli, "log", log)
    _, folder = _invoke(tmp_path, Pair())
    assert log.events == [("info", "data_quality_summary_written", {"path": str(folder / SUMMARY)})]


def test_the_pipelines_exception_is_never_masked_by_a_failed_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def cannot_write(*_: object, **__: object) -> None:
        raise OSError("disk full")

    log = _Log()
    monkeypatch.setattr(cli, "_write_quality_summary", cannot_write)
    monkeypatch.setattr(cli, "log", log)
    result, folder = _invoke(tmp_path, Pair().set("RREL30", "abc", record=1))
    assert type(result.exception) is PipelineDataError  # the pipeline's, unchanged
    assert result.exception.reason == "non_numeric_value"
    assert result.exception.__context__ is None and result.exception.__cause__ is None
    assert result.exit_code == 1
    assert not folder.exists()
    assert [(e[1], e[2]["step"]) for e in log.events] == [
        ("data_quality_assessment_failed", "write_summary")
    ]
    # And when the pipeline succeeds, a failed write still leaves the run a success.
    result, folder = _invoke(tmp_path, Pair())
    assert result.exit_code == 0 and (folder / WORKBOOK).exists()


def test_the_write_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "x.json"
    write_json({"a": 1}, path)
    write_json({"a": 2}, path)  # replaces, as the workbook is replaced
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 2}
    assert [p.name for p in path.parent.iterdir()] == ["x.json"]  # no temporary file left
