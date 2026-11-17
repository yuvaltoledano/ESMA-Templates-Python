"""The command line writes the report files and changes nothing else."""

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
DETAIL = "2024-06-30 DEAL Flattened loans and collaterals data quality detail.csv"
DROPPED = "2024-06-30 DEAL Flattened loans and collaterals dropped loans.csv"
REPORT = "2024-06-30 DEAL Flattened loans and collaterals data quality report.xlsx"
REPORT_FILES = sorted([SUMMARY, DETAIL, DROPPED, REPORT])


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
    assert sorted(p.name for p in folder.iterdir()) == sorted([WORKBOOK, *REPORT_FILES])
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
    assert sorted(p.name for p in folder.iterdir()) == REPORT_FILES  # no workbook, as before
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
    assert sorted(p.name for p in folder.iterdir()) == REPORT_FILES
    summary = _summary(folder)
    assert summary["execution"] == "pipeline_failed"
    assert summary["pipeline"]["reason_code"] == "non_numeric_value"
    ids = {f["finding_id"] for f in summary["findings"]}
    assert {"P5:violation:loans:RREL30", "P6(a):violation:collateral:RREC13"} <= ids


def test_unknown_cutoff_in_the_name_when_none_is_established(tmp_path: Path) -> None:
    pair = Pair().set("RREL6", "ND1").set("Pool_Cutoff_Date", "")
    result, folder = _invoke(tmp_path, pair)
    assert result.exit_code == 1  # the pipeline needs the cut-off, as before
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        name.replace("2024-06-30", "unknown-cutoff") for name in REPORT_FILES
    )


def test_an_interrupted_run_writes_no_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted(**_: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_pipeline", interrupted)
    result, folder = _invoke(tmp_path, Pair())
    assert result.exit_code == 1  # click's Abort, as for any Ctrl-C
    assert list(folder.iterdir()) == []  # no report, and the working directory is gone

    def exits(**_: object) -> None:
        raise SystemExit(3)

    monkeypatch.setattr(cli, "run_pipeline", exits)
    result, folder = _invoke(tmp_path, Pair())
    assert result.exit_code == 3
    assert list(folder.iterdir()) == []


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
    monkeypatch.setattr(cli, "run_assessment", broken)
    monkeypatch.setattr(cli, "log", log)
    result, folder = _invoke(tmp_path, Pair())
    assert result.exit_code == 0 and result.exception is None
    assert result.output.strip().splitlines() == [str(folder / WORKBOOK)]
    assert (folder / WORKBOOK).exists()
    errors = [e for e in log.events if e[0] == "error"]
    # The event carries the step and the exception's type, never its message.
    assert errors == [("error", "data_quality_assessment_failed",
                       {"step": "assess_files", "error_type": "RuntimeError"})]
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
    assert list(folder.iterdir()) == []
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


# -- the four report files ------------------------------------------------------

# A value put in a text cell. Lower case is a finding (P7(b)), so the value is
# a row of the detail; it must reach the files and never a log.
MARKER = "marker7q"


def _texts(folder: Path) -> dict[str, str]:
    """Everything a person can read in the four files, as text, by file."""
    from openpyxl import load_workbook

    texts = {name: (folder / name).read_text(encoding="utf-8") for name in (SUMMARY, DETAIL, DROPPED)}
    book = load_workbook(folder / REPORT, read_only=True)
    texts[REPORT] = "\n".join(
        str(value) for sheet in book.worksheets for row in sheet.iter_rows(values_only=True)
        for value in row if value is not None
    )
    book.close()
    return texts


@pytest.mark.parametrize("kind", ["normal", "dry run", "pipeline raised"])
def test_the_four_files_on_each_kind_of_run(tmp_path: Path, kind: str) -> None:
    pair = Pair().set("RREL22", MARKER, record=1).set("RREL69", "RDMD", record=2)
    if kind == "pipeline raised":
        pair.set("RREL30", "abc", record=1)
    result, folder = _invoke(tmp_path, pair, *(["--dry-run"] if kind == "dry run" else []))
    assert result.exit_code == (1 if kind == "pipeline raised" else 0)
    present = sorted(p.name for p in folder.iterdir())
    assert present == sorted([*REPORT_FILES, *([WORKBOOK] if kind == "normal" else [])])
    summary = _summary(folder)
    assert summary["schema_version"] == "2" and summary["provenance"]["rule_set_version"] == "2026.11.2"
    files = {entry["kind"]: entry for entry in summary["detail_files"]}
    assert [entry["name"] for entry in summary["detail_files"]] == [DETAIL, DROPPED, REPORT]
    assert {entry["state"] for entry in files.values()} == {"available"}
    for name, entry in ((DETAIL, files["detail"]), (DROPPED, files["dropped loans"]),
                        (REPORT, files["report workbook"])):
        assert entry["size"] == (folder / name).stat().st_size > 0
    # The detail CSV holds every finding row; the workbook's sheet says how many it shows.
    detail_rows = (folder / DETAIL).read_text(encoding="utf-8").count("\n") - 2
    assert files["detail"]["rows"] == detail_rows == sum(f["rows"] for f in summary["findings"])
    assert files["report workbook"]["finding_rows_included"] == detail_rows
    assert files["report workbook"]["sheets"] == [
        "DQ Summary", "DQ Fields", "DQ Columns", "DQ Checks", "DQ Findings", "DQ Rows", "DQ Dropped loans",
    ]
    assert files["dropped loans"]["rows"] == summary["dropped_loans"]["loans"]
    assert summary["dropped_loans"]["loans"] == (1 if kind == "pipeline raised" else 0)
    assert [line["records"] for line in summary["waterfall"]["lines"][:5]] == (
        [2, 1, 1, 1, 0] if kind == "pipeline raised" else [2, 1, 1, 0, 1]
    )
    texts = _texts(folder)
    assert MARKER in texts[DETAIL] and MARKER in texts[REPORT]  # the value is in the detail
    assert DETAIL in texts[REPORT] and DROPPED in texts[REPORT]  # the workbook names the other files
    # Nothing in any file says the pool was blocked, rejected, withheld or eligible.
    for name, text in texts.items():
        lowered = text.lower().replace("accepted alternative", "")
        for word in ("blocked", "rejected", "eligible", "accepted", "withheld", "was stopped"):
            assert word not in lowered, (name, word)
    assert not [p for p in folder.iterdir() if p.name.endswith(".part") or p.name.startswith(".dq-work-")]


def test_a_report_workbook_that_cannot_be_written_leaves_the_other_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A directory stands where the report workbook is to be written, so the
    # writing process cannot put the file there. Its name holds the marker.
    blocked = tmp_path / "out" / "DEAL" / REPORT
    (blocked / MARKER).mkdir(parents=True)
    log = _Log()
    monkeypatch.setattr(cli, "log", log)
    result, folder = _invoke(tmp_path, Pair().set("RREL22", MARKER, record=1))
    assert result.exit_code == 0 and result.exception is None
    assert sorted(p.name for p in folder.iterdir()) == sorted([WORKBOOK, *REPORT_FILES])
    assert blocked.is_dir() and (folder / DETAIL).is_file() and (folder / DROPPED).is_file()
    files = _summary(folder)["detail_files"]
    assert [entry["state"] for entry in files] == ["available", "available", "failed"]
    workbook = files[2]
    assert (workbook["name"], workbook["kind"]) == (REPORT, "report workbook")
    assert workbook["reason"].split(":")[0] in ("IsADirectoryError", "OSError", "PermissionError")
    # One log event, with the step and the exception's type; the message is in the summary only.
    assert log.events == [
        ("error", "data_quality_assessment_failed",
         {"step": "write_report_workbook", "error_type": workbook["reason"].split(":")[0]}),
        ("info", "data_quality_summary_written", {"path": str(folder / SUMMARY)}),
    ]
    assert MARKER not in repr(log.events)


def test_no_cell_value_reaches_the_log_of_a_verbose_run(tmp_path: Path) -> None:
    pair = Pair().set("RREL22", MARKER, record=1).set("RREL30", MARKER, record=2)  # the loader stops
    loans, collaterals = pair.write(tmp_path)
    result = CliRunner().invoke(
        cli.main,
        ["run", "--loans", str(loans), "--collaterals", str(collaterals), "--deal", "DEAL",
         "--output", str(tmp_path / "out"), "--verbose"],
    )
    assert result.exit_code == 1
    events = [line for line in result.output.splitlines() if "data_quality" in line]
    assert len(events) == 1 and "data_quality_summary_written" in events[0]
    assert MARKER not in "\n".join(events)
    assert MARKER in (tmp_path / "out" / "DEAL" / DETAIL).read_text(encoding="utf-8")
