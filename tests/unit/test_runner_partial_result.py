"""`run_pipeline` gives an exception that leaves it the stages that
completed, and changes nothing else about the exception."""

from __future__ import annotations

import traceback
from pathlib import Path

import polars as pl
import pytest

from esma_milan import runner
from esma_milan.errors import PipelineDataError
from esma_milan.runner import PipelineResult, run_pipeline

SYNTHETIC = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic"
STAGES = ("stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7")


def _run(loans: Path | None = None) -> PipelineResult:
    return run_pipeline(
        loans_file_path=loans or SYNTHETIC / "loans.csv",
        collaterals_file_path=SYNTHETIC / "collaterals.csv",
        deal_name="D",
        dry_run=True,
        verbose=False,
    )


def _completed(exc: BaseException) -> list[str]:
    partial = exc.partial_result  # type: ignore[attr-defined]
    assert isinstance(partial, PipelineResult)
    # Nothing a completed run alone can state is on a partial result.
    assert partial.output_path is None and partial.execution_summary is None
    assert partial.chosen_aggregation_method is None and partial.workbook_bytes is None
    return [name for name in STAGES if getattr(partial, name) is not None]


@pytest.mark.parametrize("failing", range(1, 8))
def test_a_typed_failure_at_each_stage_carries_exactly_the_stages_before_it(
    monkeypatch: pytest.MonkeyPatch, failing: int
) -> None:
    raised = PipelineDataError("the stage's own message", stage="stage3", reason="no_data_rows")

    def fail(*args: object, **kwargs: object) -> None:
        raise raised

    monkeypatch.setattr(runner, f"run_stage{failing}", fail)
    with pytest.raises(PipelineDataError) as excinfo:
        _run()
    # The same object, with the same arguments, message, stage and reason.
    assert excinfo.value is raised
    assert excinfo.value.args == ("the stage's own message",)
    assert str(excinfo.value) == "the stage's own message"
    assert (excinfo.value.stage, excinfo.value.reason) == ("stage3", "no_data_rows")
    assert _completed(excinfo.value) == list(STAGES[: failing - 1])
    partial = excinfo.value.partial_result  # type: ignore[attr-defined]
    # The record indices exist once Stage 3 has completed.
    assert (partial.loan_record_index is not None) is (failing > 3)
    assert (partial.collateral_record_index is not None) is (failing > 3)
    # The traceback still ends where the exception was raised.
    assert traceback.extract_tb(excinfo.value.__traceback__)[-1].name == "fail"


def test_the_stage_outputs_on_the_partial_result_are_the_stages_own(monkeypatch: pytest.MonkeyPatch) -> None:
    complete = _run()

    def fail(*args: object, **kwargs: object) -> None:
        raise PipelineDataError("m", stage="stage3", reason="no_data_rows")

    monkeypatch.setattr(runner, "run_stage4", fail)
    with pytest.raises(PipelineDataError) as excinfo:
        _run()
    partial = excinfo.value.partial_result  # type: ignore[attr-defined]
    assert complete.stage2 is not None and complete.stage3 is not None
    assert partial.stage2.loans.equals(complete.stage2.loans)
    assert partial.stage3.properties.equals(complete.stage3.properties)
    assert complete.loan_record_index is not None
    assert partial.loan_record_index.equals(complete.loan_record_index)


def test_the_loaders_failure_carries_no_stage(tmp_path: Path) -> None:
    loans = tmp_path / "loans.csv"
    header = (SYNTHETIC / "loans.csv").read_text().split("\n")[0]
    loans.write_text(header + "\n")  # a header and no record
    with pytest.raises(PipelineDataError) as excinfo:
        _run(loans)
    assert excinfo.value.reason == "no_data_rows"
    assert str(excinfo.value) == "ERROR: No data rows found in file: loans.csv"
    assert _completed(excinfo.value) == []


def test_the_runners_own_cutoff_check_carries_every_stage(tmp_path: Path) -> None:
    loans = tmp_path / "loans.csv"
    loans.write_text((SYNTHETIC / "loans.csv").read_text().replace("2024-06-30", "ND"))
    with pytest.raises(PipelineDataError) as excinfo:
        _run(loans)
    assert str(excinfo.value) == "pool_cutoff_date is missing in loans file"
    assert (excinfo.value.stage, excinfo.value.reason) == ("runner", "missing_pool_cutoff_date")
    assert _completed(excinfo.value) == list(STAGES)


def test_an_untyped_error_carries_one_too(monkeypatch: pytest.MonkeyPatch) -> None:
    raised = pl.exceptions.SchemaError("a Polars error from inside Stage 2")

    def fail(*args: object, **kwargs: object) -> None:
        raise raised

    monkeypatch.setattr(runner, "run_stage2", fail)
    with pytest.raises(pl.exceptions.SchemaError) as excinfo:
        _run()
    assert excinfo.value is raised and type(excinfo.value) is pl.exceptions.SchemaError
    assert _completed(excinfo.value) == ["stage1"]


def test_an_interruption_carries_one_and_stays_an_interruption(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "run_stage3", fail)
    with pytest.raises(KeyboardInterrupt) as excinfo:
        _run()
    assert _completed(excinfo.value) == ["stage1", "stage2"]


def test_an_exception_that_refuses_the_attribute_leaves_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    class SealedError(Exception):
        def __setattr__(self, name: str, value: object) -> None:
            raise TypeError("no attribute can be set on this exception")

    raised = SealedError("sealed")

    def fail(*args: object, **kwargs: object) -> None:
        raise raised

    monkeypatch.setattr(runner, "run_stage2", fail)
    with pytest.raises(SealedError) as excinfo:
        _run()
    assert excinfo.value is raised and excinfo.value.args == ("sealed",)
    assert not hasattr(excinfo.value, "partial_result")


def test_a_completed_run_returns_what_it_returned_before() -> None:
    result = _run()
    assert not hasattr(result, "partial_result")
    assert all(getattr(result, name) is not None for name in STAGES)
    assert result.execution_summary is not None
