"""Tests for run_pipeline's `on_progress` callback and the Excel row guard.

The callback reports a strictly increasing stage index up to the total
with the labels from PIPELINE_STAGE_LABELS (eleven on a full run, nine
on a dry run). Running without it is unchanged: the parity tests are
that gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import esma_milan.runner as runner_module
from esma_milan.runner import (
    EXCEL_MAX_ROWS,
    PIPELINE_STAGE_LABELS,
    OutputTooLargeError,
    run_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = REPO_ROOT / "tests" / "fixtures" / "synthetic"


def _run(tmp_path: Path, **kwargs: object) -> list[tuple[int, int, str]]:
    seen: list[tuple[int, int, str]] = []
    run_pipeline(
        loans_file_path=SYNTHETIC / "loans.csv",
        collaterals_file_path=SYNTHETIC / "collaterals.csv",
        taxonomy_file_path=SYNTHETIC / "taxonomy.xlsx",
        deal_name="SYN",
        output_dir=tmp_path,
        verbose=False,
        on_progress=lambda i, t, label: seen.append((i, t, label)),
        **kwargs,  # type: ignore[arg-type]
    )
    return seen


def test_labels_are_eleven_and_named_as_documented() -> None:
    assert len(PIPELINE_STAGE_LABELS) == 11
    assert PIPELINE_STAGE_LABELS[0] == "Stage 1: read and clean"
    assert PIPELINE_STAGE_LABELS[6] == "Stage 7: flatten and derived fields"
    assert PIPELINE_STAGE_LABELS[-1] == "Writing workbook"


def test_full_run_reports_every_stage_in_order(tmp_path: Path) -> None:
    seen = _run(tmp_path)
    assert [i for i, _, _ in seen] == list(range(1, 12))
    assert {t for _, t, _ in seen} == {11}
    assert [label for _, _, label in seen] == list(PIPELINE_STAGE_LABELS)


def test_dry_run_reports_nine_stages(tmp_path: Path) -> None:
    seen = _run(tmp_path, dry_run=True)
    assert [i for i, _, _ in seen] == list(range(1, 10))
    assert seen[-1][2] == "Stage 10: execution summary"


def test_callback_defaults_to_none_and_is_optional(tmp_path: Path) -> None:
    result = run_pipeline(
        loans_file_path=SYNTHETIC / "loans.csv",
        collaterals_file_path=SYNTHETIC / "collaterals.csv",
        taxonomy_file_path=SYNTHETIC / "taxonomy.xlsx",
        deal_name="SYN",
        output_dir=tmp_path,
        verbose=False,
    )
    assert result.output_path is not None and result.output_path.exists()


def test_row_guard_raises_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert EXCEL_MAX_ROWS == 1_048_576
    monkeypatch.setattr(runner_module, "EXCEL_MAX_ROWS", 10)
    seen: list[tuple[int, int, str]] = []
    with pytest.raises(OutputTooLargeError) as excinfo:
        run_pipeline(
            loans_file_path=SYNTHETIC / "loans.csv",
            collaterals_file_path=SYNTHETIC / "collaterals.csv",
            taxonomy_file_path=SYNTHETIC / "taxonomy.xlsx",
            deal_name="SYN",
            output_dir=tmp_path,
            verbose=False,
            on_progress=lambda i, t, label: seen.append((i, t, label)),
        )
    err = excinfo.value
    assert isinstance(err, ValueError)
    assert err.limit == 10 and err.rows > 10
    assert err.sheet == "Execution Summary"  # first sheet in workbook order to exceed 10 rows
    assert "1,048,576" not in str(err) and "10 rows" in str(err)
    # Nothing was written and the "Writing workbook" boundary was never reached.
    assert not any(tmp_path.rglob("*.xlsx"))
    assert seen[-1][0] == 10


def test_row_guard_passes_a_sheet_exactly_at_the_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The limit counts the header row: a sheet with (limit - 1) data rows fits."""
    # Execution Summary has 43 rows on the synthetic fixture (+1 header = 44);
    # the largest sheet, Combined flattened pool, has 8 (+1). A limit of 44
    # must pass while 43 must fail.
    monkeypatch.setattr(runner_module, "EXCEL_MAX_ROWS", 44)
    result = run_pipeline(
        loans_file_path=SYNTHETIC / "loans.csv",
        collaterals_file_path=SYNTHETIC / "collaterals.csv",
        taxonomy_file_path=SYNTHETIC / "taxonomy.xlsx",
        deal_name="SYN",
        output_dir=tmp_path / "ok",
        verbose=False,
    )
    assert result.output_path is not None
    monkeypatch.setattr(runner_module, "EXCEL_MAX_ROWS", 43)
    with pytest.raises(OutputTooLargeError):
        run_pipeline(
            loans_file_path=SYNTHETIC / "loans.csv",
            collaterals_file_path=SYNTHETIC / "collaterals.csv",
            taxonomy_file_path=SYNTHETIC / "taxonomy.xlsx",
            deal_name="SYN",
            output_dir=tmp_path / "fail",
            verbose=False,
        )
