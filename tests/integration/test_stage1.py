"""Stage 1 integration test against the staged synthetic fixture.

Pins the shape and key properties of the cleaned loans / properties
tables that fall out of Stage 1. Used to lock down filter/parse
behaviour without yet needing a fully-implemented pipeline.

Per the user's directive on Stage 2 (and applied here at Stage 1 too):
the dropped-loan branch coverage lives in tests/unit/test_filters.py
when Stage 2 lands. This integration test exercises the happy path.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from esma_milan.config import (
    LOAN_DATE_COLUMNS,
    PROPERTY_DATE_COLUMNS,
    REQUIRED_LOAN_COLUMNS,
    REQUIRED_PROPERTY_COLUMNS,
)
from esma_milan.pipeline.stage1 import Stage1Output, run_stage1
from esma_milan.runner import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = REPO_ROOT / "tests" / "fixtures" / "synthetic"


@pytest.fixture(scope="module")
def stage1_synthetic() -> Stage1Output:
    return run_stage1(
        loans_path=SYNTHETIC / "loans.csv",
        collaterals_path=SYNTHETIC / "collaterals.csv",
        taxonomy_path=SYNTHETIC / "taxonomy.xlsx",
    )


def test_stage1_loans_shape(stage1_synthetic: Stage1Output) -> None:
    """The synthetic CSV has 8 loans and Stage 1 doesn't drop any of
    them - filtering happens in Stage 2."""
    assert stage1_synthetic.loans.height == 8


def test_stage1_properties_shape(stage1_synthetic: Stage1Output) -> None:
    """12 collateral rows in the synthetic CSV; Stage 1 doesn't filter."""
    assert stage1_synthetic.properties.height == 12


def test_stage1_required_loan_columns_present(stage1_synthetic: Stage1Output) -> None:
    """Stage 1 must validate required columns - if any were missing it
    would have raised."""
    for col in REQUIRED_LOAN_COLUMNS:
        assert col in stage1_synthetic.loans.columns, f"missing required loan col: {col}"


def test_stage1_required_property_columns_present(stage1_synthetic: Stage1Output) -> None:
    for col in REQUIRED_PROPERTY_COLUMNS:
        assert col in stage1_synthetic.properties.columns, (
            f"missing required property col: {col}"
        )


def test_stage1_loan_date_columns_are_date_typed(stage1_synthetic: Stage1Output) -> None:
    """Every present LOAN_DATE_COLUMNS entry must come out as Polars
    Date (or null) after parse_iso_or_excel_date."""
    for col in LOAN_DATE_COLUMNS:
        if col in stage1_synthetic.loans.columns:
            assert stage1_synthetic.loans.schema[col] == pl.Date, (
                f"{col} not Date-typed: got {stage1_synthetic.loans.schema[col]}"
            )


def test_stage1_property_date_columns_are_date_typed(stage1_synthetic: Stage1Output) -> None:
    for col in PROPERTY_DATE_COLUMNS:
        if col in stage1_synthetic.properties.columns:
            assert stage1_synthetic.properties.schema[col] == pl.Date, (
                f"{col} not Date-typed: got {stage1_synthetic.properties.schema[col]}"
            )


def test_stage1_origination_date_round_trip(stage1_synthetic: Stage1Output) -> None:
    """ORIG_001 in the synthetic CSV has origination_date='2019-03-15'.
    After Stage 1 it must surface as a Polars date(2019, 3, 15)."""
    row = stage1_synthetic.loans.filter(
        pl.col("original_underlying_exposure_identifier") == "ORIG_001"
    )
    assert row.height == 1
    assert row[0, "origination_date"] == date(2019, 3, 15)


def test_stage1_pool_cutoff_date_consistent(stage1_synthetic: Stage1Output) -> None:
    """Every loan in the synthetic fixture shares pool_cutoff_date
    2024-06-30. Stage 1 must surface this for the downstream filename
    derivation."""
    cutoffs = stage1_synthetic.loans["pool_cutoff_date"].drop_nulls().unique().to_list()
    assert cutoffs == [date(2024, 6, 30)]


def test_stage1_id_columns_kept_as_string(stage1_synthetic: Stage1Output) -> None:
    """The character_cols schema override must keep ID columns as
    String even when the values are integer-looking."""
    assert stage1_synthetic.loans.schema["original_underlying_exposure_identifier"] == pl.String
    assert stage1_synthetic.loans.schema["new_underlying_exposure_identifier"] == pl.String


def test_run_pipeline_against_synthetic_fixture(tmp_path: Path) -> None:
    """End-to-end smoke test: run_pipeline executes Stage 1, derives the
    filename from pool_cutoff_date, and produces a workbook on disk.
    Sheet content remains a stub (parity gated by SHEET_STATUS until
    Stage 5)."""
    result = run_pipeline(
        loans_file_path=SYNTHETIC / "loans.csv",
        collaterals_file_path=SYNTHETIC / "collaterals.csv",
        taxonomy_file_path=SYNTHETIC / "taxonomy.xlsx",
        deal_name="SYNTHETIC_FIXTURE",
        output_dir=tmp_path,
        verbose=False,
    )
    assert result.output_path is not None
    # Filename must use the pool_cutoff_date from loans, matching R's
    # r_reference/R/pipeline.R:935 pattern.
    assert result.output_path.name == "2024-06-30 SYNTHETIC_FIXTURE Flattened loans and collaterals.xlsx"
    assert result.output_path.exists()
    assert result.stage1 is not None
    assert result.stage1.loans.height == 8


def test_run_pipeline_dry_run_skips_workbook_write(tmp_path: Path) -> None:
    result = run_pipeline(
        loans_file_path=SYNTHETIC / "loans.csv",
        collaterals_file_path=SYNTHETIC / "collaterals.csv",
        taxonomy_file_path=SYNTHETIC / "taxonomy.xlsx",
        deal_name="SYNTHETIC_FIXTURE",
        output_dir=tmp_path,
        dry_run=True,
        verbose=False,
    )
    assert result.output_path is None
    assert result.stage1 is not None
    assert result.stage1.loans.height == 8
