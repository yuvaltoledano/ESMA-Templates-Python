"""Tests for esma_milan.io_layer.read_csv.read_and_clean.

Mirrors a subset of the R tests in test-utils.R::"read_and_clean..." and
exercises:

- successful read against the staged synthetic loans.csv / collaterals.csv
- ESMA NA tokens (ND, ND1..ND5, NA, "") are converted to nulls
- character_cols schema override forces ID columns to String
- the taxonomy rename is applied to source columns whose name is a key,
  and untouched columns pass through unchanged
- duplicate column names after rename produce a hard error
- a missing file raises FileNotFoundError
- an empty file (header only, no data) raises ValueError
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from esma_milan.io_layer.read_csv import read_and_clean

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = REPO_ROOT / "tests" / "fixtures" / "synthetic"


# ---------------------------------------------------------------------------
# Against the staged synthetic fixture
# ---------------------------------------------------------------------------


def test_reads_synthetic_loans() -> None:
    # Synthetic loans CSV uses already-snake_case headers (no taxonomy
    # rename needed); we still pass the taxonomy and exercise the
    # passthrough path.
    df = read_and_clean(
        SYNTHETIC / "loans.csv",
        taxonomy={},
        character_cols=("RREL2", "RREL3", "RREL4", "RREL5"),
    )
    assert df.height == 8
    # Spot-check a few well-known columns.
    assert "calc_loan_id" not in df.columns  # not yet generated
    assert "original_underlying_exposure_identifier" in df.columns
    assert "current_principal_balance" in df.columns
    assert "account_status" in df.columns
    # Account status is read as String (not categorical, no coercion).
    assert df.schema["account_status"] == pl.String


def test_reads_synthetic_collaterals() -> None:
    df = read_and_clean(
        SYNTHETIC / "collaterals.csv",
        taxonomy={},
        character_cols=("RREC2", "RREC3", "RREC4"),
    )
    assert df.height == 12
    assert "underlying_exposure_identifier" in df.columns
    assert "current_valuation_amount" in df.columns
    assert "property_type" in df.columns


# ---------------------------------------------------------------------------
# NA-token handling
# ---------------------------------------------------------------------------


def test_na_tokens_converted_to_null(tmp_path: Path) -> None:
    """Mirrors test-utils.R::"read_and_clean treats ND and ND3 as NA values".

    All seven configured NA tokens ("", NA, ND, ND1..ND5) plus the
    canonical empty cell must map to Polars null.
    """
    p = tmp_path / "loans.csv"
    p.write_text(
        "id,balance\n"
        "L1,100\n"     # control
        "L2,NA\n"      # NA token
        "L3,ND\n"
        "L4,ND1\n"
        "L5,ND2\n"
        "L6,ND3\n"
        "L7,ND4\n"
        "L8,ND5\n"
        "L9,\n"        # empty
    )
    df = read_and_clean(p, taxonomy={})
    balances = df["balance"].to_list()
    assert balances[0] == 100  # control
    assert all(b is None for b in balances[1:]), balances


# ---------------------------------------------------------------------------
# character_cols schema override
# ---------------------------------------------------------------------------


def test_character_cols_keeps_numeric_looking_ids_as_string(tmp_path: Path) -> None:
    p = tmp_path / "loans.csv"
    p.write_text("RREL2,RREL30\n12345,100\n67890,200\n")
    df = read_and_clean(
        p,
        taxonomy={"RREL2": "Original Underlying Exposure Identifier", "RREL30": "Current Principal Balance"},
        character_cols=("RREL2",),
    )
    # RREL2 was forced to String even though every value is integer-looking.
    assert df.schema["original_underlying_exposure_identifier"] == pl.String
    assert df["original_underlying_exposure_identifier"].to_list() == ["12345", "67890"]
    # RREL30 was inferred as Int64 by Polars' usual rules.
    assert df["current_principal_balance"].to_list() == [100, 200]


# ---------------------------------------------------------------------------
# Taxonomy rename + clean_names
# ---------------------------------------------------------------------------


def test_taxonomy_rename_and_clean_names_applied(tmp_path: Path) -> None:
    p = tmp_path / "loans.csv"
    p.write_text("RREL2,RREL30,SomeOther Column\nL1,100,xx\n")
    df = read_and_clean(
        p,
        taxonomy={
            "RREL2": "Original Underlying Exposure Identifier",
            "RREL30": "Current Principal Balance",
        },
    )
    # RREL2 / RREL30 renamed via taxonomy then cleaned.
    assert "original_underlying_exposure_identifier" in df.columns
    assert "current_principal_balance" in df.columns
    # Untaxonomy'd column passes through clean_names directly.
    assert "someother_column" in df.columns


def test_unmapped_columns_pass_through_clean_names(tmp_path: Path) -> None:
    p = tmp_path / "loans.csv"
    p.write_text("pool_cutoff_date,Some Column\n2024-01-01,x\n")
    df = read_and_clean(p, taxonomy={"DOES_NOT_APPLY": "x"})
    assert df.columns == ["pool_cutoff_date", "some_column"]


# ---------------------------------------------------------------------------
# Duplicate column names hard error
# ---------------------------------------------------------------------------


def test_duplicate_after_rename_raises(tmp_path: Path) -> None:
    """Mirrors r_reference/R/utils.R:104-112 - the taxonomy mapping has two
    source codes pointing at the same target name, producing two columns
    with the same name after rename. This is a critical data-quality
    issue and must halt the pipeline."""
    p = tmp_path / "loans.csv"
    p.write_text("RREL2,RREL3\nA,B\n")
    with pytest.raises(ValueError, match="Duplicated column names"):
        read_and_clean(
            p,
            taxonomy={
                "RREL2": "Loan Identifier",
                "RREL3": "Loan Identifier",  # collision
            },
        )


def test_duplicate_after_clean_names_raises(tmp_path: Path) -> None:
    """Two columns whose clean_names happen to collide (e.g. one already
    snake-cased, one Title Case that snake-cases to the same name)."""
    p = tmp_path / "loans.csv"
    p.write_text("Loan Identifier,loan_identifier\nA,B\n")
    with pytest.raises(ValueError, match="Duplicated column names"):
        read_and_clean(p, taxonomy={})


# ---------------------------------------------------------------------------
# File-level errors
# ---------------------------------------------------------------------------


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_and_clean(tmp_path / "missing.csv", taxonomy={})


def test_empty_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("a,b\n")  # header only
    with pytest.raises(ValueError, match="No data rows"):
        read_and_clean(p, taxonomy={})
