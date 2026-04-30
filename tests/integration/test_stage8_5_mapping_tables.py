"""Stage 8.5 integration tests against the staged synthetic fixture.

Pins the synthetic-fixture-specific dedup contracts that the unit-level
tests cover with synthetic in-memory frames. Anchored on actual
synthetic-fixture loan/borrower/property identifiers so any future
fixture regeneration surfaces here.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl
import pytest

from esma_milan.pipeline.classification import run_stage5
from esma_milan.pipeline.enriched import compose_loans_enriched
from esma_milan.pipeline.filters import run_stage2
from esma_milan.pipeline.graph import run_stage4
from esma_milan.pipeline.identifiers import run_stage3
from esma_milan.pipeline.mapping_tables import compose_mapping_tables
from esma_milan.pipeline.stage1 import run_stage1

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = REPO_ROOT / "tests" / "fixtures" / "synthetic"


@pytest.fixture(scope="module")
def synthetic_mapping_tables() -> dict[str, pl.DataFrame]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        s1 = run_stage1(
            loans_path=SYNTHETIC / "loans.csv",
            collaterals_path=SYNTHETIC / "collaterals.csv",
            taxonomy_path=SYNTHETIC / "taxonomy.xlsx",
        )
        s2 = run_stage2(s1.loans, s1.properties)
        s3 = run_stage3(s2.loans, s2.properties)
        s4 = run_stage4(s3.loans, s3.properties)
        s5 = run_stage5(s4)
        le = compose_loans_enriched(s3.loans, s4.loan_groups, s5.classifications)
        return compose_mapping_tables(s3.loans, s3.properties, le)


# ---------------------------------------------------------------------------
# Borrower dedup contract (the explicit dedup test from the user's plan)
# ---------------------------------------------------------------------------


def test_synthetic_shared_borrowers_dedup_loans(
    synthetic_mapping_tables: dict[str, pl.DataFrame],
) -> None:
    """BORIG_03 is shared between LOAN_003 and LOAN_004.
    BORIG_04 is shared between LOAN_005 and LOAN_006.
    Both must emit a single row with two loan_id slots."""
    df = synthetic_mapping_tables["Borrowers to loans"]
    by_borrower = {row[0]: row[1:] for row in df.iter_rows()}
    assert by_borrower["BORIG_03"] == ("NEW_003", "NEW_004")
    assert by_borrower["BORIG_04"] == ("NEW_005", "NEW_006")


def test_synthetic_shared_borrowers_dedup_properties(
    synthetic_mapping_tables: dict[str, pl.DataFrame],
) -> None:
    """BORIG_03's two loans (LOAN_003+004) both reference PROP_004 ->
    distinct() collapses to a single property_id slot.

    BORIG_04's two loans (LOAN_005+006) reference PROP_005 + PROP_006 ->
    two distinct property_id slots.
    """
    df = synthetic_mapping_tables["Borrowers to properties"]
    by_borrower = {row[0]: row[1:] for row in df.iter_rows()}
    assert by_borrower["BORIG_03"] == ("PORIG_04", None), (
        "BORIG_03's two loans both reference PROP_004; the second slot "
        "must be null after dedup, not duplicated."
    )
    assert by_borrower["BORIG_04"] == ("PORIG_05", "PORIG_06")


# ---------------------------------------------------------------------------
# Non-dedup control (per user direction — verify dedup is selective, not
# blanket)
# ---------------------------------------------------------------------------


def test_synthetic_single_loan_borrowers_have_loan_id_1_only(
    synthetic_mapping_tables: dict[str, pl.DataFrame],
) -> None:
    """BORIG_01, _02, _05, _06 each own exactly ONE loan in the synthetic
    fixture. Their rows must have loan_id_1 populated and loan_id_2 null.
    Without this control, a regression that over-deduplicates ALL
    borrowers to a single loan_id would still pass the
    shared-borrower test above."""
    df = synthetic_mapping_tables["Borrowers to loans"]
    by_borrower = {row[0]: row[1:] for row in df.iter_rows()}
    expected_single_loan = {
        "BORIG_01": ("NEW_001", None),
        "BORIG_02": ("NEW_002", None),
        "BORIG_05": ("NEW_007", None),
        "BORIG_06": ("NEW_008", None),
    }
    for borrower, expected in expected_single_loan.items():
        assert by_borrower[borrower] == expected, (
            f"{borrower}: expected {expected}, got {by_borrower[borrower]}"
        )


# ---------------------------------------------------------------------------
# Sheet shape sanity (post-dedup row counts) — guards against regressions
# in the distinct() step
# ---------------------------------------------------------------------------


def test_synthetic_mapping_table_row_counts(
    synthetic_mapping_tables: dict[str, pl.DataFrame],
) -> None:
    """8 loans, 8 unique properties, 6 unique borrowers in the synthetic
    fixture (2 borrowers each own 2 loans, the other 4 own 1 loan)."""
    assert synthetic_mapping_tables["Loans to properties"].height == 8
    assert synthetic_mapping_tables["Properties to loans"].height == 8
    assert synthetic_mapping_tables["Borrowers to loans"].height == 6
    assert synthetic_mapping_tables["Borrowers to properties"].height == 6


def test_synthetic_loans_to_properties_classification_tail_present(
    synthetic_mapping_tables: dict[str, pl.DataFrame],
) -> None:
    """Loans-to-properties is the only mapping table with a classification
    tail (collateral_group_id, cross_collateralized_set, full_set,
    structure_type)."""
    df = synthetic_mapping_tables["Loans to properties"]
    expected_tail = {
        "collateral_group_id",
        "cross_collateralized_set",
        "full_set",
        "structure_type",
    }
    assert expected_tail.issubset(set(df.columns))
    # The other three tables should NOT have these columns.
    for sheet_name in (
        "Properties to loans",
        "Borrowers to loans",
        "Borrowers to properties",
    ):
        df_other = synthetic_mapping_tables[sheet_name]
        assert not (expected_tail & set(df_other.columns)), (
            f"{sheet_name} unexpectedly carries classification tail columns: "
            f"{expected_tail & set(df_other.columns)}"
        )
