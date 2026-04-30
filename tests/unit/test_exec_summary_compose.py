"""Tests for Stage-10 / C-2 compose_execution_summary.

Mirrors r_reference/R/pipeline.R:625-895 at the assembly level.
Synthetic byte-equal parity is the contract; the tests below pin
invariants that survive across changing pipeline outputs:

  - Total row count = 38 base + n_structure_types breakdown rows
    (defensive count check before list-equality so failures debug
    faster).
  - 38 base-row Metric labels match the canonical list in order
    (list-equality, not set-equality).
  - All-Utf8 schema (both columns).
  - Specific fixture-derived values pinned: Deal Name, Pool Cut-off
    Date, Number Of Borrowers, Original Balance, Effective Number
    of Borrowers, WA Interest Rate, plus the structure-breakdown
    rows.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import openpyxl
import polars as pl
import pytest

from esma_milan.pipeline.exec_summary import (
    _BASE_METRIC_LABELS,
    compose_execution_summary,
)
from esma_milan.runner import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "synthetic"


@pytest.fixture(scope="module")
def synthetic_summary_df(tmp_path_factory: pytest.TempPathFactory) -> pl.DataFrame:
    """Run the pipeline once and pull the assembled summary frame from
    the runner output. We re-load via openpyxl rather than re-composing
    so the test exercises the actual writer path (rows + cell values
    that hit the workbook), not just the in-memory composer output.
    """
    out_dir = tmp_path_factory.mktemp("exec_summary_run")
    result = run_pipeline(
        loans_file_path=SYNTHETIC_FIXTURE / "loans.csv",
        collaterals_file_path=SYNTHETIC_FIXTURE / "collaterals.csv",
        taxonomy_file_path=SYNTHETIC_FIXTURE / "taxonomy.xlsx",
        deal_name="SYNTHETIC_FIXTURE",
        output_dir=out_dir,
        verbose=False,
    )
    assert result.output_path is not None
    wb = openpyxl.load_workbook(result.output_path, read_only=True, data_only=True)
    ws = wb["Execution Summary"]
    ws.reset_dimensions()
    rows = list(ws.iter_rows(values_only=True))
    return pl.DataFrame(
        {
            "Metric": [str(r[0]) for r in rows[1:]],
            "Value": [str(r[1]) for r in rows[1:]],
        },
        schema={"Metric": pl.Utf8, "Value": pl.Utf8},
    )


# ---------------------------------------------------------------------------
# Defensive row-count check (pin #5 from the plan)
# ---------------------------------------------------------------------------


def test_execution_summary_row_count_is_38_plus_structure_breakdown(
    synthetic_summary_df: pl.DataFrame,
) -> None:
    """Defensive total-row assertion.

    "Expected 43 rows, got 39" debugs faster than per-row list diff.
    Synthetic fixture has 5 structure types -> 38 base + 5 = 43 data
    rows (excluding header).
    """
    n_structure_types = sum(
        1 for m in synthetic_summary_df["Metric"].to_list()
        if m.startswith("Loan Count: ")
    )
    assert n_structure_types == 5
    assert synthetic_summary_df.height == 38 + n_structure_types == 43


# ---------------------------------------------------------------------------
# 38 base-row Metric labels in order
# ---------------------------------------------------------------------------


def test_base_metric_labels_match_canonical_order(
    synthetic_summary_df: pl.DataFrame,
) -> None:
    """List-equality on the 38 base rows. Failure surfaces as
    "position N: got X, expected Y" via the pytest list-diff."""
    base_metrics = synthetic_summary_df["Metric"].to_list()[:38]
    assert base_metrics == list(_BASE_METRIC_LABELS)


# ---------------------------------------------------------------------------
# All-Utf8 schema
# ---------------------------------------------------------------------------


def test_compose_execution_summary_output_schema_is_all_utf8() -> None:
    """compose_execution_summary returns a 2-col Utf8 frame so the
    workbook writer's openpyxl native-serialisation produces strings
    in every cell - matching R's openxlsx output."""
    # Build a minimal MILAN-shaped input so the composer runs.
    milan = pl.DataFrame(
        {col: ["ND"] for col in (
            "Loan OB", "Loan CB", "External Prior Ranks CB",
            "Pari Passu Ranking Loans (Not In Pool)",
            "Interest Rate", "Months In Arrears",
            "Borrower Identifier", "Property Identifier", "Loan Identifier",
            "Interest Rate Type",
            "Interest Reset Date", "Origination Date", "Maturity Date",
            "Principal Payment Type", "Loan Purpose", "Employment Type",
            "Borrower Type",
            "Additional data 9 - calc_structure_type",
        )},
        schema=dict.fromkeys(("Loan OB", "Loan CB", "External Prior Ranks CB", "Pari Passu Ranking Loans (Not In Pool)", "Interest Rate", "Months In Arrears", "Borrower Identifier", "Property Identifier", "Loan Identifier", "Interest Rate Type", "Interest Reset Date", "Origination Date", "Maturity Date", "Principal Payment Type", "Loan Purpose", "Employment Type", "Borrower Type", "Additional data 9 - calc_structure_type"), pl.Utf8),
    )
    loans_enriched = pl.DataFrame({"calc_borrower_id": ["B1"]})
    properties_enriched = pl.DataFrame({"calc_property_id": ["P1"]})
    combined_flattened = pl.DataFrame(
        {"calc_current_LTV": [0.5], "calc_original_LTV": [0.7]},
    )
    out = compose_execution_summary(
        deal_name="X",
        chosen_aggregation="by_loan",
        pool_cutoff_date=date(2024, 6, 30),
        milan_pool=milan,
        loans_enriched=loans_enriched,
        properties_enriched=properties_enriched,
        combined_flattened=combined_flattened,
    )
    assert out.schema == {"Metric": pl.Utf8, "Value": pl.Utf8}


# ---------------------------------------------------------------------------
# Specific fixture-pinned values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric,expected_value",
    [
        ("Deal Name", "SYNTHETIC_FIXTURE"),
        ("Pool Cut-off Date", "2024-06-30"),
        ("Aggregation Method", "by_loan"),
        ("Original Balance", "1,865,000.00"),
        ("Current Balance", "1,550,000.00"),
        ("External Prior Ranks", "100,000.00"),
        ("External Equal Ranks", "90,000.00"),
        ("Number Of Borrowers", "6"),
        ("Number Of Complete Loans", "5"),
        ("Number Of Loan Parts", "8"),
        ("Number Of Properties", "8"),
        ("Top 20 Borrower Exposure", "100.00%"),
        ("Effective Number of Borrowers", "5.55"),
        ("WA Interest Rate", "2.65%"),
        ("Maximum Maturity Date", "2052-04-18"),
        ("WA Original LTV", "74.39%"),
        ("WA Current LTV", "61.62%"),
        ("Loan Count: 1: one loan → one property", "1"),
        ("Loan Count: 5: Cross-collateralised set", "2"),
    ],
)
def test_synthetic_summary_value_pinned(
    synthetic_summary_df: pl.DataFrame,
    metric: str,
    expected_value: str,
) -> None:
    """Spot-checks across the metric set:
      - DIRECT (Deal Name, Pool Cut-off Date, Aggregation Method)
      - SUM (External Prior Ranks - tests _milan_as_num ND-handling)
      - SUM (External Equal Ranks - tests _milan_as_num "0"-handling
            via the asymmetric default; #9 anchor)
      - COUNT (Number Of Borrowers, Number Of Complete Loans, etc.)
      - top-20 / effective borrowers
      - WA Interest Rate (weighted_mean)
      - max maturity date
      - WA LTVs (weighted_mean using combined_flattened, not MILAN)
      - structure breakdown rows
    """
    df = synthetic_summary_df.filter(pl.col("Metric") == metric)
    assert df.height == 1, f"metric {metric!r} not found in summary"
    actual = df["Value"][0]
    assert actual == expected_value, (
        f"metric {metric!r}: expected {expected_value!r}, got {actual!r}"
    )
