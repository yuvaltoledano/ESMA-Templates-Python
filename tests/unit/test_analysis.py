"""Unit tests for `esma_milan.analysis` and the analysis-mode API path.

Two layers:

  1. Per-function tests against tiny hand-built Polars frames where the
     expected counts/balances/percentages are derived inline from the
     fixture (not copied from a first run). The synthetic fixture is
     small enough that copying observed output would make the tests
     tautological - a bug that changed the function's behaviour would
     simply update both the function and the expected output and slip
     through unnoticed.

  2. One integration test against the real synthetic fixture through
     the HTTP TestClient, asserting response shape + a small subset of
     values derived from the raw CSV.

Plus the analysis_only/dry_run mutual-exclusion 400, and the
missing-column path that exercises the per-stratification error
isolation contract.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from esma_milan.analysis import (
    STRATIFICATIONS,
    run_all_stratifications,
)
from esma_milan.analysis.labels import (
    IR_TYPE_LABELS,
    LOAN_PURPOSE_LABELS,
    OCCUPANCY_LABELS,
    UNK_LABEL,
)
from esma_milan.analysis.stratifications import (
    LTV_LABELS,
    SEASONING_LABELS,
    stratify_current_ltv,
    stratify_geographic,
    stratify_interest_rate_type,
    stratify_loan_purpose,
    stratify_occupancy,
    stratify_seasoning,
)
from esma_milan.api.server import app, limiter

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = REPO_ROOT / "tests" / "fixtures" / "synthetic"

_LOANS_BYTES = (SYNTHETIC / "loans.csv").read_bytes()
_COLLATERALS_BYTES = (SYNTHETIC / "collaterals.csv").read_bytes()
_TAXONOMY_BYTES = (SYNTHETIC / "taxonomy.xlsx").read_bytes()
_XLSX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> None:
    """Reset slowapi's in-memory rate-limit counters between tests.

    Mirrors the same fixture in test_api.py - this file shares the
    process-wide Limiter on `app`, so without a reset the integration
    test interacts with whatever counter state the previous test left.
    """
    limiter.reset()


# ---------------------------------------------------------------------------
# Per-function tests on hand-built frames
# ---------------------------------------------------------------------------


def _frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    """Build a Polars frame from a list of dicts.

    Polars is strict about dtypes when a column has all-null values; the
    explicit constructor here lets each test pin its frame shape locally
    without poisoning the dtype of e.g. an unrelated column.
    """
    return pl.DataFrame(rows)


def test_interest_rate_type_labels_each_code_faithfully() -> None:
    """3 FLIF, 2 FXPR, 1 FLCF, 1 OTHR -> three populated rows for
    those codes plus all the other ESMA codes at count=0. Each row's
    label is the verbatim "CODE — Description" from the ESMA taxonomy
    (no derived categories, no collapsing). The OTHR row carries the
    OTHR data faithfully - it's a valid ESMA code with its own meaning,
    not a catch-all bucket."""
    df = _frame([
        {"interest_rate_type": "FLIF", "current_principal_balance": 100.0},
        {"interest_rate_type": "FLIF", "current_principal_balance": 200.0},
        {"interest_rate_type": "FLIF", "current_principal_balance": 50.0},
        {"interest_rate_type": "FXPR", "current_principal_balance": 300.0},
        {"interest_rate_type": "FXPR", "current_principal_balance": 100.0},
        {"interest_rate_type": "FLCF", "current_principal_balance": 400.0},
        {"interest_rate_type": "OTHR", "current_principal_balance": 25.0},
    ])

    s = stratify_interest_rate_type(df)

    assert s.error is None
    assert s.chart_type == "pie"
    # All 13 ESMA codes always appear, in IR_TYPE_LABELS insertion order.
    assert [r.label for r in s.rows] == list(IR_TYPE_LABELS.values())

    by_label = {r.label: r for r in s.rows}
    total_count = 7
    total_balance = 100 + 200 + 50 + 300 + 100 + 400 + 25

    # Independently derived expected values, keyed by full label.
    assert by_label[IR_TYPE_LABELS["FLIF"]].count == 3
    assert by_label[IR_TYPE_LABELS["FLIF"]].balance == pytest.approx(100 + 200 + 50)
    assert by_label[IR_TYPE_LABELS["FXPR"]].count == 2
    assert by_label[IR_TYPE_LABELS["FXPR"]].balance == pytest.approx(300 + 100)
    assert by_label[IR_TYPE_LABELS["FLCF"]].count == 1
    assert by_label[IR_TYPE_LABELS["FLCF"]].balance == pytest.approx(400.0)
    # OTHR data stays in OTHR (it's a valid ESMA code), NOT a generic
    # catch-all. UNK is reserved for nulls / non-taxonomy codes.
    assert by_label[IR_TYPE_LABELS["OTHR"]].count == 1
    assert by_label[IR_TYPE_LABELS["OTHR"]].balance == pytest.approx(25.0)

    # The codes not present in the fixture are still emitted as
    # count=0 rows (layout-stable across pools).
    for code in ("FXRL", "FINX", "FLFL", "CAPP", "FLCA", "DISC", "SWIC", "OBLS", "MODE"):
        assert by_label[IR_TYPE_LABELS[code]].count == 0
        assert by_label[IR_TYPE_LABELS[code]].balance == pytest.approx(0.0)

    # No UNK row when there are no nulls or off-taxonomy codes.
    assert UNK_LABEL not in by_label

    # Percentages are decimals against the table total.
    assert by_label[IR_TYPE_LABELS["FLIF"]].count_pct == pytest.approx(3 / total_count)
    assert by_label[IR_TYPE_LABELS["FLIF"]].balance_pct == pytest.approx(
        (100 + 200 + 50) / total_balance
    )

    assert s.total.count == total_count
    assert s.total.balance == pytest.approx(total_balance)


def test_categorical_emits_unk_row_only_when_nulls_or_off_taxonomy_present() -> None:
    """Nulls and codes outside the published ESMA taxonomy route to the
    UNK row; the row is emitted only when its count is > 0 so clean
    data leaves the table at the spec-code length. Confirms both the
    appearance condition AND its absence on clean data via a paired
    setup."""
    # Clean: only valid codes, no nulls. UNK row must NOT appear.
    clean = _frame([
        {"interest_rate_type": "FLIF", "current_principal_balance": 100.0},
        {"interest_rate_type": "FXRL", "current_principal_balance": 50.0},
    ])
    clean_labels = [r.label for r in stratify_interest_rate_type(clean).rows]
    assert UNK_LABEL not in clean_labels
    # Spec-code row count is constant regardless of which codes appear.
    assert len(clean_labels) == len(IR_TYPE_LABELS)

    # Dirty: a null and a code outside the taxonomy. Both feed the UNK row.
    dirty = _frame([
        {"interest_rate_type": "FLIF", "current_principal_balance": 100.0},
        {"interest_rate_type": None, "current_principal_balance": 50.0},
        {"interest_rate_type": "ZZZZ", "current_principal_balance": 25.0},
    ])
    dirty_rows = stratify_interest_rate_type(dirty).rows
    dirty_by_label = {r.label: r for r in dirty_rows}
    assert UNK_LABEL in dirty_by_label
    assert dirty_by_label[UNK_LABEL].count == 2
    assert dirty_by_label[UNK_LABEL].balance == pytest.approx(50 + 25)
    # The UNK row is appended after the spec codes.
    assert dirty_rows[-1].label == UNK_LABEL


def test_seasoning_buckets_apply_closed_right_convention() -> None:
    """Per the brief, lower edge exclusive, upper edge inclusive
    (closed="right"); the first bucket includes 0. Test values
    straddle every break boundary: 12 -> first bucket (right edge
    of 12 inclusive); 13 -> second bucket (just over 12); 36 ->
    third bucket (right edge of 36 inclusive); 121 -> last bucket
    (strictly >120). Source column is calc_seasoning in years."""
    df = _frame([
        {"calc_seasoning": 0.0, "current_principal_balance": 10.0},   # -> 0-12
        {"calc_seasoning": 1.0, "current_principal_balance": 10.0},   # 12 mo -> 0-12
        {"calc_seasoning": 13 / 12, "current_principal_balance": 10.0},  # ~13 mo -> 13-24
        {"calc_seasoning": 3.0, "current_principal_balance": 10.0},   # 36 mo -> 25-36
        {"calc_seasoning": 5.0, "current_principal_balance": 10.0},   # 60 mo -> 37-60
        {"calc_seasoning": 10.0, "current_principal_balance": 10.0},  # 120 mo -> 61-120
        {"calc_seasoning": 121 / 12, "current_principal_balance": 10.0},  # >120 -> 120+
    ])

    s = stratify_seasoning(df)
    assert s.error is None
    assert s.chart_type == "bar"
    assert [r.label for r in s.rows] == list(SEASONING_LABELS)

    by_label = {r.label: r for r in s.rows}
    assert by_label["0-12 months"].count == 2     # 0 mo + 12 mo
    assert by_label["13-24 months"].count == 1
    assert by_label["25-36 months"].count == 1
    assert by_label["37-60 months"].count == 1
    assert by_label["61-120 months"].count == 1
    assert by_label["120+ months"].count == 1


def test_seasoning_missing_values_excluded_with_note() -> None:
    """Null seasoning rows are excluded from the bucket rows but
    counted in the pool total; a `note` surfaces the count rather
    than silently dropping them."""
    df = _frame([
        {"calc_seasoning": 2.0, "current_principal_balance": 100.0},
        {"calc_seasoning": None, "current_principal_balance": 50.0},
        {"calc_seasoning": None, "current_principal_balance": 25.0},
    ])
    s = stratify_seasoning(df)

    assert s.note is not None
    assert "2 loans" in s.note
    # Bucket total counts only non-null rows; pool total includes nulls.
    bucket_count_sum = sum(r.count for r in s.rows)
    assert bucket_count_sum == 1
    assert s.total.count == 3
    # Total balance includes the missing-seasoning rows.
    assert s.total.balance == pytest.approx(175.0)


def test_current_ltv_buckets_apply_closed_right_at_decimal_breaks() -> None:
    """LTV buckets at 0.50 / 0.70 / 0.80 / 0.90 / 1.00, closed="right".
    Values at exactly a break go to that bucket; values immediately
    above flip to the next."""
    df = _frame([
        {"calc_current_LTV": 0.30, "current_principal_balance": 1.0},  # <=50
        {"calc_current_LTV": 0.50, "current_principal_balance": 1.0},  # <=50 (right-closed)
        {"calc_current_LTV": 0.65, "current_principal_balance": 1.0},  # 50-70
        {"calc_current_LTV": 0.70, "current_principal_balance": 1.0},  # 50-70 (right-closed)
        {"calc_current_LTV": 0.75, "current_principal_balance": 1.0},  # 70-80
        {"calc_current_LTV": 0.85, "current_principal_balance": 1.0},  # 80-90
        {"calc_current_LTV": 0.95, "current_principal_balance": 1.0},  # 90-100
        {"calc_current_LTV": 1.05, "current_principal_balance": 1.0},  # >100
    ])

    s = stratify_current_ltv(df)
    assert [r.label for r in s.rows] == list(LTV_LABELS)
    by_label = {r.label: r for r in s.rows}
    assert by_label["<=50%"].count == 2
    assert by_label["50-70%"].count == 2
    assert by_label["70-80%"].count == 1
    assert by_label["80-90%"].count == 1
    assert by_label["90-100%"].count == 1
    assert by_label[">100%"].count == 1


def test_geographic_top_n_then_other() -> None:
    """First 10 regions by current balance appear individually;
    the rest collapse into a single "Other" row."""
    rows: list[dict[str, object]] = []
    # 12 distinct regions, balances 100, 200, ..., 1200 (so order by
    # balance descending is REG-12, REG-11, ..., REG-01).
    for i in range(1, 13):
        rows.append({
            "geographic_region_collateral": f"REG-{i:02d}",
            "current_principal_balance": float(i * 100),
        })
    df = _frame(rows)
    s = stratify_geographic(df)

    # 10 top rows + 1 Other row = 11.
    assert len(s.rows) == 11
    assert s.rows[0].label == "REG-12"
    assert s.rows[0].balance == pytest.approx(1200.0)
    # The two smallest (REG-01, REG-02) get rolled into Other.
    other_row = s.rows[-1]
    assert other_row.label == "Other"
    assert other_row.count == 2
    assert other_row.balance == pytest.approx(100.0 + 200.0)


def test_geographic_nulls_route_to_other() -> None:
    """Null region rows merge into the Other bucket with any tail
    overflow, rather than appearing as a NULL row in the table."""
    df = _frame([
        {"geographic_region_collateral": "NL-NH", "current_principal_balance": 100.0},
        {"geographic_region_collateral": None, "current_principal_balance": 25.0},
    ])
    s = stratify_geographic(df)
    by_label = {r.label: r for r in s.rows}
    assert "NL-NH" in by_label
    assert "Other" in by_label
    assert by_label["Other"].count == 1
    assert by_label["Other"].balance == pytest.approx(25.0)


def test_loan_purpose_labels_each_code_faithfully() -> None:
    """Each ESMA purpose code appears as its own row with the
    canonical "CODE — Description" label. No collapsing of RMRT and
    RMEQ into a "Refinance" bucket - they're distinct ESMA codes and
    surface distinctly."""
    df = _frame([
        {"purpose": "PURC", "current_principal_balance": 100.0},
        {"purpose": "RMRT", "current_principal_balance": 50.0},
        {"purpose": "RMEQ", "current_principal_balance": 25.0},
        {"purpose": "EQRE", "current_principal_balance": 30.0},
        {"purpose": "CNST", "current_principal_balance": 40.0},
        {"purpose": "DCON", "current_principal_balance": 10.0},
    ])
    s = stratify_loan_purpose(df)

    # All 13 ESMA purpose codes always emitted, in taxonomy order.
    assert [r.label for r in s.rows] == list(LOAN_PURPOSE_LABELS.values())

    by_label = {r.label: r for r in s.rows}
    assert by_label[LOAN_PURPOSE_LABELS["PURC"]].count == 1
    # RMRT and RMEQ are distinct rows now, NOT collapsed.
    assert by_label[LOAN_PURPOSE_LABELS["RMRT"]].count == 1
    assert by_label[LOAN_PURPOSE_LABELS["RMRT"]].balance == pytest.approx(50.0)
    assert by_label[LOAN_PURPOSE_LABELS["RMEQ"]].count == 1
    assert by_label[LOAN_PURPOSE_LABELS["RMEQ"]].balance == pytest.approx(25.0)
    assert by_label[LOAN_PURPOSE_LABELS["EQRE"]].count == 1
    assert by_label[LOAN_PURPOSE_LABELS["CNST"]].count == 1
    assert by_label[LOAN_PURPOSE_LABELS["DCON"]].count == 1

    # The codes added to the label map for taxonomy completeness but
    # absent from the fixture still show as count=0 rows.
    for code in ("RENV", "BSFN", "CMRT", "IMRT", "RGBY", "GSPL", "OTHR"):
        assert by_label[LOAN_PURPOSE_LABELS[code]].count == 0


def test_occupancy_labels_each_code_faithfully() -> None:
    """POWN, TLET, HOLD, FOWN, OTHR each appear as their own row -
    no derived "Investment"/"Second Home"/"Owner-Occupied" categories.
    POWN ("Partially Owner Occupied") is its own row, not merged into
    a broader investment bucket."""
    df = _frame([
        {"occupancy_type": "FOWN", "current_principal_balance": 100.0},
        {"occupancy_type": "FOWN", "current_principal_balance": 200.0},
        {"occupancy_type": "HOLD", "current_principal_balance": 50.0},
        {"occupancy_type": "TLET", "current_principal_balance": 75.0},
        {"occupancy_type": "POWN", "current_principal_balance": 25.0},
    ])
    s = stratify_occupancy(df)

    assert [r.label for r in s.rows] == list(OCCUPANCY_LABELS.values())

    by_label = {r.label: r for r in s.rows}
    assert by_label[OCCUPANCY_LABELS["FOWN"]].count == 2
    assert by_label[OCCUPANCY_LABELS["FOWN"]].balance == pytest.approx(300.0)
    assert by_label[OCCUPANCY_LABELS["HOLD"]].count == 1
    # POWN and TLET are distinct rows now, NOT merged into Investment.
    assert by_label[OCCUPANCY_LABELS["POWN"]].count == 1
    assert by_label[OCCUPANCY_LABELS["POWN"]].balance == pytest.approx(25.0)
    assert by_label[OCCUPANCY_LABELS["TLET"]].count == 1
    assert by_label[OCCUPANCY_LABELS["TLET"]].balance == pytest.approx(75.0)
    assert by_label[OCCUPANCY_LABELS["OTHR"]].count == 0


# ---------------------------------------------------------------------------
# Missing-column error isolation
# ---------------------------------------------------------------------------


def test_missing_column_returns_error_in_one_strat_others_succeed() -> None:
    """The brief specifies: if a stratification can't be computed
    (e.g. column missing on this particular dataset), the corresponding
    entry should carry `error: "..."` rather than blow up the whole
    response. Build a frame with every column except `occupancy_type`
    and assert that only the occupancy entry has `error` populated."""
    df = _frame([{
        "interest_rate_type": "FLIF",
        "purpose": "PURC",
        "geographic_region_collateral": "NL-NH",
        "calc_seasoning": 2.0,
        "calc_current_LTV": 0.65,
        "current_principal_balance": 100.0,
        # NOTE: no occupancy_type column.
    }])

    out = run_all_stratifications(df)

    assert set(out.keys()) == set(STRATIFICATIONS.keys())
    assert out["occupancy"].error is not None
    assert "occupancy_type" in out["occupancy"].error
    # Everything else still produced rows.
    for key in ("interest_rate_type", "seasoning", "current_ltv", "geographic", "loan_purpose"):
        assert out[key].error is None, f"{key} unexpectedly errored"
        assert len(out[key].rows) > 0, f"{key} unexpectedly produced no rows"


# ---------------------------------------------------------------------------
# HTTP integration
# ---------------------------------------------------------------------------


client = TestClient(app)


def _process_files() -> dict[str, tuple[str, bytes, str]]:
    """Files mapping for a synthetic-fixture /api/process request.

    Includes the taxonomy upload because the r_reference submodule
    (which holds the default taxonomy) may not be initialised in every
    test environment - shipping the synthetic taxonomy keeps the test
    self-contained.
    """
    return {
        "loans": ("loans.csv", _LOANS_BYTES, "text/csv"),
        "collaterals": ("collaterals.csv", _COLLATERALS_BYTES, "text/csv"),
        "taxonomy": ("taxonomy.xlsx", _TAXONOMY_BYTES, _XLSX_MEDIA_TYPE),
    }


def test_analysis_only_returns_full_shape_against_synthetic_fixture() -> None:
    """POST /api/process with `analysis_only=true` returns a JSON
    AnalysisResponse with all six stratification keys populated.

    Asserts only the shape + invariants derivable from the raw CSV
    (loan_count > 0, summary positive, six keys present, per-row total
    counts equal between strats since every cut sees the same pool).
    Specific bucket values aren't pinned here: the per-function tests
    above lock those.
    """
    response = client.post(
        "/api/process",
        files=_process_files(),
        data={"deal_name": "SYNTHETIC_FIXTURE", "analysis_only": "true"},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/json")

    body = response.json()
    assert body["deal_name"] == "SYNTHETIC_FIXTURE"

    summary = body["summary"]
    assert summary["loan_count"] > 0
    assert summary["total_current_balance"] > 0
    assert summary["chosen_aggregation"] in ("by_loan", "by_group")

    expected_keys = {
        "interest_rate_type",
        "seasoning",
        "current_ltv",
        "geographic",
        "loan_purpose",
        "occupancy",
    }
    assert set(body["stratifications"].keys()) == expected_keys

    # Every stratification on the same pool sees the same loan count.
    # Categorical strats route nulls to "Other" so row sum == loan_count;
    # bucketed strats may exclude rows with null source values from the
    # bucket rows but the total still equals the pool size.
    for key, strat in body["stratifications"].items():
        assert strat["error"] is None, f"{key} errored: {strat['error']}"
        assert strat["total"]["count"] == summary["loan_count"], (
            f"{key} total count {strat['total']['count']} != pool loan_count {summary['loan_count']}"
        )


def test_analysis_only_response_includes_execution_summary() -> None:
    """The analysis response carries the Execution Summary (Sheet 1) as
    a list of `{label, value}` rows, with pre-formatted R-faithful
    strings (currencies as "1,234,567.89", percentages as "12.34%").

    Asserts:
      - row count = 38 base metrics + N per-structure-type breakdown
        rows, where N is independently derived from the synthetic
        loans CSV's account_status filter (Stage 2 keeps ARRE/PERF/
        RARR/RNAR), then by counting distinct structure types in the
        resulting groups. Hard-coded here as 5 (synthetic fixture is
        constructed with one of each structure type 1-5);
      - "Deal Name" row reflects the request's deal_name verbatim;
      - "Current Balance" row equals the independently-computed
        formatted sum of current_principal_balance on the loans
        passing Stage 2's active-status filter.
    """
    response = client.post(
        "/api/process",
        files=_process_files(),
        data={"deal_name": "EXEC_SUMMARY_TEST", "analysis_only": "true"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "execution_summary" in body

    rows = body["execution_summary"]
    by_label = {row["label"]: row["value"] for row in rows}

    # 38 base metric rows + 5 structure-type breakdown rows
    # (synthetic fixture is built with one group of each type 1-5;
    # the per-structure-type row count is the only variable component
    # of the row total).
    assert len(rows) == 38 + 5

    # Deal name flows through verbatim - exercises the label-passthrough
    # path and is the only row that depends on request input.
    assert by_label["Deal Name"] == "EXEC_SUMMARY_TEST"

    # "Current Balance" is the formatted sum of current_principal_balance
    # over Stage-2-active loans. Independently computed from the raw CSV
    # here (sum on all 8 rows since the synthetic fixture's account_status
    # is all ARRE/PERF - i.e. all active), then formatted with the same
    # R-faithful `_fmt_comma` helper the production code uses.
    raw_loans = pl.read_csv(SYNTHETIC / "loans.csv")
    active = raw_loans.filter(
        pl.col("account_status").is_in(["ARRE", "PERF", "RARR", "RNAR"])
    )
    total_cb = float(active["current_principal_balance"].sum() or 0.0)
    expected_cb_str = f"{total_cb:,.2f}"
    assert by_label["Current Balance"] == expected_cb_str


def test_analysis_only_with_dry_run_is_400_mutually_exclusive() -> None:
    """Per the agreed contract: analysis_only=true AND dry_run=true
    is a user error (400 validation_error) since the two intents are
    incompatible. The error names both fields so the client can fix it."""
    response = client.post(
        "/api/process",
        files=_process_files(),
        data={
            "deal_name": "SYNTHETIC_FIXTURE",
            "analysis_only": "true",
            "dry_run": "true",
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "validation_error"
    assert "mutually exclusive" in body["message"]
    assert body["details"]["fields"] == ["analysis_only", "dry_run"]
