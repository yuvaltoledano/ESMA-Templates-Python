"""Tests for Stage-9 / B-5 compose_milan_pool.

Mirrors r_reference/R/milan_mapping.R::map_to_milan() at the assembly
level. The byte-equality contract against the synthetic fixture is
covered by tests/parity/test_parity_synthetic.py; the tests below
pin invariants that survive across changing pipeline outputs:

  - 175-column list-equality (not set-equality, so position errors
    surface as "position N: got X, expected Y" rather than "missing
    somewhere").
  - All-Utf8 output schema (catches Polars silently preserving numeric
    dtypes through with_columns / select if a future helper forgets a
    cast).
  - Final null -> "ND" fill: a deliberately-null source cell that the
    field-level helpers don't otherwise substitute reaches the final
    fill and emits "ND".
  - Column-order mismatch raises (R-repo issue #8 deviation).
  - _ensure_source_columns warning text matches R verbatim.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import polars as pl
import pytest

from esma_milan.pipeline.milan_map import (
    EXPECTED_SOURCE_COLS,
    MILAN_EXPECTED_OUTPUT_COLS,
    _ensure_source_columns,
    compose_milan_pool,
)


def _minimal_input_row() -> dict[str, object]:
    """Build a single-row input frame with every column the composer
    expects, populated with ND tokens by default. Tests override
    individual fields to exercise specific branches.
    """
    row: dict[str, object] = dict.fromkeys(EXPECTED_SOURCE_COLS, "ND")
    # Make the few columns that drive ranking / IDs non-ND so the
    # composer doesn't trip on group-by edge cases.
    row["calc_loan_id"] = "L1"
    row["calc_borrower_id"] = "B1"
    row["calc_main_property_id"] = "P1"
    return row


def _df(rows: list[Mapping[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        [dict(r) for r in rows],
        schema=dict.fromkeys(EXPECTED_SOURCE_COLS, pl.Utf8),
    )


# ---------------------------------------------------------------------------
# 175-column list-equality
# ---------------------------------------------------------------------------


def test_compose_milan_pool_emits_exactly_175_columns_in_canonical_order() -> None:
    """List-equality, position-by-position. Failure surfaces as
    "position N: got X, expected Y" rather than "set mismatch somewhere"."""
    df = compose_milan_pool(_df([_minimal_input_row()]))
    assert list(df.columns) == list(MILAN_EXPECTED_OUTPUT_COLS), (
        f"first divergence at position "
        f"{next(i for i, (a, e) in enumerate(zip(df.columns, MILAN_EXPECTED_OUTPUT_COLS, strict=False)) if a != e)}"
    )
    assert len(df.columns) == 175


# ---------------------------------------------------------------------------
# All-Utf8 schema
# ---------------------------------------------------------------------------


def test_compose_milan_pool_output_schema_is_all_utf8() -> None:
    """Every one of the 175 columns is pl.Utf8.

    Catches refactors where Polars silently preserves numeric / Date /
    Bool dtypes through select(...) without an explicit cast. The
    workbook writer expects Utf8; numeric leakage would corrupt the
    written cells (e.g. a Float64 column would write 17-digit values
    instead of R's %.15g).
    """
    df = compose_milan_pool(_df([_minimal_input_row()]))
    bad = [(col, dtype) for col, dtype in zip(df.columns, df.dtypes, strict=True)
           if dtype != pl.Utf8]
    assert bad == [], (
        f"non-Utf8 columns leaked through compose_milan_pool: {bad[:5]}"
    )


# ---------------------------------------------------------------------------
# Final null -> "ND" fill
# ---------------------------------------------------------------------------


def test_compose_milan_pool_final_fill_replaces_remaining_nulls_with_nd() -> None:
    """Construct a row with deliberate nulls in DIRECT pass-through
    columns (which don't go through field-level ND substitution).
    The final fill should turn those into "ND" string cells.

    The DIRECT pass-throughs that don't sub ND themselves include
    `originator_name`, `rrel30_currency`, `original_principal_balance`,
    `payment_due`, etc. Set them to null; assert the corresponding
    output cells are "ND".
    """
    row = _minimal_input_row()
    row["originator_name"] = None
    row["rrel30_currency"] = None
    row["original_principal_balance"] = None
    row["payment_due"] = None

    df = compose_milan_pool(_df([row]))
    assert df["Originator Identifier"].to_list() == ["ND"]
    assert df["Loan Currency"].to_list() == ["ND"]
    assert df["Loan OB"].to_list() == ["ND"]
    assert df["Periodic Payment"].to_list() == ["ND"]

    # Sanity: nothing in the entire output is null after the fill.
    null_counts = df.null_count().row(0)
    assert sum(null_counts) == 0, (
        f"some columns still have nulls after final fill: "
        f"{[(c, n) for c, n in zip(df.columns, null_counts, strict=True) if n > 0]}"
    )


# ---------------------------------------------------------------------------
# Column-order raise (R-repo issue #8 deviation)
# ---------------------------------------------------------------------------


def test_column_order_validation_raises_with_actionable_message() -> None:
    """If the canonical column list ever drifts from the composer's
    output, ValueError is raised. Mirrors r_reference/R/milan_mapping.R:
    1066-1080 except R warns and continues; we raise. See R-repo issue
    tracker entry #8.

    Test by monkey-patching MILAN_EXPECTED_OUTPUT_COLS with a deliberately
    wrong list; the raise should surface with a "position N: got X,
    expected Y" hint.
    """
    from esma_milan.pipeline import milan_map

    orig = milan_map.MILAN_EXPECTED_OUTPUT_COLS
    bad_list = ("WRONG_FIRST_COLUMN", *orig[1:])
    milan_map.MILAN_EXPECTED_OUTPUT_COLS = bad_list
    try:
        with pytest.raises(ValueError) as exc_info:
            compose_milan_pool(_df([_minimal_input_row()]))
        msg = str(exc_info.value)
        assert "175-column layout" in msg
        assert "position 0" in msg
        assert "WRONG_FIRST_COLUMN" in msg
        assert "Originator Identifier" in msg
    finally:
        milan_map.MILAN_EXPECTED_OUTPUT_COLS = orig


# ---------------------------------------------------------------------------
# _ensure_source_columns warning text matches R verbatim
# ---------------------------------------------------------------------------


def test_ensure_source_columns_warning_text_matches_r_verbatim() -> None:
    """Greps r_reference/R/milan_mapping.R:315-318 for the warning() call:

        warning(
          "MILAN mapping: the following source columns are missing and will be ",
          "set to 'ND': ", paste(missing, collapse = ", ")
        )

    R's `warning(...)` with multiple arguments concatenates without
    separator (paste0 semantic). Final string:
      "MILAN mapping: the following source columns are missing and
       will be set to 'ND': X, Y, Z"
    """
    df = pl.DataFrame({"a": ["1"]})
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        _ensure_source_columns(df, ["a", "X", "Y", "Z"])
    assert len(records) == 1
    msg = str(records[0].message)
    assert msg == (
        "MILAN mapping: the following source columns are missing "
        "and will be set to 'ND': X, Y, Z"
    )


# ---------------------------------------------------------------------------
# Empty-input warning (mirrors milan_mapping.R:394-396)
# ---------------------------------------------------------------------------


def test_compose_milan_pool_warns_on_empty_input() -> None:
    """0-row input -> warns "MILAN mapping: input data frame has 0 rows"."""
    empty_data: dict[str, list[object]] = {col: [] for col in EXPECTED_SOURCE_COLS}
    df = pl.DataFrame(
        empty_data,
        schema=dict.fromkeys(EXPECTED_SOURCE_COLS, pl.Utf8),
    )
    with pytest.warns(UserWarning, match="MILAN mapping: input data frame has 0 rows"):
        out = compose_milan_pool(df)
    # Empty input -> empty output, but still 175 columns.
    assert out.height == 0
    assert list(out.columns) == list(MILAN_EXPECTED_OUTPUT_COLS)
