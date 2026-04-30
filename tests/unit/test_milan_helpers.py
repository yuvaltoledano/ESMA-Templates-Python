"""Tests for Stage-9 / B-1 MILAN helpers.

Mirrors r_reference/R/milan_mapping.R helpers:

  - safe_as_num               (lines 332-335)
  - divide_by_100             (lines 343-346)
  - code_map_lookup           (lines 96-104)
  - lookup_base_index         (lines 350-378)
  - ensure_source_columns     (lines 312-324)

Plus the MILAN_CODE_MAPS / MILAN_BASE_INDEX_TENOR code-map constants.

The full-helper round-trip exercises every entry in every map so any
future drift between this module and the R source is caught immediately.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence

import polars as pl
import pytest

from esma_milan.pipeline.milan_map import (
    MILAN_BASE_INDEX_TENOR,
    MILAN_CODE_MAPS,
    _code_map_lookup_expr,
    _divide_by_100_expr,
    _ensure_source_columns,
    _lookup_base_index_expr,
    _safe_as_num_expr,
)


def _apply(
    expr_factory: Callable[[pl.Expr], pl.Expr],
    values: Sequence[object | None],
) -> list[object]:
    df = pl.DataFrame({"x": list(values)}, schema={"x": pl.Utf8})
    return df.select(expr_factory(pl.col("x")).alias("out"))["out"].to_list()


# ---------------------------------------------------------------------------
# _safe_as_num_expr
# ---------------------------------------------------------------------------


def test_safe_as_num_parses_numeric_strings() -> None:
    out = _apply(_safe_as_num_expr, ["1", "2.5", "-3.0", "100000"])
    assert out == [1.0, 2.5, -3.0, 100000.0]


def test_safe_as_num_returns_null_for_nd_tokens() -> None:
    out = _apply(_safe_as_num_expr, ["ND", "ND1", "ND5", "NA", "", "  "])
    assert out == [None] * 6


def test_safe_as_num_returns_null_for_unparseable() -> None:
    out = _apply(_safe_as_num_expr, ["abc", "1.2.3"])
    assert out == [None, None]


def test_safe_as_num_passes_through_null() -> None:
    out = _apply(_safe_as_num_expr, [None, "5"])
    assert out == [None, 5.0]


# ---------------------------------------------------------------------------
# _divide_by_100_expr
# ---------------------------------------------------------------------------


def test_divide_by_100_basic() -> None:
    out = _apply(_divide_by_100_expr, ["12.5", "100", "0"])
    assert out == ["0.125", "1.0", "0.0"]


def test_divide_by_100_nd_passes_through() -> None:
    out = _apply(_divide_by_100_expr, ["ND", "ND3", "", None])
    assert out == ["ND", "ND", "ND", "ND"]


def test_divide_by_100_unparseable_becomes_nd() -> None:
    out = _apply(_divide_by_100_expr, ["abc"])
    assert out == ["ND"]


# ---------------------------------------------------------------------------
# _code_map_lookup_expr
# ---------------------------------------------------------------------------


def test_code_map_lookup_known_keys_round_trip() -> None:
    """Every key in every map must round-trip to its mapped value."""
    for map_name, mapping in MILAN_CODE_MAPS.items():
        keys = list(mapping.keys())
        expected = [mapping[k] for k in keys]

        def factory(c: pl.Expr, m: dict[str, str] = mapping) -> pl.Expr:
            return _code_map_lookup_expr(c, m)

        out = _apply(factory, keys)
        assert out == expected, f"map {map_name!r} round-trip mismatch"


def test_code_map_lookup_unknown_becomes_nd() -> None:
    mapping = MILAN_CODE_MAPS["origination_channel"]
    out = _apply(
        lambda c: _code_map_lookup_expr(c, mapping),
        ["UNKNOWN_CODE", "XXXX", "999"],
    )
    assert out == ["ND", "ND", "ND"]


def test_code_map_lookup_nd_tokens_become_nd() -> None:
    mapping = MILAN_CODE_MAPS["origination_channel"]
    out = _apply(
        lambda c: _code_map_lookup_expr(c, mapping),
        ["ND", "ND1", "ND5", "NA", "", "  ", None],
    )
    assert out == ["ND"] * 7


# ---------------------------------------------------------------------------
# _lookup_base_index_expr
# ---------------------------------------------------------------------------


def _apply_two(idx: Sequence[str | None], ten: Sequence[str | None]) -> list[str]:
    df = pl.DataFrame(
        {"i": list(idx), "t": list(ten)},
        schema={"i": pl.Utf8, "t": pl.Utf8},
    )
    return df.select(
        _lookup_base_index_expr(pl.col("i"), pl.col("t")).alias("out")
    )["out"].to_list()


def test_base_index_libo_euri_two_key_full_table() -> None:
    """Every (LIBO/EURI, tenor) pair in the table maps to its code."""
    pairs = [
        ("LIBO", "MNTH", "1"), ("LIBO", "QUTR", "3"),
        ("LIBO", "SEMI", "5"), ("LIBO", "YEAR", "7"),
        ("EURI", "MNTH", "2"), ("EURI", "QUTR", "4"),
        ("EURI", "SEMI", "6"), ("EURI", "YEAR", "8"),
    ]
    idx = [p[0] for p in pairs]
    ten = [p[1] for p in pairs]
    expected = [p[2] for p in pairs]
    assert _apply_two(idx, ten) == expected
    # Sanity: the constant table must enumerate exactly these 8 entries.
    assert set(MILAN_BASE_INDEX_TENOR.keys()) == {f"{p[0]}|{p[1]}" for p in pairs}


def test_base_index_libo_unknown_tenor_is_nd_not_single_key_fallback() -> None:
    """LIBO with an unknown tenor stays in the two-key branch and emits "ND".

    Critical: do NOT fall through to base_index_single["LIBO"] - that would
    resolve to "13" (since LIBO is in the LIBO/EURI short-circuit set, not
    the single-key map, but defensively the test pins the contract).
    """
    assert _apply_two(["LIBO"], ["XXXX"]) == ["ND"]
    assert _apply_two(["EURI"], ["BAD"]) == ["ND"]
    assert _apply_two(["LIBO"], ["ND"]) == ["ND"]
    assert _apply_two(["LIBO"], [None]) == ["ND"]


def test_base_index_single_key_lookup_for_non_libo_euri() -> None:
    # Single-key map: every key resolves to its code.
    single_map = MILAN_CODE_MAPS["base_index_single"]
    keys = list(single_map.keys())
    expected = [single_map[k] for k in keys]
    out = _apply_two(keys, [None] * len(keys))
    assert out == expected


def test_base_index_unknown_non_libo_euri_becomes_nd_default() -> None:
    """Non-LIBO/EURI codes that aren't in the single-key map fall to "ND" via default.

    Note: the single-key map *includes* an "OTHR" -> "13" entry, so the
    only way to get "ND" out of this branch is a code that isn't in the
    map at all (e.g. "ZZZZ").
    """
    assert _apply_two(["ZZZZ"], ["MNTH"]) == ["ND"]


def test_base_index_nd_index_is_nd_regardless_of_tenor() -> None:
    out = _apply_two(["ND", "ND1", "", "  ", None], ["MNTH"] * 5)
    assert out == ["ND"] * 5


# ---------------------------------------------------------------------------
# _ensure_source_columns
# ---------------------------------------------------------------------------


def test_ensure_source_columns_no_op_when_all_present() -> None:
    df = pl.DataFrame({"a": ["1"], "b": ["2"]})
    out = _ensure_source_columns(df, ["a", "b"])
    assert out.columns == ["a", "b"]
    assert out.height == 1


def test_ensure_source_columns_warns_and_fills_missing() -> None:
    df = pl.DataFrame({"a": ["1", "2"]})
    with pytest.warns(UserWarning) as records:
        out = _ensure_source_columns(df, ["a", "b", "c"])

    assert len(records) == 1
    msg = str(records[0].message)
    # Verbatim text from r_reference/R/milan_mapping.R:316-318:
    assert (
        msg == "MILAN mapping: the following source columns are missing "
        "and will be set to 'ND': b, c"
    )

    # Filled columns are string "ND" everywhere.
    assert out.columns == ["a", "b", "c"]
    assert out["b"].to_list() == ["ND", "ND"]
    assert out["c"].to_list() == ["ND", "ND"]


def test_ensure_source_columns_preserves_order_in_missing_list() -> None:
    """Missing columns are reported in the order they appear in `expected`,
    not the order the original frame is missing them in."""
    df = pl.DataFrame({"a": ["1"]})
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        _ensure_source_columns(df, ["x", "a", "y", "z"])
    assert len(records) == 1
    assert "x, y, z" in str(records[0].message)
