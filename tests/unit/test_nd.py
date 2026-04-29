"""Tests for esma_milan.nd.is_nd().

Mirrors the contract of r_reference/R/milan_mapping.R:19-22 (is_nd) plus
the specific assertions in test-milan-mapping.R::"is_nd detects all ND
codes and NA values".
"""

from __future__ import annotations

import math

from esma_milan.nd import is_nd


def test_recognises_all_nd_codes() -> None:
    for v in ("ND", "ND1", "ND2", "ND3", "ND4", "ND5", "NA"):
        assert is_nd(v), f"{v!r} should be ND"


def test_recognises_python_none_and_nan() -> None:
    assert is_nd(None)
    assert is_nd(math.nan)


def test_recognises_empty_and_whitespace_strings() -> None:
    assert is_nd("")
    assert is_nd("   ")
    assert is_nd("\t")
    assert is_nd("\n")
    assert is_nd(" \t\n ")


def test_does_not_match_real_values() -> None:
    for v in ("PERF", "123", "some_value", "0", "ARRE"):
        assert not is_nd(v), f"{v!r} should not be ND"


def test_does_not_match_numeric_or_bool_values() -> None:
    # R's is.na() returns FALSE for finite numeric and logical values.
    assert not is_nd(0)
    assert not is_nd(1)
    assert not is_nd(0.0)
    assert not is_nd(False)
    assert not is_nd(True)
