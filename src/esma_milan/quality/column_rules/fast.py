"""The fast path: values that are proven plain are assessed together.

Mirrors nothing in the R reference.

The engine evaluates each distinct value of a column once, in Python.
On a large pool most distinct values are plain: an amount such as
`150000.00`, an identifier such as `LOAN000123`. For such a value no rule
of its field can emit a row that depends on the value itself, so all the
plain values of a column that share the same other cells (currency
companion, collateral type, cut-off) are one group: the rules run once on
a representative, and their marks and any row that depends only on those
other cells (P13) apply to every record of the group.

`plain_expr` is the proof. It is conservative: a value is plain only when
every rule in the table below is impossible for it; anything else takes
the ordinary path. A field with a rule that is not in the table is never
plain (`_SLOW_FIELDS`, and every format type not listed).

  rule            plain number                     plain text
  P3a-P3d         digits only: a present value     not blank, no ND token, no marker
  P3e, P3g        cannot start ND4; not a token    does not start ND4; not a token the
                  the loader nulls                 loader nulls
  P3f             RREC8/12/16: plain only on a     -
                  property record
  P4(a)           no surrounding whitespace        (not a text rule; padded text is
                                                   not plain all the same)
  P5              inside the lexical form and      -
                  the digit limits
  P6(a)           no sign                          -
  P6(c), P6(d)    RREL59, RREL12: never plain      -
  P7(a), P7(b)    -                                the test is the rule
  P7(c)           -                                RREL1, RREC1: never plain
  P8, P12 cell    zero and placeholders excluded   -
  P13             depends on the other cells only  -
  P9, P8 and P12  from the profile counts, which the engine fills for plain values
  per column
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal

import polars as pl

from esma_milan.config import (
    CSV_NA_TOKENS,
    DQ_INVALID_MISSING_MARKER_PATTERN,
    DQ_INVALID_MISSING_MARKERS,
)
from esma_milan.quality.catalogue import (
    CONDITIONAL_ND5_FIELDS,
    PROPERTY_COLLATERAL_TYPES,
    FieldEntry,
)
from esma_milan.quality.column_rules.plausibility import PLACEHOLDER_CELLS
from esma_milan.quality.states import ND_TOKENS, XML_WHITESPACE

# Fields with a value rule the table does not cover.
_SLOW_FIELDS: frozenset[str] = frozenset({"RREL59", "RREL1", "RREC1"})

# Digits before and after the point that are certainly inside the facets:
# 13 + 5 = 18 for an amount, 5 + 6 = 11 for a percentage.
_NUMBER_PATTERNS: dict[str, str] = {
    "MONETARY": r"^[0-9]{1,13}(\.[0-9]{1,5})?$",
    "PERCENTAGE": r"^[0-9]{1,5}(\.[0-9]{1,6})?$",
    "INTEGER": r"^[0-9]{1,4}$",
}
_NON_ZERO = r"[1-9]"
_TEXT_PATTERN = r"^[\x20-\x60\x7b-\x7e]+$"
AT_LEAST_ONE = r"^[0-9]*[1-9][0-9]*(\.|$)"
"""A plain number whose absolute value is 1 or more: a non-zero digit
before the point."""


def _placeholder_pattern(value: Decimal) -> str:
    """Every plain spelling of a placeholder value: leading zeros and
    trailing fractional zeros allowed."""
    integer, _, fraction = format(value, "f").partition(".")
    fraction = fraction.rstrip("0")
    tail = rf"\.{re.escape(fraction)}0*" if fraction else r"(\.0*)?"
    return rf"^0*{re.escape(integer.lstrip('0'))}{tail}$"


def plain_expr(
    entry: FieldEntry, extras: Mapping[str, str], column: str = "_v"
) -> pl.Expr | None:
    """True for the cells of `column` that are proven plain for `entry`;
    None when no value of the field can be. `extras` names the other cells
    available as columns (`collateral_type` -> its column name)."""
    if entry.code in _SLOW_FIELDS:
        return None
    value = pl.col(column)
    if entry.kind in _NUMBER_PATTERNS:
        plain = value.str.contains(_NUMBER_PATTERNS[entry.kind]) & value.str.contains(_NON_ZERO)
        for placeholder in PLACEHOLDER_CELLS.get(entry.code, ()):
            if placeholder != 0:
                plain = plain & ~value.str.contains(_placeholder_pattern(placeholder))
    elif entry.kind == "ALPHANUM":
        assert entry.max_length is not None
        not_a_token = sorted(ND_TOKENS | set(CSV_NA_TOKENS))
        plain = (
            value.str.contains(_TEXT_PATTERN)
            & (value.str.len_chars() <= entry.max_length)
            & (value.str.strip_chars(XML_WHITESPACE) == value)
            & ~value.is_in(not_a_token)
            & ~value.str.starts_with("ND4")
            & ~value.str.to_uppercase().is_in(sorted(m.upper() for m in DQ_INVALID_MISSING_MARKERS))
            & ~value.str.contains(DQ_INVALID_MISSING_MARKER_PATTERN)
        )
    else:
        return None
    if entry.code in CONDITIONAL_ND5_FIELDS:
        if "collateral_type" not in extras:
            return None
        plain = plain & pl.col(extras["collateral_type"]).is_in(sorted(PROPERTY_COLLATERAL_TYPES))
    return plain
