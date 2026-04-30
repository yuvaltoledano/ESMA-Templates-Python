"""Shared ND-detection helper.

Mirrors `is_nd()` in r_reference/R/milan_mapping.R:19. Returns True for any
value that ESMA convention treats as "no data": Python's None, the case-
preserved strings "NA"/"ND"/"ND1".."ND5", and any whitespace-only string.

Used by date parsing, identifier selection, valuation logic, and most of
the MILAN field mapping.
"""

from __future__ import annotations

from esma_milan.config import ND_STRINGS


def is_nd(value: object) -> bool:
    """Return True if `value` is ESMA-equivalent to "no data".

    Mirrors r_reference/R/milan_mapping.R:19-22:

        is_nd <- function(x) {
          is.na(x) | x %in% c("ND", "ND1", "ND2", "ND3", "ND4", "ND5", "NA") |
            (!is.na(x) & trimws(x) == "")
        }
    """
    if value is None:
        return True
    # Match R's vectorised is.na() for scalars: only Python's None or NaN
    # qualify. Booleans and zero are NOT NA in R, so explicitly exclude them.
    if isinstance(value, float) and value != value:  # NaN check
        return True
    if not isinstance(value, str):
        return False
    if value in ND_STRINGS:
        return True
    return value.strip() == ""
