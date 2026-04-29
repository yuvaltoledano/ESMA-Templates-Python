"""snake_case cleaner for column names.

Mirrors r_reference's reliance on `janitor::clean_names()`. ESMA column
headers come in two shapes that this helper has to flatten to a single
snake_case form:

  1. Pre-cleaned snake_case names already present in some source files
     (passes through unchanged).
  2. Taxonomy-renamed Title Case names with spaces, dashes, commas,
     parens, slashes, percent signs, etc. ("Original Underlying Exposure
     Identifier", "Property Postal, Region code or Nuts code", "MIG %",
     "ND1-ND4 allowed?").

The algorithm matches janitor's `snake` case for the substring shapes
present in the ESMA taxonomy and synthetic/real input headers (verified
by spot-checking against the staged taxonomy):

  1. Replace '%' with '_percent_' and '#' with '_number_' (janitor's
     replace map default).
  2. Lowercase.
  3. Map any non-alphanumeric character to '_'.
  4. Collapse runs of consecutive '_' to a single '_'.
  5. Strip leading/trailing '_'.

Acronym/camelCase splitting is intentionally NOT implemented because the
ESMA headers do not exercise it; if a future header introduces e.g.
"RREL30Currency", we'll add it then with a unit test pinning the
expected output.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def clean_name(value: str) -> str:
    """Clean a single column name to snake_case."""
    s = value.replace("%", "_percent_").replace("#", "_number_")
    s = s.lower()
    s = _NON_ALNUM.sub("_", s)
    return s.strip("_")


def clean_names(values: Iterable[str]) -> list[str]:
    """Clean an iterable of column names. Order preserved."""
    return [clean_name(v) for v in values]
