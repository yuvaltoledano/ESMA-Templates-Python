"""Date parsing helpers.

Two helpers ported from r_reference/R/utils.R:

  parse_iso_or_excel_date()              utils.R:605-667
  collapse_multi_date_restructuring()    utils.R:688-716

Both return Polars Series so they can be dropped straight into a Polars
DataFrame column. Internal scalar/list logic is exposed via
`parse_iso_or_excel_date_values()` and `_collapse_cell()` for direct unit
testing without a Polars wrapper.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable
from datetime import date, datetime, timedelta

import polars as pl
import structlog

from esma_milan.nd import is_nd

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# parse_iso_or_excel_date
# ---------------------------------------------------------------------------

# ISO date regex (matches r_reference/R/utils.R:628):
#   ^\d{4}[-/]\d{2}[-/]\d{2}(\s.*)?$
ISO_DATE_RE = re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}(?:\s.*)?$")

# Excel-serial regex (matches r_reference/R/utils.R:636):
#   ^-?\d+(\.\d+)?$
SERIAL_RE = re.compile(r"^-?\d+(?:\.\d+)?$")

# Excel 1900 system: origin = 1899-12-30 absorbs the 1900 leap-year bug.
# r_reference/R/utils.R:640
EXCEL_EPOCH: date = date(1899, 12, 30)

DEFAULT_SERIAL_RANGE: tuple[date, date] = (date(1900, 1, 1), date(2100, 1, 1))


def parse_iso_or_excel_date_values(
    values: Iterable[object],
    col_name: str,
    serial_range: tuple[date, date] = DEFAULT_SERIAL_RANGE,
) -> list[date | None]:
    """Parse a sequence of mixed ISO-string / Excel-serial / Date values.

    Returns a list of `datetime.date | None` of the same length.

    Strategies, applied in order per element (matches utils.R:605-667):
      0. None -> None.
      1. Already a `datetime.date` (or subclass `datetime.datetime`) -> use
         the date component as-is.
      2. Coerce to string, strip; if the string is ND-equivalent (NA, ND,
         ND1..ND5, blank) -> None.
      3. ISO format YYYY-MM-DD or YYYY/MM/DD (optionally followed by space
         + arbitrary text, like a time): parse the leading 10 chars.
      4. Pure integer/decimal: Excel serial number; convert via
         EXCEL_EPOCH + timedelta(days=N). Reject if outside `serial_range`
         (those rows fall through to the bad-input warning).
      5. Anything else -> None, collected as a "bad value" for warning.

    A single warning is emitted at the end of the call enumerating up to
    5 distinct bad values. Mirrors the R warning text format.
    """
    out: list[date | None] = []
    bad_values_seen: list[str] = []
    bad_values_set: set[str] = set()

    for v in values:
        # Strategy 0/1: None / already-date passthrough
        if v is None:
            out.append(None)
            continue
        # `datetime` is a subclass of `date`; check it first so we keep
        # only the date component rather than the full timestamp.
        if isinstance(v, datetime):
            out.append(v.date())
            continue
        if isinstance(v, date):
            out.append(v)
            continue

        # Strategy 2: ND/blank -> None
        stripped = v.strip() if isinstance(v, str) else str(v).strip()
        if is_nd(stripped):
            out.append(None)
            continue

        # Strategy 3: ISO format
        if ISO_DATE_RE.match(stripped):
            iso10 = stripped[:10].replace("/", "-")
            try:
                out.append(date.fromisoformat(iso10))
                continue
            except ValueError:
                # Looked ISO-shaped but not valid (e.g. 2023-13-01).
                # Falls through to bad-value handling.
                pass

        # Strategy 4: Excel serial
        if SERIAL_RE.match(stripped):
            try:
                serial = float(stripped)
                # `EXCEL_EPOCH + timedelta(days=N)` yields a `date` (the
                # fractional days, if any, are silently truncated when
                # added to a date). This matches R's as.Date(numeric)
                # behaviour with origin=1899-12-30. OverflowError fires
                # for very large negative serials (e.g. -999999) because
                # the result would precede date.min; treat those as
                # bad-value, same as R's range-rejection path.
                parsed = EXCEL_EPOCH + timedelta(days=serial)
            except (ValueError, OverflowError):
                pass
            else:
                if serial_range[0] <= parsed <= serial_range[1]:
                    out.append(parsed)
                    continue

        # Strategy 5: bad input
        out.append(None)
        if stripped not in bad_values_set:
            bad_values_set.add(stripped)
            bad_values_seen.append(stripped)

    if bad_values_seen:
        sample = ", ".join(bad_values_seen[:5])
        warnings.warn(
            f"parse_iso_or_excel_date: column {col_name!r} contains "
            f"{len(bad_values_seen)} distinct value(s) that are neither ISO "
            f"dates nor Excel serials in the range "
            f"{serial_range[0].isoformat()}..{serial_range[1].isoformat()}. "
            f"These rows were set to NA. Examples (up to 5): [{sample}].",
            UserWarning,
            stacklevel=2,
        )

    return out


def parse_iso_or_excel_date(
    series: pl.Series,
    col_name: str,
    serial_range: tuple[date, date] = DEFAULT_SERIAL_RANGE,
) -> pl.Series:
    """Polars wrapper around parse_iso_or_excel_date_values.

    If `series` already has Date dtype (passthrough case from
    utils.R:613), returns it unchanged.
    """
    if series.dtype == pl.Date:
        return series
    parsed = parse_iso_or_excel_date_values(
        series.to_list(), col_name, serial_range
    )
    return pl.Series(name=series.name, values=parsed, dtype=pl.Date)


# ---------------------------------------------------------------------------
# collapse_multi_date_restructuring
# ---------------------------------------------------------------------------

# Comma OR semicolon is the delimiter (utils.R:691).
_DELIM_RE = re.compile(r"[,;]")


def collapse_multi_date_restructuring(
    values: Iterable[object],
) -> list[str | None]:
    """Collapse comma- or semicolon-separated cells to their max ISO date.

    Mirrors r_reference/R/utils.R:688-716.

    - Cells with no `,` or `;` delimiter pass through unchanged (as-is, i.e.
      NOT re-formatted). None / NaN / non-string scalars pass through too.
    - Cells with a delimiter: split, trim, drop empty tokens, parse each
      surviving token as an ISO date, take the max, write back as ISO.
    - All-unparseable delimited cells -> None.

    Emits a single structlog INFO log at end of call with the count of
    collapsed cells and (if any) unparseable cells, matching the R
    `message()` output in spirit.
    """
    # Coerce non-character inputs to strings up-front, matching R's
    # `if (!is.character(x)) x <- as.character(x)`. This means a `factor`
    # input gets stringified before processing.
    items: list[object] = list(values)

    n_multi = 0
    n_unparseable = 0
    out: list[str | None] = []

    for item in items:
        if item is None:
            out.append(None)
            continue
        if isinstance(item, float) and item != item:  # NaN
            out.append(None)
            continue
        s = str(item)
        if not _DELIM_RE.search(s):
            out.append(s)
            continue

        n_multi += 1
        tokens = [t.strip() for t in _DELIM_RE.split(s)]
        tokens = [t for t in tokens if t]  # nzchar filter
        parsed_dates: list[date] = []
        for tok in tokens:
            try:
                parsed_dates.append(date.fromisoformat(tok))
            except ValueError:
                continue
        if not parsed_dates:
            n_unparseable += 1
            out.append(None)
        else:
            out.append(max(parsed_dates).isoformat())

    if n_multi > 0:
        log.info(
            "collapse_multi_date_restructuring",
            n_collapsed=n_multi,
            n_unparseable=n_unparseable,
        )

    return out
