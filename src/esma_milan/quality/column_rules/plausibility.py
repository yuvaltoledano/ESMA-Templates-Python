"""Plausibility: P8 (possible placeholder), P9 (cardinality), P12 (windows).

Mirrors nothing in the R reference (specification 6.2). None of these
asserts an error: each result is a question or a context row, worded
"possible placeholder; confirm whether this is a measurement" or
"unusual; please confirm". Nothing is converted to ND5 and no value is
rescaled.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

from esma_milan.config import (
    DQ_DATE_MAX_YEARS_AFTER_CUTOFF,
    DQ_DATE_MIN_YEAR,
    DQ_PERCENT_SCALE_EXCLUDED_FIELDS,
    DQ_POSSIBLE_PLACEHOLDERS,
)
from esma_milan.quality.audit_read import RawFile
from esma_milan.quality.catalogue import FieldEntry
from esma_milan.quality.column_rules.context import CellContext, RuleOutput
from esma_milan.quality.result import Finding, Mark, RuleOutcome

if TYPE_CHECKING:
    from esma_milan.quality.column_rules import ColumnProfile

_cells = DQ_POSSIBLE_PLACEHOLDERS["cells"]
assert isinstance(_cells, dict)
PLACEHOLDER_CELLS: dict[str, tuple[Decimal, ...]] = {
    code: tuple(Decimal(v) for v in values) for code, values in _cells.items()
}
ALL_ZERO_COLUMNS = tuple(DQ_POSSIBLE_PLACEHOLDERS["all_zero_columns"])
SINGLE_TOKEN_COLUMNS = tuple(DQ_POSSIBLE_PLACEHOLDERS["single_token_columns"])


def _severity(entry: FieldEntry) -> str:
    return "Warn" if entry.feeds_milan else "Info"


def _years_after(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year + years)
    except ValueError:  # 29 February
        return day.replace(year=day.year + years, day=28)


def plausibility_rules(c: CellContext) -> RuleOutput:
    entry, value = c.entry, c.value
    if not (value.present and value.ok):
        return
    if entry.code in PLACEHOLDER_CELLS:
        assert isinstance(value.parsed, Decimal)
        yield Mark("P8")
        if value.parsed in PLACEHOLDER_CELLS[entry.code]:
            yield Finding("P8", "question", "possible_placeholder_value", severity=_severity(entry),
                          expected="possible placeholder; confirm whether this is a measurement")
    if entry.kind == "DATE":
        assert isinstance(value.parsed, date)
        yield Mark("P12")
        if value.parsed.year < DQ_DATE_MIN_YEAR:
            yield Finding("P12", "question", "date_before_window", severity=_severity(entry),
                          expected=f"unusual: before {DQ_DATE_MIN_YEAR}; please confirm")
        elif c.cutoff is not None and value.parsed > _years_after(
            c.cutoff, DQ_DATE_MAX_YEARS_AFTER_CUTOFF
        ):
            yield Finding("P12", "question", "date_after_window", severity=_severity(entry),
                          expected=f"unusual: more than {DQ_DATE_MAX_YEARS_AFTER_CUTOFF} years "
                                   "after the cut-off; please confirm", note=c.cutoff_note)
    if entry.code == "RREC8":
        yield Mark("P12")
        if value.parsed == 0:
            yield Finding("P12", "question", "zero_is_not_a_lien_position",
                          severity=_severity(entry),
                          expected="unusual: 0 is not a lien position; please confirm")


def column_rows(
    raw: RawFile, profiles: Sequence[ColumnProfile]
) -> tuple[list[dict[str, object]], list[RuleOutcome]]:
    """P8, P9 and P12 results about a whole column (one row each)."""
    rows: list[dict[str, object]] = []
    outcomes: list[RuleOutcome] = []
    by_code = {p.entry.code: p for p in profiles}

    def emit(profile: ColumnProfile, rule: str, kind: str, reason: str, expected: str,
             severity: str, fired: bool) -> None:
        outcome = RuleOutcome(rule, raw.file, profile.entry.code, applicable=1, evaluated=1)
        if fired:
            outcome.other_kinds[kind] += 1
            rows.append({
                "file": raw.file, "column_position": profile.column.position,
                "header": profile.column.header, "field": profile.entry.code, "rule_id": rule,
                "result_kind": kind, "severity": severity, "reason": reason, "expected": expected,
            })
        outcomes.append(outcome)

    for profile in profiles:
        entry, code = profile.entry, profile.entry.code
        if code in ALL_ZERO_COLUMNS:
            fired = (
                profile.n_present > 0 and profile.n_distinct_invalid == 0 and profile.n_nonzero == 0
            )
            if code == "RREC21" and fired:
                sale = by_code.get("RREC20")
                fired = sale is None or sale.n_valid == 0
            emit(profile, "P8", "question", "zero_on_every_record",
                 "possible placeholder; confirm whether this is a measurement (0 on every "
                 "record that reports a value)", _severity(entry), fired)
        if code in SINGLE_TOKEN_COLUMNS:
            token = profile.first_value or ""
            fired = (
                raw.n_records > 1 and profile.n_distinct == 1
                and profile.n_present == raw.n_records
                and not any(ch.isspace() for ch in token)
            )
            emit(profile, "P8", "question", "single_token_on_every_record",
                 "possible placeholder; confirm whether this is a measurement (one token on "
                 "every record)", _severity(entry), fired)
        emit(profile, "P9", "context", "constant_column",
             f"one value on all {profile.n_present} record(s) that report a value", "Info",
             profile.n_distinct == 1 and profile.n_present > 1)
        if entry.kind == "PERCENTAGE" and code not in DQ_PERCENT_SCALE_EXCLUDED_FIELDS:
            fired = profile.n_valid > 0 and profile.n_abs_at_least_one == 0 and profile.n_nonzero > 0
            emit(profile, "P12", "question", "percentage_values_all_below_1",
                 "unusual: every value is below 1; the column may be a fraction and not "
                 "percentage units; please confirm", "Warn", fired)
    return rows, outcomes
