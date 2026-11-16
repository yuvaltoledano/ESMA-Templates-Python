"""Value domains and reference lists: P6 and P11.

Mirrors nothing in the R reference (specification 6.2, rules P6 and P11).
A domain rule exists only where the field's definition or a published
validation rule states one; there is no blanket sign or range rule by type.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

from esma_milan.quality.catalogue import RULES
from esma_milan.quality.column_rules.context import CellContext, RuleOutput
from esma_milan.quality.refdata import ReferenceData
from esma_milan.quality.result import Finding, Mark, RuleOutcome
from esma_milan.quality.states import XML_WHITESPACE

if TYPE_CHECKING:
    from esma_milan.quality.column_rules import ColumnProfile

NOT_A_NUMBER = "value not a valid number"
NOT_A_DATE = "value not a valid date"
NOT_A_CODE = "value not in the code's form"
YEAR_FROM_LOANS_NOTE = "classification year taken from the loans file"


def domain_rules(c: CellContext) -> RuleOutput:
    entry, value = c.entry, c.value
    if not value.present:
        return

    if entry.not_below_zero:
        if not value.ok:
            yield Mark("P6(a)", not_evaluable=NOT_A_NUMBER)
        else:
            assert isinstance(value.parsed, Decimal)
            yield Mark("P6(a)")
            if value.parsed < 0:
                yield Finding("P6(a)", "violation", "below_zero", expected="not below 0")

    if entry.not_after_cutoff:
        if not value.ok:
            yield Mark("P6(b)", not_evaluable=NOT_A_DATE)
        elif c.cutoff is None:
            yield Mark("P6(b)", not_evaluable=c.cutoff_reason)
        else:
            assert isinstance(value.parsed, date)
            yield Mark("P6(b)")
            if value.parsed > c.cutoff:
                yield Finding("P6(b)", "violation", "after_cutoff",
                              expected=f"not later than {c.cutoff.isoformat()}",
                              note=c.cutoff_note)

    if entry.code == "RREL59":
        if not value.ok:
            yield Mark("P6(c)", not_evaluable=NOT_A_NUMBER)
        else:
            assert isinstance(value.parsed, Decimal)
            yield Mark("P6(c)")
            if not 0 <= value.parsed <= 100:
                yield Finding("P6(c)", "violation", "out_of_range", expected="between 0 and 100")
    if entry.code == "RREL12":
        if not value.ok:
            yield Mark("P6(d)", not_evaluable="value not a valid year")
        else:
            assert isinstance(value.parsed, int)
            yield Mark("P6(d)")
            if not 1980 < value.parsed < 2100:
                yield Finding("P6(d)", "violation", "out_of_range",
                              expected="greater than 1980 and less than 2100")

    if entry.kind in ("COUNTRY", "CURRENCY"):
        if value.ok:
            yield Mark("P11")
        elif value.reason == "not_in_list":
            yield Mark("P11")
            yield Finding("P11", "violation", "not_in_list", expected=value.expected)
        else:
            yield Mark("P11", not_evaluable=NOT_A_CODE)
    elif entry.kind == "NUTS":
        yield from _nuts(c)


def _nuts(c: CellContext) -> RuleOutput:
    """NUTS level 3 membership for the classification year, or a country
    code followed by ZZZ. A year that is not bundled, or a country the
    bundled year does not cover, is a limited assessment, never 'invalid'."""
    if not c.value.ok:
        yield Mark("P11", not_evaluable=NOT_A_CODE)
        return
    code, ref = c.raw, c.ref
    if code.endswith("ZZZ"):
        yield Mark("P11")
        if code[:2] not in ref.countries and code[:2] not in ref.nuts_prefixes:
            yield Finding("P11", "violation", "zzz_prefix_not_a_country",
                          expected="an ISO 3166-1 code or a NUTS country prefix before ZZZ")
        return
    note: str | None = None
    year: int | None
    if c.entry.file == "loans":
        text = (c.extras.get("year") or "").strip(XML_WHITESPACE)
        year = int(text) if len(text) == 4 and text.isascii() and text.isdigit() else None
    else:
        year, note = c.pool.classification_year, YEAR_FROM_LOANS_NOTE
    if year is None:
        limit = "classification year (RREL12) not established"
    elif year not in ref.nuts3:
        limit = f"NUTS {year} is not bundled"
    elif code[:2] not in ref.nuts_countries[year]:
        limit = f"country {code[:2]} is not covered by the bundled NUTS {year} list"
    else:
        yield Mark("P11")
        if code not in ref.nuts3[year]:
            yield Finding("P11", "violation", "not_a_nuts3_code_of_the_year",
                          expected=f"a NUTS {year} level 3 code", note=note)
        return
    yield Mark("P11", not_evaluable=limit)
    yield Finding("P11", "limitation", "limited_assessment_shape_only", expected=limit, note=note)


def classification_year_results(
    year_profiles: Sequence[ColumnProfile],
    region_profiles: Sequence[ColumnProfile],
    ref: ReferenceData,
) -> tuple[list[dict[str, object]], list[RuleOutcome]]:
    """P11 on RREL12 as a fact of the submission: one classification year,
    and a year the lists cover. When every region code of both files is a
    country-plus-ZZZ code any year is accepted (Q&A 1313)."""
    rows: list[dict[str, object]] = []
    outcomes: list[RuleOutcome] = []
    regions = [v for p in region_profiles for v, parsed in p.parsed.items() if parsed.ok]
    only_zzz = bool(regions) and all(v.endswith("ZZZ") for v in regions)
    for profile in year_profiles:
        years = {
            int(str(parsed.parsed)): profile.present_values[raw]
            for raw, parsed in profile.parsed.items()
            if parsed.ok
        }
        outcome = RuleOutcome("P11", "loans", "RREL12", applicable=sum(years.values()))
        base: dict[str, object] = {
            "file": "loans", "column_position": profile.column.position,
            "header": profile.column.header, "field": "RREL12", "rule_id": "P11",
            "severity": RULES["P11"].severity, "value_state": "present",
        }
        if len(years) > 1:
            listed = ", ".join(f"{y} on {n} record(s)" for y, n in sorted(years.items()))
            outcome.failed = sum(years.values())
            rows.append({**base, "result_kind": "violation",
                         "reason": "more_than_one_classification_year",
                         "expected": f"one classification year across the submission; found {listed}"})
        for year, n in sorted(years.items()):
            if year in ref.nuts3 or only_zzz:
                outcome.evaluated += n
                continue
            reason = f"NUTS {year} is not bundled"
            outcome.not_evaluable += n
            outcome.not_evaluable_reasons[reason] += n
            outcome.other_kinds["limitation"] += n
            rows.append({**base, "result_kind": "limitation",
                         "reason": "limited_assessment_shape_only", "raw_value": str(year),
                         "expected": f"{reason}; {n} record(s)"})
        outcomes.append(outcome)
    return rows, outcomes
