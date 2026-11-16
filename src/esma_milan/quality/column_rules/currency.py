"""Currency evidence: P13, and the rules of a currency companion column.

Mirrors nothing in the R reference (specification 6.2, "Currency
evidence"). The regulatory requirement is that the currency of every
amount is stated; which CSV column states it is this application's layout:

  case 1  the amount's own companion column holds a valid ISO 4217 code
  case 2  no usable companion, but the documented dedicated field does
          (RREL28 for RREL29 and RREL30; RREL18 for RREL16; RREC22 for
          RREC13). An absent companion column is then a layout note.
  case 3  both supplied and different: C30's, a later release. Until
          then the companion states the currency (case 1).
  case 4  neither supplies a valid code: the currency is not stated.
  case 5  a companion column this layout expects is absent while the
          currency is stated another way: the same layout note.

The mappings stay narrow: RREC22 describes RREC13 only, RREL18 primary
income only, RREL28 the two principal balances only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from esma_milan.quality.audit_read import RawFile
from esma_milan.quality.catalogue import FIELDS
from esma_milan.quality.column_rules.context import CellContext, RuleOutput
from esma_milan.quality.result import Finding, Mark, RuleOutcome
from esma_milan.quality.states import classify

if TYPE_CHECKING:
    from esma_milan.quality.column_rules import ColumnProfile

AMOUNT_NOT_A_NUMBER = "amount not a valid number"


def currency_evidence(c: CellContext) -> RuleOutput:
    """P13 on a cell of a {MONETARY} field."""
    entry, value = c.entry, c.value
    if entry.kind != "MONETARY" or not value.present:
        return
    if not value.ok:
        yield Mark("P13", not_evaluable=AMOUNT_NOT_A_NUMBER)
        return
    yield Mark("P13")
    companion, dedicated = c.extras.get("companion"), c.extras.get("dedicated")
    if companion in c.ref.currencies or dedicated in c.ref.currencies:
        return
    sources = [f"the companion column {entry.code}_CURRENCY"]
    if entry.dedicated_currency:
        sources.append(entry.dedicated_currency)
    yield Finding("P13", "violation", "currency_not_stated",
                  expected="a valid ISO 4217 code in " + " or ".join(sources))


def companion_rules(c: CellContext) -> RuleOutput:
    """Format (P5) and membership (P11) of a companion cell, where its
    amount is present. An empty or ND companion is not a finding here: P13
    decides whether the currency is stated another way."""
    amount_raw = c.extras.get("amount")
    if amount_raw is None:
        return
    amount = classify(FIELDS[c.entry.code.removesuffix("_CURRENCY")], amount_raw, c.ref)
    if not (amount.present and amount.ok) or not c.value.present:
        return
    yield Mark("P5")
    if c.value.ok:
        yield Mark("P11")
    elif c.value.reason == "not_in_list":
        yield Mark("P11")
        yield Finding("P11", "violation", "not_in_list", expected=c.value.expected)
    else:
        yield Mark("P11", not_evaluable="value not in the code's form")
        yield Finding("P5", "violation", c.value.reason or "shape", expected=c.value.expected)


def layout_rows(
    raw: RawFile, profiles: Sequence[ColumnProfile]
) -> tuple[list[dict[str, object]], list[RuleOutcome]]:
    """One note per {MONETARY} field whose companion column is absent
    (cases 2 and 5): this application's CSV layout, basis A, not a
    compliance matter."""
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for profile in profiles:
        entry = profile.entry
        if entry.kind != "MONETARY" or entry.code in seen or raw.companion_of(entry.code):
            continue
        seen.add(entry.code)
        rows.append({
            "file": raw.file, "column_position": profile.column.position,
            "header": profile.column.header, "field": entry.code, "rule_id": "P13",
            "result_kind": "context", "severity": "Info",
            "reason": "companion_column_absent",
            "expected": f"layout note: no {entry.code}_CURRENCY column; the currency of each "
                        "amount is assessed from the other permitted evidence",
        })
    return rows, []
