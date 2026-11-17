"""The ND rules: P3a to P3g (P1 is in the engine, it has no cell).

Mirrors nothing in the R reference (specification 6.1).
"""

from __future__ import annotations

from esma_milan.config import CSV_NA_TOKENS, EXCLUDED_COLLATERAL_TYPES
from esma_milan.quality.catalogue import (
    CONDITIONAL_ND5_FIELDS,
    FIELDS,
    NUMERIC_KINDS,
    PROPERTY_COLLATERAL_TYPES,
    UNDETERMINED_COLLATERAL_TYPES,
)
from esma_milan.quality.column_rules.context import CellContext, RuleOutput
from esma_milan.quality.result import Finding, Mark
from esma_milan.quality.states import classify

# Why P3f cannot say whether ND5 on a property record is right, by what the
# link to the loan record gives (`links.collateral_link_context`).
_ND5_ON_PROPERTY_LIMITS: dict[str, tuple[str, str]] = {
    "default, recoveries not determinable": (
        "nd5_on_property_default_recoveries_not_determinable",
        "the loan is in default and the files do not show whether recoveries have ended",
    ),
    "not established": (
        "nd5_on_property_lifecycle_not_established",
        "the lifecycle of the loan is not established (its status is empty, invalid or OTHR)",
    ),
    "linked loans whose lifecycles differ": (
        "nd5_on_property_linked_loans_differ",
        "the record belongs to loans whose lifecycles differ",
    ),
    "no loan record linked": (
        "nd5_on_property_no_loan_record_linked",
        "no loan record is linked to the collateral record",
    ),
}
COLLATERAL_TYPE_UNDETERMINED = "collateral type OTHR or MIXD: applicability not established"
COLLATERAL_TYPE_NOT_ESTABLISHED = "RREC5 is not a valid code on the record"
COLLATERAL_TYPE_ABSENT = "RREC5 is not in the file"


def nd_rules(c: CellContext) -> RuleOutput:
    entry, state = c.entry, c.value.state

    yield Mark("P3a")
    if state in ("nd1", "nd2", "nd3", "nd4_dated") and not entry.nd1_4_allowed:
        yield Finding("P3a", "violation", "nd1_4_not_allowed",
                      expected="a value" + (" or ND5" if entry.nd5_allowed else ""),
                      nd4_date=c.value.nd4_date)
    yield Mark("P3b")
    if state == "nd5" and not entry.nd5_allowed:
        yield Finding("P3b", "violation", "nd5_not_allowed",
                      expected="a value" + (" or ND1 to ND4" if entry.nd1_4_allowed else ""))
    yield Mark("P3c")
    if state == "empty":
        yield Finding("P3c", "violation", "empty", expected="a value or a permitted ND option")
    yield Mark("P3d")
    if state == "invalid_marker":
        yield Finding("P3d", "violation", "invalid_missing_marker",
                      expected="a value or a permitted ND option")

    if state == "nd4_malformed":
        yield Mark("P3e")
        yield Finding("P3e", "violation", "nd4_malformed", expected="ND4-YYYY-MM-DD")
    elif state == "nd4_dated":
        assert c.value.nd4_date is not None
        if c.cutoff is None:
            yield Mark("P3e", not_evaluable=c.cutoff_reason)
        else:
            yield Mark("P3e")
            if c.value.nd4_date <= c.cutoff:
                yield Finding("P3e", "violation", "nd4_date_not_after_cutoff",
                              expected=f"a date after {c.cutoff.isoformat()}",
                              note=c.cutoff_note, nd4_date=c.value.nd4_date)

    if entry.code in CONDITIONAL_ND5_FIELDS:
        yield from _conditional_nd5(c)
    yield from _loader_effect(c)


def _conditional_nd5(c: CellContext) -> RuleOutput:
    """P3f for RREC7, RREC8, RREC9, RREC12 and RREC16 against RREC5 of the
    same record."""
    entry, value = c.entry, c.value
    if value.state not in ("nd5", "present"):
        return
    collateral_type = c.extras.get("collateral_type")
    if collateral_type is None:
        yield Mark("P3f", not_evaluable=COLLATERAL_TYPE_ABSENT)
        return
    type_value = classify(FIELDS["RREC5"], collateral_type, c.ref)
    if not (type_value.present and type_value.ok):
        yield Mark("P3f", not_evaluable=COLLATERAL_TYPE_NOT_ESTABLISHED)
        yield Finding("P3f", "limitation", "collateral_type_not_established")
    elif collateral_type in UNDETERMINED_COLLATERAL_TYPES:
        yield Mark("P3f", not_evaluable=COLLATERAL_TYPE_UNDETERMINED)
        yield Finding("P3f", "limitation", "collateral_type_undetermined")
    elif collateral_type in PROPERTY_COLLATERAL_TYPES:
        if value.state != "nd5":
            yield Mark("P3f")
            return
        # Whether ND5 is the prescribed final-report value (Q&A 1296) or a
        # questioned use depends on the lifecycle the loan claims (E17(b); for
        # RREC9 the regulatory part of E2).
        lifecycle = c.extras.get("lifecycle") or "no loan record linked"
        if lifecycle == "final-inactive (presumed)":
            yield Mark("P3f")
        elif lifecycle == "reporting-active (presumed)":
            yield Mark("P3f")
            yield Finding("P3f", "question", "nd5_on_property_of_a_reporting_active_exposure",
                          expected="a value: the collateral is property and the loan is "
                                   "presumed active for reporting")
        else:
            reason, why = _ND5_ON_PROPERTY_LIMITS[lifecycle]
            yield Mark("P3f", not_evaluable=why)
            yield Finding("P3f", "limitation", reason, expected=why)
    else:
        assert collateral_type in EXCLUDED_COLLATERAL_TYPES
        yield Mark("P3f")
        guarantee_lien = entry.code == "RREC8" and collateral_type == "GUAR"  # Q&A 1323(c)
        if value.state == "present" and not guarantee_lien:
            yield Finding("P3f", "violation", "value_on_non_property_collateral",
                          expected=f"ND5 (collateral type {collateral_type} is not property)")


def _loader_effect(c: CellContext) -> RuleOutput:
    """P3g: what the pipeline's loader does with this cell, where that is
    not what the file states. A dated ND4 in a numeric field makes the
    loader stop; that is A5's row (`family_a`), one defect and one row."""
    entry, value = c.entry, c.value
    yield Mark("P3g")
    if value.present and value.ok and c.raw in CSV_NA_TOKENS:
        yield Finding("P3g", "context", "valid_value_read_as_missing",
                      expected="the loader reads this text as missing in every column")
    elif value.state == "nd4_dated":
        if entry.kind == "DATE":
            yield Finding("P3g", "context", "dated_nd4_set_to_missing_with_a_warning",
                          nd4_date=value.nd4_date)
        elif entry.kind not in NUMERIC_KINDS or entry.optional:
            yield Finding("P3g", "context", "dated_nd4_kept_as_text", nd4_date=value.nd4_date)
    elif value.present and entry.kind == "DATE" and value.reason in (
        "alternative_date_form", "excel_serial"
    ):
        yield Finding("P3g", "context", "date_parsed_by_the_pipeline",
                      expected="the pipeline reads this form although it is not YYYY-MM-DD")
