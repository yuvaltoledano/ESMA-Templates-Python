"""Format rules: P4, P5, P7 and the RREL65 validator.

Mirrors nothing in the R reference (specification 6.2).
"""

from __future__ import annotations

import re

from esma_milan.quality.catalogue import WHITESPACE_COLLAPSING_KINDS
from esma_milan.quality.column_rules.context import CellContext, RuleOutput
from esma_milan.quality.result import Finding, Mark
from esma_milan.quality.states import ND_TOKENS, XML_WHITESPACE, classify, lei_fault, real_date

_ELEMENT_SEPARATOR = re.compile(r"[,;]")
# P7(b): ASCII from space to tilde, without the lower-case letters.
_ASCII_CAPITALS = re.compile(r"^[\x20-\x60\x7b-\x7e]*$")
# P7(c): Article 11(1): LEI, N, four-digit year, two-digit number.
_UNIQUE_IDENTIFIER = re.compile(r"^[A-Z0-9]{18}[0-9]{2}N[0-9]{4}[0-9]{2}$")


def format_rules(c: CellContext) -> RuleOutput:
    entry, value = c.entry, c.value
    if not value.present:
        return
    if entry.kind == "ALPHANUM":
        yield from _text_rules(c)
        return
    if entry.code == "RREL65" and _ELEMENT_SEPARATOR.search(c.raw):
        yield from _restructuring_dates(c)
        return

    yield Mark("P5")
    collapses = entry.kind in WHITESPACE_COLLAPSING_KINDS
    if collapses:
        yield Mark("P4(a)")
    if entry.kind == "LIST" and entry.feeds_milan:
        yield Mark("P4(d)")
    if value.ok:
        if value.alternative == "xml_whitespace":
            # Valid for the XSD type; nonconforming only under this
            # application's CSV convention.
            yield Finding("P4(a)", "accepted alternative", "xml_whitespace",
                          expected="the value without surrounding whitespace")
        elif value.alternative == "xml_boolean":
            yield Finding("P5", "accepted alternative", "xml_boolean",
                          expected="Y or N (the template form)")
    elif value.reason != "not_in_list":  # membership is P11's
        assert value.reason is not None
        yield Finding("P5", "violation", value.reason, expected=value.expected)
        if entry.kind == "LIST" and entry.feeds_milan and value.reason == "trim_or_case":
            yield Finding("P4(d)", "violation", "code_not_recognised_by_the_pipeline",
                          expected="the pipeline compares this code exactly; the value as "
                                   "written is not recognised")


def _text_rules(c: CellContext) -> RuleOutput:
    entry, raw = c.entry, c.raw
    assert entry.max_length is not None
    yield Mark("P7(a)")
    if len(raw) > entry.max_length:
        yield Finding("P7(a)", "violation", "too_long",
                      expected=f"at most {entry.max_length} characters (found {len(raw)})")
    yield Mark("P7(b)")
    if not _ASCII_CAPITALS.match(raw):
        yield Finding("P7(b)", "violation", "not_ascii_capitals",
                      expected="ASCII characters without lower-case letters")
    if entry.code in ("RREL1", "RREC1"):
        yield Mark("P7(c)")
        if not _UNIQUE_IDENTIFIER.match(raw):
            yield Finding("P7(c)", "violation", "identifier_pattern",
                          expected="LEI, N, a four-digit year and a two-digit number")
        elif lei_fault(raw[:20]) is not None:
            yield Finding("P7(c)", "violation", "identifier_lei_check_digits",
                          expected="the first 20 characters are an LEI with valid check digits")


def _restructuring_dates(c: CellContext) -> RuleOutput:
    """RREL65 holding several elements in one cell (specification 6.3).

    The XML carries repeated date elements; `,` or `;` inside one CSV
    cell is this application's convention. Each element is a valid date, a
    recognised ND state or an invalid fragment. Three questions are kept
    apart: the validity of each element (P5; P6(b) for a date; P3e for a
    dated ND4), an unexplained mixture of a date and an ND option (P3f,
    a question), and what the pipeline's split selects (P3g).
    """
    elements = [e.strip(XML_WHITESPACE) for e in _ELEMENT_SEPARATOR.split(c.raw)]
    dates = [d for d in map(real_date, elements) if d is not None]
    nd_dates = []
    invalid: list[str] = []
    n_nd = 0
    for element in elements:
        if real_date(element) is not None:
            continue
        state = classify(c.entry, element, c.ref).state
        if element in ND_TOKENS:
            n_nd += 1
        elif state == "nd4_dated":
            n_nd += 1
            nd_dates.append(real_date(element[4:]))
        else:
            invalid.append(element)

    yield Mark("P5")
    if invalid:
        shown = ", ".join(repr(e) for e in invalid[:5])
        yield Finding("P5", "violation", "invalid_fragment",
                      expected=f"each element a date YYYY-MM-DD or an ND option; invalid: {shown}")
    yield Mark("P3f")
    if n_nd and dates:
        yield Finding("P3f", "question", "nd_option_beside_a_date",
                      expected="mixture not explained: an ND option beside a real date")
    if dates:
        if c.cutoff is None:
            yield Mark("P6(b)", not_evaluable=c.cutoff_reason)
        else:
            yield Mark("P6(b)")
            if max(dates) > c.cutoff:
                yield Finding("P6(b)", "violation", "after_cutoff",
                              expected=f"every date not later than {c.cutoff.isoformat()}",
                              note=c.cutoff_note)
    if nd_dates:
        if c.cutoff is None:
            yield Mark("P3e", not_evaluable=c.cutoff_reason)
        else:
            yield Mark("P3e")
            if any(d is not None and d <= c.cutoff for d in nd_dates):
                yield Finding("P3e", "violation", "nd4_date_not_after_cutoff",
                              expected=f"a date after {c.cutoff.isoformat()}", note=c.cutoff_note)
    selected = max(dates).isoformat() if dates else "no date (the cell is read as missing)"
    yield Finding("P3g", "context", "pipeline_selects_latest_date",
                  expected=f"the pipeline splits the cell and keeps {selected}; "
                           "other elements disappear from its value")
