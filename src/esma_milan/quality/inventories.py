"""The two inventories (specification 5.3 and 6.4).

Mirrors nothing in the R reference.

  actual columns    one row per physical column of a file, in file order:
                    how its header was resolved and a generic profile.
  expected fields   one row per template field (84 loan, 23 collateral),
                    present or absent, with its value states and the
                    results of every rule on it; plus a row for each
                    optional ESMA element that was supplied.

Sample values are cells of the user's own file. They go into the summary
beside the workbook and are never logged.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import polars as pl

from esma_milan.config import DQ_SAMPLE_VALUE_MAX_CHARS, DQ_SAMPLE_VALUES
from esma_milan.quality.audit_read import ColumnResolution, RawFile
from esma_milan.quality.catalogue import RULES, FileKind, fields_of
from esma_milan.quality.column_rules import ColumnProfile
from esma_milan.quality.result import RuleOutcome
from esma_milan.quality.states import STATES

_STATEMENTS: dict[str, str] = {
    "unmapped": "no taxonomy mapping established; not assessed for compliance",
    "optional_element": "recognised optional ESMA element, assessed",
    "export_extension": "known export extension; not a template field",
    "currency_companion": "currency companion of the field; assessed with the amount",
}
_SEVERITY_RANK = {"Info": 1, "Warn": 2, "Stop": 3}
LINEAGE_NOT_IMPLEMENTED = "not implemented in this release (arrives with P10)"


def _cut(value: str) -> str:
    if len(value) <= DQ_SAMPLE_VALUE_MAX_CHARS:
        return value
    return f"{value[:DQ_SAMPLE_VALUE_MAX_CHARS]}... ({len(value)} characters)"


def actual_columns(raw: RawFile) -> list[dict[str, object]]:
    """One row per physical column, with a generic profile."""
    assert raw.frame is not None
    rows: list[dict[str, object]] = []
    for column in raw.columns:
        values = raw.frame.select(pl.col(column.frame_column).alias("v"))
        filled = values.filter(pl.col("v").is_not_null() & (pl.col("v").str.strip_chars() != ""))
        top = (
            filled.group_by("v").len().sort(["len", "v"], descending=[True, False])
            .head(DQ_SAMPLE_VALUES)
        )
        longest = filled.select(pl.col("v").str.len_chars().max()).item()
        rows.append({
            "position": column.position,
            "header": column.header,
            "resolved_as": column.how,
            "field": column.field,
            "pipeline_column_name": column.pipeline_name,
            "records": raw.n_records,
            "empty": raw.n_records - filled.height,
            "distinct_values": filled.select(pl.col("v").n_unique()).item() if filled.height else 0,
            "longest_value": int(longest) if longest is not None else 0,
            "sample_values": [
                {"value": _cut(value), "records": n} for value, n in top.iter_rows()
            ],
            "statement": _STATEMENTS.get(column.how),
        })
    return rows


def _headers(columns: Sequence[ColumnResolution]) -> list[dict[str, object]]:
    return [{"position": c.position, "header": c.header, "resolved_as": c.how} for c in columns]


def expected_fields(
    file: FileKind,
    profiles: Sequence[ColumnProfile],
    outcomes: Sequence[RuleOutcome],
    results: pl.DataFrame,
) -> list[dict[str, object]]:
    """One row per template field of `file`, and one per supplied optional
    element. A field supplied in two columns is one row; its counts are
    the sums over its columns."""
    by_field: dict[str, list[ColumnProfile]] = defaultdict(list)
    for profile in profiles:
        by_field[profile.entry.code].append(profile)
    outcomes_by_field: dict[str, list[RuleOutcome]] = defaultdict(list)
    for outcome in outcomes:
        if outcome.field is not None:
            outcomes_by_field[outcome.field].append(outcome)
    counted = (
        results.filter(pl.col("file") == file)
        .group_by("field", "rule_id", "result_kind", "severity")
        .agg(pl.len().alias("n"))
    )
    rule_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    worst: dict[str, dict[str, str]] = defaultdict(dict)
    for field_code, rule_id, kind, severity, n in counted.iter_rows():
        rule_counts[field_code][rule_id][kind] = rule_counts[field_code][rule_id].get(kind, 0) + n
        basis = RULES[rule_id].basis
        if _SEVERITY_RANK[severity] > _SEVERITY_RANK.get(worst[field_code].get(basis, ""), 0):
            worst[field_code][basis] = severity

    entries = [*fields_of(file), *(e for e in fields_of(file, optional=True) if e.code in by_field)]
    rows: list[dict[str, object]] = []
    for entry in entries:
        mine = by_field.get(entry.code, [])
        states = {state: sum(p.states[state] for p in mine) for state in STATES}
        retained: dict[str, int] | str = "not available in this release"
        if mine and all(p.states_retained is not None for p in mine):
            retained = {
                state: sum(p.states_retained[state] for p in mine if p.states_retained is not None)
                for state in STATES
            }
        distribution: dict[str, int] = {}
        for p in mine:
            for value, n in p.distribution.items():
                distribution[value] = distribution.get(value, 0) + n
        examples: list[str] = []
        for p in mine:
            examples.extend(_cut(v) for v in p.non_compliant_examples if _cut(v) not in examples)
        merged = _merged(outcomes_by_field.get(entry.code, []))
        rows.append({
            "file": file,
            "field": entry.code,
            "name": entry.name,
            "taxonomy_format": entry.taxonomy_format,
            "nd1_4_allowed": entry.nd1_4_allowed,
            "nd5_allowed": entry.nd5_allowed,
            "optional_element": entry.optional,
            "critical": entry.critical,
            "present": bool(mine),
            "columns": _headers([p.column for p in mine]),
            "value_states": states if mine else None,
            "value_states_retained_pool": retained if mine else None,
            "format_conformity": {
                "compliant": sum(p.format_counts["compliant"] for p in mine),
                "alternative_form": sum(p.format_counts["alternative"] for p in mine),
                "non_compliant": sum(p.format_counts["non_compliant"] for p in mine),
                "non_compliant_examples": examples[:5],
            } if mine else None,
            "distinct_present_values": sum(p.n_distinct for p in mine) if mine else None,
            "code_distribution": dict(sorted(distribution.items())) if entry.kind == "LIST" and mine else None,
            "surrounding_whitespace": rule_counts[entry.code].get("P4(a)", {}).get("accepted alternative", 0),
            "results": {
                rule: dict(sorted(kinds.items()))
                for rule, kinds in sorted(rule_counts[entry.code].items())
            },
            "outcomes": {
                o.rule_id: {
                    "applicable": o.applicable, "evaluated": o.evaluated, "failed": o.failed,
                    "not_evaluable": o.not_evaluable, "label": o.label,
                }
                for o in merged
                if o.label != "passed"
            },
            "rules_passed": [o.rule_id for o in merged if o.label == "passed"],
            "worst_severity_by_basis": dict(sorted(worst[entry.code].items())),
            "pipeline_use_and_lineage": LINEAGE_NOT_IMPLEMENTED,
        })
    return rows


def _merged(outcomes: Sequence[RuleOutcome]) -> list[RuleOutcome]:
    """One outcome per rule (a field supplied in two columns has two)."""
    merged: dict[str, RuleOutcome] = {}
    for outcome in outcomes:
        if outcome.rule_id not in merged:
            merged[outcome.rule_id] = RuleOutcome(outcome.rule_id, outcome.file, outcome.field)
        merged[outcome.rule_id].add(outcome)
    return [merged[k] for k in sorted(merged)]
