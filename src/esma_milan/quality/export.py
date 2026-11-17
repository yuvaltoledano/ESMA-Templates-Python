"""The data quality summary file.

Mirrors nothing in the R reference (specification 10.1, 10.3, 10.4).

`summary_dict` turns a `QualityReport` into the JSON document the command
line writes beside the workbook: the provenance, the compliance and
completeness judgments, the plausibility count, the outcome of every rule
with its four coverage counts, the finding groups, the two inventories and
the population counts. Row-level results are not in it; they are the
detail file of a later release.

Wording follows the result kind first and the basis second (3.2): only a
violation of a reporting requirement says "does not comply with"; a
plausibility result never asserts an error; a pipeline-constraint row says
what the pipeline does with the value.

The parts of the result contract that later releases fill are present and
say "not implemented in this release". There is no policy outcome, and
nothing in the file says a pool was blocked, rejected or eligible.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import polars as pl

import esma_milan
from esma_milan import config
from esma_milan.io_layer.read_taxonomy import bundled_taxonomy_path
from esma_milan.quality.assess import (
    FileAudit,
    QualityReport,
    inventory_of_expected_fields,
)
from esma_milan.quality.catalogue import FIELDS, RULES, SCOPE_STATEMENTS, sentence_clause
from esma_milan.quality.populations import LIFECYCLES, POPULATIONS
from esma_milan.quality.refdata import load_reference_data
from esma_milan.quality.result import RuleOutcome

SCHEMA_VERSION = "1"
NOT_IMPLEMENTED = "not implemented in this release"
SUMMARY_SUFFIX = " data quality summary.json"
UNKNOWN_CUTOFF = "unknown-cutoff"
ASSESSMENT_INTERRUPTED = "assessment interrupted by a technical fault"

# Rules whose results are not rows of their own.
_REPORTED_ELSEWHERE = {
    "P2": "the value-state counts of the expected-field inventory",
    "P4(c)": "P5 rows with the reason control_character (one defect, one row)",
    "A4": "P1 rows (one defect, one row)",
    "A14": "the record counts of the summary's file block",
}

_THRESHOLDS = (
    "DQ_CRITICAL_FIELDS", "DQ_CRITICAL_P_RULES", "DQ_CRITICAL_P6A_FIELDS",
    "DQ_CRITICAL_P6B_FIELDS", "DQ_INVALID_MISSING_MARKERS",
    "DQ_INVALID_MISSING_MARKER_PATTERN", "DQ_POSSIBLE_PLACEHOLDERS", "DQ_DATE_MIN_YEAR",
    "DQ_DATE_MAX_YEARS_AFTER_CUTOFF", "DQ_PERCENT_SCALE_EXCLUDED_FIELDS",
)


def summary_filename(deal_name: str, cutoff: str | None) -> str:
    """`<cut-off> <deal> Flattened loans and collaterals data quality
    summary.json`: the workbook's name without its extension, then the
    suffix. `unknown-cutoff` stands in when no cut-off is established."""
    return f"{cutoff or UNKNOWN_CUTOFF} {deal_name} Flattened loans and collaterals{SUMMARY_SUFFIX}"


def finding_sentence(rule_id: str, kind: str, label: str, field: str | None = None) -> str:
    """One sentence for a finding group: the field, then the clause the
    catalogue holds for the rule and the result kind."""
    return f"{label}: {sentence_clause(rule_id, kind, field)}."


def _field_label(field_code: str | None) -> str:
    if field_code is None:
        return "the file"
    if field_code in FIELDS:
        return FIELDS[field_code].label
    amount = field_code.removesuffix("_CURRENCY")
    return f"currency companion of {FIELDS[amount].label}" if amount in FIELDS else field_code


def _jsonable(value: object) -> object:
    if isinstance(value, (frozenset, set)):
        return sorted(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return value


def _rules(outcomes: tuple[RuleOutcome, ...]) -> list[dict[str, object]]:
    totals: dict[str, RuleOutcome] = {}
    for outcome in outcomes:
        if outcome.rule_id not in totals:
            totals[outcome.rule_id] = RuleOutcome(outcome.rule_id, "both", None)
        totals[outcome.rule_id].add(outcome)
    rows: list[dict[str, object]] = []
    for rule in RULES.values():
        if rule.status == "not_implemented":
            continue
        total = totals.get(rule.rule_id) or RuleOutcome(rule.rule_id, "both", None)
        rows.append({
            "rule_id": rule.rule_id, "title": rule.title, "basis": rule.basis,
            "regulatory_source": rule.regulatory_source, "severity": rule.severity,
            "applies_to": rule.applies_to, "counts_on_critical_fields": rule.counts,
            "status": rule.status, "not_implemented_part": rule.not_implemented_part,
            "applicable": total.applicable, "evaluated": total.evaluated,
            "failed": total.failed, "not_evaluable": total.not_evaluable,
            "not_evaluable_reasons": dict(sorted(total.not_evaluable_reasons.items())),
            "other_result_kinds": dict(sorted(total.other_kinds.items())),
            "label": total.label,
        })
        if rule.rule_id in _REPORTED_ELSEWHERE:
            rows[-1]["label"] = f"reported as {_REPORTED_ELSEWHERE[rule.rule_id]}"
    return rows


def _rules_not_implemented() -> dict[str, object]:
    """Every catalogue rule that this release does not run, by what it is
    waiting for. None of them was evaluated; none is a pass."""
    groups: dict[str, list[str]] = {}
    for rule in RULES.values():
        if rule.status == "not_implemented":
            groups.setdefault(rule.not_implemented_part or NOT_IMPLEMENTED, []).append(rule.rule_id)
    return {
        "label": f"not evaluated: {NOT_IMPLEMENTED}",
        "count": sum(len(ids) for ids in groups.values()),
        "rules": groups,
    }


def _findings(counts: pl.DataFrame) -> list[dict[str, object]]:
    """The finding groups, from the counts of the result rows (the store's
    index): the detail itself is not read. A group is one rule, result
    kind, file and field. Its records are counted per column and reason
    and added, which is exact unless one field is supplied in two columns
    with findings on the same record in both."""
    grouped = (
        counts.group_by("rule_id", "result_kind", "file", "field")
        .agg(
            pl.col("rows").sum(),
            pl.col("records").sum(),
            pl.col("column_level").all(),
            pl.col("severity").unique().sort().alias("severities"),
            pl.struct("reason", "rows").alias("reasons"),
        )
        .sort("file", "field", "rule_id", "result_kind")
    )
    out: list[dict[str, object]] = []
    for row in grouped.iter_rows(named=True):
        rule = RULES[row["rule_id"]]
        out.append({
            "finding_id": f"{row['rule_id']}:{row['result_kind']}:{row['file']}:{row['field']}",
            "rule_id": row["rule_id"], "result_kind": row["result_kind"], "file": row["file"],
            "field": row["field"], "basis": rule.basis, "severities": row["severities"],
            "defect_key": rule.defect_key.format(field=row["field"], file=row["file"]),
            "enters_compliance_judgment": row["result_kind"] == "violation"
            and rule.regulatory_source is not None,
            "rows": row["rows"],
            "records": None if row["column_level"] else row["records"],
            "reasons": _reason_counts(row["reasons"]),
            "sentence": finding_sentence(
                row["rule_id"], row["result_kind"],
                "the pool" if row["field"] is None and row["file"] == "pool"
                else _field_label(row["field"]),
                row["field"],
            ),
        })
    return out


def _reason_counts(reasons: list[dict[str, object]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for item in reasons:
        totals[str(item["reason"])] += int(str(item["rows"]))
    return dict(sorted(totals.items(), key=lambda x: (-x[1], x[0])))


def _file_block(audit: FileAudit) -> dict[str, object]:
    block: dict[str, object] = {
        "file": audit.file, "name": audit.name, "sha256": audit.sha256, "size": audit.size,
        "readable": audit.readable,
    }
    if audit.unreadable is not None:
        u = audit.unreadable
        block["unreadable"] = {
            "statement": "the file could not be split into records; nothing that needs its "
                         "records was run and no counts are given for it",
            "reason": u.reason, "detail": u.detail, "record": u.record,
            "physical_line": u.physical_line, "header_fields": u.header_fields,
            "long_rows": [
                {"record": r.record, "physical_line": r.physical_line, "fields": r.fields}
                for r in u.long_rows
            ],
        }
    else:
        block.update(
            records=audit.n_records, columns=len(audit.columns),
            cells_with_repaired_encoding=audit.encoding_repaired.height,
        )
    return block


def _provenance(report: QualityReport) -> dict[str, object]:
    assessment = report.assessment
    started = assessment.started_at.isoformat()
    digest = hashlib.sha256(
        "|".join((assessment.loans.sha256, assessment.collateral.sha256,
                  config.DQ_RULESET_VERSION, started)).encode()
    ).hexdigest()[:16]
    manifest = load_reference_data().manifest
    lists = manifest["lists"]
    assert isinstance(lists, dict)
    return {
        "assessment_id": digest,
        "inputs": [
            {"file": a.file, "name": a.name, "sha256": a.sha256, "size": a.size}
            for a in assessment.files
        ],
        "options": dict(report.options),
        "package_version": esma_milan.__version__,
        "commit": None,
        "commit_note": "the build's commit is not recorded in this release",
        "rule_set_version": config.DQ_RULESET_VERSION,
        "taxonomy_sha256": hashlib.sha256(bundled_taxonomy_path().read_bytes()).hexdigest(),
        "esma_package_version": config.DQ_ESMA_PACKAGE_VERSION,
        "reference_lists": {
            name: {k: entry[k] for k in ("standard", "source", "address", "version",
                                         "source_sha256", "codes") if k in entry}
            | ({"countries": entry["countries"]} if "countries" in entry else {})
            for name, entry in lists.items()
        },
        "reference_lists_fetched": manifest["fetched"],
        "thresholds": {name: _jsonable(getattr(config, name)) for name in _THRESHOLDS},
        "assessment_started_utc": started,
        "as_of_date": assessment.as_of.isoformat(),
    }


def summary_dict(report: QualityReport) -> dict[str, object]:
    """The summary document (see the module docstring)."""
    assessment = report.assessment
    outcomes = assessment.outcomes
    sourced = {r.rule_id for r in RULES.values() if r.regulatory_source is not None}
    basis_p = {r.rule_id for r in RULES.values() if r.basis == "P"}

    counts = assessment.counts
    sourced_violations = counts.filter(
        (pl.col("result_kind") == "violation") & pl.col("rule_id").is_in(sorted(sourced))
    )
    v_rows = int(sourced_violations["rows"].sum())
    v_rules = sourced_violations["rule_id"].n_unique()
    v_fields = sourced_violations.select("file", "field").unique().height
    v_column_level = int(sourced_violations.filter(pl.col("column_level"))["rows"].sum())
    v_records = assessment.records_with_compliance_violation
    if v_rows == 0:
        v_rules = v_fields = 0
    not_evaluable = [o for o in outcomes if o.rule_id in sourced and o.not_evaluable]
    partly = {r.rule_id: r.not_implemented_part for r in RULES.values() if r.status == "partly_implemented"}
    unreadable = [a.file for a in assessment.files if not a.readable]

    reasons_by_rule: dict[str, Counter[str]] = {}
    for outcome in outcomes:
        if outcome.not_evaluable:
            reasons_by_rule.setdefault(outcome.rule_id, Counter()).update(outcome.not_evaluable_reasons)

    pipeline = report.pipeline
    pipeline_block: dict[str, object] = {"completed": pipeline.completed}
    if not pipeline.completed and pipeline.reason is not None:
        pipeline_block.update(
            statement="the pipeline failed; no MILAN workbook was produced by this run",
            error_type=pipeline.error_type, message=pipeline.error_message,
            stage=pipeline.stage, reason_code=pipeline.reason,
        )
    elif not pipeline.completed:
        # Not a typed data error: there is no reason code, and the message
        # of an arbitrary exception can quote cells of the file, so only
        # the exception's type is given.
        pipeline_block.update(
            statement="the pipeline failed with an error that carries no reason code "
                      f"({pipeline.error_type}); no MILAN workbook was produced by this run",
            error_type=pipeline.error_type,
        )
    workbook = pipeline.workbook_path
    if workbook is not None:
        workbook_artifact: dict[str, object] = {
            "name": workbook.name, "kind": "MILAN workbook", "state": "available",
            "size": workbook.stat().st_size if workbook.exists() else None,
            "media_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }
    elif pipeline.completed:
        workbook_artifact = {"name": None, "kind": "MILAN workbook", "state": "not requested"}
    else:
        workbook_artifact = {"name": None, "kind": "MILAN workbook", "state": "failed",
                             "reason": "the pipeline did not complete"}

    populations: dict[str, object] = {"statement": "loan records by population (specification 8.1)"}
    lifecycle: dict[str, int] | str = "not available: the loans file could not be split into records"
    records = assessment.records("loans", ["population", "lifecycle"])
    if records is not None:
        by_population = Counter(records["population"].to_list())
        comparison = report.stage2_comparison
        populations.update(
            counts={label: by_population.get(label, 0) for label in POPULATIONS},
            labels="observed (equal to the Stage 2 result on every record)"
            if report.populations_observed else "predicted from the Stage 2 predicate",
            stage2_comparison={
                "evaluated": comparison.evaluated, "reason": comparison.reason,
                "records": comparison.records, "retained_by_audit": comparison.retained_by_audit,
                "retained_by_stage": comparison.retained_by_stage,
                "records_that_differ": len(comparison.differing_records),
                "first_differing_records": list(comparison.differing_records[:20]),
                "state": "agrees" if comparison.agrees
                else "reconciliation incomplete" if comparison.evaluated else "not evaluated",
            } if comparison is not None else None,
        )
        seen = Counter(records["lifecycle"].to_list())
        lifecycle = {label: seen.get(label, 0) for label in LIFECYCLES}
    else:
        populations["counts"] = "not available: the loans file could not be split into records"

    return {
        "schema_version": SCHEMA_VERSION,
        "assessment_id": _provenance(report)["assessment_id"],
        "execution": "completed" if pipeline.completed else "pipeline_failed",
        "assessment": {"state": "ran"},
        "assessment_completeness": {
            "state": "file_unreadable" if unreadable else "partial",
            "files_unreadable": unreadable,
            "statement": "This release assesses the per-column rules only. Rules not "
                         "implemented in this release were not run; they are listed apart "
                         "from rules that could not be evaluated on these files.",
            "rules_not_implemented_in_this_release": _rules_not_implemented(),
            "rules_partly_implemented": partly,
            "rules_not_evaluable_on_these_files": {
                rule: {"records": sum(reasons.values()), "reasons": dict(sorted(reasons.items()))}
                for rule, reasons in sorted(reasons_by_rule.items())
            },
        },
        "compliance": {
            "violations": {
                "state": "present" if v_rows else "none",
                "rules": v_rules,
                "fields": v_fields,
                "records": v_records,
                "column_level_findings": v_column_level,
            },
            "coverage": {
                "state": "partial",
                "checks_with_records_not_evaluable": len({o.rule_id for o in not_evaluable}),
                "records_not_evaluable": sum(o.not_evaluable for o in not_evaluable),
                "note": "partial in every report of this release: the cross-column, "
                        "cross-record and cross-file rules are not implemented yet",
            },
        },
        "plausibility": {
            "observations": int(
                counts.filter(
                    (pl.col("result_kind") == "question") & pl.col("rule_id").is_in(sorted(basis_p))
                )["rows"].sum()
            ),
        },
        "policy_outcome": NOT_IMPLEMENTED,
        "release_permitted": NOT_IMPLEMENTED,
        "override": NOT_IMPLEMENTED,
        "waterfall": NOT_IMPLEMENTED,
        "dropped_loans": NOT_IMPLEMENTED,
        "lineage": NOT_IMPLEMENTED,
        "detail_files": NOT_IMPLEMENTED,
        "requested_outputs": {
            "workbook": bool(not report.options.get("dry_run", False)), "summary": True,
        },
        "artifacts": [
            workbook_artifact,
            {"name": "this file", "kind": "data quality summary", "state": "available",
             "media_type": "application/json"},
        ],
        "pipeline": pipeline_block,
        "files": [_file_block(a) for a in assessment.files],
        "rules": _rules(outcomes),
        "findings": _findings(assessment.counts),
        "expected_fields": inventory_of_expected_fields(assessment),
        "actual_columns": {a.file: list(a.actual_columns) for a in assessment.files},
        "populations": populations,
        "identity_confidence": {
            "selected_loan_identifier": report.selected_loan_identifier
            or "not available: Stage 3 did not report a selection",
            "counts": dict(sorted(report.identity_confidence.items())),
        },
        "lifecycle": {"statement": "an operational presumption from the status and RREL9 the "
                                   "record itself reports (specification 3.3)",
                      "counts": lifecycle},
        "scope_statements": list(SCOPE_STATEMENTS),
        "provenance": _provenance(report),
    }


def interrupted_summary(reason: str, pipeline_completed: bool | None) -> dict[str, object]:
    """The summary when the assessment itself failed: a technical fault,
    never a statement about the data."""
    return {
        "schema_version": SCHEMA_VERSION,
        "execution": "completed" if pipeline_completed else "pipeline_failed",
        "assessment": {"state": "interrupted", "statement": ASSESSMENT_INTERRUPTED, "reason": reason},
        "assessment_completeness": {"state": "not assessed"},
        "policy_outcome": NOT_IMPLEMENTED,
    }


def render_json(value: object, indent: int = 0, width: int = 150) -> str:
    """JSON with two-space indentation in which any object or array that
    fits on one line is written on one line. The summary has many small
    objects (counts per state, per rule); this keeps it readable and a
    quarter of the size of fully indented JSON."""
    compact = json.dumps(value, ensure_ascii=False, default=str)
    if not isinstance(value, (dict, list)) or len(compact) + indent <= width or not value:
        return compact
    pad, inner = " " * indent, " " * (indent + 2)
    if isinstance(value, dict):
        parts = [
            f"{inner}{json.dumps(str(k), ensure_ascii=False)}: {render_json(v, indent + 2, width)}"
            for k, v in value.items()
        ]
        return "{\n" + ",\n".join(parts) + f"\n{pad}}}"
    if all(not isinstance(v, (dict, list)) for v in value):
        # A long list of scalars is wrapped, not written one per line.
        lines, line = [], inner
        for item in (json.dumps(v, ensure_ascii=False, default=str) for v in value):
            if len(line) + len(item) + 2 > width and line.strip():
                lines.append(line.rstrip())
                line = inner
            line += item + ", "
        lines.append(line.rstrip().rstrip(","))
        return "[\n" + "\n".join(lines) + f"\n{pad}]"
    parts = [f"{inner}{render_json(v, indent + 2, width)}" for v in value]
    return "[\n" + ",\n".join(parts) + f"\n{pad}]"


def write_json(document: dict[str, object], path: Path) -> None:
    """Write to a temporary name in the same directory, then rename, so a
    half-written file is never seen under the final name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(render_json(document) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_summary(report: QualityReport, path: Path) -> None:
    write_json(summary_dict(report), path)
