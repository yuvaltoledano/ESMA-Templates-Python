"""The reconciliation waterfall, the dropped-loans list, G8 and A17.

Mirrors nothing in the R reference (specification 8.2, 8.5, 7.1 A17, 7.6 G8).

The waterfall follows every loan record from the file to the MILAN output:

    1  loan records submitted
    2  less: eligibility unknown; inactive (of which: status not recognised)
    3  active status, before the balance filter
    4  less: zero, negative, missing, unparseable balance; active status
       not recognised
    5  active, retained                     reconciled with Stage 2 (G8)
    6  less: lost after Stage 2, by the bookkeeping priority below
    7  loans in the MILAN output            reconciled with 5 less 6 (G8)

Each line says whether it was **observed** from a stage's own result or
**predicted** from the stage's predicate applied by the audit, because the
stage did not run or its result is not available. When the pipeline raised,
the stages that completed before the failure are taken from the exception's
`partial_result` (`runner.run_pipeline`).

A loan lost after Stage 2 can have several problems. It is assigned to one
line by a fixed bookkeeping priority, so the lines add up; the priority is a
convention and not the established cause, and the dropped-loans list gives
the assigned line and every other cause seen on the loan:

    1  no collateral record at all
    2  collateral records only of excluded types
    3  a collateral record without a collateral type (the Python pipeline
       drops such a record today; the R reference keeps it)
    4  a property record without a property type
    5  a property record without a valuation above 0
    6  the selected identifier is missing on the loan record
    7  no match under the selected identifier convention
    8  removed with a group by a later stage (0 today)

Lines 6 and 7 need Stage 3's choice of identifier; they are "not available"
when Stage 3 did not complete, and they cannot occur on a completed run
until issue #45 is fixed (Stage 3 raises on such a pool).

Balances are the loan balance as the pipeline's loader reads it (the number
the pipeline itself adds up), per currency of RREL30 and never added across
currencies. The exact text of each balance is in the detail files.

A difference at line 5 or line 7 is "reconciliation incomplete" with both
figures. It is a fault of this report or of the pipeline, never a cause.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from esma_milan.quality.catalogue import RULES
from esma_milan.quality.links import CONVENTIONS
from esma_milan.quality.populations import (
    COLLATERAL_REASONS,
    CURRENCY_NOT_STATED,
    LOAN_ID_COLUMNS,
    POPULATIONS,
    RETAINED,
)
from esma_milan.quality.refdata import load_reference_data
from esma_milan.quality.result import (
    RESULT_SCHEMA,
    RuleOutcome,
    concat_results,
    empty_results,
    results_frame,
)

if TYPE_CHECKING:
    from esma_milan.runner import PipelineResult

OBSERVED = "observed"
PREDICTED = "predicted"
NOT_AVAILABLE_STAGE3 = "not available: Stage 3 did not complete"
NOT_AVAILABLE_OUTPUT = "not available: the pipeline did not complete"
INCOMPLETE = "reconciliation incomplete"
AGREES = "agrees"

NOT_STATED_LABEL = "currency not stated"

# Line 2 and line 4, by population label (specification 8.1).
_LINE2: tuple[tuple[str, str], ...] = (
    ("eligibility unknown", POPULATIONS[8]),
    ("inactive", POPULATIONS[6]),
    ("inactive, status not recognised", POPULATIONS[7]),
)
_LINE4: tuple[tuple[str, str], ...] = (
    ("zero balance", POPULATIONS[1]),
    ("negative balance", POPULATIONS[2]),
    ("balance missing", POPULATIONS[3]),
    ("balance unparseable", POPULATIONS[4]),
    ("active status not recognised", POPULATIONS[5]),
)

# The bookkeeping priority of line 6: (key, label).
CAUSES: tuple[tuple[str, str], ...] = (
    ("no_collateral_record", "no collateral record at all"),
    ("only_excluded_types", "collateral records only of excluded types"),
    ("collateral_type_missing",
     "a collateral record without a collateral type (the Python pipeline drops such a record "
     "today; the R reference keeps it)"),
    ("property_type_missing", "a property record without a property type"),
    ("no_positive_valuation", "a property record without a valuation above 0"),
    ("selected_identifier_missing", "the selected identifier is missing on the loan record"),
    ("no_match_under_selected_convention", "no match under the selected identifier convention"),
    ("removed_by_a_later_stage", "removed with a group by a later stage"),
)
CAUSE_LABELS: dict[str, str] = dict(CAUSES)
CAUSE_NOT_ESTABLISHED = "cause not established"
# Seen on a loan and never a line of its own.
ALSO_EXCLUDED_TYPE = "a collateral record of an excluded type"
ALSO_TRIMMED_ONLY = "a collateral record matches only after trimming whitespace"

DROPPED_SCHEMA: dict[str, pl.DataType] = {
    "record": pl.UInt32(),
    "physical_line": pl.UInt32(),
    "loan_identifier_original": pl.String(),
    "loan_identifier_new": pl.String(),
    "obligor_identifier_new": pl.String(),
    "status": pl.String(),
    "waterfall_step": pl.UInt8(),
    "assigned_line": pl.String(),
    "other_causes": pl.String(),
    "observed_or_predicted": pl.String(),
    "balance": pl.String(),
    "currency": pl.String(),
    "collateral_records": pl.String(),
    "collateral_records_kept_by_stage2": pl.String(),
}

# A17: the sets of amounts the pipeline combines on one loan (specification
# 8.5). The first set, RREL30 of all retained loans, is tested across the pool.
_LOAN_SETS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("ranking_amounts",
     "Ranking, External Prior Ranks, Pari Passu and Flexible Loan Amount",
     ("RREL30", "RREL31", "RREL32", "RREL33")),
    ("total_income", "Total Income", ("RREL16", "RREL20")),
)
STAGE7_DID_NOT_COMPLETE = "Stage 7 did not complete"


@dataclass(frozen=True)
class Waterfall:
    """What `build_waterfall` returns."""

    lines: tuple[Mapping[str, object], ...]
    """The seven lines, ready for the summary."""
    dropped: pl.DataFrame
    """One row per loan lost at line 4 or line 6 (`DROPPED_SCHEMA`)."""
    reconciliation: Mapping[str, object]
    """G8: line 5 against Stage 2 and line 7 against the output."""
    collateral_counts: Mapping[str, object]
    results: pl.DataFrame
    """The result rows of G8 and A17."""
    outcomes: tuple[RuleOutcome, ...]

    @property
    def dropped_counts(self) -> dict[str, object]:
        by_line = (
            self.dropped.group_by("waterfall_step", "assigned_line", "observed_or_predicted")
            .len().sort("waterfall_step", "assigned_line")
        )
        return {
            "loans": self.dropped.height,
            "by_line": [
                {"waterfall_step": step, "assigned_line": line, "basis": basis, "loans": n}
                for step, line, basis, n in by_line.iter_rows()
            ],
        }


def _measure(frame: pl.DataFrame) -> dict[str, object]:
    """Records, and the balance per currency of RREL30 as the loader reads
    it. Totals are never added across currencies."""
    with_amount = frame.filter(pl.col("balance").is_not_null())
    by_currency = (
        with_amount.group_by(pl.col("cur_RREL30").cast(pl.String).alias("currency"))
        .agg(pl.col("balance").sum().alias("balance"), pl.len().alias("records"))
        .sort("currency")
    )
    balance: dict[str, object] = {}
    for currency, total, records in by_currency.iter_rows():
        label = NOT_STATED_LABEL if currency in (None, CURRENCY_NOT_STATED) else currency
        balance[label] = {"balance": round(total, 5), "records": records}
    return {
        "records": frame.height,
        "balance_by_currency": balance,
        "records_without_a_balance": frame.height - with_amount.height,
    }


def _records(positions: pl.Series, of: pl.Series) -> list[int]:
    """1-based file records for 0-based positions in a stage's input frame
    whose rows are the file positions `of`."""
    return [int(of[i]) + 1 for i in positions.to_list()]


def _join(values: Sequence[int]) -> str:
    return ";".join(str(v) for v in sorted(values))


def _collateral_reasons(collateral: pl.DataFrame, result: PipelineResult | None) -> tuple[pl.DataFrame, str]:
    """Per collateral record: the four Stage 2 reasons and whether Stage 2
    kept the record, observed from the stage or predicted by the audit."""
    stage2 = result.stage2 if result is not None else None
    if (
        stage2 is not None and stage2.property_drop_reasons is not None
        and stage2.property_rows is not None
        and stage2.property_drop_reasons.height == collateral.height
    ):
        kept = set(stage2.property_rows.to_list())
        return (
            stage2.property_drop_reasons.select(COLLATERAL_REASONS).with_columns(
                collateral["record"].alias("collateral_record"),
                pl.Series("kept", [i in kept for i in range(collateral.height)], dtype=pl.Boolean),
            ),
            OBSERVED,
        )
    return (
        collateral.select(
            pl.col("record").alias("collateral_record"),
            *[pl.col(f"s2_{name}").alias(name) for name in COLLATERAL_REASONS],
        ).with_columns((~pl.any_horizontal(COLLATERAL_REASONS)).alias("kept")),
        PREDICTED,
    )


def _per_loan(links: pl.DataFrame, reasons: pl.DataFrame, convention: str | None) -> pl.DataFrame:
    """Per loan record: what its linked collateral records look like. The
    links counted are raw matches; under `convention` when Stage 3 chose
    one, under either convention when it did not."""
    raw = links.filter(pl.col("match") == "raw")
    chosen = raw.filter(pl.col("convention") == convention) if convention else raw
    chosen = chosen.select("loan_record", "collateral_record").unique().join(reasons, on="collateral_record")
    per_chosen = chosen.group_by("loan_record").agg(
        pl.len().alias("n_chosen"),
        pl.col("collateral_type_excluded").all().alias("all_excluded"),
        pl.col("collateral_type_excluded").any().alias("any_excluded"),
        pl.col("collateral_type_missing").any().alias("any_type_missing"),
        pl.col("property_type_missing").any().alias("any_property_type_missing"),
        pl.col("no_positive_valuation").any().alias("any_no_valuation"),
        pl.col("kept").any().alias("any_kept"),
        pl.col("collateral_record").sort().alias("collateral_records"),
        pl.col("collateral_record").filter(pl.col("kept")).sort().alias("kept_records"),
    )
    per_any = raw.group_by("loan_record").agg(pl.col("collateral_record").n_unique().alias("n_any"))
    trimmed = (
        links.filter(pl.col("match") == "trimmed").group_by("loan_record")
        .agg(pl.len().alias("n_trimmed"))
    )
    return per_any.join(per_chosen, on="loan_record", how="full", coalesce=True).join(
        trimmed, on="loan_record", how="full", coalesce=True
    )


def _assign(candidates: pl.DataFrame, *, selection_known: bool) -> pl.DataFrame:
    """Give every candidate loan its line by the bookkeeping priority, and
    every other cause seen on it."""
    n_any = pl.col("n_any").fill_null(0)
    n_chosen = pl.col("n_chosen").fill_null(0)
    tests: list[tuple[str, pl.Expr]] = [
        ("no_collateral_record", n_any == 0),
        ("only_excluded_types", (n_chosen > 0) & pl.col("all_excluded").fill_null(False)),
        ("collateral_type_missing", pl.col("any_type_missing").fill_null(False)),
        ("property_type_missing", pl.col("any_property_type_missing").fill_null(False)),
        ("no_positive_valuation", pl.col("any_no_valuation").fill_null(False)),
    ]
    if selection_known:
        tests += [
            ("selected_identifier_missing", pl.col("identifier_missing")),
            ("no_match_under_selected_convention", (n_chosen == 0) & (n_any > 0)),
        ]
    assigned = pl.lit(CAUSE_NOT_ESTABLISHED)
    for key, test in reversed(tests):
        assigned = pl.when(test).then(pl.lit(key)).otherwise(assigned)
    also = [
        *[pl.when(test).then(pl.lit(CAUSE_LABELS[key])).otherwise(None) for key, test in tests],
        pl.when(pl.col("any_excluded").fill_null(False) & ~pl.col("all_excluded").fill_null(False))
        .then(pl.lit(ALSO_EXCLUDED_TYPE)).otherwise(None),
        pl.when((n_any == 0) & (pl.col("n_trimmed").fill_null(0) > 0))
        .then(pl.lit(ALSO_TRIMMED_ONLY)).otherwise(None),
    ]
    with_cause = candidates.with_columns(assigned.alias("cause"))
    label = pl.col("cause").replace_strict(CAUSE_LABELS, default=pl.col("cause"), return_dtype=pl.String)
    return with_cause.with_columns(
        label.alias("assigned_line"),
        pl.concat_list(also).list.drop_nulls().list.set_difference(pl.concat_list(label))
        .list.join("; ").alias("other_causes"),
    )


def _dropped_rows(frame: pl.DataFrame, step: int, basis: str) -> pl.DataFrame:
    """Rows of the dropped-loans list. `frame` has the loan record columns
    and `assigned_line`, `other_causes`, and optionally the collateral
    record lists."""
    def records(column: str) -> pl.Expr:
        if column not in frame.columns:
            return pl.lit("")
        return pl.col(column).fill_null([]).cast(pl.List(pl.String)).list.join(";")

    return frame.select(
        "record",
        "physical_line",
        pl.col("RREL2").alias("loan_identifier_original"),
        pl.col("RREL3").alias("loan_identifier_new"),
        pl.col("RREL5").alias("obligor_identifier_new"),
        "status",
        pl.lit(step).alias("waterfall_step"),
        "assigned_line",
        "other_causes",
        pl.lit(basis).alias("observed_or_predicted"),
        pl.col("balance_text").alias("balance"),
        pl.when(pl.col("cur_RREL30").cast(pl.String) == CURRENCY_NOT_STATED).then(None)
        .otherwise(pl.col("cur_RREL30").cast(pl.String)).alias("currency"),
        records("collateral_records").alias("collateral_records"),
        records("kept_records").alias("collateral_records_kept_by_stage2"),
    ).cast(DROPPED_SCHEMA)  # type: ignore[arg-type]


def build_waterfall(
    loans: pl.DataFrame,
    collateral: pl.DataFrame | None,
    links: pl.DataFrame | None,
    result: PipelineResult | None,
    *,
    completed: bool,
) -> Waterfall:
    """Build the waterfall from the per-record frames of the audit
    (`populations.loan_records`, `collateral_records`), the links, and the
    pipeline's result: the whole result when it completed, the partial
    result of its exception when it raised, None when there is neither.

    `collateral` and `links` are None when the collateral file could not
    be split into records; line 6 is then not available."""
    stage2 = result.stage2 if result is not None else None
    stage3 = result.stage3 if result is not None else None
    decisions = stage3.decisions if stage3 is not None else None
    stage7 = result.stage7 if result is not None and completed else None
    population = pl.col("population")
    rows: list[dict[str, object]] = []
    frames: list[pl.DataFrame] = []
    outcomes: list[RuleOutcome] = []

    # -- lines 1 to 5 -------------------------------------------------------
    audit_retained = loans.filter(population == RETAINED)
    stage_retained: list[int] | None = None
    if stage2 is not None and stage2.loan_rows is not None:
        stage_retained = [i + 1 for i in stage2.loan_rows.to_list()]
    reconciled5 = stage_retained is not None and stage_retained == audit_retained["record"].to_list()
    basis = OBSERVED if reconciled5 else PREDICTED
    line2 = loans.filter(population.is_in([p for _, p in _LINE2]))
    line3 = loans.filter(~population.is_in([p for _, p in _LINE2]))
    line4 = loans.filter(population.is_in([p for _, p in _LINE4]))
    lines: list[dict[str, object]] = [
        {"line": 1, "label": "loan records submitted", "basis": OBSERVED, **_measure(loans)},
        {
            "line": 2, "label": "less: not an active status", "basis": basis, **_measure(line2),
            "parts": [
                {"label": "eligibility unknown", **_measure(loans.filter(population == POPULATIONS[8]))},
                {
                    "label": "inactive",
                    **_measure(loans.filter(population.is_in([POPULATIONS[6], POPULATIONS[7]]))),
                    "of_which_status_not_recognised": _measure(loans.filter(population == POPULATIONS[7])),
                },
            ],
        },
        {"line": 3, "label": "active status, before the balance filter", "basis": basis, **_measure(line3)},
        {
            "line": 4, "label": "less: not retained by the balance filter", "basis": basis,
            **_measure(line4),
            "parts": [
                {"label": label, **_measure(loans.filter(population == name))} for label, name in _LINE4
            ],
        },
    ]
    line5: dict[str, object] = {
        "line": 5, "label": "active, retained", "basis": basis, **_measure(audit_retained),
    }
    step5: dict[str, object] = {
        "audit": audit_retained.height,
        "stage2": len(stage_retained) if stage_retained is not None else None,
        "state": AGREES if reconciled5
        else INCOMPLETE if stage_retained is not None
        else "not evaluated: the pipeline did not return a Stage 2 result",
    }
    if stage_retained is not None and not reconciled5:
        differing = sorted(set(stage_retained) ^ set(audit_retained["record"].to_list()))
        step5["records_that_differ"] = len(differing)
        step5["first_differing_records"] = differing[:20]
    line5["reconciliation_with_stage2"] = step5
    lines.append(line5)

    line4_basis = PREDICTED if not reconciled5 else OBSERVED
    labels4 = {name: label for label, name in _LINE4}
    dropped4 = line4.with_columns(
        population.replace_strict(labels4, return_dtype=pl.String).alias("assigned_line"),
        pl.lit("").alias("other_causes"),
    )
    dropped: list[pl.DataFrame] = [
        # An unparseable balance is always predicted: the loader stops at it.
        _dropped_rows(dropped4.filter(population != POPULATIONS[4]), 4, line4_basis),
        _dropped_rows(dropped4.filter(population == POPULATIONS[4]), 4, PREDICTED),
    ]

    # -- line 6 -------------------------------------------------------------
    # The loans Stage 2 kept are the stage's own when it ran, else the audit's.
    retained = (
        loans.filter(pl.col("record").is_in(stage_retained)) if stage_retained is not None
        else audit_retained
    )
    selected: str | None = None
    convention: str | None = None
    if decisions is not None and decisions.loan_id.chosen_column is not None:
        selected = LOAN_ID_COLUMNS.get(decisions.loan_id.chosen_column)
        convention = next((name for name, code in CONVENTIONS.items() if code == selected), None)
    line6: dict[str, object] = {"line": 6, "label": "less: lost after Stage 2"}
    lost_records: list[int] | None = None
    collateral_basis: str | None = None
    if collateral is None or links is None:
        line6.update(
            basis="not available: the collateral file could not be split into records",
            records=None, parts=[],
        )
    else:
        reasons, collateral_basis = _collateral_reasons(collateral, result)
        per_loan = _per_loan(links, reasons, convention)
        observed6 = (
            decisions is not None and stage2 is not None and stage2.loan_rows is not None
            and convention is not None
        )
        if observed6:
            assert decisions is not None and stage2 is not None and stage2.loan_rows is not None
            missing = _records(decisions.loans_missing_calc_loan_id, stage2.loan_rows)
            without = _records(decisions.loans_without_collateral, stage2.loan_rows)
            candidates = (
                retained.filter(pl.col("record").is_in(missing + without))
                .join(per_loan, left_on="record", right_on="loan_record", how="left")
                .with_columns(pl.col("record").is_in(missing).alias("identifier_missing"))
            )
        else:
            # Predicted: the loans lost under either identifier convention,
            # that is, with no collateral record at all or with none that
            # Stage 2's collateral predicate keeps.
            candidates = (
                retained.join(per_loan, left_on="record", right_on="loan_record", how="left")
                .filter((pl.col("n_any").fill_null(0) == 0) | ~pl.col("any_kept").fill_null(False))
                .with_columns(pl.lit(False).alias("identifier_missing"))
            )
        assigned = _assign(candidates, selection_known=observed6)
        basis6 = OBSERVED if observed6 else PREDICTED
        parts: list[dict[str, object]] = []
        for order, (key, label) in enumerate(CAUSES, start=1):
            part: dict[str, object] = {"order": order, "label": label}
            if key == "removed_by_a_later_stage":
                # No stage after Stage 3 removes a loan today. A difference
                # at line 7 is a reconciliation fault, not this line.
                if stage7 is not None:
                    part.update(basis=OBSERVED, **_measure(assigned.clear()))
                else:
                    part.update(basis=NOT_AVAILABLE_OUTPUT, records=None)
            elif not observed6 and order >= 6:
                part.update(basis=NOT_AVAILABLE_STAGE3, records=None)
            else:
                part.update(basis=basis6, **_measure(assigned.filter(pl.col("cause") == key)))
            parts.append(part)
        unexplained = assigned.filter(pl.col("cause") == CAUSE_NOT_ESTABLISHED)
        if unexplained.height:
            parts.append({
                "label": f"{CAUSE_NOT_ESTABLISHED} ({INCOMPLETE})", "basis": basis6,
                **_measure(unexplained),
            })
        line6.update(
            basis=basis6 if observed6 else
            "predicted: the loans that would be lost under either identifier convention; the "
            "lines that need Stage 3's choice are not available",
            collateral_reasons=collateral_basis,
            selected_loan_identifier=selected,
            **_measure(assigned),
            parts=parts,
        )
        lost_records = assigned["record"].to_list()
        dropped.append(_dropped_rows(assigned, 6, basis6))
    lines.append(line6)

    # -- line 7 -------------------------------------------------------------
    line7: dict[str, object] = {"line": 7, "label": "loans in the MILAN output"}
    step7: dict[str, object]
    if stage7 is None or result is None or result.loan_record_index is None:
        line7.update(basis=NOT_AVAILABLE_OUTPUT, records=None)
        step7 = {"state": "not evaluated: the pipeline did not complete"}
    else:
        in_output = [i + 1 for i in result.loan_record_index.to_list()]
        output_rows = stage7.combined_flattened.height
        line7.update(basis=OBSERVED, **_measure(loans.filter(pl.col("record").is_in(in_output))))
        line7["records"] = output_rows
        expected = retained.height - len(lost_records) if lost_records is not None else None
        agrees = (
            expected == output_rows
            and lost_records is not None
            and sorted(in_output) == sorted(set(retained["record"].to_list()) - set(lost_records))
        )
        step7 = {
            "line_5_less_line_6": expected, "output": output_rows,
            "state": AGREES if agrees else INCOMPLETE,
        }
    line7["reconciliation_with_line_5_less_line_6"] = step7
    lines.append(line7)

    # -- G8 -----------------------------------------------------------------
    g8 = RuleOutcome("G8", "pool", None, applicable=2)
    for step, state in ((5, step5), (7, step7)):
        if str(state["state"]).startswith("not evaluated"):
            g8.not_evaluable += 1
            g8.not_evaluable_reasons[str(state["state"]).removeprefix("not evaluated: ")] += 1
            continue
        g8.evaluated += 1
        if state["state"] == INCOMPLETE:
            g8.failed += 1
            figures = (
                f"the audit retains {state['audit']} loan record(s) and Stage 2 {state['stage2']}, "
                f"and {state['records_that_differ']} record(s) differ"
                if step == 5 else
                f"line 5 less line 6 is {state['line_5_less_line_6']} and the output holds "
                f"{state['output']} loan(s)"
            )
            rows.append({
                "file": "pool", "rule_id": "G8", "result_kind": "limitation",
                "severity": RULES["G8"].severity, "reason": f"line_{step}_reconciliation_incomplete",
                "expected": f"equal figures at line {step} of the waterfall; {figures}. This is a "
                            "fault of the report or of the pipeline, not a finding about the data",
            })
    outcomes.append(g8)

    # -- A17 ----------------------------------------------------------------
    a17_rows, a17_frames, a17_outcomes = _currency_compatibility(
        retained, collateral, links, result if completed else None, convention,
    )
    rows.extend(a17_rows)
    frames.extend(a17_frames)
    outcomes.extend(a17_outcomes)

    return Waterfall(
        lines=tuple(lines),
        dropped=pl.concat(dropped).sort("waterfall_step", "record") if dropped
        else pl.DataFrame(schema=DROPPED_SCHEMA),
        reconciliation={"line_5": step5, "line_7": step7},
        collateral_counts=_collateral_counts(collateral, result, completed),
        results=concat_results([results_frame(rows), *frames]),
        outcomes=tuple(outcomes),
    )


def _collateral_counts(
    collateral: pl.DataFrame | None, result: PipelineResult | None, completed: bool
) -> dict[str, object]:
    """Collateral counts, each labelled as what it is (specification 8.2)."""
    counts: dict[str, object] = {}
    if collateral is not None:
        identifier = pl.col("RREC4")
        counts["raw collateral records"] = collateral.height
        counts["distinct collateral identifiers (RREC4, as written)"] = collateral.filter(
            identifier.is_not_null() & (identifier.str.strip_chars() != "")
        )["RREC4"].n_unique()
    if result is not None and result.stage2 is not None:
        counts["property rows retained by Stage 2"] = result.stage2.properties.height
    if result is not None and result.stage3 is not None:
        counts["property rows linked to a retained loan (Stage 3)"] = result.stage3.properties.height
    if result is not None and result.stage4 is not None and completed:
        counts["graph groups"] = result.stage4.collateral_groups["collateral_group_id"].n_unique()
    return counts


# ---------------------------------------------------------------------------
# A17
# ---------------------------------------------------------------------------


def _a17_record_rows(frame: pl.DataFrame, field: str, reason: str, expected: str, note: pl.Expr) -> pl.DataFrame:
    """A17 rows on loan records. `frame` has `record` and `physical_line`."""
    if frame.height == 0:
        return empty_results()
    return frame.select(
        pl.lit("loans").alias("file"), "record", "physical_line",
        pl.lit(None, dtype=pl.UInt16).alias("column_position"),
        pl.lit(None, dtype=pl.String).alias("header"),
        pl.lit(field).alias("field"), pl.lit("A17").alias("rule_id"),
        pl.lit("violation").alias("result_kind"), pl.lit(RULES["A17"].severity).alias("severity"),
        pl.lit(None, dtype=pl.String).alias("value_state"),
        pl.lit(None, dtype=pl.String).alias("raw_value"),
        pl.lit(None, dtype=pl.Date).alias("nd4_date"),
        pl.lit(expected).alias("expected"), pl.lit(reason).alias("reason"), note.alias("note"),
    ).cast(RESULT_SCHEMA)  # type: ignore[arg-type]


def _stated(codes: Sequence[str]) -> pl.Expr:
    """Per record: the distinct currencies stated for the amounts `codes`
    that the loader reads as numbers, as a sorted list of text."""
    return (
        pl.concat_list([
            pl.when(pl.col(f"cur_{code}").cast(pl.String) == CURRENCY_NOT_STATED).then(None)
            .otherwise(pl.col(f"cur_{code}").cast(pl.String))
            for code in codes
        ])
        .list.drop_nulls().list.unique().list.sort()
    )


def _unstated(codes: Sequence[str]) -> pl.Expr:
    """Per record: an amount of `codes` is a number and its currency is not stated."""
    return pl.any_horizontal([
        (pl.col(f"cur_{code}").cast(pl.String) == CURRENCY_NOT_STATED).fill_null(False) for code in codes
    ])


def _currency_compatibility(
    retained: pl.DataFrame,
    collateral: pl.DataFrame | None,
    links: pl.DataFrame | None,
    result: PipelineResult | None,
    convention: str | None,
) -> tuple[list[dict[str, object]], list[pl.DataFrame], list[RuleOutcome]]:
    """A17: among the retained loans and the valuations selected for them,
    more than one currency, or a code that is not a valid currency, within a
    set of amounts the pipeline adds, subtracts, divides or allocates
    together. A currency that is not stated cannot be compared: such
    records are counted as not evaluable (P13 reports them)."""
    valid = sorted(load_reference_data().currencies)
    rows: list[dict[str, object]] = []
    frames: list[pl.DataFrame] = []
    outcomes: list[RuleOutcome] = []
    not_stated = "the currency of an amount is not stated"

    def pool_row(reason: str, expected: str) -> None:
        rows.append({
            "file": "pool", "rule_id": "A17", "result_kind": "violation",
            "severity": RULES["A17"].severity, "reason": reason, "expected": expected,
        })

    # Set 1: RREL30 of all retained loans (pool totals, allocation weights, group balances).
    currency = pl.col("cur_RREL30").cast(pl.String)
    with_amount = retained.filter(currency.is_not_null())
    stated = with_amount.filter(currency != CURRENCY_NOT_STATED)
    counts = stated.group_by(currency.alias("c")).len().sort(["len", "c"], descending=[True, False])
    outcome = RuleOutcome(
        "A17", "loans", "RREL30", applicable=with_amount.height, evaluated=stated.height,
        not_evaluable=with_amount.height - stated.height,
    )
    if outcome.not_evaluable:
        outcome.not_evaluable_reasons[not_stated] = outcome.not_evaluable
    if counts.height:
        modal = next((c for c in counts["c"].to_list() if c in valid), None)
        invalid = stated.filter(~currency.is_in(valid))
        other = stated.filter(currency.is_in(valid) & (currency != modal))
        found = ", ".join(f"{c}: {n} record(s)" for c, n in counts.iter_rows())
        expected = (
            "one valid currency for the current principal balance (RREL30) of every retained "
            f"loan, which the pipeline adds up and allocates by without conversion; found {found}"
        )
        if counts.height > 1 or invalid.height:
            pool_row("pool_balances_in_more_than_one_currency", expected)
        frames.append(_a17_record_rows(
            other, "RREL30", "balance_currency_differs_from_the_pool",
            f"{modal}, the currency of most retained loans' RREL30", pl.format("currency {}", currency),
        ))
        frames.append(_a17_record_rows(
            invalid, "RREL30", "balance_currency_not_a_valid_code",
            "a valid ISO 4217 code for the currency of RREL30", pl.format("currency {}", currency),
        ))
        outcome.failed = other.height + invalid.height
    outcomes.append(outcome)

    # Sets 2 and 3: amounts combined on one loan.
    for key, calculation, codes in _LOAN_SETS:
        seen = retained.with_columns(_stated(codes).alias("_c"), _unstated(codes).alias("_u"))
        present = seen.filter(pl.any_horizontal([pl.col(f"cur_{c}").is_not_null() for c in codes]))
        mixed = present.filter(pl.col("_c").list.len() > 1)
        outcome = RuleOutcome(
            "A17", "loans", codes[0], applicable=present.height,
            evaluated=present.height - int(present["_u"].sum()),
            not_evaluable=int(present["_u"].sum()), failed=mixed.height,
        )
        if outcome.not_evaluable:
            outcome.not_evaluable_reasons[not_stated] = outcome.not_evaluable
        frames.append(_a17_record_rows(
            mixed, codes[0], f"{key}_in_more_than_one_currency",
            f"one currency for {', '.join(codes)} of a loan, which the pipeline combines for "
            f"{calculation} without conversion",
            pl.format("currencies {}", pl.col("_c").list.join(", ")),
        ))
        outcomes.append(outcome)

    # Set 4: the valuations actually selected, with RREL29 and RREL30.
    outcome = RuleOutcome("A17", "loans", "RREL29", applicable=retained.height)
    stage7 = result.stage7 if result is not None else None
    provenance = stage7.provenance if stage7 is not None else None
    if (
        provenance is None or result is None or result.collateral_record_index is None
        or result.loan_record_index is None or collateral is None or links is None
    ):
        outcome.not_evaluable = retained.height
        if retained.height:
            outcome.not_evaluable_reasons[STAGE7_DID_NOT_COMPLETE] = retained.height
        outcomes.append(outcome)
        return rows, frames, outcomes

    # A provenance record is a row of the Stage 3 properties; its record in
    # the collateral file is `collateral_record_index` + 1.
    file_record = (result.collateral_record_index + 1).cast(pl.UInt32)
    selection = pl.DataFrame({
        "collateral_record": file_record.gather(provenance.records["record"]),
        "source": provenance.records["source"],
    })
    valuations = selection.join(
        collateral.select(
            pl.col("record").alias("collateral_record"),
            pl.col("cur_RREC13").cast(pl.String), pl.col("cur_RREC17").cast(pl.String),
        ),
        on="collateral_record",
    ).select(
        "collateral_record",
        pl.when(pl.col("source") == "current").then(pl.col("cur_RREC13"))
        .otherwise(pl.col("cur_RREC17")).alias("valuation_currency"),
    )
    raw = links.filter(pl.col("match") == "raw")
    if convention is not None:
        raw = raw.filter(pl.col("convention") == convention)
    per_loan = (
        raw.select("loan_record", "collateral_record").unique()
        .join(valuations, on="collateral_record")
        .group_by("loan_record").agg(
            pl.col("valuation_currency").filter(
                pl.col("valuation_currency").is_not_null()
                & (pl.col("valuation_currency") != CURRENCY_NOT_STATED)
            ).unique().alias("_v"),
            (pl.col("valuation_currency") == CURRENCY_NOT_STATED).any().fill_null(False).alias("_vu"),
        )
    )
    in_output = (result.loan_record_index + 1).cast(pl.UInt32).to_list()
    loans_out = (
        retained.filter(pl.col("record").is_in(in_output))
        .join(per_loan, left_on="record", right_on="loan_record", how="left")
        .with_columns(
            pl.concat_list(_stated(("RREL29", "RREL30")), pl.col("_v").fill_null([]))
            .list.unique().list.sort().alias("_c"),
            (_unstated(("RREL29", "RREL30")) | pl.col("_vu").fill_null(False)).alias("_u"),
        )
    )
    mixed = loans_out.filter(pl.col("_c").list.len() > 1)
    outcome.applicable = loans_out.height
    outcome.not_evaluable = int(loans_out["_u"].sum())
    outcome.evaluated = loans_out.height - outcome.not_evaluable
    outcome.failed = mixed.height
    if outcome.not_evaluable:
        outcome.not_evaluable_reasons[not_stated] = outcome.not_evaluable
    frames.append(_a17_record_rows(
        mixed, "RREL29", "selected_valuations_in_more_than_one_currency",
        "one currency for the valuation selected for each collateral record of the loan and for "
        "the loan's RREL29 and RREL30, which the pipeline combines for Property Value and the "
        "calculated loan-to-value figures without conversion",
        pl.format("currencies {}", pl.col("_c").list.join(", ")),
    ))
    across = (
        valuations.filter(
            pl.col("valuation_currency").is_not_null()
            & (pl.col("valuation_currency") != CURRENCY_NOT_STATED)
        )
        .group_by("valuation_currency").len().sort(["len", "valuation_currency"], descending=[True, False])
    )
    if across.height > 1 or across.filter(~pl.col("valuation_currency").is_in(valid)).height:
        found = ", ".join(f"{c}: {n} record(s)" for c, n in across.iter_rows())
        pool_row(
            "selected_valuations_in_more_than_one_currency",
            "one valid currency for the valuations selected across the pool, which the pipeline "
            f"aggregates within a group without conversion; found {found}",
        )
    outcomes.append(outcome)
    return rows, frames, outcomes


__all__ = ["CAUSES", "DROPPED_SCHEMA", "Waterfall", "build_waterfall"]
