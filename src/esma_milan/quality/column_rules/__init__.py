"""Per-column rules (specification section 6) and the engine that runs them.

Mirrors nothing in the R reference.

Every resolved column of a file is assessed on every record. A rule is a
generator over a `CellContext`: one distinct combination of the cell's
text, the record's cut-off date and the other cells the field's rules
read. It yields a `Mark` for each rule that applies to those records and a
`Finding` for each result. The engine evaluates each distinct combination
once and joins the findings back to the records, so every affected record
has its row; nothing is sampled and no rule stops at its first finding.

Two supplied columns that resolve to one field are assessed separately and
keep separate rows (the row carries the physical column position).
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

import polars as pl

from esma_milan.quality.audit_read import RESOLVED_TO_FIELD, ColumnResolution, RawFile
from esma_milan.quality.catalogue import (
    CONDITIONAL_ND5_FIELDS,
    FIELDS,
    NUMERIC_KINDS,
    RULES,
    FieldEntry,
    fields_of,
)
from esma_milan.quality.column_rules.context import (
    CellContext,
    PoolFacts,
    RuleOutput,
    cutoff_columns,
)
from esma_milan.quality.column_rules.currency import (
    companion_rules,
    currency_evidence,
    layout_rows,
)
from esma_milan.quality.column_rules.domain import domain_rules
from esma_milan.quality.column_rules.fast import AT_LEAST_ONE, plain_expr
from esma_milan.quality.column_rules.formats import format_rules
from esma_milan.quality.column_rules.nd import nd_rules
from esma_milan.quality.column_rules.plausibility import column_rows, plausibility_rules
from esma_milan.quality.refdata import ReferenceData, load_reference_data
from esma_milan.quality.result import (
    RESULT_SCHEMA,
    Finding,
    Mark,
    RuleOutcome,
    concat_results,
    results_frame,
)
from esma_milan.quality.states import Classified, classify
from esma_milan.quality.store import ResultStore

Rule = Callable[[CellContext], RuleOutput]

FIELD_RULES: tuple[Rule, ...] = (
    nd_rules, format_rules, domain_rules, currency_evidence, plausibility_rules,
)
"""Run, in order, on every cell of a column resolved to a field."""
COMPANION_RULES: tuple[Rule, ...] = (companion_rules,)
"""Run on every cell of a currency companion column."""

_DISTRIBUTION_KINDS = frozenset({"LIST", "BOOLEAN", "YEAR", "COUNTRY", "CURRENCY"})

# Stands for any proven-plain amount when a companion column is assessed.
_A_PLAIN_AMOUNT = "1"

# A currency companion is assessed as a {CURRENCYCODE_3} value.
_COMPANION_TEMPLATE = FIELDS["RREL28"]


@dataclass
class ColumnProfile:
    """What the inventories and the column-level rules need from one
    assessed column (P2 is this profile). It holds counts, never the
    column's values, so its size does not grow with the file."""

    column: ColumnResolution
    entry: FieldEntry
    states: Counter[str] = field(default_factory=Counter)
    states_retained: Counter[str] | None = None
    format_counts: Counter[str] = field(default_factory=Counter)
    """compliant, alternative, non_compliant: present values only."""
    non_compliant_examples: list[str] = field(default_factory=list)
    n_present: int = 0
    """Records with a present value."""
    n_distinct: int = 0
    """Distinct present raw values."""
    n_distinct_invalid: int = 0
    n_valid: int = 0
    """Records whose present value is valid for the field."""
    n_nonzero: int = 0
    """Records with a valid number other than 0."""
    n_abs_at_least_one: int = 0
    """Records with a valid number of absolute value 1 or more."""
    n_zzz: int = 0
    """Records with a valid region code ending in ZZZ."""
    first_value: str | None = None
    """The first distinct present value (the value, when `n_distinct` is 1)."""
    distribution: Counter[str] = field(default_factory=Counter)
    """Valid values and their record counts, for the kinds with a small
    domain (code lists, booleans, years, country and currency codes)."""

    def observe_plain(self, first: str, n: int, n_at_least_one: int) -> None:
        """`n` records with proven-plain values (present, valid, and for a
        number not zero), of which `n_at_least_one` are 1 or more."""
        self.n_present += n
        self.n_valid += n
        if self.first_value is None:
            self.first_value = first
        if self.entry.kind in NUMERIC_KINDS:
            self.n_nonzero += n
            self.n_abs_at_least_one += n_at_least_one

    def observe(self, raw: str, value: Classified, n: int) -> None:
        """`n` records with the present value `raw`. The distinct counts
        are set by the engine once the column is done."""
        self.n_present += n
        if self.first_value is None:
            self.first_value = raw
        if not value.ok:
            return
        self.n_valid += n
        if isinstance(value.parsed, Decimal):
            if value.parsed != 0:
                self.n_nonzero += n
            if abs(value.parsed) >= 1:
                self.n_abs_at_least_one += n
        if self.entry.kind == "NUTS" and raw.endswith("ZZZ"):
            self.n_zzz += n
        if self.entry.kind in _DISTRIBUTION_KINDS:
            self.distribution[str(value.parsed) if self.entry.kind == "YEAR" else raw] += n


@dataclass
class ColumnAssessment:
    results: pl.DataFrame
    outcomes: list[RuleOutcome]
    profiles: list[ColumnProfile]


def companion_entry(amount: FieldEntry) -> FieldEntry:
    return dataclasses.replace(
        _COMPANION_TEMPLATE,
        code=f"{amount.code}_CURRENCY",
        file=amount.file,
        name=f"currency of {amount.label}",
        nd1_4_allowed=True,
        nd5_allowed=True,
        critical=False,
        feeds_milan=False,
        optional=amount.optional,
    )


def _extras(raw: RawFile, entry: FieldEntry) -> dict[str, ColumnResolution | None]:
    """The other columns of the record that `entry`'s rules read."""
    out: dict[str, ColumnResolution | None] = {}
    if entry.code in CONDITIONAL_ND5_FIELDS:
        out["collateral_type"] = raw.column_of("RREC5")
    if entry.kind == "NUTS" and raw.file == "loans":
        out["year"] = raw.column_of("RREL12")
    if entry.kind == "MONETARY":
        out["companion"] = raw.companion_of(entry.code)
        out["dedicated"] = (
            raw.column_of(entry.dedicated_currency) if entry.dedicated_currency else None
        )
    return out


def _evaluate(
    raw: RawFile,
    column: ColumnResolution,
    entry: FieldEntry,
    extras: Mapping[str, ColumnResolution | None],
    rules: Sequence[Rule],
    cut: pl.DataFrame,
    cut_note: str | None,
    retained: pl.Series | None,
    pool: PoolFacts,
    ref: ReferenceData,
    fast_path: bool,
    derived: Mapping[str, pl.Series],
) -> tuple[pl.DataFrame, dict[str, RuleOutcome], ColumnProfile]:
    assert raw.frame is not None
    present_extras = {name: col for name, col in extras.items() if col is not None}
    keys = [
        "_v", *[f"_x_{name}" for name in present_extras], *[f"_x_{name}" for name in derived],
        "_cut", "_cut_reason",
    ]
    work = raw.frame.select(
        "record",
        "physical_line",
        pl.col(column.frame_column).fill_null("").alias("_v"),
        *[
            pl.col(col.frame_column).fill_null("").alias(f"_x_{name}")
            for name, col in present_extras.items()
        ],
    ).with_columns(
        cut["_cut"],
        cut["_cut_reason"],
        (retained if retained is not None else pl.lit(False)).alias("_ret"),
        *[series.alias(f"_x_{name}") for name, series in derived.items()],
    )
    # Proven-plain values are grouped by the other cells only (fast.py);
    # every other value is its own group.
    plain = (
        plain_expr(entry, {name: f"_x_{name}" for name in present_extras})
        if fast_path and column.how in RESOLVED_TO_FIELD
        else None
    )
    if fast_path and column.how == "currency_companion" and "amount" in present_extras:
        # A companion's rules ask only whether its amount is a valid number.
        # Every proven-plain amount is, so they stand in for one another.
        amount_entry = FIELDS[entry.code.removesuffix("_CURRENCY")]
        plain_amount = plain_expr(amount_entry, {}, column="_x_amount")
        if plain_amount is not None:
            work = work.with_columns(
                pl.when(plain_amount).then(pl.lit(_A_PLAIN_AMOUNT)).otherwise(pl.col("_x_amount"))
                .alias("_x_amount")
            )
    work = work.with_columns((plain if plain is not None else pl.lit(False)).alias("_plain"))
    work = work.with_columns(
        pl.when("_plain").then(pl.lit(None, dtype=pl.String)).otherwise(pl.col("_v")).alias("_gv")
    )
    keys = ["_plain", "_gv", *keys[1:]]
    groups = (
        work.group_by(keys, maintain_order=True)
        .agg(
            pl.len().alias("_n"),
            pl.col("_ret").sum().alias("_n_ret"),
            pl.col("_v").first().alias("_first"),
            pl.col("_v").str.contains(AT_LEAST_ONE).sum().alias("_n_one"),
        )
        .with_row_index("_g")
    )

    profile = ColumnProfile(column=column, entry=entry)
    if retained is not None:
        profile.states_retained = Counter()
    outcomes: dict[str, RuleOutcome] = {}
    found: list[tuple[int, str, Finding]] = []
    present_values: set[str] = set()
    invalid_values: set[str] = set()
    for row in groups.iter_rows(named=True):
        n: int = row["_n"]
        text: str = row["_first"]
        value = classify(entry, text, ref)
        profile.states[value.state] += n
        if profile.states_retained is not None:
            profile.states_retained[value.state] += row["_n_ret"]
        if row["_plain"]:
            assert value.present and value.ok, (entry.code, "a plain value must be valid")
            profile.observe_plain(text, n, row["_n_one"] if entry.kind in NUMERIC_KINDS else 0)
        elif value.present:
            profile.observe(text, value, n)
            present_values.add(text)
            if not value.ok:
                invalid_values.add(text)
        context = CellContext(
            entry=entry,
            ref=ref,
            raw=text,
            value=value,
            cutoff=row["_cut"],
            cutoff_reason=row["_cut_reason"],
            cutoff_note=cut_note,
            extras={
                **{name: (row[f"_x_{name}"] if name in present_extras else None) for name in extras},
                **{name: row[f"_x_{name}"] for name in derived},
            },
            pool=pool,
        )
        marked: set[str] = set()
        failed: set[str] = set()
        for rule in rules:
            for item in rule(context):
                outcome = outcomes.get(item.rule_id)
                if outcome is None:
                    outcome = outcomes[item.rule_id] = RuleOutcome(
                        item.rule_id, raw.file, entry.code
                    )
                if isinstance(item, Mark):
                    if item.rule_id in marked:
                        continue
                    marked.add(item.rule_id)
                    outcome.applicable += n
                    if item.not_evaluable is None:
                        outcome.evaluated += n
                    else:
                        outcome.not_evaluable += n
                        outcome.not_evaluable_reasons[item.not_evaluable] += n
                    continue
                found.append((row["_g"], value.state, item))
                if item.result_kind == "violation":
                    if item.rule_id not in failed:
                        failed.add(item.rule_id)
                        outcome.failed += n
                else:
                    outcome.other_kinds[item.result_kind] += n
                if value.present and item.rule_id in ("P5", "P7(a)", "P7(b)", "P7(c)"):
                    kind = "non_compliant" if item.result_kind == "violation" else "alternative"
                    profile.format_counts[kind] += n
                    examples = profile.non_compliant_examples
                    if kind == "non_compliant" and len(examples) < 5 and text not in examples:
                        examples.append(text)
    # Distinct present values of the column: the plain ones and the others,
    # a value counted once however many groups it falls in.
    plain_values = work.filter("_plain")["_v"].unique()
    profile.n_distinct = len(plain_values) + len(present_values) - (
        int(plain_values.is_in(sorted(present_values)).sum()) if present_values else 0
    )
    profile.n_distinct_invalid = len(invalid_values)
    profile.format_counts["compliant"] = (
        profile.n_present
        - profile.format_counts["non_compliant"]
        - profile.format_counts["alternative"]
    )

    if not found:
        return pl.DataFrame(schema=RESULT_SCHEMA), outcomes, profile
    findings = pl.DataFrame(
        {
            "_g": [g for g, _, _ in found],
            "rule_id": [f.rule_id for _, _, f in found],
            "result_kind": [f.result_kind for _, _, f in found],
            "severity": [f.severity or RULES[f.rule_id].severity for _, _, f in found],
            "value_state": [state for _, state, _ in found],
            "nd4_date": [f.nd4_date for _, _, f in found],
            "expected": [f.expected for _, _, f in found],
            "reason": [f.reason for _, _, f in found],
            "note": [f.note for _, _, f in found],
        },
        schema_overrides={"_g": pl.UInt32, "nd4_date": pl.Date, "expected": pl.String, "note": pl.String},
    )
    rows = (
        work.join(groups.select(*keys, "_g"), on=keys, nulls_equal=True)
        .join(findings, on="_g")
        .sort("record", "rule_id")
        .select(
            pl.lit(raw.file).alias("file"),
            "record",
            "physical_line",
            pl.lit(column.position, dtype=pl.UInt16).alias("column_position"),
            pl.lit(column.header).alias("header"),
            pl.lit(entry.code).alias("field"),
            "rule_id",
            "result_kind",
            "severity",
            "value_state",
            pl.col("_v").alias("raw_value"),
            "nd4_date",
            "expected",
            "reason",
            "note",
        )
        .cast(RESULT_SCHEMA)  # type: ignore[arg-type]
    )
    return rows, outcomes, profile


def column_level_rows(
    raw: RawFile, profiles: Sequence[ColumnProfile], pool: PoolFacts
) -> tuple[list[dict[str, object]], list[RuleOutcome]]:
    """Results about a whole column (no record): P1 (a template field
    without a column), the currency layout note, P8, P9 and P12."""
    rows: list[dict[str, object]] = []
    outcomes: list[RuleOutcome] = []
    present = {p.entry.code for p in profiles}
    for entry in fields_of(raw.file):
        outcome = RuleOutcome("P1", raw.file, entry.code, applicable=1, evaluated=1)
        if entry.code not in present:
            outcome.failed = 1
            rows.append(
                {
                    "file": raw.file,
                    "field": entry.code,
                    "rule_id": "P1",
                    "result_kind": "violation",
                    "severity": RULES["P1"].severity,
                    "value_state": "field_absent",
                    "reason": "field_absent",
                    "expected": "a column for the field (an ND allowance does not make it optional)",
                }
            )
        outcomes.append(outcome)
    for more_rows, more_outcomes in (layout_rows(raw, profiles), column_rows(raw, profiles)):
        rows.extend(more_rows)
        outcomes.extend(more_outcomes)
    return rows, outcomes


def assess_columns(
    raw: RawFile,
    pool: PoolFacts,
    *,
    retained: pl.Series | None = None,
    ref: ReferenceData | None = None,
    store: ResultStore | None = None,
    fast_path: bool = True,
    linked_lifecycle: pl.Series | None = None,
) -> ColumnAssessment:
    """Run every per-column rule on every resolved column of `raw`.

    `retained` (one Boolean per record) adds the value-state profile of
    the retained active pool to each column's profile.

    With a `store`, each column's rows are written to it as soon as the
    column is evaluated and are not kept: `results` of the returned
    assessment is then empty. Without one, the rows are returned in
    memory (the rule tests, small files).

    `linked_lifecycle` (collateral only, one text per record, from
    `links.collateral_link_context`) is the claimed lifecycle of the loan
    each record belongs to; the conditional ND5 rule (P3f) reads it.
    Without it P3f treats every record as linked to no loan.

    `fast_path=False` evaluates every distinct value on its own. It exists
    for the test that the two paths give the same rows, outcomes and
    profiles; it is not an option of the command line.
    """
    assert raw.frame is not None, "an unreadable file has no columns to assess"
    ref = ref or load_reference_data()
    cut, cut_note = cutoff_columns(raw, pool)
    frames: list[pl.DataFrame] = []
    outcomes: list[RuleOutcome] = []
    profiles: list[ColumnProfile] = []
    for column in raw.columns:
        if column.field is None:
            continue
        if column.how in RESOLVED_TO_FIELD:
            entry = FIELDS[column.field]
            extras = _extras(raw, entry)
            rules: Sequence[Rule] = FIELD_RULES
        else:  # currency companion
            entry = companion_entry(FIELDS[column.field])
            extras = {"amount": raw.column_of(column.field)}
            rules = COMPANION_RULES
        derived: dict[str, pl.Series] = {}
        if linked_lifecycle is not None and entry.code in CONDITIONAL_ND5_FIELDS:
            derived["lifecycle"] = linked_lifecycle
        rows, column_outcomes, profile = _evaluate(
            raw, column, entry, extras, rules, cut, cut_note, retained, pool, ref, fast_path,
            derived,
        )
        if store is not None:
            store.write_results(f"{raw.file}-{column.position:04d}", rows)
        else:
            frames.append(rows)
        outcomes.extend(column_outcomes.values())
        if column.how in RESOLVED_TO_FIELD:
            profiles.append(profile)
    level_rows, level_outcomes = column_level_rows(raw, profiles, pool)
    if store is not None:
        store.write_results(f"{raw.file}-columns", results_frame(level_rows))
    else:
        frames.append(results_frame(level_rows))
    outcomes.extend(level_outcomes)
    return ColumnAssessment(concat_results(frames), outcomes, profiles)
