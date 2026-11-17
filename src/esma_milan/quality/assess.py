"""The assessment of the two files, and the report completed from the run.

Mirrors nothing in the R reference (specification 5, 6, 8.1, 10.1).

Two steps, so the report exists whether or not the pipeline succeeds:

  assess_files      reads each file in turn, runs every per-column rule,
                    labels every record, builds the inventories, and frees
                    the file's text before it reads the next. It needs
                    nothing from the pipeline and is meant to run before
                    it, so the two never hold their frames at once.
  complete_report   adds what only the run can say: whether the pipeline
                    completed or failed (and with which reason code),
                    the comparison of the population labels with Stage 2's
                    own output, and the identifier column Stage 3 chose.

Nothing here blocks, rejects or withholds anything. The report never
states a policy outcome; that arrives with the acceptance rules of a later
release.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from esma_milan.errors import PipelineDataError
from esma_milan.quality.audit_read import ColumnResolution, RawFile, Unreadable, read_file
from esma_milan.quality.catalogue import RULES, FileKind
from esma_milan.quality.column_rules import ColumnProfile, assess_columns
from esma_milan.quality.column_rules.context import PoolFacts, pool_facts
from esma_milan.quality.column_rules.domain import classification_year_results
from esma_milan.quality.inventories import actual_columns, expected_fields
from esma_milan.quality.links import build_links, collateral_link_context
from esma_milan.quality.populations import (
    LOAN_ID_COLUMNS,
    RETAINED,
    Stage2Comparison,
    collateral_records,
    compare_with_stage2,
    identity_confidence,
    loan_records,
)
from esma_milan.quality.refdata import load_reference_data
from esma_milan.quality.result import RuleOutcome, concat_results, empty_results, results_frame
from esma_milan.quality.store import ResultStore, result_counts

if TYPE_CHECKING:
    from esma_milan.runner import PipelineResult


@dataclass(frozen=True)
class FileAudit:
    """Everything the audit keeps about one file. The text frame is not
    kept: `results` holds every finding row and `records` one narrow row
    per record."""

    file: FileKind
    name: str
    sha256: str
    size: int
    unreadable: Unreadable | None
    headers: tuple[str, ...]
    columns: tuple[ColumnResolution, ...]
    n_records: int
    encoding_repaired: pl.DataFrame
    results: pl.DataFrame
    """Every finding row of the file; empty when the rows are in the store
    (`FileAssessment.store_root`)."""
    outcomes: tuple[RuleOutcome, ...]
    profiles: tuple[ColumnProfile, ...]
    actual_columns: tuple[Mapping[str, object], ...]
    records: pl.DataFrame | None
    """Per-record identifiers and labels (`populations`); None when the
    file is unreadable or the frame is in the store."""

    @property
    def readable(self) -> bool:
        return self.unreadable is None


@dataclass(frozen=True)
class FileAssessment:
    """The result of `assess_files`."""

    loans: FileAudit
    collateral: FileAudit
    pool: PoolFacts
    pool_results: pl.DataFrame
    pool_outcomes: tuple[RuleOutcome, ...]
    started_at: datetime
    as_of: date
    """The date used for "today" (UTC date of the run's start)."""
    store_root: Path | None = None
    """The working directory of the on-disk store, when the assessment was
    made with one. The result rows and the per-record frames are then
    there and not on this object."""
    links: pl.DataFrame | None = None
    """The links between collateral and loan records (`links.build_links`);
    None when they are in the store or a file was unreadable."""
    records_with_compliance_violation: int = 0
    """Distinct records, over both files, with at least one violation of a
    rule that has a regulatory source. It is the one figure of the summary
    that needs the detail itself, so it is counted here, where the detail
    is being written, and not again after the pipeline has run."""

    @property
    def files(self) -> tuple[FileAudit, FileAudit]:
        return (self.loans, self.collateral)

    @property
    def store(self) -> ResultStore | None:
        return ResultStore.open(self.store_root) if self.store_root is not None else None

    def scan_results(self) -> pl.LazyFrame:
        """Every result row of both files and of the pool, lazily."""
        store = self.store
        if store is not None:
            return store.scan_results()
        return concat_results([self.loans.results, self.collateral.results, self.pool_results]).lazy()

    @property
    def results(self) -> pl.DataFrame:
        """Every result row, in memory. For tests and small files; the
        exporters stream `scan_results()` instead."""
        return self.scan_results().collect()

    @property
    def counts(self) -> pl.DataFrame:
        """Rows and records per file, field, rule, kind, severity and reason."""
        store = self.store
        return store.index if store is not None else result_counts(self.results)

    def scan_links(self) -> pl.LazyFrame | None:
        """The links, from memory or from the store; None when there are none."""
        if self.links is not None:
            return self.links.lazy()
        store = self.store
        return store.scan_links() if store is not None and store.has_links() else None

    def records(self, file: FileKind, columns: list[str] | None = None) -> pl.DataFrame | None:
        """The per-record frame of one file (optionally some columns), from
        the store or from memory; None when the file was unreadable."""
        audit = self.loans if file == "loans" else self.collateral
        if audit.records is not None:
            return audit.records.select(columns) if columns else audit.records
        store = self.store
        if store is None or not store.has_records(file):
            return None
        scan = store.scan_records(file)
        return (scan.select(columns) if columns else scan).collect()

    @property
    def outcomes(self) -> tuple[RuleOutcome, ...]:
        return (*self.loans.outcomes, *self.collateral.outcomes, *self.pool_outcomes)


@dataclass(frozen=True)
class PipelineOutcome:
    """What the run did, as a fact. Never a statement about the pool."""

    completed: bool
    error_type: str | None = None
    error_message: str | None = None
    stage: str | None = None
    """For a `PipelineDataError`: where it was detected."""
    reason: str | None = None
    """For a `PipelineDataError`: its reason code."""
    workbook_path: Path | None = None


@dataclass(frozen=True)
class QualityReport:
    """The report object. The command line holds it; `PipelineResult` does
    not carry it, so it exists when the pipeline raises."""

    assessment: FileAssessment
    pipeline: PipelineOutcome
    stage2_comparison: Stage2Comparison | None
    selected_loan_identifier: str | None
    """`RREL2` or `RREL3` when Stage 3 reported its choice."""
    identity_confidence: Mapping[str, int]
    options: Mapping[str, object] = field(default_factory=dict)

    @property
    def populations_observed(self) -> bool:
        """True when the population labels were confirmed against Stage 2;
        otherwise they are predicted from the stage's predicate."""
        return self.stage2_comparison is not None and self.stage2_comparison.agrees


def _audit(
    path: Path,
    file: FileKind,
    pool: PoolFacts | None,
    store: ResultStore | None,
    loans_records: pl.DataFrame | None = None,
) -> tuple[FileAudit, PoolFacts, pl.DataFrame | None, pl.DataFrame | None]:
    """Read and assess one file. Returns the audit, the pool facts, the
    per-record frame and, for the collateral file, the links to the loan
    records (`loans_records` is the loans file's per-record frame)."""
    raw: RawFile = read_file(path, file)
    if raw.frame is None:
        return (
            FileAudit(
                file=file, name=raw.name, sha256=raw.sha256, size=raw.size,
                unreadable=raw.unreadable, headers=raw.headers, columns=(), n_records=0,
                encoding_repaired=raw.encoding_repaired, results=empty_results(), outcomes=(),
                profiles=(), actual_columns=(), records=None,
            ),
            pool or PoolFacts(),
            None,
            None,
        )
    if pool is None:
        pool = pool_facts(raw)
    retained: pl.Series | None = None
    linked_lifecycle: pl.Series | None = None
    links: pl.DataFrame | None = None
    if file == "loans":
        records = loan_records(raw, pool)
        retained = records["population"] == RETAINED
    else:
        records = collateral_records(raw)
        if loans_records is not None:
            # The links, and through them the lifecycle of each record's loan
            # (P3f) and whether the record belongs to the retained pool.
            links = build_links(loans_records, records)
            context = collateral_link_context(links, loans_records, raw.n_records)
            records = records.with_columns(
                context["linked_lifecycle"], context["linked_retained"],
                context["links_new"], context["links_original"],
            )
            retained = context["linked_retained"]
            linked_lifecycle = context["linked_lifecycle"]
            if store is not None:
                store.write_links(links)
    assessed = assess_columns(
        raw, pool, retained=retained, ref=load_reference_data(), store=store,
        linked_lifecycle=linked_lifecycle,
    )
    if store is not None:
        store.write_records(file, records)
        store.write_repairs(file, raw.encoding_repaired)
    return (
        FileAudit(
            file=file, name=raw.name, sha256=raw.sha256, size=raw.size, unreadable=None,
            headers=raw.headers, columns=raw.columns, n_records=raw.n_records,
            encoding_repaired=raw.encoding_repaired,
            results=assessed.results,
            outcomes=tuple(assessed.outcomes), profiles=tuple(assessed.profiles),
            actual_columns=tuple(actual_columns(raw)),
            records=records if store is None else None,
        ),
        pool,
        records,
        links,
    )


def assess_files(
    loans: Path,
    collaterals: Path,
    *,
    started_at: datetime | None = None,
    store: ResultStore | None = None,
) -> FileAssessment:
    """Read and assess the two uploaded files. Independent of the
    pipeline; one file's text is freed before the other is read. A file
    that cannot be split into records is reported as such and the other
    file is still assessed.

    With a `store`, every part of the detail is written to it as soon as
    it exists, and the returned object holds no result row and no
    per-record frame. Without one, everything is returned in memory."""
    started = started_at or datetime.now(UTC)
    loans_audit, pool, loans_records, _ = _audit(loans, "loans", None, store)
    collateral_audit, _, _, links = _audit(
        collaterals, "collateral", pool if loans_audit.readable else PoolFacts(), store,
        loans_records,
    )
    profiles = (*loans_audit.profiles, *collateral_audit.profiles)
    rows, outcomes = classification_year_results(
        [p for p in profiles if p.entry.code == "RREL12"],
        [p for p in profiles if p.entry.kind == "NUTS"],
        load_reference_data(),
    )
    pool_results = results_frame(rows)
    if store is not None:
        store.write_results("pool", pool_results)
        pool_results = empty_results()
    detail = (
        store.scan_results()
        if store is not None
        else concat_results([loans_audit.results, collateral_audit.results, pool_results]).lazy()
    )
    sourced = sorted(r.rule_id for r in RULES.values() if r.regulatory_source is not None)
    n_violating = (
        detail.filter(
            (pl.col("result_kind") == "violation")
            & pl.col("rule_id").cast(pl.String).is_in(sourced)
            & pl.col("record").is_not_null()
        )
        .select(pl.col("file").cast(pl.String), "record")
        .unique()
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    return FileAssessment(
        loans=loans_audit,
        collateral=collateral_audit,
        pool=pool,
        pool_results=pool_results,
        records_with_compliance_violation=n_violating,
        links=links if store is None else None,
        pool_outcomes=tuple(outcomes),
        started_at=started,
        as_of=started.astimezone(UTC).date(),
        store_root=store.root if store is not None else None,
    )


def inventory_of_expected_fields(assessment: FileAssessment) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    counts = assessment.counts
    for audit in assessment.files:
        if audit.readable:
            rows.extend(expected_fields(audit.file, audit.profiles, assessment.outcomes, counts))
    return rows


def complete_report(
    assessment: FileAssessment,
    *,
    pipeline: PipelineResult | None,
    error: BaseException | None,
    options: Mapping[str, object] | None = None,
) -> QualityReport:
    """Complete the report from the run: exactly one of `pipeline` (it
    returned) and `error` (it raised) is given."""
    if (pipeline is None) == (error is None):
        raise ValueError("complete_report needs the pipeline's result or its exception, not both")
    if pipeline is not None:
        outcome = PipelineOutcome(completed=True, workbook_path=pipeline.output_path)
    else:
        assert error is not None
        data_fault = error if isinstance(error, PipelineDataError) else None
        outcome = PipelineOutcome(
            completed=False,
            error_type=type(error).__name__,
            error_message=str(error),
            stage=data_fault.stage if data_fault else None,
            reason=data_fault.reason if data_fault else None,
        )

    comparison: Stage2Comparison | None = None
    selected: str | None = None
    identity: Counter[str] = Counter()
    records = assessment.records(
        "loans",
        ["record", "population", "status_not_active", "balance_not_positive",
         "balance_unparseable", "RREL2_duplicate", "RREL2_colliding", "RREL3_duplicate",
         "RREL3_colliding"],
    )
    if records is not None:
        comparison = compare_with_stage2(records, pipeline.stage2 if pipeline else None)
        decisions = pipeline.stage3.decisions if pipeline and pipeline.stage3 else None
        if decisions is not None and decisions.loan_id.chosen_column is not None:
            selected = LOAN_ID_COLUMNS.get(decisions.loan_id.chosen_column)
        identity = Counter(identity_confidence(records, selected).to_list())
    return QualityReport(
        assessment=assessment,
        pipeline=outcome,
        stage2_comparison=comparison,
        selected_loan_identifier=selected,
        identity_confidence=dict(identity),
        options=dict(options or {}),
    )
