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
from esma_milan.quality.catalogue import FileKind
from esma_milan.quality.column_rules import ColumnProfile, assess_columns
from esma_milan.quality.column_rules.context import PoolFacts, pool_facts
from esma_milan.quality.column_rules.domain import classification_year_results
from esma_milan.quality.inventories import actual_columns, expected_fields
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
    outcomes: tuple[RuleOutcome, ...]
    profiles: tuple[ColumnProfile, ...]
    actual_columns: tuple[Mapping[str, object], ...]
    records: pl.DataFrame | None
    """Per-record identifiers and labels (`populations`); None when the
    file is unreadable."""

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

    @property
    def files(self) -> tuple[FileAudit, FileAudit]:
        return (self.loans, self.collateral)

    @property
    def results(self) -> pl.DataFrame:
        """Every result row of both files and of the pool."""
        return concat_results([self.loans.results, self.collateral.results, self.pool_results])

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


def _audit(path: Path, file: FileKind, pool: PoolFacts | None) -> tuple[FileAudit, PoolFacts]:
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
        )
    if pool is None:
        pool = pool_facts(raw)
    retained: pl.Series | None = None
    if file == "loans":
        records = loan_records(raw, pool)
        retained = records["population"] == RETAINED
    else:
        records = collateral_records(raw)
    assessed = assess_columns(raw, pool, retained=retained, ref=load_reference_data())
    return (
        FileAudit(
            file=file, name=raw.name, sha256=raw.sha256, size=raw.size, unreadable=None,
            headers=raw.headers, columns=raw.columns, n_records=raw.n_records,
            encoding_repaired=raw.encoding_repaired, results=assessed.results,
            outcomes=tuple(assessed.outcomes), profiles=tuple(assessed.profiles),
            actual_columns=tuple(actual_columns(raw)), records=records,
        ),
        pool,
    )


def assess_files(
    loans: Path, collaterals: Path, *, started_at: datetime | None = None
) -> FileAssessment:
    """Read and assess the two uploaded files. Independent of the
    pipeline; one file's text is freed before the other is read. A file
    that cannot be split into records is reported as such and the other
    file is still assessed."""
    started = started_at or datetime.now(UTC)
    loans_audit, pool = _audit(loans, "loans", None)
    collateral_audit, _ = _audit(collaterals, "collateral", pool if loans_audit.readable else PoolFacts())
    profiles = (*loans_audit.profiles, *collateral_audit.profiles)
    rows, outcomes = classification_year_results(
        [p for p in profiles if p.entry.code == "RREL12"],
        [p for p in profiles if p.entry.kind == "NUTS"],
        load_reference_data(),
    )
    return FileAssessment(
        loans=loans_audit,
        collateral=collateral_audit,
        pool=pool,
        pool_results=results_frame(rows),
        pool_outcomes=tuple(outcomes),
        started_at=started,
        as_of=started.astimezone(UTC).date(),
    )


def inventory_of_expected_fields(assessment: FileAssessment) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    results = assessment.results
    for audit in assessment.files:
        if audit.readable:
            rows.extend(expected_fields(audit.file, audit.profiles, assessment.outcomes, results))
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
    records = assessment.loans.records
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
