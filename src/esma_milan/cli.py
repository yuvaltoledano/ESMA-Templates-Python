"""ESMA -> MILAN CLI.

Mirrors r_reference/R/main.R's CLI surface with two deliberate,
documented deviations:

- The same --loans/--collaterals/--deal/--output flags, plus
  --aggregation, --min-coverage, --dry-run and --verbose/--quiet, which
  R exposes only as function arguments.
- No --taxonomy flag. R takes the ESMA taxonomy workbook as a path
  argument; here it is a fixed ESMA artefact bundled with the package
  (esma_milan/data/esma_rre_taxonomy.xlsx, byte-identical to the R
  reference's copy), so the CLI runs from any directory and nothing can
  substitute a different taxonomy.

One addition has no counterpart in R: the data quality report
(`esma_milan.quality`). The two files are assessed before the pipeline
runs, in a process of their own (`quality.child`) that writes the
row-level detail to a working directory and exits, so the conversion runs
in a process that holds none of the audit's memory. The report files are
written beside the workbook when the pipeline returns or raises, a dry run
included, and the working directory is removed at the end, also after a
failure and after Ctrl-C. An interrupted run writes no report.

It changes nothing else: the pipeline is called with the same arguments,
the exception it raises leaves the command unchanged, and standard output
is still the workbook's path. A fault in the assessment itself (a bug in a
rule, a dead assessment process, a working directory that cannot be
created) is logged once, with the step and the exception's type and never
its message, which could quote a cell of the user's file; it never reaches
the pipeline or the exit status.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import click
import structlog

from esma_milan.quality import (
    FileAssessment,
    complete_report,
    summary_filename,
    write_summary,
)
from esma_milan.quality.child import run_assessment
from esma_milan.quality.export import interrupted_summary, write_json
from esma_milan.quality.store import ResultStore
from esma_milan.runner import PipelineResult, run_pipeline

log = structlog.get_logger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
    )


def _guarded[T](step: str, action: Callable[[], T]) -> tuple[T | None, str | None]:
    """Run one step of the data quality assessment. A fault in it is logged
    once and returned as text; it never propagates, so the conversion and
    the command's exit status are what they would be without the report.

    The log event carries the step and the exception's type (and the exit
    status of a dead assessment process), never the message: a message can
    quote a cell of the user's file. The message is returned for the
    summary, which stays beside the workbook."""
    try:
        return action(), None
    except Exception as exc:
        fields: dict[str, object] = {"step": step, "error_type": type(exc).__name__}
        exit_status = getattr(exc, "exit_status", None)
        if exit_status is not None:
            fields["exit_status"] = exit_status
        log.error("data_quality_assessment_failed", **fields)
        return None, f"{type(exc).__name__}: {exc}"


def _summary_cutoff(result: PipelineResult | None, assessment: FileAssessment | None) -> str | None:
    """The cut-off date in the summary's name: the one in the workbook's
    name when the pipeline returned, else the single cut-off the audit
    established, else None."""
    if result is not None and result.stage1 is not None:
        dates = result.stage1.loans["pool_cutoff_date"].drop_nulls()
        if len(dates) and isinstance(dates[0], date):
            return dates[0].isoformat()
    if assessment is not None and assessment.pool.loans_cutoff is not None:
        return assessment.pool.loans_cutoff.isoformat()
    return None


def _write_quality_summary(
    assessment: FileAssessment | None,
    assessment_fault: str | None,
    result: PipelineResult | None,
    error: Exception | None,
    output_dir: Path,
    deal_name: str,
    options: dict[str, object],
) -> None:
    path = output_dir / deal_name / summary_filename(deal_name, _summary_cutoff(result, assessment))
    if assessment is None:
        write_json(interrupted_summary(assessment_fault or "unknown", error is None), path)
    else:
        report = complete_report(assessment, pipeline=result, error=error, options=options)
        write_summary(report, path)
    log.info("data_quality_summary_written", path=str(path))


@click.group()
@click.version_option()
def main() -> None:
    """ESMA -> MILAN pipeline."""


@main.command("run")
@click.option("--loans", "loans_file", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--collaterals",
    "collaterals_file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option("--deal", "deal_name", type=str, required=True)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(path_type=Path),
    default=Path("data/clean"),
    show_default=True,
)
@click.option(
    "--aggregation",
    type=click.Choice(["auto", "by_loan", "by_group"]),
    default="auto",
    show_default=True,
)
@click.option(
    "--min-coverage",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Minimum acceptable coverage of the chosen calc_loan_id column. "
    "Defaults to DEFAULT_MIN_LOAN_ID_COVERAGE (0.85).",
)
@click.option("--dry-run", is_flag=True, default=False, help="Skip writing the output workbook.")
@click.option("--verbose/--quiet", default=True, show_default=True)
def run(
    loans_file: Path,
    collaterals_file: Path,
    deal_name: str,
    output_dir: Path,
    aggregation: str,
    min_coverage: float | None,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Run the pipeline against a pair of ESMA CSVs."""
    _configure_logging(verbose)

    aggregation_method = None if aggregation == "auto" else aggregation

    options: dict[str, object] = {
        "deal_name": deal_name,
        "aggregation": aggregation,
        "min_coverage": min_coverage,
        "dry_run": dry_run,
    }

    # The working directory of the report's detail, unique to this run. If
    # none can be created the assessment is skipped and the pipeline runs
    # as it does without the report.
    store, store_fault = _guarded(
        "create_store", lambda: ResultStore.create(output_dir / deal_name)
    )
    result: PipelineResult | None = None
    error: Exception | None = None
    try:
        # The files are assessed first, in a process of their own, so the
        # conversion starts in a process that holds none of the audit's memory.
        assessment: FileAssessment | None = None
        assessment_fault = store_fault
        if store is not None:
            started = datetime.now(UTC)
            working = store
            assessment, assessment_fault = _guarded(
                "assess_files",
                lambda: run_assessment(loans_file, collaterals_file, working, started),
            )
        try:
            result = run_pipeline(
                loans_file_path=loans_file,
                collaterals_file_path=collaterals_file,
                deal_name=deal_name,
                output_dir=output_dir,
                aggregation_method=aggregation_method,
                min_coverage=min_coverage,
                dry_run=dry_run,
                verbose=verbose,
            )
        except Exception as exc:
            error = exc
            raise
        finally:
            # Neither a result nor an Exception: the run was interrupted
            # (Ctrl-C, SystemExit). No report is written for a run that did
            # not finish. Otherwise it is written, guarded, so the
            # pipeline's own exception is the one that leaves the command.
            if store is not None and (result is not None or error is not None):
                _guarded(
                    "write_summary",
                    lambda: _write_quality_summary(
                        assessment, assessment_fault, result, error, output_dir, deal_name, options
                    ),
                )
    finally:
        if store is not None:
            _guarded("remove_store", store.remove)

    assert result is not None
    if result.output_path is not None:
        click.echo(str(result.output_path))


if __name__ == "__main__":
    main()
