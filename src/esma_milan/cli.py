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

One addition has no counterpart in R: the data quality summary
(`esma_milan.quality`). The two files are assessed before the pipeline
runs, and `<workbook name> data quality summary.json` is written beside
the workbook when the pipeline returns or raises, a dry run included. It
changes nothing else: the pipeline is called with the same arguments, the
exception it raises leaves the command unchanged, and standard output is
still the workbook's path. A fault in the assessment itself is logged
once and never reaches the pipeline or the exit status.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date
from pathlib import Path

import click
import structlog

from esma_milan.quality import (
    FileAssessment,
    assess_files,
    complete_report,
    summary_filename,
    write_summary,
)
from esma_milan.quality.export import interrupted_summary, write_json
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
    the command's exit status are what they would be without the report."""
    try:
        return action(), None
    except Exception as exc:
        log.error("data_quality_assessment_failed", step=step, error=repr(exc))
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

    # The files are assessed first, so the audit's text frames are freed
    # before the pipeline builds its own.
    assessment, assessment_fault = _guarded(
        "assess_files", lambda: assess_files(loans_file, collaterals_file)
    )
    options: dict[str, object] = {
        "deal_name": deal_name,
        "aggregation": aggregation,
        "min_coverage": min_coverage,
        "dry_run": dry_run,
    }

    result: PipelineResult | None = None
    error: Exception | None = None
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
        # (Ctrl-C, SystemExit). No summary is written for a run that did
        # not finish. Otherwise the summary is written, guarded, so the
        # pipeline's own exception is the one that leaves the command.
        if result is not None or error is not None:
            _guarded(
                "write_summary",
                lambda: _write_quality_summary(
                    assessment, assessment_fault, result, error, output_dir, deal_name, options
                ),
            )

    assert result is not None
    if result.output_path is not None:
        click.echo(str(result.output_path))


if __name__ == "__main__":
    main()
