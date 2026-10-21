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
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import structlog

from esma_milan.runner import run_pipeline


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

    if result.output_path is not None:
        click.echo(str(result.output_path))


if __name__ == "__main__":
    main()
