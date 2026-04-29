"""ESMA -> MILAN CLI. Strict superset of r_reference/R/main.R's CLI surface
(see §10 of project brief - deliberate documented deviation).
"""

from __future__ import annotations

import logging
import sys
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
    "--taxonomy",
    "taxonomy_file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to ESMA taxonomy XLSX. Defaults to r_reference/inputs/ESMA template taxonomy.xlsx.",
)
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
    taxonomy_file: Path | None,
    output_dir: Path,
    aggregation: str,
    min_coverage: float | None,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Run the pipeline against a pair of ESMA CSVs."""
    _configure_logging(verbose)

    if taxonomy_file is None:
        # Default to the taxonomy bundled with the R reference. Project brief
        # treats r_reference/ as read-only but reading from it is fine.
        taxonomy_file = Path("r_reference/inputs/ESMA template taxonomy.xlsx")
        if not taxonomy_file.exists():
            click.echo(
                f"Error: --taxonomy not provided and default {taxonomy_file} not found.",
                err=True,
            )
            sys.exit(2)

    aggregation_method = None if aggregation == "auto" else aggregation

    result = run_pipeline(
        loans_file_path=loans_file,
        collaterals_file_path=collaterals_file,
        taxonomy_file_path=taxonomy_file,
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
