"""Single library entry point for the pipeline.

Same function backs both the CLI and the FastAPI service. As stages port
over, each stage's driver is composed in here in §9 order. Stages still
to land write a no-op slot in the workbook so the parity harness can
keep diffing against the staged R reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from esma_milan.io_layer.write_workbook import write_stub_workbook
from esma_milan.pipeline.stage1 import Stage1Output, run_stage1

if TYPE_CHECKING:
    import polars as pl

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    """Return value from run_pipeline()."""

    output_path: Path | None
    """Path to the workbook on disk, or None if dry_run was True."""

    stage1: Stage1Output | None = None
    """Cleaned tables from Stage 1 (None if Stage 1 didn't run, e.g. on
    error before then). Exposed so integration tests can assert on
    intermediate state without re-running the full pipeline."""


def run_pipeline(
    *,
    loans_file_path: Path,
    collaterals_file_path: Path,
    taxonomy_file_path: Path,
    deal_name: str,
    output_dir: Path,
    aggregation_method: str | None = None,
    min_coverage: float | None = None,
    interactive_mode: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> PipelineResult:
    """Run the ESMA -> MILAN pipeline.

    Currently implements Stage 1 only; remaining stages add their
    contribution to the output workbook as they land in §9 order. Until
    Stage 10 is in, the workbook is a 10-sheet empty stub so the parity
    harness can keep diffing.
    """
    if verbose:
        log.info(
            "pipeline_start",
            deal_name=deal_name,
            loans=str(loans_file_path),
            collaterals=str(collaterals_file_path),
        )

    # --- Stage 1: read & clean -------------------------------------------
    stage1 = run_stage1(
        loans_path=loans_file_path,
        collaterals_path=collaterals_file_path,
        taxonomy_path=taxonomy_file_path,
    )

    # TODO Stages 2..10. Until they land, the output workbook stays an
    # empty 10-sheet stub.

    if dry_run:
        return PipelineResult(output_path=None, stage1=stage1)

    # The final filename uses the pool_cutoff_date from loans (matches
    # r_reference/R/pipeline.R:611-621). Stage 1's parsed loans table
    # carries it as a Date column; pluck the first non-null value.
    cutoff = _first_non_null_date(stage1.loans, "pool_cutoff_date")
    if cutoff is None:
        raise ValueError("pool_cutoff_date is missing in loans file")
    cutoff_str = cutoff.isoformat()

    deal_dir = output_dir / deal_name
    deal_dir.mkdir(parents=True, exist_ok=True)
    output_path = deal_dir / f"{cutoff_str} {deal_name} Flattened loans and collaterals.xlsx"
    write_stub_workbook(output_path)

    if verbose:
        log.info("pipeline_workbook_written", path=str(output_path))

    return PipelineResult(output_path=output_path, stage1=stage1)


def _first_non_null_date(df: pl.DataFrame, col: str) -> date | None:
    """Return the first non-null Date value in `col`, or None."""
    if col not in df.columns:
        return None
    series = df[col].drop_nulls()
    if len(series) == 0:
        return None
    value = series[0]
    if isinstance(value, date):
        return value
    raise TypeError(
        f"_first_non_null_date: column {col!r} should hold dates, got {type(value).__name__}"
    )
