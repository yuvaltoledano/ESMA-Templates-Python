"""Single library entry point for the pipeline.

Same function backs both the CLI and the FastAPI service. The current
implementation is a no-op stub: it writes an empty 10-sheet workbook so the
parity harness can demonstrate clean diffs against the staged R expected
output. The pipeline stages will be ported in §9 order, sheet by sheet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog

from esma_milan.io_layer.write_workbook import write_stub_workbook

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    """Return value from run_pipeline()."""

    output_path: Path | None
    """Path to the workbook on disk, or None if dry_run was True."""


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

    NO-OP STUB: writes an empty 10-sheet workbook. Full implementation
    lands in §9, stage by stage.

    Filename pattern matches r_reference/R/pipeline.R:935.
    """
    if verbose:
        log.info(
            "pipeline_start_stub",
            deal_name=deal_name,
            loans=str(loans_file_path),
            collaterals=str(collaterals_file_path),
        )

    if dry_run:
        return PipelineResult(output_path=None)

    # The R pipeline derives the filename from the first pool_cutoff_date in
    # the loans file. The stub doesn't read the file; we use a placeholder
    # that the parity harness will treat as part of the diff.
    deal_dir = output_dir / deal_name
    deal_dir.mkdir(parents=True, exist_ok=True)
    output_path = deal_dir / f"STUB {deal_name} Flattened loans and collaterals.xlsx"
    write_stub_workbook(output_path)

    if verbose:
        log.info("pipeline_stub_written", path=str(output_path))

    return PipelineResult(output_path=output_path)
