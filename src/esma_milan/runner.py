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

import polars as pl
import structlog

from esma_milan.config import DEFAULT_MIN_LOAN_ID_COVERAGE
from esma_milan.io_layer.write_workbook import write_pipeline_workbook
from esma_milan.pipeline.classification import Stage5Output, run_stage5
from esma_milan.pipeline.enriched import (
    compose_loans_enriched,
    compose_properties_enriched,
)
from esma_milan.pipeline.filters import Stage2Output, run_stage2
from esma_milan.pipeline.graph import GraphResult, run_stage4
from esma_milan.pipeline.identifiers import Stage3Output, run_stage3
from esma_milan.pipeline.stage1 import Stage1Output, run_stage1
from esma_milan.pipeline.valuation import Stage6Output, run_stage6

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

    stage2: Stage2Output | None = None
    """Filtered tables from Stage 2 (None if Stage 2 didn't run)."""

    stage3: Stage3Output | None = None
    """ID-augmented + intersection-filtered tables from Stage 3 (None if
    Stage 3 didn't run). Carries calc_loan_id, calc_borrower_id on
    loans and calc_property_id on properties."""

    stage4: GraphResult | None = None
    """Bipartite-graph result from Stage 4 (None if Stage 4 didn't run).
    Carries loan_groups, collateral_groups, and edges_with_group, all
    keyed on a deterministic 1-indexed collateral_group_id."""

    stage5: Stage5Output | None = None
    """Group classifications from Stage 5 (None if Stage 5 didn't run).
    Carries the (collateral_group_id, loans, collaterals, is_full_set,
    structure_type) frame written to the "Group classifications" sheet."""

    stage6: Stage6Output | None = None
    """Auto-detected aggregation method from Stage 6 (None if Stage 6
    didn't run). When the user passed an explicit `aggregation_method`
    argument, Stage 6 runs anyway as a sanity check; the explicit
    argument wins for the actual flatten in Stage 7."""

    chosen_aggregation_method: str | None = None
    """The aggregation method actually used downstream. Equals the
    user's explicit `aggregation_method` argument when supplied;
    otherwise equals `stage6.detected_method` when that's "by_loan" or
    "by_group", with a fallback to "by_loan" on "ambiguous" + non-
    interactive (matches r_reference/R/pipeline.R:469-485)."""


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

    # --- Stage 2: filter active loans + RRE properties -------------------
    stage2 = run_stage2(stage1.loans, stage1.properties)

    # --- Stage 3: select calc_loan_id + intersection filter + IDs --------
    stage3 = run_stage3(
        stage2.loans,
        stage2.properties,
        min_coverage=(
            min_coverage
            if min_coverage is not None
            else DEFAULT_MIN_LOAN_ID_COVERAGE
        ),
    )

    # --- Stage 4: bipartite graph + connected components -----------------
    stage4 = run_stage4(stage3.loans, stage3.properties)

    # --- Stage 5: classify groups into structure types 1-5 ---------------
    stage5 = run_stage5(stage4)

    # --- Stage 6: detect aggregation method (by_loan vs by_group) --------
    # Compose loans_enriched / properties_enriched for Stage 6's input:
    # the cleaned tables joined with their collateral_group_id from the
    # Stage 4 graph. Mirrors r_reference/R/pipeline.R:405-425.
    loans_enriched = stage3.loans.join(
        stage4.loan_groups, on="calc_loan_id", how="left"
    )
    properties_enriched = stage3.properties.join(
        stage4.collateral_groups, on="calc_property_id", how="left"
    )
    stage6 = run_stage6(loans_enriched, properties_enriched)

    # Resolve the aggregation method actually used downstream. Mirrors
    # r_reference/R/pipeline.R:469-485:
    #   - explicit argument wins when supplied
    #   - else use stage6.detected_method when it's by_loan / by_group
    #   - else (ambiguous + non-interactive) fall back to "by_loan"
    if aggregation_method is not None:
        chosen_aggregation = aggregation_method
    elif stage6.detected_method in ("by_loan", "by_group"):
        chosen_aggregation = stage6.detected_method
    else:
        # Ambiguous + non-interactive: R defaults to by_loan with a warning.
        chosen_aggregation = "by_loan"
        log.warning(
            "aggregation_method_ambiguous_defaulting_to_by_loan",
            stage6_message=stage6.message,
        )
    if verbose:
        log.info("chosen_aggregation_method", method=chosen_aggregation)

    # TODO Stages 7..10. Until they land, the output workbook only writes
    # the sheets contributed by the stages that have shipped (currently
    # just "Group classifications"); the other 9 stay empty so the
    # parity harness's sheet-order check still passes.

    if dry_run:
        return PipelineResult(
            output_path=None,
            stage1=stage1,
            stage2=stage2,
            stage3=stage3,
            stage4=stage4,
            stage5=stage5,
            stage6=stage6,
            chosen_aggregation_method=chosen_aggregation,
        )

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

    # Compose loans_enriched / properties_enriched for Sheets 6 and 7.
    # Mirrors the joins at r_reference/R/pipeline.R:405-450; column
    # order is fully determined by the input frames + Polars' deterministic
    # left-join semantics. See pipeline.enriched.
    loans_enriched_for_writer = compose_loans_enriched(
        stage3.loans, stage4.loan_groups, stage5.classifications
    )
    properties_enriched_for_writer = compose_properties_enriched(
        stage3.properties, stage4.collateral_groups, stage5.classifications
    )

    write_pipeline_workbook(
        output_path,
        populated_sheets={
            "Cleaned ESMA loans": loans_enriched_for_writer,
            "Cleaned ESMA properties": properties_enriched_for_writer,
            "Group classifications": stage5.classifications,
        },
    )

    if verbose:
        log.info("pipeline_workbook_written", path=str(output_path))

    return PipelineResult(
        output_path=output_path,
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
        stage4=stage4,
        stage5=stage5,
        stage6=stage6,
        chosen_aggregation_method=chosen_aggregation,
    )


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
