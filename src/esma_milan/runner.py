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
from esma_milan.io_layer.write_workbook import (
    build_workbook_bytes,
    write_pipeline_workbook,
)
from esma_milan.pipeline.classification import Stage5Output, run_stage5
from esma_milan.pipeline.enriched import (
    compose_loans_enriched,
    compose_properties_enriched,
)
from esma_milan.pipeline.exec_summary import compose_execution_summary
from esma_milan.pipeline.filters import Stage2Output, run_stage2
from esma_milan.pipeline.flatten import Stage7Output, run_stage7
from esma_milan.pipeline.graph import GraphResult, run_stage4
from esma_milan.pipeline.identifiers import Stage3Output, run_stage3
from esma_milan.pipeline.mapping_tables import compose_mapping_tables
from esma_milan.pipeline.milan_map import compose_milan_pool
from esma_milan.pipeline.stage1 import Stage1Output, run_stage1
from esma_milan.pipeline.valuation import Stage6Output, run_stage6

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    """Return value from run_pipeline()."""

    output_path: Path | None
    """Path to the workbook on disk. None when dry_run was True, or when
    run_pipeline was called in in-memory mode (output_dir=None) - in
    that case the workbook is carried as `workbook_bytes` instead."""

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

    stage7: Stage7Output | None = None
    """Combined flattened pool from Stage 7 (None if Stage 7 didn't run).
    The frame is byte-equal to the R fixture's "Combined flattened pool"
    sheet column-for-column; calc_current_LTV / calc_original_LTV /
    calc_seasoning land here, and the calc_ rename + reorder from R's
    pipeline.R:553-592 has been applied."""

    chosen_aggregation_method: str | None = None
    """The aggregation method actually used downstream. Equals the
    user's explicit `aggregation_method` argument when supplied;
    otherwise equals `stage6.detected_method` when that's "by_loan" or
    "by_group", with a fallback to "by_loan" on "ambiguous" + non-
    interactive (matches r_reference/R/pipeline.R:469-485)."""

    workbook_bytes: bytes | None = None
    """The 10-sheet workbook as in-memory XLSX bytes. Populated only
    when run_pipeline was called in in-memory mode (output_dir=None)
    and dry_run was False - this is the path the FastAPI service takes
    so the workbook never touches disk on the server. None on disk-
    write runs and on dry_run."""

    output_filename: str | None = None
    """The workbook's bare filename (no directory), e.g.
    "2024-06-30 DEAL Flattened loans and collaterals.xlsx". Populated
    alongside `workbook_bytes` in in-memory mode; matches the filename
    used for the on-disk output. None on disk-write runs and dry_run."""

    execution_summary: pl.DataFrame | None = None
    """The composed Execution Summary (Sheet 1) as a 2-column
    `(Metric: Utf8, Value: Utf8)` Polars frame with pre-formatted
    string values. Populated whenever Stages 1-7 succeed, regardless
    of `dry_run`: the analysis-mode HTTP path needs the summary
    structure even when no workbook is written, and the ~60 ms cost on
    a 1k-loan pool is invisible next to the I/O overhead it would
    otherwise gate. None only when an earlier stage failed before
    Stage 10 could run."""


def run_pipeline(
    *,
    loans_file_path: Path,
    collaterals_file_path: Path,
    taxonomy_file_path: Path,
    deal_name: str,
    output_dir: Path | None = None,
    aggregation_method: str | None = None,
    min_coverage: float | None = None,
    interactive_mode: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> PipelineResult:
    """Run the ESMA -> MILAN pipeline.

    Output destination is selected by `output_dir`:
      - a Path  -> the workbook is written to disk under
        `output_dir/<deal_name>/` and returned as `result.output_path`
        (the CLI's `--output` behaviour).
      - None    -> the workbook is built in memory and returned as
        `result.workbook_bytes` + `result.output_filename`, never
        touching disk (the FastAPI service's no-disk-persistence path).
    `dry_run=True` skips workbook composition entirely regardless of
    `output_dir`.
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

    # Compose loans_enriched / properties_enriched ONCE here. These are
    # used by Stage 6 (detect_aggregation_method - extra cols are
    # harmless), Stage 7 (flatten requires the full classification
    # cols), and the workbook writer (Sheets 6/7). Mirrors the join
    # sequence at r_reference/R/pipeline.R:405-450.
    loans_enriched = compose_loans_enriched(
        stage3.loans, stage4.loan_groups, stage5.classifications
    )
    properties_enriched = compose_properties_enriched(
        stage3.properties, stage4.collateral_groups, stage5.classifications
    )

    # --- Stage 6: detect aggregation method (by_loan vs by_group) --------
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

    # --- Stage 7: flatten + derived fields (Combined flattened pool) -----
    # `chosen_aggregation` is a str at runtime constrained to {"by_loan",
    # "by_group"} by the resolution above; cast for mypy strict.
    from typing import cast

    from esma_milan.pipeline.flatten import AggregationMethod

    stage7 = run_stage7(
        loans_enriched,
        properties_enriched,
        stage5.classifications,
        aggregation_method=cast(AggregationMethod, chosen_aggregation),
    )

    # The final filename uses the pool_cutoff_date from loans (matches
    # r_reference/R/pipeline.R:611-621). Stage 1's parsed loans table
    # carries it as a Date column; pluck the first non-null value.
    cutoff = _first_non_null_date(stage1.loans, "pool_cutoff_date")
    if cutoff is None:
        raise ValueError("pool_cutoff_date is missing in loans file")
    cutoff_str = cutoff.isoformat()
    output_filename = f"{cutoff_str} {deal_name} Flattened loans and collaterals.xlsx"

    # Stage 9: compose the 175-column MILAN template pool (Sheet 10) from
    # the Stage-7 combined_flattened frame. Mirrors map_to_milan() in
    # r_reference/R/milan_mapping.R.
    # Stage 10: compose the Execution Summary (Sheet 1) from the MILAN
    # pool + enriched loans/properties + combined_flattened. Mirrors
    # the summary block in r_reference/R/pipeline.R:625-895.
    # Both run on every code path - dry_run skips only the workbook
    # write below, since the analysis-mode HTTP response needs the
    # summary structure too and the ~60 ms cost is invisible.
    milan_pool = compose_milan_pool(stage7.combined_flattened)
    execution_summary = compose_execution_summary(
        deal_name=deal_name,
        chosen_aggregation=chosen_aggregation,
        pool_cutoff_date=cutoff,
        milan_pool=milan_pool,
        loans_enriched=loans_enriched,
        properties_enriched=properties_enriched,
        combined_flattened=stage7.combined_flattened,
    )

    if dry_run:
        return PipelineResult(
            output_path=None,
            stage1=stage1,
            stage2=stage2,
            stage3=stage3,
            stage4=stage4,
            stage5=stage5,
            stage6=stage6,
            stage7=stage7,
            chosen_aggregation_method=chosen_aggregation,
            execution_summary=execution_summary,
        )

    # Stage 8.5: compose the four mapping tables (Sheets 1-4) from the
    # post-Stage-3 loans/properties + post-Stage-5 loans_enriched (the
    # latter for the classification tail on "Loans to properties").
    mapping_tables = compose_mapping_tables(
        stage3.loans, stage3.properties, loans_enriched
    )

    populated_sheets = {
        "Execution Summary": execution_summary,  # Stage 10: Sheet 1
        "Cleaned ESMA loans": loans_enriched,
        "Cleaned ESMA properties": properties_enriched,
        "Group classifications": stage5.classifications,
        "Combined flattened pool": stage7.combined_flattened,
        "MILAN template pool": milan_pool,  # Stage 9: Sheet 10
        **mapping_tables,  # Stage 8.5: Sheets 1-4 (mapping tables)
    }

    if output_dir is None:
        # In-memory mode (FastAPI service): build the workbook as bytes,
        # never touching disk. output_path stays None; the caller reads
        # workbook_bytes + output_filename off the result.
        workbook_bytes = build_workbook_bytes(populated_sheets=populated_sheets)
        if verbose:
            log.info(
                "pipeline_workbook_built_in_memory",
                filename=output_filename,
                n_bytes=len(workbook_bytes),
            )
        return PipelineResult(
            output_path=None,
            stage1=stage1,
            stage2=stage2,
            stage3=stage3,
            stage4=stage4,
            stage5=stage5,
            stage6=stage6,
            stage7=stage7,
            chosen_aggregation_method=chosen_aggregation,
            workbook_bytes=workbook_bytes,
            output_filename=output_filename,
            execution_summary=execution_summary,
        )

    deal_dir = output_dir / deal_name
    deal_dir.mkdir(parents=True, exist_ok=True)
    output_path = deal_dir / output_filename

    write_pipeline_workbook(output_path, populated_sheets=populated_sheets)

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
        stage7=stage7,
        chosen_aggregation_method=chosen_aggregation,
        execution_summary=execution_summary,
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
