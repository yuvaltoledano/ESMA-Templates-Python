"""Stage 1: read and clean.

Composes the I/O-layer helpers into the orchestration described in
r_reference/R/pipeline.R:130-200:

  1. Load taxonomy.
  2. Read loans + collaterals CSVs with the appropriate character_cols
     overrides; apply taxonomy rename + clean_names; convert NA tokens.
  3. Drop the metadata columns sec_id, unique_identifier, data_cut_off_date.
  4. Drop the loan-side and collateral-side currency-companion columns.
  5. Pre-collapse date_of_restructuring (multi-date cells -> max ISO).
  6. Convert every listed date column on each table via
     parse_iso_or_excel_date (handles mixed ISO + Excel-serial inputs).
  7. Validate the required columns are present on each table.

Returns a frozen `Stage1Output` carrying both cleaned tables. Stages 2+
operate on these.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import structlog

from esma_milan.config import (
    ALWAYS_DROPPED_COLUMNS,
    COLLATERALS_CHARACTER_COLS,
    COLLATERALS_CURRENCY_COMPANIONS,
    LOAN_DATE_COLUMNS,
    LOANS_CHARACTER_COLS,
    LOANS_CURRENCY_COMPANIONS,
    PROPERTY_DATE_COLUMNS,
    REQUIRED_LOAN_COLUMNS,
    REQUIRED_PROPERTY_COLUMNS,
)
from esma_milan.io_layer.date_parsing import (
    collapse_multi_date_restructuring,
    parse_iso_or_excel_date,
)
from esma_milan.io_layer.read_csv import read_and_clean
from esma_milan.io_layer.read_taxonomy import load_taxonomy
from esma_milan.pipeline.validate import validate_required_columns

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Stage1Output:
    """Output of Stage 1: cleaned loans and properties tables."""

    loans: pl.DataFrame
    properties: pl.DataFrame
    taxonomy: dict[str, str]


def run_stage1(
    *,
    loans_path: Path,
    collaterals_path: Path,
    taxonomy_path: Path,
) -> Stage1Output:
    """Execute Stage 1 against the three input paths and return cleaned tables."""
    log.info(
        "stage1_start",
        loans=loans_path.name,
        collaterals=collaterals_path.name,
        taxonomy=taxonomy_path.name,
    )

    taxonomy = load_taxonomy(taxonomy_path)

    # --- Read both CSVs ---------------------------------------------------
    loans = read_and_clean(loans_path, taxonomy, character_cols=LOANS_CHARACTER_COLS)
    properties = read_and_clean(
        collaterals_path, taxonomy, character_cols=COLLATERALS_CHARACTER_COLS
    )

    # --- Drop metadata + currency-companion columns -----------------------
    # `any_of` semantics: drop columns that are present, ignore those absent.
    loans_drop = [
        c for c in (*ALWAYS_DROPPED_COLUMNS, *LOANS_CURRENCY_COMPANIONS)
        if c in loans.columns
    ]
    properties_drop = [
        c for c in (*ALWAYS_DROPPED_COLUMNS, *COLLATERALS_CURRENCY_COMPANIONS)
        if c in properties.columns
    ]
    if loans_drop:
        loans = loans.drop(loans_drop)
    if properties_drop:
        properties = properties.drop(properties_drop)

    # --- Pre-collapse date_of_restructuring -------------------------------
    # Some originators pack multiple restructuring dates into a single
    # cell as ',' or ';' delimited tokens. Collapse to the max parseable
    # ISO date BEFORE the generic date parser runs (mirrors
    # r_reference/R/pipeline.R:174-178).
    if "date_of_restructuring" in loans.columns:
        # The collapse helper takes any iterable; we feed it the column's
        # current values (whatever dtype Polars inferred) and write the
        # results back as String so the downstream date parser can pick
        # them up uniformly.
        collapsed = collapse_multi_date_restructuring(
            loans["date_of_restructuring"].to_list()
        )
        loans = loans.with_columns(
            pl.Series("date_of_restructuring", collapsed, dtype=pl.String)
        )

    # --- Date column normalisation ----------------------------------------
    # Mixed ISO strings + Excel serials in the same column are common when
    # CSVs round-trip through Excel. parse_iso_or_excel_date handles both;
    # see r_reference/R/utils.R:605-667.
    for col in LOAN_DATE_COLUMNS:
        if col in loans.columns:
            loans = loans.with_columns(parse_iso_or_excel_date(loans[col], col))
    for col in PROPERTY_DATE_COLUMNS:
        if col in properties.columns:
            properties = properties.with_columns(
                parse_iso_or_excel_date(properties[col], col)
            )

    # --- Validate required columns ----------------------------------------
    validate_required_columns(loans, REQUIRED_LOAN_COLUMNS, "loans_cleaned")
    validate_required_columns(properties, REQUIRED_PROPERTY_COLUMNS, "properties_cleaned")

    log.info(
        "stage1_complete",
        loans_rows=loans.height,
        loans_cols=loans.width,
        properties_rows=properties.height,
        properties_cols=properties.width,
    )

    return Stage1Output(loans=loans, properties=properties, taxonomy=taxonomy)
