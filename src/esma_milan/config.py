"""Pipeline constants. Mirrors r_reference/R/utils.R exactly.

Any change here is a deviation from the R reference and must be flagged
in writing per §10 / §11 of the project brief.
"""

from __future__ import annotations

# Minimum valid property value - values at or below this are considered
# invalid/placeholder and trigger the Stage-2 valuation flip in flattening.
# r_reference/R/utils.R:5
MIN_VALID_PROPERTY_VALUE: int = 10

# Maximum number of properties per loan before issuing a warning.
# r_reference/R/utils.R:8
MAX_PROPERTIES_PER_LOAN_WARNING_THRESHOLD: int = 50

# Active loan account statuses (ESMA codes). Loans whose account_status is
# not in this set are dropped at Stage 2.
# r_reference/R/utils.R:11
ACTIVE_LOAN_STATUSES: frozenset[str] = frozenset({"PERF", "ARRE", "RARR", "RNAR"})

# Valuation methods that indicate full inspection (FIEI/FOEI preferred).
# r_reference/R/utils.R:14
VALUATION_METHODS_FULL_INSPECTION: frozenset[str] = frozenset({"FIEI", "FOEI"})

# Collateral types to exclude from analysis. These are non-residential-real-
# estate codes from the ESMA RMBS taxonomy. Any property row carrying one of
# these codes is dropped before the loan-collateral graph is built.
# r_reference/R/utils.R:22-33  (19 codes - the prompt's "20" was a miscount)
EXCLUDED_COLLATERAL_TYPES: frozenset[str] = frozenset({
    # Guarantees
    "GUAR",
    # Vehicles
    "CARX", "CMTR", "RALV", "NACM", "NALV", "AERO", "OTHV",
    # Equipment
    "MCHT", "INDE", "OFEQ", "ITEQ", "MDEQ", "ENEQ", "OTHE",
    # Financial assets
    "SECU", "OTFA",
    # Inventory & individual / other
    "OTGI", "INDV",
})

# Maximum filename length for filesystem compatibility.
# r_reference/R/utils.R:36
MAX_FILENAME_LENGTH: int = 255

# Default minimum coverage of the chosen calc_loan_id column against the
# collaterals file. Coverage below this threshold errors; coverage at or
# above the threshold (but below 1.0) warns and continues.
# r_reference/R/utils.R:43
DEFAULT_MIN_LOAN_ID_COVERAGE: float = 0.85

# CSV NA tokens passed to the loader. Matches readr's `na` argument in
# r_reference/R/utils.R:75 / 82. The MD doc at docs/ESMA_PROCESSING_DOCUMENTATION.md
# omits "ND" and "ND3"; the code is the truth.
CSV_NA_TOKENS: tuple[str, ...] = ("", "NA", "ND", "ND1", "ND2", "ND3", "ND4", "ND5")

# ND-equivalent string set used by `is_nd()` in milan_mapping.R:19. Note this
# is a superset of CSV_NA_TOKENS - is_nd() also matches whitespace-only
# strings via a separate trimws() check.
ND_STRINGS: frozenset[str] = frozenset({"ND", "ND1", "ND2", "ND3", "ND4", "ND5", "NA"})

# Currency-companion columns dropped from the loans table after the taxonomy
# rename. Auxiliary columns that specify the currency unit of monetary fields.
# r_reference/R/pipeline.R:142-148
LOANS_CURRENCY_COMPANIONS: tuple[str, ...] = (
    "rrel16_currency", "rrel20_currency", "rrel29_currency",
    "rrel31_currency", "rrel32_currency", "rrel33_currency",
    "rrel41_currency", "rrel77_currency", "rrel39_currency",
    "rrel67_currency", "rrel71_currency", "rrel74_currency",
    "rrel73_currency", "rrel61_currency", "rrel64_currency",
)

# Currency-companion columns dropped from the collaterals table.
# r_reference/R/pipeline.R:158-160
COLLATERALS_CURRENCY_COMPANIONS: tuple[str, ...] = (
    "rrec17_currency", "rrec13_currency", "rrec21_currency",
)

# Always-dropped metadata columns from both files.
# r_reference/R/pipeline.R:155 / 167
ALWAYS_DROPPED_COLUMNS: tuple[str, ...] = (
    "sec_id", "unique_identifier", "data_cut_off_date",
)

# Columns forced to character type when reading the loans CSV (the four
# loan/borrower ID columns - prevents readr from numerically coercing IDs
# that look like integers).
# r_reference/R/pipeline.R:153
LOANS_CHARACTER_COLS: tuple[str, ...] = ("RREL2", "RREL3", "RREL4", "RREL5")

# Columns forced to character type when reading the collaterals CSV.
# r_reference/R/pipeline.R:165
COLLATERALS_CHARACTER_COLS: tuple[str, ...] = ("RREC2", "RREC3", "RREC4")

# Date columns to normalise via parse_iso_or_excel_date on the loans table.
# r_reference/R/pipeline.R:184-189
LOAN_DATE_COLUMNS: tuple[str, ...] = (
    "pool_cutoff_date", "origination_date", "maturity_date",
    "date_last_in_arrears", "interest_revision_date_1",
    "date_of_repurchase", "date_of_restructuring",
    "redemption_date", "default_date",
)

# Date columns to normalise on the collaterals table.
# r_reference/R/pipeline.R:194-197
PROPERTY_DATE_COLUMNS: tuple[str, ...] = (
    "original_valuation_date", "current_valuation_date",
    "property_pool_cutoff_date", "pool_cutoff_date",
)

# Required columns asserted after Stage 1 cleaning.
# r_reference/R/pipeline.R:208-214
REQUIRED_LOAN_COLUMNS: tuple[str, ...] = (
    "new_underlying_exposure_identifier", "account_status",
    "pool_cutoff_date", "current_principal_balance",
    "original_underlying_exposure_identifier", "new_obligor_identifier",
    "origination_date",
)

# r_reference/R/pipeline.R:220-223
REQUIRED_PROPERTY_COLUMNS: tuple[str, ...] = (
    "underlying_exposure_identifier", "new_collateral_identifier",
    "property_type", "collateral_type", "current_valuation_amount",
)

# 10-sheet output workbook order. r_reference/R/pipeline.R:903-914
OUTPUT_SHEET_ORDER: tuple[str, ...] = (
    "Execution Summary",
    "Loans to properties",
    "Properties to loans",
    "Borrowers to loans",
    "Borrowers to properties",
    "Cleaned ESMA loans",
    "Cleaned ESMA properties",
    "Group classifications",
    "Combined flattened pool",
    "MILAN template pool",
)

# Sheets whose date columns are written as Excel serial numbers
# (because openxlsx::writeData serialises R Date type as numeric serials).
# Sheets NOT in this set use ISO date strings.
# Confirmed against tests/fixtures/synthetic/expected_r_output.xlsx.
SHEETS_WITH_EXCEL_SERIAL_DATES: frozenset[str] = frozenset({
    "Cleaned ESMA loans",
    "Cleaned ESMA properties",
    "Combined flattened pool",
})
