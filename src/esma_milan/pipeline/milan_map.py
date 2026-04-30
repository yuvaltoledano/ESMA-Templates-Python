"""MILAN template mapping: 175 fields, in canonical Master Mapping order.

Mirrors r_reference/R/milan_mapping.R. Stage 9 lands the field mapping
incrementally:

    B-1: helpers + code-map constants                    (this commit)
    B-2: per-property ranking + cumulative-sum CB fields
    B-3: date-based derivations (Months Current, ...)
    B-4: INFER + remaining CALC fields
    B-5: assembly + runner wire-in + parity flip

The Python port matches R behaviour except for two documented fail-loud
divergences (R-repo issue tracker entries #6 and #8): silent failure
modes that R hides behind tryCatch / warning are surfaced here as
exceptions. Each divergence is annotated with a one-line comment at the
relevant site referencing the tracker entry.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import polars as pl

from esma_milan.nd import is_nd_expr

# Canonical 175-column output order. Copied verbatim from
# r_reference/R/milan_mapping.R:199-304. Do not reorder; downstream parity
# tests depend on this exact sequence.
MILAN_EXPECTED_OUTPUT_COLS: tuple[str, ...] = (
    "Originator Identifier", "Servicer Identifier",
    "Borrower Identifier", "Property Identifier", "Loan Identifier",
    "Loan Currency", "Loan OB", "Loan CB",
    "Origination Date", "Maturity Date",
    "Origination Channel", "Loan Purpose",
    "Principal Payment Frequency", "Interest Payment Frequency",
    "Principal Payment Type", "Periodic Payment",
    "Ranking", "Total Prior Ranks CB", "Total Prior Ranks OB",
    "External Prior Ranks CB", "Pari Passu Ranking Loans (Not In Pool)",
    "Retained Amount", "Further Advances", "Flexible Loan Amount",
    "Grace Period", "Type of Payment Holiday",
    "Length of Payment Holiday Allowed In Months",
    "Interest Rate Type", "Base Index",
    "Interest Rate", "Full Margin", "Margin After Reset Date",
    "Interest Reset Date",
    "Months Current", "Arrears Amount", "Months In Arrears",
    "Number of Payments Before Securitisation",
    "Social Programme Type",
    "FX Risk Mitigated", "Inflation-Indexed Loan",
    "Exchange Rate At Loan Origination",
    "Borrower Type", "Borrower Residency", "Borrower Birth Year",
    "Employment Type", "Internal Credit Score", "Credit Score - Canada",
    "Borrower Postal/Zip Code", "Number of Borrowers",
    "Income Borrower 1", "Income Verification For Primary Income",
    "Income Borrower 2", "Income Verification For Secondary Income",
    "Guarantor's Income", "Total Income", "DTI Ratio",
    "Loan To Employee", "Number of Properties Owned By Borrower",
    "Regulated",
    "Prior Missed On Previous Mortgage",
    "Live Units of Adverse Credit", "Satisfied Units of Adverse Credit",
    "Total Amount of CCJs or Balance of Adverse Credit",
    "Number of Other Adverse Credit Events",
    "Prior Personally Bankruptcy Or IVA",
    "Prior Repossessions On Previous Mortgage", "Prior Litigation",
    "Deposit Amount", "Property Type", "Property Value", "Purchase Price",
    "Recourse", "Property Valuation Type", "Property Valuation Date",
    "Confidence Interval For Original AVM Valuation",
    "Provider of Original AVM Valuation",
    "Property Postal, Region code or Nuts code",
    "Occupancy Type", "Rental Income", "Metro / Non Metro",
    "Construction Year",
    "Mortgage Inscription - Property 1", "Mortgage Mandate - Property 1",
    "Property Identifier Property 2", "Property Type - Property 2",
    "Property Value Property 2", "Property Valuation Type - Property 2",
    "Property Valuation date Property 2", "Property Postcode Property 2",
    "Occupancy Type - Property 2",
    "Mortgage Inscription Property 2", "Mortgage mandate Property 2",
    "Property Identifier Property 3", "Property Type - Property 3",
    "Property Value Property 3", "Property Valuation Type - Property 3",
    "Property Valuation date Property 3", "Property Postcode Property 3",
    "Occupancy Type - Property 3",
    "Mortgage Inscription Property 3", "Mortgage mandate Property 3",
    "Property Identifier Property 4", "Property Type - Property 4",
    "Property Value Property 4", "Property Valuation Type - Property 4",
    "Property Valuation date Property 4", "Property Postcode Property 4",
    "Occupancy Type - Property 4",
    "Mortgage Inscription Property 4", "Mortgage mandate Property 4",
    "Property Identifier Property 5", "Property Type - Property 5",
    "Property Value Property 5", "Property Valuation Type - Property 5",
    "Property Valuation date Property 5", "Property Postcode Property 5",
    "Occupancy Type - Property 5",
    "Mortgage Inscription Property 5", "Mortgage mandate Property 5",
    "Original LTV", "Current LTV",
    "Energy Performance Certificate Value",
    "Energy Performance Certificate Provider Name",
    "Additional Collateral Type", "Additional Collateral Provider",
    "Additional Collateral Value",
    "Account Status", "Repurchase Date",
    "Restructured Loan", "Restructuring Date", "Restructuring Type",
    "Redemption Date",
    "Default amount", "Default date",
    "Reason for Default or Foreclosure",
    "Allocated Losses", "Cumulative Recoveries", "Sale Price",
    "MIG Provider", "Type of Guarantee Provider",
    "MIG %", "Initial MCI Coverage Amount", "Current MCI Coverage Amount",
    "Additional data 1 - calc_loan_id",
    "Additional data 2 - calc_borrower_id",
    "Additional data 3 - calc_main_property_id",
    "Additional data 4 - calc_collateral_group_id",
    "Additional data 5 - calc_nr_loans_in_group",
    "Additional data 6 - calc_nr_properties_in_group",
    "Additional data 7 - calc_full_set",
    "Additional data 8 - calc_cross_collateralized_set",
    "Additional data 9 - calc_structure_type",
    "Additional data 10 - calc_aggregated_property_value",
    "Additional data 11 - calc_original_LTV",
    "Additional data 12 - calc_current_LTV",
    "Additional data 13 - calc_seasoning",
    "Additional data 14 - lien",
    "Additional data 15 - prior_principal_balances",
    "Additional data 16 - pari_passu_underlying_exposures",
    "Additional data 17 - original_valuation_method",
    "Additional data 18 - original_valuation_date",
    "Additional data 19 - current_valuation_method",
    "Additional data 20 - current_valuation_date",
    "Additional data 21 - new_underlying_exposure_identifier",
    "Additional data 22 - original_underlying_exposure_identifier",
    "Additional data 23 - new_obligor_identifier",
    "Additional data 24 - original_obligor_identifier",
    "Additional data 25 - new_collateral_identifier",
    "Additional data 26 - original_collateral_identifier",
    "column for additional data 27", "column for additional data 28",
    "column for additional data 29", "column for additional data 30",
    "column for additional data 31", "column for additional data 32",
)

assert len(MILAN_EXPECTED_OUTPUT_COLS) == 175, (
    f"MILAN_EXPECTED_OUTPUT_COLS must have 175 entries, has {len(MILAN_EXPECTED_OUTPUT_COLS)}"
)


# ---------------------------------------------------------------------------
# Code Maps - inline definitions from the Code Maps sheet
# ---------------------------------------------------------------------------
#
# Verbatim copy of MILAN_CODE_MAPS in r_reference/R/milan_mapping.R:109-186.
# Each map turns an ESMA source code into a MILAN code; codes not in the
# map fall through to "ND" via _code_map_lookup_expr.

MILAN_CODE_MAPS: dict[str, dict[str, str]] = {
    "origination_channel": {
        "BRAN": "1", "DRCT": "2", "BROK": "3", "WEBI": "4",
        "TPAC": "5", "TPTC": "6", "OTHR": "7",
    },
    "purpose": {
        "PURC": "1", "RMRT": "2", "RENV": "3", "EQRE": "4",
        "CNST": "5", "DCON": "6", "RMEQ": "7",
        "BSFN": "8", "CMRT": "8", "IMRT": "8", "RGBY": "8",
        "GSPL": "8", "OTHR": "8",
    },
    "principal_payment_frequency": {
        "MNTH": "1", "QUTR": "2", "SEMI": "3", "YEAR": "4", "OTHR": "5",
    },
    "interest_payment_frequency": {
        "MNTH": "1", "QUTR": "2", "SEMI": "3", "YEAR": "4", "OTHR": "5",
    },
    "principal_payment_type": {
        "FRXX": "1", "DEXX": "2", "FIXE": "2", "BLLT": "8", "OTHR": "ND",
    },
    "interest_rate_type": {
        "FLIF": "1", "FINX": "2", "FXRL": "3", "FXPR": "4",
        "FLCF": "5", "FLFL": "6", "CAPP": "6", "FLCA": "6",
        "DISC": "7", "SWIC": "8", "OBLS": "8", "MODE": "8", "OTHR": "8",
    },
    # Base Index single-key (non-LIBO/EURI codes). LIBO/EURI go through
    # the two-key tenor lookup in MILAN_BASE_INDEX_TENOR.
    "base_index_single": {
        "BOER": "9", "ECBR": "10", "LDOR": "11",
        "TREA": "13", "MAAA": "13", "FUSW": "13", "LIBI": "13",
        "SWAP": "13", "PFAN": "13", "EONA": "13", "EONS": "13",
        "EUUS": "13", "EUCH": "13", "TIBO": "13", "ISDA": "13",
        "GCFR": "13", "STBO": "13", "BBSW": "13", "JIBA": "16",
        "BUBO": "13", "CDOR": "13", "CIBO": "13", "MOSP": "13",
        "NIBO": "13", "PRBO": "13", "TLBO": "13", "WIBO": "13",
        "OTHR": "13",
    },
    "employment_type": {
        "EMRS": "1", "EMBL": "2", "EMUK": "1", "SFEM": "3",
        "NOEM": "4", "PNNR": "5", "STNT": "6", "UNEM": "6", "OTHR": "6",
    },
    "income_verification": {
        "VRFD": "1", "SCNF": "2", "SCRG": "2",
        "NVRF": "3", "SCRT": "4", "OTHR": "5",
    },
    "property_type": {
        "RHOS": "1", "RFLT": "2", "RBGL": "1", "RTHS": "1",
        "MULF": "5", "PCMM": "6", "BIZZ": "7", "LAND": "8", "OTHR": "10",
    },
    "occupancy_type": {
        "FOWN": "1", "POWN": "3", "TLET": "3", "HOLD": "2", "OTHR": "4",
    },
    "energy_performance_certificate_value": {
        "EPCA": "1", "EPCB": "2", "EPCC": "3", "EPCD": "4",
        "EPCE": "5", "EPCF": "6", "EPCG": "7", "OTHR": "8",
    },
    "account_status": {
        "PERF": "1", "ARRE": "2",
        "DFLT": "3", "NDFT": "3", "DTCR": "3", "DADB": "3",
        "RDMD": "4", "REBR": "5", "REDF": "6", "RERE": "7",
        "RESS": "8", "REOT": "9", "RNAR": "10", "RARR": "10",
        "OTHR": "ND",
    },
    "reason_for_default": {
        "UPXX": "1", "PDXX": "2", "UPPD": "3", "OTHR": "ND",
    },
    "guarantee_provider_type": {
        "NGUA": "1", "FAML": "2", "IOTH": "3", "GOVE": "4",
        "BANK": "5", "INSU": "6", "NHGX": "7", "FGAS": "8",
        "CATN": "9", "OTHR": "10",
    },
    "property_valuation_type": {
        "FIEI": "1", "FOEI": "2", "DRVB": "3", "AUVM": "4",
        "IDXD": "5", "DKTP": "6", "MAEA": "9", "TXAT": "8", "OTHR": "11",
    },
}


# Two-key Base Index lookup for LIBO/EURI x tenor. Mirrors
# MILAN_BASE_INDEX_TENOR in r_reference/R/milan_mapping.R:188-193.
# Keyed by "<index>|<tenor>" so the lookup composes cleanly with a
# pl.format() composite key in _lookup_base_index_expr.
MILAN_BASE_INDEX_TENOR: dict[str, str] = {
    "LIBO|MNTH": "1", "LIBO|QUTR": "3", "LIBO|SEMI": "5", "LIBO|YEAR": "7",
    "EURI|MNTH": "2", "EURI|QUTR": "4", "EURI|SEMI": "6", "EURI|YEAR": "8",
}


# ---------------------------------------------------------------------------
# Helpers (Polars expressions)
# ---------------------------------------------------------------------------


def _safe_as_num_expr(col: pl.Expr) -> pl.Expr:
    """Char/numeric -> Float64, with ND tokens collapsing to null.

    Mirrors r_reference/R/milan_mapping.R:332-335 (`safe_as_num`):

        x_chr <- as.character(x)
        if_else(is_nd(x_chr), NA_real_, as.numeric(x_chr))

    Polars' cast(Float64, strict=False) returns null on parse failure -
    matches `suppressWarnings(as.numeric(x))`, where R returns NA on
    non-numeric strings.
    """
    s = col.cast(pl.Utf8, strict=False)
    return (
        pl.when(is_nd_expr(s))
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(s.cast(pl.Float64, strict=False))
    )


def _divide_by_100_expr(col: pl.Expr) -> pl.Expr:
    """ND-passthrough divide-by-100, formatted as character.

    Mirrors r_reference/R/milan_mapping.R:343-346 (`divide_by_100`):

        val <- safe_as_num(x)
        if_else(is.na(val), "ND", as.character(val / 100))

    Polars Float -> Utf8 cast formats with up to ~17 significant digits
    and no trailing zero-stripping; for typical interest-rate inputs
    (e.g. 12.5 -> 0.125) this matches R's `as.character()` behaviour
    one-for-one. Pin via fixture rather than asserting general string-
    format equivalence.
    """
    val = _safe_as_num_expr(col)
    return (
        pl.when(val.is_null())
        .then(pl.lit("ND"))
        .otherwise((val / 100).cast(pl.Utf8))
    )


def _code_map_lookup_expr(col: pl.Expr, mapping: dict[str, str]) -> pl.Expr:
    """Single-key code-map lookup with ND passthrough.

    Mirrors r_reference/R/milan_mapping.R:96-104 (`code_map_lookup`):

        ND      -> "ND"
        known   -> mapping[code]
        unknown -> "ND"

    Polars' replace_strict(default=...) collapses both the unknown and
    ND branches into "ND". The explicit ND branch keeps the contract
    obvious - and shields against any future divergence between the
    `is_nd_expr` semantic and Polars' equality-based mapping lookup.
    """
    s = col.cast(pl.Utf8, strict=False)
    return (
        pl.when(is_nd_expr(s))
        .then(pl.lit("ND"))
        .otherwise(s.replace_strict(mapping, default="ND"))
    )


def _lookup_base_index_expr(index_col: pl.Expr, tenor_col: pl.Expr) -> pl.Expr:
    """Two-key Base Index lookup for LIBO/EURI x tenor; single-key fallback.

    Mirrors r_reference/R/milan_mapping.R:350-378 (`lookup_base_index`):

        ND index            -> "ND"
        LIBO/EURI x tenor   -> tenor table; unknown tenor -> "ND"
        any other non-ND    -> base_index_single map (default "ND")

    Critical: LIBO/EURI with an unknown tenor lands in the two-key
    branch and emits "ND" - it does NOT fall through to the single-key
    map (which would otherwise resolve LIBO/EURI to "13"). Matches R's
    `single_mask <- !nd_mask & !libo_euri_mask` short-circuit.
    """
    s_idx = index_col.cast(pl.Utf8, strict=False)
    s_ten = tenor_col.cast(pl.Utf8, strict=False)
    nd = is_nd_expr(s_idx)
    is_libo_euri = (~nd) & s_idx.is_in(["LIBO", "EURI"])

    composite = pl.format("{}|{}", s_idx, s_ten)
    two_key = composite.replace_strict(MILAN_BASE_INDEX_TENOR, default="ND")
    single_key = s_idx.replace_strict(
        MILAN_CODE_MAPS["base_index_single"], default="ND"
    )

    return (
        pl.when(nd).then(pl.lit("ND"))
        .when(is_libo_euri).then(two_key)
        .otherwise(single_key)
    )


def _ensure_source_columns(df: pl.DataFrame, expected: Sequence[str]) -> pl.DataFrame:
    """Warn-and-fill missing columns with the string "ND".

    Mirrors r_reference/R/milan_mapping.R:312-324 (`ensure_source_columns`).
    Warning text is verbatim so log-grep parity with R holds.
    """
    missing = [c for c in expected if c not in df.columns]
    if not missing:
        return df
    warnings.warn(
        "MILAN mapping: the following source columns are missing and will be "
        f"set to 'ND': {', '.join(missing)}",
        UserWarning,
        stacklevel=2,
    )
    return df.with_columns([pl.lit("ND").alias(c) for c in missing])
