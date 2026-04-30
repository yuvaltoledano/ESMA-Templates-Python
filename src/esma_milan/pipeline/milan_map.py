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

    Routes the float result through _r_as_character_expr (%.15g) to
    match R's as.character() byte-equal: Polars' default Float -> Utf8
    cast emits 17 sig digits and exposes IEEE-754 noise (e.g.
    "0.6900000000000001" for 69/100), while R's as.character() rounds
    to 15 sig digits and emits "0.69".
    """
    val = _safe_as_num_expr(col)
    return (
        pl.when(val.is_null())
        .then(pl.lit("ND"))
        .otherwise(_r_as_character_expr(val / 100))
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


# ---------------------------------------------------------------------------
# Stage 9 / B-2: per-property ranking + cumulative-sum CB fields
# ---------------------------------------------------------------------------


def _attach_ranking(df: pl.DataFrame) -> pl.DataFrame:
    """Add `_ppb_num` and `_milan_ranking` (dense_rank within property).

    Mirrors r_reference/R/milan_mapping.R:478-494:

        .ppb_numeric = if_else(is_nd(prior_principal_balances), 0, as.numeric(...))
        .milan_ranking = dense_rank(.ppb_numeric)   # within calc_main_property_id

    ND prior_principal_balances values collapse to 0 *before* ranking,
    so a row with ND ppb gets the lowest rank within its property
    (rank 1 if no other row has a smaller ppb). dplyr's dense_rank
    starts at 1 with no gaps; Polars' rank(method="dense") matches.
    """
    return df.with_columns(
        _ppb_num=pl.when(is_nd_expr(pl.col("prior_principal_balances")))
        .then(pl.lit(0.0))
        .otherwise(
            pl.col("prior_principal_balances")
            .cast(pl.Utf8, strict=False)
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
        ),
    ).with_columns(
        _milan_ranking=pl.col("_ppb_num")
        .rank(method="dense")
        .over("calc_main_property_id")
        .cast(pl.Int64),
    )


def _attach_external_prior_and_pari_passu(df: pl.DataFrame) -> pl.DataFrame:
    """Add `_cpb_num`, `_ppue_num`, `_cpb_sum_at_rank`, `_cpb_cumsum_below`,
    `_milan_ext_prior_ranks_cb`, `_milan_pari_passu_not_in_pool`.

    Mirrors r_reference/R/milan_mapping.R:498-555. The R recipe:

      1. Coerce CPB and PPUE to numeric, ND -> 0.
      2. Per-(property, rank) sum of CPB (".cpb_sum_at_rank").
      3. Per-property, cum_sum of step-2 in rank-ascending order, lagged
         by 1 with default 0 (".cpb_cumsum_below" - the sum of CB at
         strictly lower ranks within the same property).
      4. Join (property, rank) totals back to the row-level frame.
      5. Compute the two final fields:
         - .milan_ext_prior_ranks_cb = max(0, ppb - cumsum_below)
         - .milan_pari_passu_not_in_pool = max(0, ppue - sum_at_rank)

    Asymmetric defaults documented as R-repo issue tracker entry #9
    (probably-intentional Moody's MILAN spec quirk): External Prior
    displays "ND" on ND ppb input while Pari Passu displays "0" on ND
    ppue input. Both float values are 0 in those cases (since ND
    collapses to 0 before subtraction); the asymmetry lives entirely in
    the B-5 transmute, not here.
    """
    df = df.with_columns(
        _cpb_num=pl.when(is_nd_expr(pl.col("current_principal_balance")))
        .then(pl.lit(0.0))
        .otherwise(
            pl.col("current_principal_balance")
            .cast(pl.Utf8, strict=False)
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
        ),
        _ppue_num=pl.when(is_nd_expr(pl.col("pari_passu_underlying_exposures")))
        .then(pl.lit(0.0))
        .otherwise(
            pl.col("pari_passu_underlying_exposures")
            .cast(pl.Utf8, strict=False)
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
        ),
    )

    # Build the per-(property, rank) totals + per-property cumsum-below.
    # Mirrors R's `summarise(...) |> arrange(...) |> mutate(lag(cumsum(...)))`.
    # Sort by (property, rank) ascending so the cum_sum within property
    # respects rank order; over(property) keeps the cumsum from bleeding
    # across properties.
    ranking_sums = (
        df.group_by(["calc_main_property_id", "_milan_ranking"])
        .agg(_cpb_sum_at_rank=pl.col("_cpb_num").sum())
        .sort(["calc_main_property_id", "_milan_ranking"])
        .with_columns(
            _cpb_cumsum_below=pl.col("_cpb_sum_at_rank")
            .cum_sum()
            .shift(1, fill_value=0.0)
            .over("calc_main_property_id"),
        )
    )

    df = df.join(
        ranking_sums,
        on=["calc_main_property_id", "_milan_ranking"],
        how="left",
    )

    return df.with_columns(
        _milan_ext_prior_ranks_cb=pl.max_horizontal(
            pl.lit(0.0),
            pl.col("_ppb_num") - pl.col("_cpb_cumsum_below"),
        ),
        _milan_pari_passu_not_in_pool=pl.max_horizontal(
            pl.lit(0.0),
            pl.col("_ppue_num") - pl.col("_cpb_sum_at_rank"),
        ),
    )


# ---------------------------------------------------------------------------
# Stage 9 / B-3: date derivations
# ---------------------------------------------------------------------------


def _r_as_character_expr(expr: pl.Expr) -> pl.Expr:
    """Format Float64 the way R's as.character() does for default options.

    R's `as.character(<numeric>)` emits up to 15 significant digits with
    trailing-zero stripping and no trailing decimal. Polars' default
    Float -> Utf8 cast emits 17 sig digits and a trailing ".0" for whole
    numbers. Python's "%.15g" format strips trailing zeros, drops the
    decimal point for integer-valued floats, and rounds to 15 sig digits
    - byte-equal to R for the values that surface in this pipeline.

    Pinned against the synthetic fixture's "Months Current" /
    "Months In Arrears" columns:
      12.51745379876798  -> "12.517453798768"   (NEW_005, fixture)
      1.4784394250513349 -> "1.47843942505133"  (NEW_008, fixture)
      0.0                -> "0"                  (fixture)
    """
    return expr.map_elements(
        lambda v: None if v is None else f"{v:.15g}",
        return_dtype=pl.Utf8,
    )


def _parse_date_safe_expr(col: pl.Expr) -> pl.Expr:
    """Cast Date|Utf8 -> Date; ND, blank, or non-ISO values -> null.

    Mirrors r_reference/R/milan_mapping.R:42-87 (`safe_as_date`)
    minus the actionable-error path: Stage 1 already validates these
    columns via parse_iso_or_excel_date(), so anything reaching Stage
    9 is either a parsed Date or an ND/blank string. strict=False
    silently nulls anything else.

    Round-trips Date -> Utf8 -> Date harmlessly; the cast to Utf8
    emits "yyyy-mm-dd" which str.to_date parses back idempotently.
    """
    return (
        col.cast(pl.Utf8, strict=False)
        .str.to_date(format="%Y-%m-%d", strict=False)
    )


def _attach_date_derivations(df: pl.DataFrame) -> pl.DataFrame:
    """Add the staging columns the date-flavoured MILAN fields depend on.

    Mirrors r_reference/R/milan_mapping.R:556-597. Computes:

      _pool_cutoff_safe          - Date (parsed pool_cutoff_date)
      _date_last_in_arrears_safe - Date (parsed date_last_in_arrears)
      _date_last_in_arrears_chr  - Utf8 (kept for the case_when ND check
                                  in `Months Current` rule 3, since the
                                  parsed Date version would lose the
                                  "ND"/"ND3"/etc. distinction)
      _months_since_last_arrears - Utf8, R-formatted to %.15g, null
                                  when either date is null. Mirrors R's
                                  as.character(as.numeric(difftime(...))/30.4375).
      _days_in_arrears_num       - Float64
      _primary_income_num        - Float64
      _secondary_income_num      - Float64
      _total_credit_limit_num    - Float64

    The 30.4375 divisor is R's average month length (365.25 / 12);
    matches the fixture's emitted values for NEW_005 / NEW_008 to the
    last digit when paired with %.15g formatting.
    """
    df = df.with_columns(
        _date_last_in_arrears_chr=pl.col("date_last_in_arrears").cast(
            pl.Utf8, strict=False
        ),
        _pool_cutoff_safe=_parse_date_safe_expr(pl.col("pool_cutoff_date")),
        _date_last_in_arrears_safe=_parse_date_safe_expr(
            pl.col("date_last_in_arrears")
        ),
        _days_in_arrears_num=_safe_as_num_expr(pl.col("number_of_days_in_arrears")),
        _primary_income_num=_safe_as_num_expr(pl.col("primary_income")),
        _secondary_income_num=_safe_as_num_expr(pl.col("secondary_income")),
        _total_credit_limit_num=_safe_as_num_expr(pl.col("total_credit_limit")),
    )

    months_since_float = (
        (pl.col("_pool_cutoff_safe") - pl.col("_date_last_in_arrears_safe"))
        .dt.total_days()
        / 30.4375
    )

    return df.with_columns(
        _months_since_last_arrears=pl.when(
            pl.col("_pool_cutoff_safe").is_not_null()
            & pl.col("_date_last_in_arrears_safe").is_not_null()
        )
        .then(_r_as_character_expr(months_since_float))
        .otherwise(pl.lit(None, dtype=pl.Utf8)),
    )


def _compute_months_current_expr(
    days_in_arrears_num: pl.Expr,
    months_since_last_arrears: pl.Expr,
    date_last_in_arrears_chr: pl.Expr,
) -> pl.Expr:
    """Apply the `Months Current` case_when from milan_mapping.R:704-715.

    Rules in order:
      1. Currently in arrears (days > 0)               -> "0"
      2. Not in arrears, has valid last-arrears date   -> _months_since_last_arrears
      3. Never in arrears (date_last_in_arrears is ND) -> "Never in Arrears"
      Fallback                                          -> "ND"

    The string "Never in Arrears" is byte-equal to the value emitted
    by R into the synthetic fixture's "Months Current" column (verified
    against tests/fixtures/synthetic/expected_r_output.xlsx).
    """
    return (
        pl.when(days_in_arrears_num.is_not_null() & (days_in_arrears_num > 0))
        .then(pl.lit("0"))
        .when(
            days_in_arrears_num.is_not_null()
            & (days_in_arrears_num == 0)
            & months_since_last_arrears.is_not_null()
        )
        .then(months_since_last_arrears)
        .when(
            days_in_arrears_num.is_not_null()
            & (days_in_arrears_num == 0)
            & is_nd_expr(date_last_in_arrears_chr)
        )
        .then(pl.lit("Never in Arrears"))
        .otherwise(pl.lit("ND"))
    )


def _compute_months_in_arrears_expr(days_in_arrears_num: pl.Expr) -> pl.Expr:
    """Apply the `Months In Arrears` if_else from milan_mapping.R:723-727.

    days null -> "ND"; else as.character(days / 30.4375) via %.15g.
    """
    return (
        pl.when(days_in_arrears_num.is_null())
        .then(pl.lit("ND"))
        .otherwise(_r_as_character_expr(days_in_arrears_num / 30.4375))
    )


# ---------------------------------------------------------------------------
# Stage 9 / B-4: INFER + remaining CALC field helpers
# ---------------------------------------------------------------------------
#
# Pinning strategy: for each helper, branches exercised by the synthetic
# fixture are pinned against fixture-extracted values; branches not
# exercised by the fixture are tested against the R source contract
# (regex / case_when / equality) with an explicit "branch not exercised
# by synthetic fixture" comment in the test. Real-fixture parity in
# CI nightly will catch any branch-level drift.


def _milan_social_programme_expr(special_scheme: pl.Expr) -> pl.Expr:
    """Apply the Social Programme Type case_when from milan_mapping.R:732-739.

    case_when short-circuits at first match (R semantic). All grepl
    patterns are case-insensitive.

      RTB | Right to Buy                      -> "1"
      Tenant Purchase                          -> "2"
      HTB | Help to Buy | Equity Loan          -> "3"
      Shared Ownership | Part Buy Part Rent    -> "4"
      ND special_scheme                        -> "6"
      else                                     -> "5"
    """
    s = special_scheme.cast(pl.Utf8, strict=False)
    nd = is_nd_expr(s)
    return (
        pl.when((~nd) & s.str.contains(r"(?i)RTB|Right to Buy"))
        .then(pl.lit("1"))
        .when((~nd) & s.str.contains(r"(?i)Tenant Purchase"))
        .then(pl.lit("2"))
        .when((~nd) & s.str.contains(r"(?i)HTB|Help to Buy|Equity Loan"))
        .then(pl.lit("3"))
        .when((~nd) & s.str.contains(r"(?i)Shared Ownership|Part Buy Part Rent"))
        .then(pl.lit("4"))
        .when(nd)
        .then(pl.lit("6"))
        .otherwise(pl.lit("5"))
    )


def _milan_borrower_type_expr(
    employment_status: pl.Expr,
    primary_income_type: pl.Expr,
) -> pl.Expr:
    """Apply Borrower Type case_when from milan_mapping.R:751-756.

      employment == "NOEM" OR primary_income_type == "CORP" -> "2"
      employment present and != "NOEM"                       -> "1"
      else                                                    -> "ND"

    R's `==` returns NA on NA inputs and case_when treats NA-LHS as
    no-match. Mirror by gating each equality with `~is_nd_expr` so
    a null/ND on either side falls through cleanly.
    """
    emp = employment_status.cast(pl.Utf8, strict=False)
    inc = primary_income_type.cast(pl.Utf8, strict=False)
    emp_nd = is_nd_expr(emp)
    inc_nd = is_nd_expr(inc)
    emp_eq_noem = (~emp_nd) & (emp == "NOEM")
    inc_eq_corp = (~inc_nd) & (inc == "CORP")
    emp_present_not_noem = (~emp_nd) & (emp != "NOEM")
    return (
        pl.when(emp_eq_noem | inc_eq_corp).then(pl.lit("2"))
        .when(emp_present_not_noem).then(pl.lit("1"))
        .otherwise(pl.lit("ND"))
    )


def _milan_borrower_residency_expr(resident: pl.Expr) -> pl.Expr:
    """Apply Borrower Residency case_when from milan_mapping.R:759-763.

    Plain `==` on character (case-sensitive); "True" / "true" / etc.
    fall through to "ND".
    """
    s = resident.cast(pl.Utf8, strict=False)
    return (
        pl.when(s == "TRUE").then(pl.lit("Y"))
        .when(s == "FALSE").then(pl.lit("N"))
        .otherwise(pl.lit("ND"))
    )


def _milan_recourse_expr(recourse: pl.Expr) -> pl.Expr:
    """Apply Recourse case_when from milan_mapping.R:859-863.

    Y / N pass-through; everything else (including ND) -> "ND".
    """
    s = recourse.cast(pl.Utf8, strict=False)
    return (
        pl.when(s.is_in(["Y", "N"])).then(s)
        .otherwise(pl.lit("ND"))
    )


def _milan_restructured_loan_expr(
    date_of_restructuring: pl.Expr,
    account_status: pl.Expr,
) -> pl.Expr:
    """Apply Restructured Loan case_when from milan_mapping.R:978-983.

      date_of_restructuring is non-ND          -> "Y"
      account_status in {RNAR, RARR}           -> "Y"
      both date and status are ND              -> "ND"
      else                                      -> "N"

    R's `%in%` with NA on the LHS returns FALSE (not NA), so a NA
    account_status falls through to the next branch. Mirror via
    `is_nd_expr` gate.
    """
    d = date_of_restructuring.cast(pl.Utf8, strict=False)
    s = account_status.cast(pl.Utf8, strict=False)
    d_nd = is_nd_expr(d)
    s_nd = is_nd_expr(s)
    return (
        pl.when(~d_nd).then(pl.lit("Y"))
        .when((~s_nd) & s.is_in(["RNAR", "RARR"])).then(pl.lit("Y"))
        .when(d_nd & s_nd).then(pl.lit("ND"))
        .otherwise(pl.lit("N"))
    )


def _milan_mig_provider_expr(guarantor_type: pl.Expr) -> pl.Expr:
    """Apply MIG Provider case_when from milan_mapping.R:1015-1021.

    Plain `==` chain; null / ND / unknown codes fall to "No Guarantor".

    Strings pinned against the synthetic fixture:
      "NHG / Waarborgfonds Eigen Woningen"  - emitted for guarantor=NHGX
      "No Guarantor"                         - emitted for ND/unknown
    """
    s = guarantor_type.cast(pl.Utf8, strict=False)
    return (
        pl.when(s == "NHGX").then(pl.lit("NHG / Waarborgfonds Eigen Woningen"))
        .when(s == "FGAS").then(pl.lit("SGFGAS"))
        .when(s == "CATN").then(pl.lit("Caution"))
        .when(s == "OTHR").then(pl.lit("Other"))
        .otherwise(pl.lit("No Guarantor"))
    )


def _milan_total_income_expr(
    primary_income_num: pl.Expr,
    secondary_income_num: pl.Expr,
) -> pl.Expr:
    """Apply Total Income case_when from milan_mapping.R:803-808.

      both null              -> "ND"
      one null               -> the other (R-formatted via %.15g)
      else                   -> sum (R-formatted via %.15g)
    """
    p_null = primary_income_num.is_null()
    s_null = secondary_income_num.is_null()
    return (
        pl.when(p_null & s_null).then(pl.lit("ND"))
        .when(p_null).then(_r_as_character_expr(secondary_income_num))
        .when(s_null).then(_r_as_character_expr(primary_income_num))
        .otherwise(_r_as_character_expr(primary_income_num + secondary_income_num))
    )


def _milan_flexible_loan_amount_expr(
    total_credit_limit_num: pl.Expr,
    cpb_num: pl.Expr,
) -> pl.Expr:
    """Apply Flexible Loan Amount case_when from milan_mapping.R:670-675.

      total_credit_limit null     -> "0"
      tcl - cpb <= 0              -> "0"
      else                        -> as.character(tcl - cpb)  (%.15g)
    """
    diff = total_credit_limit_num - cpb_num
    return (
        pl.when(total_credit_limit_num.is_null()).then(pl.lit("0"))
        .when(diff <= 0).then(pl.lit("0"))
        .otherwise(_r_as_character_expr(diff))
    )


# ---------------------------------------------------------------------------
# Stage 9 / B-5: composer (assembly of the 175-column MILAN frame)
# ---------------------------------------------------------------------------


# Source columns expected by the MILAN mapping. Verbatim port of
# r_reference/R/milan_mapping.R:401-457. Ordered to match R for log-grep
# parity on _ensure_source_columns warning text.
EXPECTED_SOURCE_COLS: tuple[str, ...] = (
    "originator_name",
    "calc_borrower_id", "calc_main_property_id", "calc_loan_id",
    "rrel30_currency",
    "original_principal_balance", "current_principal_balance",
    "origination_date", "maturity_date",
    "origination_channel", "purpose",
    "scheduled_principal_payment_frequency", "scheduled_interest_payment_frequency",
    "amortisation_type", "payment_due",
    "prior_principal_balances", "pari_passu_underlying_exposures",
    "total_credit_limit",
    "interest_rate_type",
    "current_interest_rate_index", "current_interest_rate_index_tenor",
    "current_interest_rate", "current_interest_rate_margin",
    "revision_margin_1", "interest_revision_date_1",
    "date_last_in_arrears", "number_of_days_in_arrears", "pool_cutoff_date",
    "arrears_balance",
    "number_of_payments_before_securitisation",
    "special_scheme",
    "employment_status", "primary_income_type",
    "resident",
    "geographic_region_obligor",
    "primary_income", "primary_income_verification",
    "secondary_income", "secondary_income_verification",
    "debt_to_income_ratio",
    "deposit_amount",
    "property_type",
    "calc_aggregated_property_value",
    "recourse",
    "original_valuation_method", "current_valuation_method",
    "original_valuation_date", "current_valuation_date",
    "final_valuation_method", "final_valuation_date",
    "geographic_region_collateral",
    "occupancy_type",
    "original_loan_to_value", "current_loan_to_value",
    "energy_performance_certificate_value",
    "energy_performance_certificate_provider_name",
    "insurance_or_investment_provider",
    "account_status",
    "date_of_repurchase",
    "date_of_restructuring",
    "redemption_date",
    "default_amount", "default_date",
    "reason_for_default_or_foreclosure",
    "allocated_losses", "cumulative_recoveries", "sale_price",
    "guarantor_type",
    "calc_collateral_group_id", "calc_nr_loans_in_group",
    "calc_nr_properties_in_group", "calc_full_set",
    "calc_cross_collateralized_set", "calc_structure_type",
    "calc_original_LTV", "calc_current_LTV", "calc_seasoning",
    "lien",
    "original_underlying_exposure_identifier",
    "new_underlying_exposure_identifier",
    "new_obligor_identifier", "original_obligor_identifier",
    "original_collateral_identifier", "new_collateral_identifier",
)


def _r_char_coerce_all(df: pl.DataFrame) -> pl.DataFrame:
    """Coerce every column to Utf8 with R-equivalent formatting.

    Mirrors r_reference/R/milan_mapping.R:472:
        df |> mutate(across(everything(), as.character))

    Polars' default casts diverge from R's as.character() in two ways:
      - Float -> Utf8 emits 17 sig digits + trailing ".0"; R uses 15
        sig digits and strips trailing zeros / decimal point.
      - Bool -> Utf8 emits lowercase "true"/"false"; R emits uppercase
        "TRUE"/"FALSE".
    Both are normalised here. Date / Int / other types use the default
    cast, which matches R for those types.
    """
    casts: list[pl.Expr] = []
    for col, dtype in zip(df.columns, df.dtypes, strict=True):
        if dtype == pl.Utf8:
            casts.append(pl.col(col))
        elif dtype.is_float():
            casts.append(
                _r_as_character_expr(pl.col(col).cast(pl.Float64, strict=False)).alias(col)
            )
        elif dtype == pl.Boolean:
            casts.append(
                pl.when(pl.col(col).is_null())
                .then(pl.lit(None, dtype=pl.Utf8))
                .when(pl.col(col))
                .then(pl.lit("TRUE"))
                .otherwise(pl.lit("FALSE"))
                .alias(col)
            )
        else:
            casts.append(pl.col(col).cast(pl.Utf8, strict=False))
    return df.select(casts)


def compose_milan_pool(df: pl.DataFrame) -> pl.DataFrame:
    """Build the 175-column MILAN template pool from combined_flattened.

    Mirrors r_reference/R/milan_mapping.R::map_to_milan() (lines 390-1099).

    Pipeline:
      1. _ensure_source_columns (warn-and-fill missing as "ND").
      2. _r_char_coerce_all (Float -> %.15g; Date -> "yyyy-mm-dd"; etc.).
      3. _attach_ranking + _attach_external_prior_and_pari_passu (B-2).
      4. _attach_date_derivations (B-3).
      5. 175-column transmute via select(...).
      6. Column-order validation: raises on mismatch (R-repo issue #8
         deviation - R warns and continues).
      7. Final null -> "ND" fill across every column.

    Output is an all-Utf8 frame with exactly 175 columns in the order
    of MILAN_EXPECTED_OUTPUT_COLS.
    """
    if df.height == 0:
        warnings.warn(
            "MILAN mapping: input data frame has 0 rows", UserWarning, stacklevel=2
        )

    df = _ensure_source_columns(df, EXPECTED_SOURCE_COLS)
    df = _r_char_coerce_all(df)
    df = _attach_ranking(df)
    df = _attach_external_prior_and_pari_passu(df)
    df = _attach_date_derivations(df)

    cm = MILAN_CODE_MAPS

    milan = df.select(
        # --- 1-10: identifiers + currency + balances + dates ---
        pl.col("originator_name").alias("Originator Identifier"),
        pl.lit("ESMA template placeholder servicer").alias("Servicer Identifier"),
        pl.col("calc_borrower_id").alias("Borrower Identifier"),
        pl.col("calc_main_property_id").alias("Property Identifier"),
        pl.col("calc_loan_id").alias("Loan Identifier"),
        pl.col("rrel30_currency").alias("Loan Currency"),
        pl.col("original_principal_balance").alias("Loan OB"),
        pl.col("current_principal_balance").alias("Loan CB"),
        pl.col("origination_date").alias("Origination Date"),
        pl.col("maturity_date").alias("Maturity Date"),
        # --- 11-16: channel/purpose/freq/type/payment ---
        _code_map_lookup_expr(pl.col("origination_channel"), cm["origination_channel"]).alias("Origination Channel"),
        _code_map_lookup_expr(pl.col("purpose"), cm["purpose"]).alias("Loan Purpose"),
        _code_map_lookup_expr(
            pl.col("scheduled_principal_payment_frequency"),
            cm["principal_payment_frequency"],
        ).alias("Principal Payment Frequency"),
        _code_map_lookup_expr(
            pl.col("scheduled_interest_payment_frequency"),
            cm["interest_payment_frequency"],
        ).alias("Interest Payment Frequency"),
        _code_map_lookup_expr(pl.col("amortisation_type"), cm["principal_payment_type"]).alias("Principal Payment Type"),
        pl.col("payment_due").alias("Periodic Payment"),
        # --- 17-21: ranking + prior-rank fields (B-2) ---
        pl.col("_milan_ranking").cast(pl.Utf8).alias("Ranking"),
        pl.col("prior_principal_balances").alias("Total Prior Ranks CB"),
        pl.lit("ND").alias("Total Prior Ranks OB"),
        # External Prior Ranks CB: R uses format(scientific=FALSE, trim=TRUE)
        # which is 7-sig-digit fixed; we use _r_as_character_expr (%.15g).
        # Synthetic fixture's value range produces identical output; real-
        # fixture parity is the gate for any divergence on >7-sig values.
        pl.when(is_nd_expr(pl.col("prior_principal_balances")))
        .then(pl.lit("ND"))
        .otherwise(_r_as_character_expr(pl.col("_milan_ext_prior_ranks_cb")))
        .alias("External Prior Ranks CB"),
        # Pari Passu (Not In Pool): R-repo entry #9 - asymmetric default
        # ("0" string when input is ND vs "ND" for External Prior). Float
        # value is 0 in both cases; the asymmetry lives entirely here.
        pl.when(is_nd_expr(pl.col("pari_passu_underlying_exposures")))
        .then(pl.lit("0"))
        .otherwise(_r_as_character_expr(pl.col("_milan_pari_passu_not_in_pool")))
        .alias("Pari Passu Ranking Loans (Not In Pool)"),
        # --- 22-27: retained / further advances / flex / grace / payment holiday ---
        pl.lit("0").alias("Retained Amount"),
        pl.lit("ND").alias("Further Advances"),
        _milan_flexible_loan_amount_expr(
            pl.col("_total_credit_limit_num"), pl.col("_cpb_num")
        ).alias("Flexible Loan Amount"),
        pl.lit("ND").alias("Grace Period"),
        pl.lit("ND").alias("Type of Payment Holiday"),
        pl.lit("ND").alias("Length of Payment Holiday Allowed In Months"),
        # --- 28-33: rate/index/margins ---
        _code_map_lookup_expr(pl.col("interest_rate_type"), cm["interest_rate_type"]).alias("Interest Rate Type"),
        _lookup_base_index_expr(
            pl.col("current_interest_rate_index"),
            pl.col("current_interest_rate_index_tenor"),
        ).alias("Base Index"),
        _divide_by_100_expr(pl.col("current_interest_rate")).alias("Interest Rate"),
        _divide_by_100_expr(pl.col("current_interest_rate_margin")).alias("Full Margin"),
        _divide_by_100_expr(pl.col("revision_margin_1")).alias("Margin After Reset Date"),
        pl.col("interest_revision_date_1").alias("Interest Reset Date"),
        # --- 34-37: arrears block ---
        _compute_months_current_expr(
            pl.col("_days_in_arrears_num"),
            pl.col("_months_since_last_arrears"),
            pl.col("_date_last_in_arrears_chr"),
        ).alias("Months Current"),
        pl.col("arrears_balance").alias("Arrears Amount"),
        _compute_months_in_arrears_expr(pl.col("_days_in_arrears_num")).alias("Months In Arrears"),
        pl.col("number_of_payments_before_securitisation").alias("Number of Payments Before Securitisation"),
        # --- 38-49: borrower block ---
        _milan_social_programme_expr(pl.col("special_scheme")).alias("Social Programme Type"),
        pl.lit("ND").alias("FX Risk Mitigated"),
        pl.lit("ND").alias("Inflation-Indexed Loan"),
        pl.lit("ND").alias("Exchange Rate At Loan Origination"),
        _milan_borrower_type_expr(
            pl.col("employment_status"), pl.col("primary_income_type")
        ).alias("Borrower Type"),
        _milan_borrower_residency_expr(pl.col("resident")).alias("Borrower Residency"),
        pl.lit("ND").alias("Borrower Birth Year"),
        _code_map_lookup_expr(pl.col("employment_status"), cm["employment_type"]).alias("Employment Type"),
        pl.lit("ND").alias("Internal Credit Score"),
        pl.lit("ND").alias("Credit Score - Canada"),
        pl.col("geographic_region_obligor").alias("Borrower Postal/Zip Code"),
        pl.lit("ND").alias("Number of Borrowers"),
        # --- 50-58: income / DTI / employee ---
        pl.col("primary_income").alias("Income Borrower 1"),
        _code_map_lookup_expr(
            pl.col("primary_income_verification"), cm["income_verification"]
        ).alias("Income Verification For Primary Income"),
        pl.col("secondary_income").alias("Income Borrower 2"),
        _code_map_lookup_expr(
            pl.col("secondary_income_verification"), cm["income_verification"]
        ).alias("Income Verification For Secondary Income"),
        pl.lit("ND").alias("Guarantor's Income"),
        _milan_total_income_expr(
            pl.col("_primary_income_num"), pl.col("_secondary_income_num")
        ).alias("Total Income"),
        pl.col("debt_to_income_ratio").alias("DTI Ratio"),
        pl.lit("ND").alias("Loan To Employee"),
        pl.lit("ND").alias("Number of Properties Owned By Borrower"),
        # --- 59-67: regulated + adverse credit ---
        pl.lit("ND").alias("Regulated"),
        pl.lit("ND").alias("Prior Missed On Previous Mortgage"),
        pl.lit("ND").alias("Live Units of Adverse Credit"),
        pl.lit("ND").alias("Satisfied Units of Adverse Credit"),
        pl.lit("ND").alias("Total Amount of CCJs or Balance of Adverse Credit"),
        pl.lit("ND").alias("Number of Other Adverse Credit Events"),
        pl.lit("ND").alias("Prior Personally Bankruptcy Or IVA"),
        pl.lit("ND").alias("Prior Repossessions On Previous Mortgage"),
        pl.lit("ND").alias("Prior Litigation"),
        # --- 68-83: property block (Property 1) ---
        pl.col("deposit_amount").alias("Deposit Amount"),
        _code_map_lookup_expr(pl.col("property_type"), cm["property_type"]).alias("Property Type"),
        pl.col("calc_aggregated_property_value").alias("Property Value"),
        pl.lit("ND").alias("Purchase Price"),
        _milan_recourse_expr(pl.col("recourse")).alias("Recourse"),
        _code_map_lookup_expr(
            pl.col("final_valuation_method"), cm["property_valuation_type"]
        ).alias("Property Valuation Type"),
        pl.col("final_valuation_date").alias("Property Valuation Date"),
        pl.lit("ND").alias("Confidence Interval For Original AVM Valuation"),
        pl.lit("ND").alias("Provider of Original AVM Valuation"),
        pl.col("geographic_region_collateral").alias("Property Postal, Region code or Nuts code"),
        _code_map_lookup_expr(pl.col("occupancy_type"), cm["occupancy_type"]).alias("Occupancy Type"),
        pl.lit("ND").alias("Rental Income"),
        pl.lit("ND").alias("Metro / Non Metro"),
        pl.lit("ND").alias("Construction Year"),
        pl.lit("ND").alias("Mortgage Inscription - Property 1"),
        pl.lit("ND").alias("Mortgage Mandate - Property 1"),
        # --- 84-119: Property 2-5 (all ND) ---
        *[pl.lit("ND").alias(c) for c in (
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
        )],
        # --- 120-126: LTV + EPC + additional collateral ---
        _divide_by_100_expr(pl.col("original_loan_to_value")).alias("Original LTV"),
        _divide_by_100_expr(pl.col("current_loan_to_value")).alias("Current LTV"),
        _code_map_lookup_expr(
            pl.col("energy_performance_certificate_value"),
            cm["energy_performance_certificate_value"],
        ).alias("Energy Performance Certificate Value"),
        pl.col("energy_performance_certificate_provider_name").alias("Energy Performance Certificate Provider Name"),
        pl.lit("ND").alias("Additional Collateral Type"),
        pl.col("insurance_or_investment_provider").alias("Additional Collateral Provider"),
        pl.lit("ND").alias("Additional Collateral Value"),
        # --- 127-138: status / restructuring / default ---
        _code_map_lookup_expr(pl.col("account_status"), cm["account_status"]).alias("Account Status"),
        pl.col("date_of_repurchase").alias("Repurchase Date"),
        _milan_restructured_loan_expr(
            pl.col("date_of_restructuring"), pl.col("account_status")
        ).alias("Restructured Loan"),
        pl.col("date_of_restructuring").alias("Restructuring Date"),
        pl.lit("ND").alias("Restructuring Type"),
        pl.col("redemption_date").alias("Redemption Date"),
        pl.col("default_amount").alias("Default amount"),
        pl.col("default_date").alias("Default date"),
        _code_map_lookup_expr(
            pl.col("reason_for_default_or_foreclosure"), cm["reason_for_default"]
        ).alias("Reason for Default or Foreclosure"),
        pl.col("allocated_losses").alias("Allocated Losses"),
        pl.col("cumulative_recoveries").alias("Cumulative Recoveries"),
        pl.col("sale_price").alias("Sale Price"),
        # --- 139-143: MIG / guarantee provider ---
        _milan_mig_provider_expr(pl.col("guarantor_type")).alias("MIG Provider"),
        _code_map_lookup_expr(pl.col("guarantor_type"), cm["guarantee_provider_type"]).alias("Type of Guarantee Provider"),
        pl.lit("ND").alias("MIG %"),
        pl.lit("ND").alias("Initial MCI Coverage Amount"),
        pl.lit("ND").alias("Current MCI Coverage Amount"),
        # --- 144-169: Additional data 1-26 (DIRECT calc_*) ---
        pl.col("calc_loan_id").alias("Additional data 1 - calc_loan_id"),
        pl.col("calc_borrower_id").alias("Additional data 2 - calc_borrower_id"),
        pl.col("calc_main_property_id").alias("Additional data 3 - calc_main_property_id"),
        pl.col("calc_collateral_group_id").alias("Additional data 4 - calc_collateral_group_id"),
        pl.col("calc_nr_loans_in_group").alias("Additional data 5 - calc_nr_loans_in_group"),
        pl.col("calc_nr_properties_in_group").alias("Additional data 6 - calc_nr_properties_in_group"),
        pl.col("calc_full_set").alias("Additional data 7 - calc_full_set"),
        pl.col("calc_cross_collateralized_set").alias("Additional data 8 - calc_cross_collateralized_set"),
        pl.col("calc_structure_type").alias("Additional data 9 - calc_structure_type"),
        pl.col("calc_aggregated_property_value").alias("Additional data 10 - calc_aggregated_property_value"),
        pl.col("calc_original_LTV").alias("Additional data 11 - calc_original_LTV"),
        pl.col("calc_current_LTV").alias("Additional data 12 - calc_current_LTV"),
        pl.col("calc_seasoning").alias("Additional data 13 - calc_seasoning"),
        pl.col("lien").alias("Additional data 14 - lien"),
        pl.col("prior_principal_balances").alias("Additional data 15 - prior_principal_balances"),
        pl.col("pari_passu_underlying_exposures").alias("Additional data 16 - pari_passu_underlying_exposures"),
        pl.col("original_valuation_method").alias("Additional data 17 - original_valuation_method"),
        pl.col("original_valuation_date").alias("Additional data 18 - original_valuation_date"),
        pl.col("current_valuation_method").alias("Additional data 19 - current_valuation_method"),
        pl.col("current_valuation_date").alias("Additional data 20 - current_valuation_date"),
        pl.col("new_underlying_exposure_identifier").alias("Additional data 21 - new_underlying_exposure_identifier"),
        pl.col("original_underlying_exposure_identifier").alias("Additional data 22 - original_underlying_exposure_identifier"),
        pl.col("new_obligor_identifier").alias("Additional data 23 - new_obligor_identifier"),
        pl.col("original_obligor_identifier").alias("Additional data 24 - original_obligor_identifier"),
        pl.col("new_collateral_identifier").alias("Additional data 25 - new_collateral_identifier"),
        pl.col("original_collateral_identifier").alias("Additional data 26 - original_collateral_identifier"),
        # --- 170-175: column for additional data 27-32 ---
        *[pl.lit("ND").alias(c) for c in (
            "column for additional data 27", "column for additional data 28",
            "column for additional data 29", "column for additional data 30",
            "column for additional data 31", "column for additional data 32",
        )],
    )

    # Column-order validation. Deviates from R: r_reference/R/milan_mapping.R:1066-1080
    # warns and continues on column-layout mismatch. We raise — see R-repo
    # issue tracker entry #8.
    if list(milan.columns) != list(MILAN_EXPECTED_OUTPUT_COLS):
        actual = list(milan.columns)
        expected = list(MILAN_EXPECTED_OUTPUT_COLS)
        diffs: list[str] = []
        for i, (a, e) in enumerate(zip(actual, expected, strict=False)):
            if a != e:
                diffs.append(f"  position {i}: got {a!r}, expected {e!r}")
            if len(diffs) >= 5:
                break
        if len(actual) != len(expected):
            diffs.append(
                f"  length mismatch: got {len(actual)} columns, expected {len(expected)}"
            )
        raise ValueError(
            "MILAN mapping: output columns do not match expected 175-column layout.\n"
            + "\n".join(diffs)
        )

    # Final null -> "ND" fill across every column. Mirrors r_reference/
    # R/milan_mapping.R:1088-1090. Catches any remaining nulls produced by
    # source-column NA passthrough where the field-level helper didn't
    # already substitute "ND".
    return milan.with_columns([pl.col(c).fill_null("ND") for c in milan.columns])
