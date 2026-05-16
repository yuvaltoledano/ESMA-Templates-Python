"""ESMA value-code -> display-label maps for the analysis stratifications.

**Principle**: ESMA-coded fields are displayed faithfully. Each row
shows the literal ESMA code alongside the canonical description from
the ESMA taxonomy. No derived categories, no collapsing into broader
buckets - a structured-finance analyst needs to see exactly what the
data says, code by code.

When ESMA standards add a new value code, add it here verbatim from
the taxonomy. Don't invent groupings.

**Source**: ``r_reference/inputs/ESMA template taxonomy.xlsx`` -
"CONTENT TO REPORT" column - for each indicated field (RREL42 for
interest_rate_type, RREL27 for purpose, RREC7 for occupancy_type).
Descriptions are taxonomy-verbatim with two pragmatic trims agreed
with the user:

  1. The qualifier "underlying exposure" - which appears in every
     interest-rate-type description - is dropped, since the field
     heading already conveys the same scope.
  2. The trailing "i.e. ..." clauses on FOWN / POWN are dropped, since
     the code + heading already disambiguate the cases.

Both trims keep the wording authoritative while preserving table
legibility.

Kept separate from ``pipeline/milan_map.py`` (which carries the
parity-sensitive ESMA -> MILAN numeric-code maps) so display concerns
never tangle with parity. The MILAN map there must stay byte-equal to
``r_reference/R/milan_mapping.R``; this file is a GUI-only concern and
can evolve freely.
"""

from __future__ import annotations

# Sentinel row label for data values that fall outside the ESMA
# taxonomy: either null/missing values, or codes the source data
# carries that are not part of the published value list (shouldn't
# happen for valid ESMA submissions, but defensive). Surfaced as a
# row only when count > 0, so for clean data this stays invisible -
# it acts as a data-quality signal rather than a permanent category.
UNK_LABEL: str = "UNK — Unknown / not in ESMA taxonomy"


# Interest Rate Type (ESMA field RREL42). Insertion order = display
# order. Source: ESMA template taxonomy, RREL42 "Content to report".
# 13 codes, taxonomy-verbatim minus the "underlying exposure" qualifier.
IR_TYPE_LABELS: dict[str, str] = {
    "FLIF": "FLIF — Floating rate (for life)",
    "FINX": "FINX — Floating rate linked to one index that will revert to another",
    "FXRL": "FXRL — Fixed rate (for life)",
    "FXPR": "FXPR — Fixed with future periodic resets",
    "FLCF": "FLCF — Fixed rate with compulsory future switch to floating",
    "FLFL": "FLFL — Floating rate with floor",
    "CAPP": "CAPP — Floating rate with cap",
    "FLCA": "FLCA — Floating rate with both floor and cap",
    "DISC": "DISC — Discount",
    "SWIC": "SWIC — Switch Optionality",
    "OBLS": "OBLS — Obligor Swapped",
    "MODE": "MODE — Modular",
    "OTHR": "OTHR — Other",
}


# Loan Purpose (ESMA field RREL27). Source: ESMA template taxonomy,
# RREL27 "Content to report". All 13 codes verbatim (no qualifier
# trims needed - descriptions are already concise).
LOAN_PURPOSE_LABELS: dict[str, str] = {
    "PURC": "PURC — Purchase",
    "RMRT": "RMRT — Remortgage",
    "RENV": "RENV — Renovation",
    "EQRE": "EQRE — Equity Release",
    "CNST": "CNST — Construction",
    "DCON": "DCON — Debt Consolidation",
    "RMEQ": "RMEQ — Remortgage with Equity Release",
    "BSFN": "BSFN — Business Funding",
    "CMRT": "CMRT — Combination Mortgage",
    "IMRT": "IMRT — Investment Mortgage",
    "RGBY": "RGBY — Right to Buy",
    "GSPL": "GSPL — Government Sponsored Loan",
    "OTHR": "OTHR — Other",
}


# Occupancy Type (ESMA field RREC7). Source: ESMA template taxonomy,
# RREC7 "Content to report". Taxonomy-verbatim minus the FOWN/POWN
# "i.e. ..." clauses, which clarify rather than define.
OCCUPANCY_LABELS: dict[str, str] = {
    "FOWN": "FOWN — Owner Occupied",
    "POWN": "POWN — Partially Owner Occupied",
    "TLET": "TLET — Non-Owner Occupied or Buy-To-Let",
    "HOLD": "HOLD — Holiday or Second Home",
    "OTHR": "OTHR — Other",
}
