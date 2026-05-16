"""ESMA-code -> display-label maps for the analysis stratifications.

Kept separate from ``pipeline/milan_map.py`` (which carries the
parity-sensitive ESMA -> MILAN numeric-code maps) so display concerns
never tangle with parity. The MILAN map there must stay byte-equal to
``r_reference/R/milan_mapping.R``; this file is a GUI-only concern and
can evolve freely.

Each map collapses the wide ESMA vocabulary into the small bucket set
called out by the Phase 1 stratifications brief. Codes not in the map
fall through to the last bucket ("Other"). Nulls in the source column
are routed to "Other" by the stratification helpers, not by the maps.
"""

from __future__ import annotations

# Interest rate type: 4 buckets per the brief (fixed / floating /
# hybrid / other). Source codes are ESMA `interest_rate_type` values;
# Fixed-then-floating variants are "hybrid"; capped/floored variants
# are still fundamentally floating.
IR_TYPE_LABELS: dict[str, str] = {
    "FLCF": "Fixed",
    "FLIF": "Floating",
    "FINX": "Floating",
    "FLFL": "Floating",
    "CAPP": "Floating",
    "FLCA": "Floating",
    "DISC": "Floating",
    "FXRL": "Hybrid",
    "FXPR": "Hybrid",
    "SWIC": "Other",
    "OBLS": "Other",
    "MODE": "Other",
    "OTHR": "Other",
}
IR_TYPE_ORDER: tuple[str, ...] = ("Fixed", "Floating", "Hybrid", "Other")

# Loan purpose: 5 buckets per the brief (purchase / refinance /
# equity release / construction / other). RMRT and RMEQ both
# carry a remortgage component and bucket as refinance; everything
# unfamiliar drops into Other.
LOAN_PURPOSE_LABELS: dict[str, str] = {
    "PURC": "Purchase",
    "RMRT": "Refinance",
    "RMEQ": "Refinance",
    "EQRE": "Equity Release",
    "CNST": "Construction",
    "RENV": "Other",
    "DCON": "Other",
    "BSFN": "Other",
    "CMRT": "Other",
    "IMRT": "Other",
    "RGBY": "Other",
    "GSPL": "Other",
    "OTHR": "Other",
}
LOAN_PURPOSE_ORDER: tuple[str, ...] = (
    "Purchase", "Refinance", "Equity Release", "Construction", "Other",
)

# Occupancy: 4 buckets per the brief (owner-occupied / second home /
# investment / other). Mirrors the MILAN grouping in
# ``pipeline/milan_map.py``: POWN (partially owner-occupied) and
# TLET (to-let) both bucket as investment.
OCCUPANCY_LABELS: dict[str, str] = {
    "FOWN": "Owner-Occupied",
    "HOLD": "Second Home",
    "TLET": "Investment",
    "POWN": "Investment",
    "OTHR": "Other",
}
OCCUPANCY_ORDER: tuple[str, ...] = (
    "Owner-Occupied", "Second Home", "Investment", "Other",
)
