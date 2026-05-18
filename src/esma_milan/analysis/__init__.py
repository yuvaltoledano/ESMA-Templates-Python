"""Pool stratifications for the GUI analysis view.

Registry pattern: every stratification is one entry in `STRATIFICATIONS`.
Adding a seventh is one new function in `stratifications.py` plus one
line here - no other module changes. `run_all_stratifications` walks
the registry and assembles the dict the API returns.

Each function is responsible for catching its own data errors and
returning a `Stratification` with `error` populated; the registry
walker does not try/except so a bug that escapes the function shape
still surfaces cleanly via the API's normal 500 handler rather than
being silently swallowed here.
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from esma_milan.analysis.stratifications import (
    stratify_amortisation_type,
    stratify_current_interest_rate,
    stratify_current_ltv,
    stratify_employment_type,
    stratify_geographic,
    stratify_interest_rate_type,
    stratify_loan_purpose,
    stratify_occupancy,
    stratify_property_type,
    stratify_property_valuation_type,
    stratify_seasoning,
)
from esma_milan.analysis.types import (
    AnalysisResult,
    AnalysisSummary,
    ExecutionSummaryRow,
    Stratification,
    StratificationRow,
    StratificationTotal,
)

# Order matters: this is the order the frontend renders tiles in. The
# Phase-2 layout groups by analytical kinship: loan-structure rate
# fields first (IR type / amortisation / current rate), then bucketed
# numeric loan metrics (seasoning / current LTV), then property-side
# (geography / property type / valuation type), then loan-purpose /
# occupancy / borrower-side employment last.
STRATIFICATIONS: dict[str, Callable[[pl.DataFrame], Stratification]] = {
    "interest_rate_type": stratify_interest_rate_type,
    "amortisation_type": stratify_amortisation_type,
    "current_interest_rate": stratify_current_interest_rate,
    "seasoning": stratify_seasoning,
    "current_ltv": stratify_current_ltv,
    "geographic": stratify_geographic,
    "property_type": stratify_property_type,
    "property_valuation_type": stratify_property_valuation_type,
    "loan_purpose": stratify_loan_purpose,
    "occupancy": stratify_occupancy,
    "employment_type": stratify_employment_type,
}


def run_all_stratifications(df: pl.DataFrame) -> dict[str, Stratification]:
    """Walk the registry against `df`, returning one entry per key.

    `df` is expected to be the Stage-7 `combined_flattened` frame: one
    row per loan-part, with main-property attributes joined on.
    """
    return {key: fn(df) for key, fn in STRATIFICATIONS.items()}


__all__ = [
    "STRATIFICATIONS",
    "AnalysisResult",
    "AnalysisSummary",
    "ExecutionSummaryRow",
    "Stratification",
    "StratificationRow",
    "StratificationTotal",
    "run_all_stratifications",
]
