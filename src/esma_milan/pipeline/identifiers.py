"""Identifier selection helpers (Stage 3).

Two helpers, mirroring r_reference/R/utils.R:

  select_calc_loan_id()    utils.R:266-442
  generate_id_column()     utils.R:173-206

`select_calc_loan_id` is the highest-parity-risk function in the pipeline
because every downstream join keys on its output. The R function has 20
dedicated test cases pinning the three-gate decision logic; this Python
port mirrors them one-for-one (see tests/unit/test_identifiers.py).

`generate_id_column` is the simpler helper used for borrower and property
IDs, where uniqueness/coverage gates do not apply - it just picks the
column with more unique non-null values, defaulting to original on tie.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import polars as pl
import structlog

from esma_milan.config import DEFAULT_MIN_LOAN_ID_COVERAGE

log = structlog.get_logger(__name__)


@dataclass
class _Candidate:
    """Internal tracking struct for one candidate ID column.

    Mirrors the per-candidate state R's select_calc_loan_id builds in its
    `candidates` list. Public API still returns just a Polars DataFrame.
    """

    name: str
    values: list[str | None]
    has_duplicates: bool = False
    all_na: bool = False
    disqualified: bool = False
    total: int = 0
    matched: int = 0
    coverage: float = 0.0
    unmatched_sample: str = ""


def _sample_values(values: list[str], n: int = 10) -> str:
    """Mirrors R's `paste(utils::head(x, n), collapse = ', ')`."""
    return ", ".join(str(v) for v in values[:n])


def _duplicate_diagnostics(values: list[str | None]) -> tuple[int, str]:
    """Return (n_duplicated, sample_string) for the non-null portion."""
    non_null = [v for v in values if v is not None]
    seen: set[str] = set()
    dups: list[str] = []
    for v in non_null:
        if v in seen and v not in dups:
            dups.append(v)
        seen.add(v)
    return len(dups), _sample_values(dups, 10)


def select_calc_loan_id(
    loans: pl.DataFrame,
    properties: pl.DataFrame,
    *,
    original_id_col: str = "original_underlying_exposure_identifier",
    new_id_col: str = "new_underlying_exposure_identifier",
    output_col_name: str = "calc_loan_id",
    property_id_col: str = "underlying_exposure_identifier",
    min_coverage: float = DEFAULT_MIN_LOAN_ID_COVERAGE,
) -> pl.DataFrame:
    """Pick the loan-ID column to use as `calc_loan_id`.

    Three-gate decision (utils.R:213-226):

      Gate 1 - Duplicate elimination + all-NA disqualification.
        A candidate with any duplicated non-null value or that is all-NA
        is disqualified. If both candidates are disqualified the function
        raises with a sample of duplicated values from each.

      Gate 2 - Coverage measurement.
        For each surviving candidate, coverage =
          n_distinct(intersect(candidate, property_ids)) /
          n_distinct(non_na_candidate)

      Gate 3 - Selection.
        - Only one survivor: it wins.
        - Both full coverage (1.0): default to original (stable tie-break).
        - One full, one partial: full wins.
        - Both partial, different coverage: higher wins.
        - Both partial, equal coverage: default to original AND emit a
          warning with unmatched samples from both candidates.

    After selection, coverage strictly below `min_coverage` raises;
    coverage in [min_coverage, 1.0) emits a warning. The chosen column
    is appended to `loans` as `output_col_name` and the modified
    DataFrame returned.

    Raises:
        ValueError: missing input column / out-of-range min_coverage /
                    both candidates disqualified / coverage below
                    min_coverage.
    """
    # -- Parameter validation ------------------------------------------------
    if original_id_col not in loans.columns:
        raise ValueError(
            f"select_calc_loan_id: loans is missing column {original_id_col!r}"
        )
    if new_id_col not in loans.columns:
        raise ValueError(
            f"select_calc_loan_id: loans is missing column {new_id_col!r}"
        )
    if property_id_col not in properties.columns:
        raise ValueError(
            f"select_calc_loan_id: properties is missing column {property_id_col!r}"
        )
    if (
        not isinstance(min_coverage, (int, float))
        or isinstance(min_coverage, bool)
        or not (0.0 <= float(min_coverage) <= 1.0)
    ):
        raise ValueError(
            "select_calc_loan_id: 'min_coverage' must be a numeric scalar in [0, 1]"
        )

    original_values: list[str | None] = loans[original_id_col].to_list()
    new_values: list[str | None] = loans[new_id_col].to_list()
    property_ids: set[str] = {
        v for v in properties[property_id_col].to_list() if v is not None
    }

    candidates = [
        _Candidate(name=original_id_col, values=original_values),
        _Candidate(name=new_id_col, values=new_values),
    ]

    # -- Gate 1: duplicate / all-NA disqualification -----------------------
    for c in candidates:
        non_null = [v for v in c.values if v is not None]
        c.all_na = len(non_null) == 0
        c.has_duplicates = len(non_null) != len(set(non_null))
        c.disqualified = c.has_duplicates or c.all_na

    if all(c.disqualified for c in candidates):
        orig, new = candidates
        orig_n_dups, orig_sample = _duplicate_diagnostics(orig.values)
        new_n_dups, new_sample = _duplicate_diagnostics(new.values)
        orig_msg = (
            "all NA"
            if orig.all_na
            else f"{orig_n_dups} duplicated values [{orig_sample}]"
        )
        new_msg = (
            "all NA"
            if new.all_na
            else f"{new_n_dups} duplicated values [{new_sample}]"
        )
        raise ValueError(
            f"select_calc_loan_id: neither candidate loan-ID column is usable. "
            f"{orig.name!r}: {orig_msg}. "
            f"{new.name!r}: {new_msg}. "
            f"This is a data-quality problem that requires manual investigation."
        )

    # -- Gate 2: coverage measurement -------------------------------------
    for c in candidates:
        if c.disqualified:
            continue
        non_null = [v for v in c.values if v is not None]
        unique_non_null = set(non_null)
        c.total = len(unique_non_null)
        c.matched = len(unique_non_null & property_ids)
        c.coverage = c.matched / c.total if c.total > 0 else 0.0
        unmatched = sorted(unique_non_null - property_ids, key=str)
        c.unmatched_sample = _sample_values(unmatched, 10)

    # -- Gate 3: selection -------------------------------------------------
    surviving_idx = [i for i, c in enumerate(candidates) if not c.disqualified]

    chosen_idx: int
    reason: str
    if len(surviving_idx) == 1:
        chosen_idx = surviving_idx[0]
        reason = "only duplicate-free candidate"
    else:
        cov1 = candidates[0].coverage
        cov2 = candidates[1].coverage
        full1 = cov1 == 1.0
        full2 = cov2 == 1.0

        if full1 and full2:
            chosen_idx, reason = 0, "both fully covered; default to original"
        elif full1:
            chosen_idx, reason = 0, "full coverage"
        elif full2:
            chosen_idx, reason = 1, "full coverage"
        elif cov1 > cov2:
            chosen_idx, reason = 0, "higher coverage than alternative"
        elif cov2 > cov1:
            chosen_idx, reason = 1, "higher coverage than alternative"
        else:
            # Equal partial coverage. Default to original (mirroring the
            # equal-full-coverage tie-break) and warn with unmatched
            # samples from both. Matches utils.R:391-397.
            chosen_idx = 0
            reason = "equal partial coverage; defaulted to original"
            warnings.warn(
                f"select_calc_loan_id: both candidate columns have equal "
                f"partial coverage ({_pct(cov1)}) against {property_id_col!r}. "
                f"Defaulted to {candidates[0].name!r}. "
                f"{candidates[0].name!r} unmatched sample: "
                f"[{candidates[0].unmatched_sample}]. "
                f"{candidates[1].name!r} unmatched sample: "
                f"[{candidates[1].unmatched_sample}].",
                UserWarning,
                stacklevel=2,
            )

    chosen = candidates[chosen_idx]

    # -- min_coverage threshold check -------------------------------------
    if chosen.coverage < min_coverage:
        raise ValueError(
            f"select_calc_loan_id: chosen column {chosen.name!r} has coverage "
            f"{chosen.matched}/{chosen.total} = {_pct(chosen.coverage)}, "
            f"below min_coverage = {_pct(min_coverage)}. "
            f"Unmatched sample: [{chosen.unmatched_sample}]."
        )

    log.info(
        "select_calc_loan_id",
        picked=chosen.name,
        reason=reason,
        original_duplicates=candidates[0].has_duplicates,
        original_coverage=f"{candidates[0].matched}/{candidates[0].total}",
        new_duplicates=candidates[1].has_duplicates,
        new_coverage=f"{candidates[1].matched}/{candidates[1].total}",
    )

    if chosen.coverage < 1.0:
        warnings.warn(
            f"select_calc_loan_id: chosen column {chosen.name!r} has partial "
            f"coverage {chosen.matched}/{chosen.total} = "
            f"{_pct(chosen.coverage)}. "
            f"Unmatched sample: [{chosen.unmatched_sample}].",
            UserWarning,
            stacklevel=2,
        )

    return loans.with_columns(
        pl.Series(output_col_name, chosen.values, dtype=pl.String)
    )


def _pct(x: float) -> str:
    """Format a fraction as percentage with 0.01 accuracy. Matches R's
    `scales::percent(x, accuracy = 0.01)` output: '99.00%', '85.71%'."""
    return f"{x * 100:.2f}%"
