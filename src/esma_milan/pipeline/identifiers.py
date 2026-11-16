"""Identifier selection helpers (Stage 3).

Two helpers + a stage driver, mirroring r_reference/R/pipeline.R:264-331
which composes them:

  select_calc_loan_id()    utils.R:266-442
  generate_id_column()     utils.R:173-206
  run_stage3()             pipeline.R:264-331 (the orchestration block)

`select_calc_loan_id` is the highest-parity-risk function in the pipeline
because every downstream join keys on its output. The R function has 20
dedicated test cases pinning the three-gate decision logic; this Python
port mirrors them one-for-one (see tests/unit/test_identifiers.py).

`generate_id_column` is the simpler helper used for borrower and property
IDs, where uniqueness/coverage gates do not apply - it just picks the
column with more unique non-null values, defaulting to original on tie
or when the two columns are element-wise identical.

`run_stage3` is the stage driver: select_calc_loan_id; drop rows whose
calc_loan_id is null (with warning); intersection-filter loans and
properties; assert symmetric coverage; generate_id_column for
calc_borrower_id and calc_property_id.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import polars as pl
import structlog

from esma_milan.config import DEFAULT_MIN_LOAN_ID_COVERAGE
from esma_milan.errors import PipelineDataError

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
    unmatched: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoanIdCandidate:
    """What `select_calc_loan_id` established about one candidate column."""

    column: str
    all_missing: bool
    has_duplicates: bool
    disqualified: bool
    """True when the column is all missing or has a duplicated value."""

    total: int
    """Distinct non-missing values. Not computed for a disqualified
    column, where it is 0 (which is why the log shows `0/0` for one)."""

    matched: int
    """How many of those values are a collateral row's loan identifier.
    Not computed for a disqualified column: 0."""

    coverage: float
    """matched / total. Not computed for a disqualified column: 0.0."""

    unmatched: tuple[str, ...]
    """Every distinct value that is no collateral row's loan identifier,
    sorted. Complete; warnings and errors show the first ten. Empty for a
    disqualified column."""

    duplicated_values: tuple[str, ...]
    """Every value that occurs on more than one loan, in the order the
    repeat is met. Complete. Empty when there is none."""


@dataclass(frozen=True)
class LoanIdSelection:
    """The loan-identifier decision of `select_calc_loan_id`."""

    candidates: tuple[LoanIdCandidate, LoanIdCandidate]
    """(original, new)."""

    chosen_column: str | None
    """The column used as `calc_loan_id`. None only on the selection
    carried by a `no_usable_loan_id_column` error, where no choice was
    made."""

    reason: str | None
    """Why it was chosen: the string logged as `reason`. None with
    `chosen_column`."""

    min_coverage: float
    """The minimum coverage the chosen column was held to."""


@dataclass(frozen=True)
class IdColumnChoice:
    """One `generate_id_column` decision (borrower or property)."""

    output_column: str
    original_column: str
    new_column: str
    chosen_column: str
    """Which of the two source columns supplied the values."""

    identical: bool
    """The two columns were equal element for element, missing values
    included."""

    n_unique_original: int
    n_unique_new: int
    decision: str
    """The string logged as `decision`."""


@dataclass(frozen=True)
class Stage3Decisions:
    """What Stage 3 decided and what it removed. Complete, not sampled.

    Every position here is a 0-based row position in the Stage 3 INPUT
    frame, which is the Stage 2 output. To get the record's position in
    the uploaded file, index `Stage2Output.loan_rows` (or
    `property_rows`) with it; `PipelineResult.loan_record_index` and
    `collateral_record_index` do that for the rows kept.
    """

    loan_id: LoanIdSelection

    loan_rows: pl.Series
    """Positions in the Stage 3 input loans frame of the loans kept,
    ascending (UInt32). Output `loans` row i is input row `loan_rows[i]`."""

    property_rows: pl.Series
    """Positions in the Stage 3 input properties frame of the collateral
    rows kept, ascending (UInt32)."""

    loans_missing_calc_loan_id: pl.Series
    """Positions in the Stage 3 input loans frame of the loans dropped
    because the chosen identifier is missing on them. The warning gives
    only their number."""

    loans_without_collateral: pl.Series
    """Positions in the Stage 3 input loans frame of the loans dropped by
    the intersection: their `calc_loan_id` is no collateral row's loan
    identifier. Nothing else reports these."""

    unmatched_loan_ids: tuple[str, ...]
    """The chosen column's values with no collateral row, sorted (the
    chosen candidate's `unmatched`)."""

    unmatched_collateral_ids: tuple[str, ...]
    """Collateral loan identifiers that are no kept loan's `calc_loan_id`,
    sorted. Empty on every successful run today: see
    `properties_removed`."""

    properties_removed: pl.Series
    """Positions in the Stage 3 input properties frame of the collateral
    rows removed by the intersection. Empty on every successful run today:
    a collateral row that the intersection would remove makes `run_stage3`
    raise `loan_identifier_mismatch` first. R drops the row instead
    (r_reference/R/pipeline.R:283-300). The fix for issue #45 will
    populate this field."""

    borrower_id: IdColumnChoice
    """The `generate_id_column` decision behind `calc_borrower_id`."""

    property_id: IdColumnChoice
    """The `generate_id_column` decision behind `calc_property_id`."""


def _sample_values(values: list[str], n: int = 10) -> str:
    """Mirrors R's `paste(utils::head(x, n), collapse = ', ')`."""
    return ", ".join(str(v) for v in values[:n])


def _duplicated_values(values: list[str | None]) -> list[str]:
    """Every non-null value that occurs more than once, each listed once,
    in the order its second occurrence is met."""
    seen: set[str] = set()
    listed: set[str] = set()
    dups: list[str] = []
    for v in values:
        if v is None:
            continue
        if v in seen and v not in listed:
            dups.append(v)
            listed.add(v)
        seen.add(v)
    return dups


def _duplicate_diagnostics(values: list[str | None]) -> tuple[int, str]:
    """Return (n_duplicated, sample_string) for the non-null portion."""
    dups = _duplicated_values(values)
    return len(dups), _sample_values(dups, 10)


def _loan_id_selection(
    candidates: list[_Candidate],
    *,
    chosen: _Candidate | None,
    reason: str | None,
    min_coverage: float,
) -> LoanIdSelection:
    """The public record of the candidates as the selection has them."""
    original, new = (
        LoanIdCandidate(
            column=c.name,
            all_missing=c.all_na,
            has_duplicates=c.has_duplicates,
            disqualified=c.disqualified,
            total=c.total,
            matched=c.matched,
            coverage=c.coverage,
            unmatched=c.unmatched,
            duplicated_values=(
                tuple(_duplicated_values(c.values)) if c.has_duplicates else ()
            ),
        )
        for c in candidates
    )
    return LoanIdSelection(
        candidates=(original, new),
        chosen_column=chosen.name if chosen is not None else None,
        reason=reason,
        min_coverage=float(min_coverage),
    )


def _select_calc_loan_id(
    loans: pl.DataFrame,
    properties: pl.DataFrame,
    *,
    original_id_col: str = "original_underlying_exposure_identifier",
    new_id_col: str = "new_underlying_exposure_identifier",
    output_col_name: str = "calc_loan_id",
    property_id_col: str = "underlying_exposure_identifier",
    min_coverage: float = DEFAULT_MIN_LOAN_ID_COVERAGE,
    _stacklevel: int = 2,
) -> tuple[pl.DataFrame, LoanIdSelection]:
    """Pick the loan-ID column to use as `calc_loan_id`.

    The body of `select_calc_loan_id`, returning the decision beside the
    frame. `_stacklevel` is the `stacklevel` of the warnings it emits, so
    a warning is attributed to the caller of the public function (or to
    `run_stage3`) whichever way this is reached.

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
        raise PipelineDataError(
            f"select_calc_loan_id: neither candidate loan-ID column is usable. "
            f"{orig.name!r}: {orig_msg}. "
            f"{new.name!r}: {new_msg}. "
            f"This is a data-quality problem that requires manual investigation.",
            stage="stage3",
            reason="no_usable_loan_id_column",
            items=tuple(
                (c.name, v) for c in candidates for v in _duplicated_values(c.values)
            ),
            context={
                "all_missing": {c.name: c.all_na for c in candidates},
                "loan_id_selection": _loan_id_selection(
                    candidates, chosen=None, reason=None, min_coverage=min_coverage
                ),
            },
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
        c.unmatched = tuple(unmatched)

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
                stacklevel=_stacklevel,
            )

    chosen = candidates[chosen_idx]

    # -- min_coverage threshold check -------------------------------------
    if chosen.coverage < min_coverage:
        raise PipelineDataError(
            f"select_calc_loan_id: chosen column {chosen.name!r} has coverage "
            f"{chosen.matched}/{chosen.total} = {_pct(chosen.coverage)}, "
            f"below min_coverage = {_pct(min_coverage)}. "
            f"Unmatched sample: [{chosen.unmatched_sample}].",
            stage="stage3",
            reason="loan_id_coverage_below_minimum",
            items=chosen.unmatched,
            context={
                "column": chosen.name,
                "matched": chosen.matched,
                "total": chosen.total,
                "coverage": chosen.coverage,
                "min_coverage": float(min_coverage),
                "loan_id_selection": _loan_id_selection(
                    candidates, chosen=chosen, reason=reason, min_coverage=min_coverage
                ),
            },
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
            stacklevel=_stacklevel,
        )

    return (
        loans.with_columns(
            pl.Series(output_col_name, chosen.values, dtype=pl.String)
        ),
        _loan_id_selection(
            candidates, chosen=chosen, reason=reason, min_coverage=min_coverage
        ),
    )


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
    """Pick the loan-ID column to use as `calc_loan_id` and return `loans`
    with it appended. See `_select_calc_loan_id` for the rule; this returns
    its frame and discards the `LoanIdSelection`."""
    frame, _ = _select_calc_loan_id(
        loans,
        properties,
        original_id_col=original_id_col,
        new_id_col=new_id_col,
        output_col_name=output_col_name,
        property_id_col=property_id_col,
        min_coverage=min_coverage,
        _stacklevel=3,
    )
    return frame


def _pct(x: float) -> str:
    """Format a fraction as percentage with 0.01 accuracy. Matches R's
    `scales::percent(x, accuracy = 0.01)` output: '99.00%', '85.71%'."""
    return f"{x * 100:.2f}%"


def _generate_id_column(
    df: pl.DataFrame,
    *,
    original_id_col: str,
    new_id_col: str,
    output_col_name: str,
) -> tuple[pl.DataFrame, IdColumnChoice]:
    """Pick between two candidate ID columns by uniqueness count.

    The body of `generate_id_column`, returning the decision beside the
    frame.

    Mirrors r_reference/R/utils.R:173-206. Used for borrower and
    property IDs where neither the duplicate-free constraint nor the
    cross-table coverage check that select_calc_loan_id enforces is
    appropriate.

    Decision (in order):
      1. Columns are element-wise identical (NAs in matching positions
         too) -> use original (stable tie-break).
      2. count_unique(non_null(original)) > count_unique(non_null(new))
           -> use original.
      3. count_unique(non_null(new)) > count_unique(non_null(original))
           -> use new.
      4. Tie -> use original.

    The chosen column's values are appended to `df` as
    `output_col_name` and the modified DataFrame returned.

    Raises:
        ValueError: if either named column is missing from `df`.
    """
    if original_id_col not in df.columns:
        raise PipelineDataError(
            f"generate_id_column: df is missing column {original_id_col!r}",
            stage="stage3",
            reason="missing_identifier_column",
            items=(original_id_col,),
            context={"output_column": output_col_name},
        )
    if new_id_col not in df.columns:
        raise ValueError(
            f"generate_id_column: df is missing column {new_id_col!r}"
        )

    original_values = df[original_id_col].to_list()
    new_values = df[new_id_col].to_list()

    # Element-wise identity check (including NAs in the same positions).
    # Polars Series equality returns False at null positions, so use
    # the materialised lists for the identity test - matches R's
    # `identical(original_ids, new_ids)` semantics.
    identical = original_values == new_values

    count_unique_original = len({v for v in original_values if v is not None})
    count_unique_new = len({v for v in new_values if v is not None})

    chosen_col = original_id_col
    if identical:
        chosen, source = original_values, "identical, using original"
    elif count_unique_original > count_unique_new:
        chosen, source = original_values, "original has more unique values"
    elif count_unique_new > count_unique_original:
        chosen, source = new_values, "new has more unique values"
        chosen_col = new_id_col
    else:
        chosen, source = original_values, "tied, defaulting to original"

    log.info(
        "generate_id_column",
        output_col=output_col_name,
        original=original_id_col,
        new=new_id_col,
        n_unique_original=count_unique_original,
        n_unique_new=count_unique_new,
        decision=source,
    )

    return (
        df.with_columns(pl.Series(output_col_name, chosen, dtype=pl.String)),
        IdColumnChoice(
            output_column=output_col_name,
            original_column=original_id_col,
            new_column=new_id_col,
            chosen_column=chosen_col,
            identical=identical,
            n_unique_original=count_unique_original,
            n_unique_new=count_unique_new,
            decision=source,
        ),
    )


def generate_id_column(
    df: pl.DataFrame,
    *,
    original_id_col: str,
    new_id_col: str,
    output_col_name: str,
) -> pl.DataFrame:
    """Pick between two candidate ID columns by uniqueness count and
    return `df` with the chosen values appended. See `_generate_id_column`
    for the rule; this returns its frame and discards the
    `IdColumnChoice`."""
    frame, _ = _generate_id_column(
        df,
        original_id_col=original_id_col,
        new_id_col=new_id_col,
        output_col_name=output_col_name,
    )
    return frame


# ---------------------------------------------------------------------------
# Stage 3 driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage3Output:
    """Output of Stage 3.

    `loans` carries the appended `calc_loan_id` and `calc_borrower_id`
    columns. `properties` carries the appended `calc_property_id`. Both
    tables are restricted to the intersection on calc_loan_id /
    underlying_exposure_identifier so downstream graph-building can
    assume symmetric coverage.
    """

    loans: pl.DataFrame
    properties: pl.DataFrame

    decisions: Stage3Decisions | None = None
    """What Stage 3 decided and removed; set by `run_stage3`."""


def run_stage3(
    loans: pl.DataFrame,
    properties: pl.DataFrame,
    *,
    min_coverage: float = DEFAULT_MIN_LOAN_ID_COVERAGE,
) -> Stage3Output:
    """Compose select_calc_loan_id + generate_id_column + intersection filter.

    Mirrors r_reference/R/pipeline.R:264-331:
      1. select_calc_loan_id picks the loan-ID column.
      2. Drop loans whose chosen calc_loan_id is null (with warning).
      3. Intersection-filter: loans whose calc_loan_id is in the property
         table's underlying_exposure_identifier set survive; properties
         whose underlying_exposure_identifier is in the calc_loan_id set
         survive.
      4. Assert both filtered tables are non-empty (raise otherwise).
      5. Belt-and-braces symmetric-coverage check: every calc_loan_id
         must appear in property loan IDs and vice versa. This block is
         unreachable under correct upstream behaviour but is kept as a
         regression guard.
      6. generate_id_column for calc_borrower_id (on loans).
      7. generate_id_column for calc_property_id (on properties).
    """
    loans, loan_id = _select_calc_loan_id(
        loans, properties, min_coverage=min_coverage, _stacklevel=2
    )
    # Positions in the input frames, carried beside the frames through the
    # filters below so the decision data can name the records removed.
    loan_positions = pl.int_range(loans.height, dtype=pl.UInt32, eager=True)
    property_positions = pl.int_range(properties.height, dtype=pl.UInt32, eager=True)

    # Drop rows with null calc_loan_id. R: pipeline.R:275-280.
    n_before = loans.height
    has_calc_loan_id = loans["calc_loan_id"].is_not_null()
    loans_missing_calc_loan_id = loan_positions.filter(~has_calc_loan_id)
    loan_positions = loan_positions.filter(has_calc_loan_id)
    loans = loans.filter(has_calc_loan_id)
    n_dropped = n_before - loans.height
    if n_dropped > 0:
        warnings.warn(
            f"Dropped {n_dropped} loans with missing calc_loan_id",
            UserWarning,
            stacklevel=2,
        )

    # Intersection filter (pipeline.R:283-287).
    property_loan_ids: set[str] = {
        v for v in properties["underlying_exposure_identifier"].to_list() if v is not None
    }
    has_collateral = loans.select(
        pl.col("calc_loan_id").is_in(list(property_loan_ids))
    ).to_series()
    loans_without_collateral = loan_positions.filter(~has_collateral)
    loan_positions = loan_positions.filter(has_collateral)
    loans = loans.filter(has_collateral)

    calc_loan_ids: set[str] = {
        v for v in loans["calc_loan_id"].to_list() if v is not None
    }
    has_loan = properties.select(
        pl.col("underlying_exposure_identifier").is_in(list(calc_loan_ids))
    ).to_series()
    properties_removed = property_positions.filter(~has_loan)
    property_positions = property_positions.filter(has_loan)
    properties = properties.filter(has_loan)

    if loans.height == 0:
        raise PipelineDataError(
            "ERROR: No active loans remaining after filtering.",
            stage="stage3",
            reason="no_loans_after_intersection",
            context={"column": loan_id.chosen_column},
        )
    if properties.height == 0:
        raise ValueError("ERROR: No properties remaining after filtering.")

    # Belt-and-braces symmetric coverage check (pipeline.R:296-317).
    # Should be unreachable under correct upstream behaviour but kept as
    # a regression guard.
    missing_in_properties = sorted(calc_loan_ids - property_loan_ids, key=str)
    missing_in_loans = sorted(property_loan_ids - calc_loan_ids, key=str)
    if missing_in_properties or missing_in_loans:
        raise PipelineDataError(
            "ERROR: Loan identifier mismatch between calc_loan_id and "
            "properties$underlying_exposure_identifier. "
            f"Missing in properties: {len(missing_in_properties)}"
            + (
                f" [{', '.join(missing_in_properties[:10])}]"
                if missing_in_properties
                else ""
            )
            + f"; missing in loans: {len(missing_in_loans)}"
            + (f" [{', '.join(missing_in_loans[:10])}]" if missing_in_loans else "")
            + ".",
            stage="stage3",
            reason="loan_identifier_mismatch",
            items=(
                *(("missing_in_properties", v) for v in missing_in_properties),
                *(("missing_in_loans", v) for v in missing_in_loans),
            ),
            context={
                "n_missing_in_properties": len(missing_in_properties),
                "n_missing_in_loans": len(missing_in_loans),
            },
        )

    # Borrower and property IDs use the simpler generate_id_column.
    loans, borrower_id = _generate_id_column(
        loans,
        original_id_col="original_obligor_identifier",
        new_id_col="new_obligor_identifier",
        output_col_name="calc_borrower_id",
    )
    properties, property_id = _generate_id_column(
        properties,
        original_id_col="original_collateral_identifier",
        new_id_col="new_collateral_identifier",
        output_col_name="calc_property_id",
    )

    log.info(
        "stage3_complete",
        n_loans=loans.height,
        n_properties=properties.height,
    )

    chosen_candidate = next(
        c for c in loan_id.candidates if c.column == loan_id.chosen_column
    )
    return Stage3Output(
        loans=loans,
        properties=properties,
        decisions=Stage3Decisions(
            loan_id=loan_id,
            loan_rows=loan_positions,
            property_rows=property_positions,
            loans_missing_calc_loan_id=loans_missing_calc_loan_id,
            loans_without_collateral=loans_without_collateral,
            unmatched_loan_ids=chosen_candidate.unmatched,
            unmatched_collateral_ids=tuple(missing_in_loans),
            properties_removed=properties_removed,
            borrower_id=borrower_id,
            property_id=property_id,
        ),
    )
