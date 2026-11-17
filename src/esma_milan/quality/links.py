"""The links between collateral records and loan records.

Mirrors nothing in the R reference (specification 5.2).

Built by the audit from its own text, without calling a stage. A
collateral record refers to a loan through RREC2. Annex 2 prescribes that
RREC2 matches RREL3, the new identifier (convention `new`); exports also
carry the original identifier there (convention `original`, RREC2 =
RREL2), and the pipeline's Stage 3 chooses between the two. Each
convention is matched twice and the results are kept apart:

    match = "raw"       the two cells are equal as they stand
    match = "trimmed"   they are equal only after trimming whitespace; the
                        pipeline does not match these (issue #46)

An empty cell and an ND token are not identifiers and link nothing.

From the links: the lifecycle of the loan a collateral record belongs to
(for the conditional ND5 rule, P3f), whether it belongs to a retained
loan, and, when Stage 3 ran, a record-by-record comparison of the loans
the stage dropped with the loans the links say it should have dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from esma_milan.config import CSV_NA_TOKENS
from esma_milan.pipeline.filters import Stage2Output
from esma_milan.pipeline.identifiers import Stage3Decisions
from esma_milan.quality.populations import LOAN_ID_COLUMNS, RETAINED
from esma_milan.quality.states import ND_TOKENS

CONVENTIONS: dict[str, str] = {"new": "RREL3", "original": "RREL2"}

LINK_SCHEMA: dict[str, pl.DataType] = {
    "collateral_record": pl.UInt32(),
    "loan_record": pl.UInt32(),
    "convention": pl.String(),
    "match": pl.String(),
}

NO_LOAN_LINKED = "no loan record linked"
LIFECYCLES_DIFFER = "linked loans whose lifecycles differ"


def _identifier(column: str) -> pl.Expr:
    """The cell when it is an identifier, else null."""
    value = pl.col(column)
    usable = value.is_not_null() & (value.str.strip_chars() != "") & ~value.is_in(sorted(ND_TOKENS))
    return pl.when(usable).then(value).otherwise(None)


def build_links(loans: pl.DataFrame, collateral: pl.DataFrame) -> pl.DataFrame:
    """One row per (collateral record, loan record, convention) that
    match. `loans` has `record`, `RREL2`, `RREL3`; `collateral` has
    `record`, `RREC2`."""
    parts: list[pl.DataFrame] = []
    right = collateral.select(
        pl.col("record").alias("collateral_record"),
        _identifier("RREC2").alias("_raw"),
        _identifier("RREC2").str.strip_chars().alias("_trimmed"),
    ).filter(pl.col("_raw").is_not_null())
    for convention, column in CONVENTIONS.items():
        left = loans.select(
            pl.col("record").alias("loan_record"),
            _identifier(column).alias("_loan_raw"),
            _identifier(column).str.strip_chars().alias("_trimmed"),
        ).filter(pl.col("_loan_raw").is_not_null())
        joined = right.join(left, on="_trimmed")
        parts.append(
            joined.select(
                "collateral_record",
                "loan_record",
                pl.lit(convention).alias("convention"),
                pl.when(pl.col("_raw") == pl.col("_loan_raw")).then(pl.lit("raw"))
                .otherwise(pl.lit("trimmed")).alias("match"),
            )
        )
    return (
        pl.concat(parts).cast(LINK_SCHEMA)  # type: ignore[arg-type]
        .sort("collateral_record", "loan_record", "convention")
    )


def collateral_link_context(
    links: pl.DataFrame, loans: pl.DataFrame, n_collateral: int
) -> pl.DataFrame:
    """Per collateral record (1..n, in order): `linked_lifecycle`, the
    claimed lifecycle of the loan it belongs to (the prescribed convention
    on the raw text first, else the other; `NO_LOAN_LINKED` or
    `LIFECYCLES_DIFFER` where that is not one label), `linked_retained`,
    whether a raw link under either convention leads to a retained loan,
    and the number of raw links under each convention."""
    raw = links.filter(pl.col("match") == "raw").join(
        loans.select(pl.col("record").alias("loan_record"), "lifecycle", "population"),
        on="loan_record",
    )
    per_convention = raw.group_by("collateral_record", "convention").agg(
        pl.col("lifecycle").n_unique().alias("_n"),
        pl.col("lifecycle").first().alias("_lifecycle"),
        pl.len().alias("_links"),
        (pl.col("population") == RETAINED).any().alias("_retained"),
    )

    def side(convention: str) -> pl.DataFrame:
        return per_convention.filter(pl.col("convention") == convention).select(
            pl.col("collateral_record").alias("record"),
            pl.when(pl.col("_n") == 1).then(pl.col("_lifecycle"))
            .otherwise(pl.lit(LIFECYCLES_DIFFER)).alias(f"_life_{convention}"),
            pl.col("_links").alias(f"links_{convention}"),
            pl.col("_retained").alias(f"_retained_{convention}"),
        )

    records = pl.DataFrame({"record": pl.int_range(1, n_collateral + 1, eager=True, dtype=pl.UInt32)})
    return (
        records.join(side("new"), on="record", how="left")
        .join(side("original"), on="record", how="left")
        .select(
            "record",
            pl.coalesce("_life_new", "_life_original", pl.lit(NO_LOAN_LINKED)).alias("linked_lifecycle"),
            (pl.col("_retained_new").fill_null(False) | pl.col("_retained_original").fill_null(False))
            .alias("linked_retained"),
            pl.col("links_new").fill_null(0),
            pl.col("links_original").fill_null(0),
        )
    )


def link_counts(links: pl.DataFrame | pl.LazyFrame, n_loans: int, n_collateral: int) -> dict[str, object]:
    """Counts for the summary: per convention and match, the collateral
    records and the loan records linked; and the records linked by nothing."""
    frame = links.lazy()
    by = (
        frame.group_by("convention", "match")
        .agg(
            pl.col("collateral_record").n_unique().alias("collateral_records"),
            pl.col("loan_record").n_unique().alias("loan_records"),
            pl.len().alias("links"),
        )
        .sort("convention", "match")
        .collect()
    )
    raw = frame.filter(pl.col("match") == "raw")
    linked_collateral = raw.select(pl.col("collateral_record").n_unique()).collect().item()
    linked_loans = raw.select(pl.col("loan_record").n_unique()).collect().item()
    return {
        "by_convention": {
            f"{row['convention']} ({CONVENTIONS[row['convention']]}), {row['match']}": {
                "collateral_records": row["collateral_records"],
                "loan_records": row["loan_records"],
                "links": row["links"],
            }
            for row in by.iter_rows(named=True)
        },
        "collateral_records_matching_no_loan_record": n_collateral - linked_collateral,
        "loan_records_with_no_collateral_record": n_loans - linked_loans,
        "note": "raw: the two cells are equal as they stand. trimmed: equal only after "
                "trimming whitespace, which the pipeline does not match. The unlinked "
                "counts are over raw links under either convention.",
    }


@dataclass(frozen=True)
class Stage3Comparison:
    """The loans Stage 3 dropped against the loans the links say it drops."""

    evaluated: bool
    reason: str | None
    selected: str | None
    """`RREL2` or `RREL3`."""
    missing_identifier: tuple[int, ...] = ()
    """Loan records (1-based) dropped for a missing selected identifier, per the stage."""
    without_collateral: tuple[int, ...] = ()
    """Loan records dropped because no kept collateral record refers to them, per the stage."""
    differing_records: tuple[int, ...] = ()

    @property
    def agrees(self) -> bool:
        return self.evaluated and not self.differing_records


def compare_with_stage3(
    loans: pl.DataFrame,
    links: pl.DataFrame,
    stage2: Stage2Output | None,
    decisions: Stage3Decisions | None,
) -> Stage3Comparison:
    """`loans` has `record`, `population`, `RREL2`, `RREL3`. The audit's
    own sets: among retained loans, the selected identifier is missing as
    the loader reads it; or no collateral record that Stage 2 kept links
    to the loan under the selected convention on the raw text."""
    if (
        stage2 is None or decisions is None or stage2.loan_rows is None
        or stage2.property_rows is None or decisions.loan_id.chosen_column is None
    ):
        return Stage3Comparison(False, "the pipeline did not return a Stage 3 result", None)
    selected = LOAN_ID_COLUMNS[decisions.loan_id.chosen_column]
    convention = next(name for name, column in CONVENTIONS.items() if column == selected)
    kept_loans = [i + 1 for i in stage2.loan_rows.to_list()]

    def records(positions: pl.Series) -> tuple[int, ...]:
        return tuple(sorted(kept_loans[i] for i in positions.to_list()))

    theirs_missing = records(decisions.loans_missing_calc_loan_id)
    theirs_without = records(decisions.loans_without_collateral)

    kept_collateral = [i + 1 for i in stage2.property_rows.to_list()]
    covered = (
        links.filter(
            (pl.col("convention") == convention) & (pl.col("match") == "raw")
            & pl.col("collateral_record").is_in(kept_collateral)
        )["loan_record"].unique().to_list()
    )
    identifier = pl.col(selected)
    retained = loans.filter(pl.col("population") == RETAINED)
    missing = identifier.is_null() | identifier.is_in(list(CSV_NA_TOKENS))
    ours_missing = tuple(sorted(retained.filter(missing)["record"].to_list()))
    ours_without = tuple(
        sorted(retained.filter(~missing & ~pl.col("record").is_in(covered))["record"].to_list())
    )
    differing = sorted(
        (set(ours_missing) ^ set(theirs_missing)) | (set(ours_without) ^ set(theirs_without))
    )
    return Stage3Comparison(
        evaluated=True, reason=None, selected=selected,
        missing_identifier=theirs_missing, without_collateral=theirs_without,
        differing_records=tuple(differing),
    )
