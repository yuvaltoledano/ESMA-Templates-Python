"""What a per-column rule is given, and the cut-off date of each record.

Mirrors nothing in the R reference.

The data cut-off date (specification 6.2, "The cut-off date"): for a loan
record, RREL6 when it is a valid date, else the export's
`Pool_Cutoff_Date` column; when both are valid and differ the cut-off is
not established for that record. For a collateral record, the file's own
`Pool_Cutoff_Date`; where the file has no such column, the loans file's
cut-off when that is a single value, and the result row says so.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date

import polars as pl

from esma_milan.quality.audit_read import RawFile
from esma_milan.quality.catalogue import FieldEntry
from esma_milan.quality.refdata import ReferenceData
from esma_milan.quality.result import Finding, Mark
from esma_milan.quality.states import XML_WHITESPACE, Classified

CUTOFF_NOT_ESTABLISHED = "cut-off not established"
CUTOFF_TWO_VALUES = "cut-off not established (two values on the record)"
CUTOFF_FROM_LOANS_NOTE = "pool cut-off taken from the loans file"


@dataclass(frozen=True)
class PoolFacts:
    """Facts about the pool that a rule on one record may use. They are
    values of the loans file as a whole, not links between records."""

    loans_cutoff: date | None = None
    """The loans file's cut-off when its records give exactly one."""
    classification_year: int | None = None
    """RREL12 when the loans file gives exactly one valid year."""


@dataclass(frozen=True)
class CellContext:
    """One distinct combination of values a rule is evaluated on."""

    entry: FieldEntry
    ref: ReferenceData
    raw: str
    value: Classified
    cutoff: date | None
    cutoff_reason: str | None
    """Why `cutoff` is None."""
    cutoff_note: str | None
    """Set when the cut-off came from the loans file."""
    extras: Mapping[str, str | None]
    """Other cells of the same record the field's rules read (by name);
    None when that column is not in the file."""
    pool: PoolFacts


RuleOutput = Iterator[Finding | Mark]


def strict_date(column: pl.Expr) -> pl.Expr:
    """`YYYY-MM-DD` after XML whitespace trimming, a real date; else null."""
    text = column.str.strip_chars(XML_WHITESPACE)
    return (
        pl.when(text.str.contains(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"))
        .then(text)
        .otherwise(None)
        .str.to_date("%Y-%m-%d", strict=False)
    )


def cutoff_columns(raw: RawFile, pool: PoolFacts) -> tuple[pl.DataFrame, str | None]:
    """`record`, `_cut` (Date) and `_cut_reason` for every record of
    `raw`, and the note to attach when the cut-off is the loans file's."""
    assert raw.frame is not None
    frame = raw.frame
    own = raw.extension("pool_cutoff_date")
    field = raw.column_of("RREL6") if raw.file == "loans" else None
    null_date = pl.lit(None, dtype=pl.Date)
    a = strict_date(pl.col(field.frame_column)) if field else null_date
    b = strict_date(pl.col(own.frame_column)) if own else null_date
    note: str | None = None
    if raw.file == "collateral" and own is None:
        cut = pl.lit(pool.loans_cutoff, dtype=pl.Date)
        reason = pl.lit(None if pool.loans_cutoff else CUTOFF_NOT_ESTABLISHED, dtype=pl.String)
        note = CUTOFF_FROM_LOANS_NOTE if pool.loans_cutoff else None
    else:
        conflict = a.is_not_null() & b.is_not_null() & (a != b)
        cut = pl.when(conflict).then(null_date).otherwise(pl.coalesce(a, b))
        reason = (
            pl.when(conflict).then(pl.lit(CUTOFF_TWO_VALUES))
            .when(pl.coalesce(a, b).is_null()).then(pl.lit(CUTOFF_NOT_ESTABLISHED))
            .otherwise(pl.lit(None, dtype=pl.String))
        )
    return frame.select("record", cut.alias("_cut"), reason.alias("_cut_reason")), note


def pool_facts(loans: RawFile) -> PoolFacts:
    """The pool-level facts the loans file establishes."""
    if loans.frame is None:
        return PoolFacts()
    cut, _ = cutoff_columns(loans, PoolFacts())
    cutoffs = cut["_cut"].drop_nulls().unique().to_list()
    year_column = loans.column_of("RREL12")
    years: list[str] = []
    if year_column is not None:
        text = pl.col(year_column.frame_column).str.strip_chars(XML_WHITESPACE)
        years = (
            loans.frame.select(text.alias("y"))
            .filter(pl.col("y").str.contains(r"^[0-9]{4}$"))["y"]
            .unique()
            .to_list()
        )
    return PoolFacts(
        loans_cutoff=cutoffs[0] if len(cutoffs) == 1 else None,
        classification_year=int(years[0]) if len(years) == 1 else None,
    )
