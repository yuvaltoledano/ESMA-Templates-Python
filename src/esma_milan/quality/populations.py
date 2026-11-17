"""Per-record labels: population, identity confidence, claimed lifecycle.

Mirrors nothing in the R reference (specification 3.3, 5.2 and 8.1).

`loan_records` / `collateral_records` are the narrow per-record frames the
audit keeps after it frees a file's text: identifiers as supplied and
trimmed, and the labels.

Population (8.1) is retention only and mirrors Stage 2. The stage's own
predicates (`pipeline.filters.loan_drop_reasons`) are applied, unchanged,
to two columns typed by the loader's rules: the status as text and the
balance as a number, each null where the loader reads the cell as missing
(`config.CSV_NA_TOKENS`). A balance the loader's cast cannot read makes
the loader stop, which no stage predicate describes. Because the typing
is re-derived here, the labels are compared record by record with
`Stage2Output` whenever the pipeline returned (`compare_with_stage2`), and
are called "predicted" whenever it did not.

Stage 2 has two reasons; the audit splits them into the nine populations:

  none true                      Active, retained
  balance_not_positive only      Active status, zero / negative / missing balance
  (the loader stops)             Active status, balance unparseable
  status_not_active              Active status not recognised | Inactive |
                                 Inactive status not recognised |
                                 Eligibility unknown

The lifecycle label is a presumption from what the record claims through
its status and RREL9, never from the values the lifecycle is later used
to check (3.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from esma_milan.config import (
    ACTIVE_LOAN_STATUSES,
    CSV_NA_TOKENS,
    EXCLUDED_COLLATERAL_TYPES,
)
from esma_milan.pipeline.filters import Stage2Output, collateral_drop_reasons, loan_drop_reasons
from esma_milan.quality.audit_read import RawFile
from esma_milan.quality.catalogue import (
    FIELDS,
    PROPERTY_COLLATERAL_TYPES,
    UNDETERMINED_COLLATERAL_TYPES,
)
from esma_milan.quality.column_rules.context import PoolFacts, cutoff_columns, strict_date
from esma_milan.quality.refdata import load_reference_data

POPULATIONS: tuple[str, ...] = (
    "Active, retained",
    "Active status, zero balance",
    "Active status, negative balance",
    "Active status, balance missing",
    "Active status, balance unparseable",
    "Active status not recognised",
    "Inactive",
    "Inactive status not recognised",
    "Eligibility unknown",
)
RETAINED = POPULATIONS[0]

LIFECYCLES: tuple[str, ...] = (
    "reporting-active (presumed)",
    "final-inactive (presumed)",
    "default, recoveries not determinable",
    "not established",
)
DEFAULT_STATUSES: frozenset[str] = frozenset({"DFLT", "NDFT", "DTCR", "DADB"})
REPURCHASE_STATUSES: frozenset[str] = frozenset({"REBR", "REDF", "RERE", "RESS", "REOT"})

IDENTITY_UNIQUE = "unique"
IDENTITY_OTHER_COLUMN = "duplicate in a candidate identifier column the pipeline did not select"
IDENTITY_AMBIGUOUS = "ambiguous: duplicate or colliding in the selected identifier column"
IDENTITY_UNKNOWN_SELECTION = (
    "duplicate or colliding in a candidate identifier column; selection not available"
)

# The amounts whose currencies A17 compares (specification 8.5), by file.
LOAN_AMOUNTS: tuple[str, ...] = ("RREL30", "RREL29", "RREL31", "RREL32", "RREL33", "RREL16", "RREL20")
COLLATERAL_AMOUNTS: tuple[str, ...] = ("RREC13", "RREC17")
CURRENCY_NOT_STATED = ""

# The reasons of Stage 2's collateral predicate the audit can apply from
# the collateral file alone (`no_active_loan` needs the loans Stage 2 kept).
COLLATERAL_REASONS: tuple[str, ...] = (
    "property_type_missing", "collateral_type_missing", "collateral_type_excluded",
    "no_positive_valuation",
)

# The cleaned column names Stage 3 reports, by ESMA code.
LOAN_ID_COLUMNS: dict[str, str] = {
    "original_underlying_exposure_identifier": "RREL2",
    "new_underlying_exposure_identifier": "RREL3",
}


def _text(raw: RawFile, field: str) -> pl.Expr:
    column = raw.column_of(field)
    return pl.col(column.frame_column) if column else pl.lit(None, dtype=pl.String)


def _loader_text(expr: pl.Expr) -> pl.Expr:
    """The cell as the loader delivers it: null for an NA token."""
    return pl.when(expr.is_null() | expr.is_in(list(CSV_NA_TOKENS))).then(None).otherwise(expr)


def _loader_number(expr: pl.Expr) -> pl.Expr:
    """The cell as the loader's numeric cast delivers it; null where the
    loader reads it as missing or cannot convert it."""
    return _loader_text(expr).str.strip_chars().cast(pl.Float64, strict=False)


def _currency(raw: RawFile, code: str) -> pl.Expr:
    """The currency stated for an amount, by the evidence of specification
    6.2: the amount's companion column if it holds a valid ISO 4217 code,
    else the dedicated field if that does, else whatever text either
    holds (an invalid code), else `CURRENCY_NOT_STATED`. Null where the
    loader reads no number in the amount."""
    valid = sorted(load_reference_data().currencies)
    sources = [raw.companion_of(code)]
    dedicated = FIELDS[code].dedicated_currency
    if dedicated is not None:
        sources.append(raw.column_of(dedicated))
    texts = [_loader_text(pl.col(c.frame_column)).str.strip_chars() for c in sources if c is not None]
    stated = [pl.when(t != "").then(t).otherwise(None) for t in texts]
    currency = pl.coalesce(
        *[pl.when(t.is_in(valid)).then(t).otherwise(None) for t in stated],
        *stated,
        pl.lit(CURRENCY_NOT_STATED),
    )
    return (
        pl.when(_loader_number(_text(raw, code)).is_not_null()).then(currency).otherwise(None)
        .cast(pl.Categorical).alias(f"cur_{code}")
    )


def _identity_flags(frame: pl.DataFrame, column: str, other: str) -> list[pl.Expr]:
    """Duplicate (the same raw value on another record) and colliding (a
    different raw value equal after trimming, or the same value on a
    different record's other identifier column)."""
    value = pl.col(column)
    filled = value.is_not_null() & (value.str.strip_chars() != "")
    trimmed = value.str.strip_chars()
    in_other = frame.filter(pl.col(other).is_not_null()).group_by(pl.col(other).alias("_k")).agg(
        pl.len().alias("_n")
    )
    counts = dict(zip(in_other["_k"].to_list(), in_other["_n"].to_list(), strict=True))
    n_other = value.replace_strict(counts, default=0, return_dtype=pl.UInt32)
    own = (pl.col(other) == value).fill_null(False).cast(pl.UInt32)
    return [
        (filled & (pl.len().over(column) > 1)).alias(f"{column}_duplicate"),
        (
            filled & ((value.n_unique().over(trimmed) > 1) | (n_other > own))
        ).alias(f"{column}_colliding"),
    ]


def loan_records(raw: RawFile, pool: PoolFacts) -> pl.DataFrame:
    """One row per loan record: identifiers, status, balance, cut-off,
    population (with the Stage 2 reasons it rests on), lifecycle."""
    assert raw.frame is not None
    cut, _ = cutoff_columns(raw, pool)
    status_codes = FIELDS["RREL69"].codes
    assert status_codes is not None
    inactive_codes = sorted(status_codes - ACTIVE_LOAN_STATUSES)
    active_codes = sorted(ACTIVE_LOAN_STATUSES)

    base = raw.frame.select(
        "record",
        "physical_line",
        *[_text(raw, f"RREL{n}").alias(f"RREL{n}") for n in (2, 3, 4, 5)],
        _text(raw, "RREL69").alias("status"),
        _text(raw, "RREL30").alias("balance_text"),
        strict_date(_text(raw, "RREL9")).alias("_rrel9"),
        *[_currency(raw, code) for code in LOAN_AMOUNTS],
    ).with_columns(cut["_cut"].alias("cutoff"))

    loader_balance = _loader_text(pl.col("balance_text"))
    typed = base.select(
        account_status=_loader_text(pl.col("status")),
        current_principal_balance=_loader_number(pl.col("balance_text")),
    )
    reasons = typed.select(**loan_drop_reasons())
    base = base.with_columns(
        reasons["status_not_active"],
        reasons["balance_not_positive"],
        typed["current_principal_balance"].alias("balance"),
        (loader_balance.is_not_null() & typed["current_principal_balance"].is_null()).alias(
            "balance_unparseable"
        ),
    )

    active = ~pl.col("status_not_active")
    near = pl.col("status").fill_null("").str.strip_chars().str.to_uppercase()
    balance = pl.col("balance")
    population = (
        pl.when(active & pl.col("balance_unparseable")).then(pl.lit(POPULATIONS[4]))
        .when(active & ~pl.col("balance_not_positive")).then(pl.lit(POPULATIONS[0]))
        .when(active & balance.is_null()).then(pl.lit(POPULATIONS[3]))
        .when(active & (balance == 0)).then(pl.lit(POPULATIONS[1]))
        .when(active).then(pl.lit(POPULATIONS[2]))
        .when(near.is_in(active_codes)).then(pl.lit(POPULATIONS[5]))
        .when(pl.col("status").is_in(inactive_codes).fill_null(False)).then(pl.lit(POPULATIONS[6]))
        .when(near.is_in(inactive_codes)).then(pl.lit(POPULATIONS[7]))
        .otherwise(pl.lit(POPULATIONS[8]))
    )

    status = pl.col("status").fill_null("")
    recovery_ended = pl.col("_rrel9").is_not_null() & (pl.col("_rrel9") <= pl.col("cutoff")).fill_null(False)
    lifecycle = (
        pl.when(status.is_in(active_codes)).then(pl.lit(LIFECYCLES[0]))
        .when((status == "RDMD") | status.is_in(sorted(REPURCHASE_STATUSES))).then(pl.lit(LIFECYCLES[1]))
        .when(status.is_in(sorted(DEFAULT_STATUSES)) & recovery_ended).then(pl.lit(LIFECYCLES[1]))
        .when(status.is_in(sorted(DEFAULT_STATUSES))).then(pl.lit(LIFECYCLES[2]))
        .otherwise(pl.lit(LIFECYCLES[3]))
    )
    rrel9_text = (
        pl.when(pl.col("_rrel9").is_null()).then(pl.lit("no valid date"))
        .when(pl.col("cutoff").is_null()).then(
            pl.format("{} (cut-off not established)", pl.col("_rrel9").cast(pl.String))
        )
        .when(pl.col("_rrel9") <= pl.col("cutoff")).then(
            pl.format("{} (not after the cut-off)", pl.col("_rrel9").cast(pl.String))
        )
        .otherwise(pl.format("{} (after the cut-off)", pl.col("_rrel9").cast(pl.String)))
    )
    needs_rrel9 = (status == "RDMD") | status.is_in(sorted(DEFAULT_STATUSES))
    evidence = (
        pl.when(status == "").then(pl.lit("status empty"))
        .when(needs_rrel9).then(pl.format("status {}; RREL9 {}", status, rrel9_text))
        .otherwise(pl.format("status {}", status))
    )

    return base.with_columns(
        *[pl.col(f"RREL{n}").str.strip_chars().alias(f"RREL{n}_trimmed") for n in (2, 3, 4, 5)],
        population.alias("population"),
        lifecycle.alias("lifecycle"),
        evidence.alias("lifecycle_evidence"),
        *_identity_flags(base, "RREL2", "RREL3"),
        *_identity_flags(base, "RREL3", "RREL2"),
    ).drop("_rrel9")


def collateral_records(raw: RawFile) -> pl.DataFrame:
    """One row per collateral record: identifiers, the collateral class
    (property, established non-property, undetermined, not established),
    the currency stated for each valuation, and the reasons of Stage 2's
    collateral predicate that the record alone establishes (`s2_<reason>`,
    predicted: the stage's own expressions on columns typed by the
    loader's rules)."""
    assert raw.frame is not None
    kind = pl.col("RREC5").fill_null("")
    typed = raw.frame.select(
        property_type=_loader_text(_text(raw, "RREC9")),
        collateral_type=_loader_text(_text(raw, "RREC5")),
        underlying_exposure_identifier=pl.lit(None, dtype=pl.String),
        original_valuation_amount=_loader_number(_text(raw, "RREC17")),
        current_valuation_amount=_loader_number(_text(raw, "RREC13")),
    )
    predicate = collateral_drop_reasons(())
    reasons = typed.select(**{f"s2_{name}": predicate[name] for name in COLLATERAL_REASONS})
    return raw.frame.select(
        "record",
        "physical_line",
        *[_text(raw, f"RREC{n}").alias(f"RREC{n}") for n in (2, 3, 4, 5)],
        *[_currency(raw, code) for code in COLLATERAL_AMOUNTS],
    ).with_columns(
        *reasons.get_columns(),
        *[pl.col(f"RREC{n}").str.strip_chars().alias(f"RREC{n}_trimmed") for n in (2, 3, 4)],
        pl.when(kind.is_in(sorted(PROPERTY_COLLATERAL_TYPES))).then(pl.lit("property"))
        .when(kind.is_in(sorted(EXCLUDED_COLLATERAL_TYPES))).then(pl.lit("established non-property"))
        .when(kind.is_in(sorted(UNDETERMINED_COLLATERAL_TYPES))).then(pl.lit("undetermined"))
        .otherwise(pl.lit("not established"))
        .alias("collateral_class"),
    )


def identity_confidence(records: pl.DataFrame, selected: str | None) -> pl.Series:
    """The identity-confidence label of every loan record. `selected` is
    the ESMA code of the column Stage 3 chose (`RREL2` or `RREL3`), or
    None when Stage 3 did not report one."""
    flagged = {
        code: pl.col(f"{code}_duplicate") | pl.col(f"{code}_colliding") for code in ("RREL2", "RREL3")
    }
    if selected is None:
        label = (
            pl.when(flagged["RREL2"] | flagged["RREL3"]).then(pl.lit(IDENTITY_UNKNOWN_SELECTION))
            .otherwise(pl.lit(IDENTITY_UNIQUE))
        )
    else:
        other = "RREL3" if selected == "RREL2" else "RREL2"
        label = (
            pl.when(flagged[selected]).then(pl.lit(IDENTITY_AMBIGUOUS))
            .when(flagged[other]).then(pl.lit(IDENTITY_OTHER_COLUMN))
            .otherwise(pl.lit(IDENTITY_UNIQUE))
        )
    return records.select(label.alias("identity_confidence")).to_series()


@dataclass(frozen=True)
class Stage2Comparison:
    """The audit's population labels against the stage's own output."""

    evaluated: bool
    reason: str | None
    """Why it was not evaluated."""
    records: int
    retained_by_audit: int
    retained_by_stage: int | None = None
    differing_records: tuple[int, ...] = ()
    """Record numbers (1-based) on which the audit's Stage 2 reasons
    differ from the stage's. Empty when the two agree."""

    @property
    def agrees(self) -> bool:
        return self.evaluated and not self.differing_records


def compare_with_stage2(records: pl.DataFrame, stage2: Stage2Output | None) -> Stage2Comparison:
    retained = int((records["population"] == RETAINED).sum())
    if stage2 is None or stage2.loan_drop_reasons is None or stage2.loan_rows is None:
        return Stage2Comparison(
            evaluated=False,
            reason="the pipeline did not return a Stage 2 result",
            records=records.height,
            retained_by_audit=retained,
        )
    theirs = stage2.loan_drop_reasons
    if theirs.height != records.height:
        return Stage2Comparison(
            evaluated=True, reason=None, records=records.height, retained_by_audit=retained,
            retained_by_stage=len(stage2.loan_rows),
            differing_records=tuple(range(1, max(theirs.height, records.height) + 1)),
        )
    dropped = theirs.select(pl.any_horizontal(pl.all())).to_series()
    differs = (
        (records["status_not_active"] != theirs["status_not_active"])
        | (records["balance_not_positive"] != theirs["balance_not_positive"])
        | records["balance_unparseable"]
        | ((records["population"] == RETAINED) != ~dropped)
    )
    kept_differs = set(records.filter(records["population"] == RETAINED)["record"].to_list()) ^ {
        i + 1 for i in stage2.loan_rows.to_list()
    }
    differing = sorted(set(records.filter(differs)["record"].to_list()) | kept_differs)
    return Stage2Comparison(
        evaluated=True, reason=None, records=records.height, retained_by_audit=retained,
        retained_by_stage=len(stage2.loan_rows), differing_records=tuple(differing),
    )
