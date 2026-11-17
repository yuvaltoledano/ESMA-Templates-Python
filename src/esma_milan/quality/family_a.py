"""Family A: findings about the files themselves (specification 7.1, 5.3).

Mirrors nothing in the R reference.

Every rule here is evaluated from the two files alone, in the assessment
process, while a file's text is in hand. A17 (currency compatibility of the
amounts the pipeline combines) needs the pipeline's own results and is in
`waterfall.py`.

Some rules are reported through something that already exists and emit no
row of their own, so that one defect is not reported twice:

    A3  header resolution     the actual-column inventory; A3 adds one
                              context row per unmapped column
    A4  expected fields       is P1
    A14 record counts         the summary's file block
    A16 a file that cannot    the unreadable outcome of the audit read; A16
        be split into records gives it its rows (every over-long record)

Severity Stop never means this run was stopped. A rule with basis L says
the pipeline cannot process what it found; a rule with basis A says the
acceptance policy of a later release would stop on it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from functools import lru_cache

import polars as pl

from esma_milan.config import CSV_NA_TOKENS, REQUIRED_LOAN_COLUMNS, REQUIRED_PROPERTY_COLUMNS
from esma_milan.io_layer.clean_names import clean_name
from esma_milan.io_layer.read_taxonomy import (
    bundled_taxonomy_path,
    load_taxonomy,
    load_taxonomy_formats,
    numeric_field_names,
)
from esma_milan.quality.audit_read import (
    RESOLVED_TO_FIELD,
    ColumnResolution,
    RawFile,
    Unreadable,
)
from esma_milan.quality.catalogue import RULES, FileKind, fields_of
from esma_milan.quality.column_rules.context import (
    CUTOFF_NOT_ESTABLISHED,
    PoolFacts,
    cutoff_columns,
    strict_date,
)
from esma_milan.quality.result import RESULT_SCHEMA, RuleOutcome, concat_results, results_frame
from esma_milan.quality.states import ND_TOKENS

LOAN_IDENTIFIERS: tuple[str, ...] = ("RREL2", "RREL3", "RREL4", "RREL5")
COLLATERAL_IDENTIFIERS: tuple[str, ...] = ("RREC2", "RREC3", "RREC4")


@lru_cache(maxsize=1)
def _numeric_pipeline_names() -> frozenset[str]:
    """The column names the loader casts to a number, in either file."""
    path = bundled_taxonomy_path()
    return numeric_field_names(load_taxonomy(path), load_taxonomy_formats(path))


@dataclass
class FileFacts:
    """What the pool-level rules (A9 to A11) need from one file."""

    file: FileKind
    cutoffs: dict[str, Counter[date]] = field(default_factory=dict)
    """Valid cut-off dates and their record counts, by source column."""
    records_without_cutoff: int = 0
    unique_identifiers: Counter[str] | None = None
    """Present values of RREL1 / RREC1 with their record counts; None when
    the file has no such column."""
    earliest_origination: date | None = None
    modal_cutoff: date | None = None
    """The cut-off date most records of the file hold (the later date
    on a tie); None when no record has one."""


def _severity(rule_id: str) -> str:
    return RULES[rule_id].severity


def _identifier(column: str) -> pl.Expr:
    """True where the cell is an identifier: not empty, not an ND token."""
    value = pl.col(column)
    return value.is_not_null() & (value.str.strip_chars() != "") & ~value.is_in(sorted(ND_TOKENS))


def _cell_rows(
    raw: RawFile,
    column: ColumnResolution,
    mask: pl.Expr,
    rule_id: str,
    kind: str,
    reason: str,
    expected: str,
    *,
    note: pl.Expr | None = None,
    severity: str | None = None,
) -> pl.DataFrame:
    """One result row per record of `raw` where `mask` holds, on one column."""
    assert raw.frame is not None
    return (
        raw.frame.filter(mask)
        .select(
            pl.lit(raw.file).alias("file"),
            "record",
            "physical_line",
            pl.lit(column.position, dtype=pl.UInt16).alias("column_position"),
            pl.lit(column.header).alias("header"),
            pl.lit(column.field, dtype=pl.String).alias("field"),
            pl.lit(rule_id).alias("rule_id"),
            pl.lit(kind).alias("result_kind"),
            pl.lit(severity or _severity(rule_id)).alias("severity"),
            pl.lit("present").alias("value_state"),
            pl.col(column.frame_column).alias("raw_value"),
            pl.lit(None, dtype=pl.Date).alias("nd4_date"),
            pl.lit(expected).alias("expected"),
            pl.lit(reason).alias("reason"),
            (note if note is not None else pl.lit(None, dtype=pl.String)).alias("note"),
        )
        .cast(RESULT_SCHEMA)  # type: ignore[arg-type]
    )


def _column_row(
    raw_file: str, rule_id: str, kind: str, reason: str, expected: str,
    column: ColumnResolution | None = None, *, field_code: str | None = None,
    raw_value: str | None = None, severity: str | None = None, note: str | None = None,
) -> dict[str, object]:
    """One result row about a column, a file or the pool (no record)."""
    return {
        "file": raw_file,
        "column_position": column.position if column else None,
        "header": column.header if column else None,
        "field": field_code if field_code is not None else (column.field if column else None),
        "rule_id": rule_id, "result_kind": kind, "severity": severity or _severity(rule_id),
        "reason": reason, "expected": expected, "raw_value": raw_value, "note": note,
    }


def _other_record(group: pl.Expr) -> pl.Expr:
    """ "also on record n": another record of the same group."""
    first, last = pl.col("record").min().over(group), pl.col("record").max().over(group)
    other = pl.when(pl.col("record") == first).then(last).otherwise(first)
    return pl.format("also on record {}", other)


def unreadable_rows(file: FileKind, unreadable: Unreadable) -> tuple[pl.DataFrame, list[RuleOutcome]]:
    """A1(a) for a file with no header, A16 for a file that cannot be
    split into records: one row for the file, and one for every record
    that is longer than the header."""
    rule_id = "A1(a)" if unreadable.reason == "no_header" else "A16"
    rows = [_column_row(file, rule_id, "violation", unreadable.reason, unreadable.detail)]
    for long_row in unreadable.long_rows:
        rows.append({
            **_column_row(
                file, "A16", "violation", "record_longer_than_the_header",
                f"{unreadable.header_fields} field(s), as the header has; this record has "
                f"{long_row.fields}",
            ),
            "record": long_row.record, "physical_line": long_row.physical_line,
        })
    outcome = RuleOutcome(rule_id, file, None, applicable=1, evaluated=1, failed=1)
    return results_frame(rows), [outcome]


def _header_rules(raw: RawFile) -> tuple[list[dict[str, object]], list[pl.DataFrame], list[RuleOutcome]]:
    """A1(b), A1(c), A2, A3 and A15: the headers, and a field supplied in
    two columns (specification 5.3)."""
    assert raw.frame is not None
    rows: list[dict[str, object]] = []
    frames: list[pl.DataFrame] = []
    outcomes: list[RuleOutcome] = []

    by_text: dict[str, list[ColumnResolution]] = defaultdict(list)
    by_name: dict[str, list[ColumnResolution]] = defaultdict(list)
    by_field: dict[str, list[ColumnResolution]] = defaultdict(list)
    for column in raw.columns:
        by_text[column.header].append(column)
        by_name[column.pipeline_name].append(column)
        if column.field is not None and column.how in RESOLVED_TO_FIELD:
            by_field[column.field].append(column)

    # A1(c): the same header text twice. The loader's parser renames the
    # later columns, so the pipeline silently uses the first.
    repeated = {text: cols for text, cols in by_text.items() if len(cols) > 1}
    outcome = RuleOutcome("A1(c)", raw.file, None, applicable=len(raw.columns), evaluated=len(raw.columns))
    for text, cols in repeated.items():
        for column in cols:
            outcome.failed += 1
            others = ", ".join(str(c.position) for c in cols if c is not column)
            rows.append(_column_row(
                raw.file, "A1(c)", "violation", "header_text_repeated",
                "each header text once; the pipeline uses the first column of a repeated "
                f"header and carries the others as {clean_name(text)}_duplicated_n",
                column, note=f"also at column {others}",
            ))
    outcomes.append(outcome)

    # A1(b): two columns the pipeline gives one name: it raises.
    outcome = RuleOutcome("A1(b)", raw.file, None, applicable=len(raw.columns), evaluated=len(raw.columns))
    for name, cols in by_name.items():
        if len(cols) > 1:
            for column in cols:
                outcome.failed += 1
                others = ", ".join(str(c.position) for c in cols if c is not column)
                rows.append(_column_row(
                    raw.file, "A1(b)", "violation", "pipeline_name_collision",
                    f"one column named {name} after the pipeline's renaming; the pipeline "
                    "cannot process a file in which two columns get one name",
                    column, note=f"also at column {others}",
                ))
    outcomes.append(outcome)

    # A2: a column the pipeline requires.
    required = REQUIRED_LOAN_COLUMNS if raw.file == "loans" else REQUIRED_PROPERTY_COLUMNS
    by_cleaned = {clean_name(f.name): f.code for f in fields_of(raw.file)}
    for name in required:
        code = by_cleaned.get(name)
        outcome = RuleOutcome("A2", raw.file, code, applicable=1, evaluated=1)
        if name not in by_name:
            outcome.failed = 1
            rows.append(_column_row(
                raw.file, "A2", "violation", "required_column_absent",
                f"a column the pipeline reads as {name}"
                + (f" ({code})" if code else " (the export's Pool_Cutoff_Date)")
                + "; the pipeline cannot process the file without it",
                field_code=code,
            ))
        outcomes.append(outcome)

    # A3: unmapped columns (the inventory shows every column's resolution).
    outcome = RuleOutcome("A3", raw.file, None, applicable=len(raw.columns), evaluated=len(raw.columns))
    for column in raw.columns:
        if column.how == "unmapped":
            outcome.other_kinds["context"] += 1
            rows.append(_column_row(
                raw.file, "A3", "context", "unmapped_column",
                "no taxonomy mapping established; not assessed for compliance", column,
            ))
    outcomes.append(outcome)

    # A15: one field in two columns: equal on every record, or not.
    for code, cols in by_field.items():
        if len(cols) < 2:
            continue
        first = cols[0]
        equal_everywhere = True
        for other in cols[1:]:
            a = pl.col(first.frame_column).fill_null("")
            b = pl.col(other.frame_column).fill_null("")
            differs = a != b
            n = int(raw.frame.select(differs.sum()).item())
            if n == 0:
                continue
            equal_everywhere = False
            expected = (
                f"the same value in both columns of {code}; the acceptance policy would stop "
                "on a field supplied twice with different values"
            )
            frames.append(_cell_rows(
                raw, first, differs, "A15(b)", "violation", "field_twice_values_differ", expected,
                note=pl.format("column {} holds {}", pl.lit(other.position), b),
            ))
            frames.append(_cell_rows(
                raw, other, differs, "A15(b)", "violation", "field_twice_values_differ", expected,
                note=pl.format("column {} holds {}", pl.lit(first.position), a),
            ))
            outcomes.append(RuleOutcome(
                "A15(b)", raw.file, code, applicable=raw.n_records, evaluated=raw.n_records, failed=n,
            ))
        if equal_everywhere:
            positions = ", ".join(str(c.position) for c in cols)
            rows.append(_column_row(
                raw.file, "A15(a)", "violation", "field_twice_values_equal",
                f"the field in one column; it is supplied in columns {positions} with equal "
                "values on every record",
                first,
            ))
            outcomes.append(RuleOutcome("A15(a)", raw.file, code, applicable=1, evaluated=1, failed=1))
    return rows, frames, outcomes


def _loader_rule(raw: RawFile) -> tuple[list[pl.DataFrame], list[RuleOutcome]]:
    """A5: cells the loader cannot convert in a column it reads as a number.

    This is the loader's own test (`io_layer.read_csv`: null tokens removed,
    surrounding whitespace trimmed, a float cast), applied to the raw column,
    so it lists exactly the cells that make the loader stop, in whichever
    column the pipeline names as a numeric field, including one the audit
    calls unmapped. It is not the format rule: `1e3` converts and is a P5
    violation; `abc` is both, and the two defects have separate keys."""
    assert raw.frame is not None
    frames: list[pl.DataFrame] = []
    outcomes: list[RuleOutcome] = []
    numeric = _numeric_pipeline_names()
    for column in raw.columns:
        if column.pipeline_name not in numeric:
            continue
        value = pl.col(column.frame_column)
        read = pl.when(value.is_null() | value.is_in(list(CSV_NA_TOKENS))).then(None).otherwise(value)
        stops = read.is_not_null() & read.str.strip_chars().cast(pl.Float64, strict=False).is_null()
        applicable, failed = raw.frame.select(
            read.is_not_null().sum().alias("n"), stops.sum().alias("bad")
        ).row(0)
        outcomes.append(RuleOutcome(
            "A5", raw.file, column.field, applicable=applicable, evaluated=applicable, failed=failed,
        ))
        if failed:
            frames.append(_cell_rows(
                raw, column, stops, "A5", "violation", "loader_cannot_convert",
                "a number or an ND token the loader recognises; the pipeline cannot process "
                "the file while this value is in a numeric field (the loader stops at it)",
            ))
    return frames, outcomes


def _repair_rule(raw: RawFile) -> tuple[list[pl.DataFrame], list[RuleOutcome]]:
    """A6: cells whose bytes were not valid UTF-8 and were repaired."""
    assert raw.frame is not None
    repaired = raw.encoding_repaired
    if repaired.height == 0:
        return [], []
    frames: list[pl.DataFrame] = []
    outcomes: list[RuleOutcome] = []
    for column in raw.columns:
        mine = repaired.filter(pl.col("column_position") == column.position)
        if mine.height == 0:
            continue
        records = mine.filter(pl.col("record") > 0)["record"].to_list()
        if records:
            frames.append(_cell_rows(
                raw, column, pl.col("record").is_in(records), "A6", "context",
                "bytes_repaired",
                "valid UTF-8; the bytes were repaired as the pipeline repairs them, and the "
                "original bytes are kept in the detail file",
            ))
        if mine.filter(pl.col("record") == 0).height:
            frames.append(results_frame([_column_row(
                raw.file, "A6", "context", "header_bytes_repaired",
                "valid UTF-8 in the header; the bytes were repaired", column,
            )]))
        outcome = RuleOutcome("A6", raw.file, column.field, applicable=raw.n_records, evaluated=raw.n_records)
        outcome.other_kinds["context"] = mine.height
        outcomes.append(outcome)
    return frames, outcomes


def _identifier_rules(raw: RawFile) -> tuple[list[pl.DataFrame], list[RuleOutcome]]:
    """A7, A8, A12 and A13: duplicates, collisions and padding of the
    identifiers. A cell that is empty or an ND token is not an identifier
    and takes no part (the ND rules report it)."""
    assert raw.frame is not None
    frame = raw.frame
    frames: list[pl.DataFrame] = []
    outcomes: list[RuleOutcome] = []

    def outcome(rule_id: str, code: str | None, applicable: int, failed: int) -> None:
        outcomes.append(RuleOutcome(
            rule_id, raw.file, code, applicable=applicable, evaluated=applicable, failed=failed,
        ))

    def count(mask: pl.Expr) -> int:
        return int(frame.select(mask.sum()).item())

    codes = LOAN_IDENTIFIERS if raw.file == "loans" else COLLATERAL_IDENTIFIERS
    present = {code: raw.column_of(code) for code in codes}

    # A13: padding (a), and different raw values equal after trimming (b).
    for code, column in present.items():
        if column is None:
            continue
        value = pl.col(column.frame_column)
        usable = _identifier(column.frame_column)
        n = count(usable)
        padded = usable & (value.str.strip_chars() != value)
        outcome("A13(a)", code, n, count(padded))
        frames.append(_cell_rows(
            raw, column, padded, "A13(a)", "violation", "identifier_padded",
            "the identifier without surrounding whitespace; the pipeline matches "
            "identifiers exactly and does not trim them today (issue #46)",
        ))
        trimmed = value.str.strip_chars()
        written = pl.when(usable).then(value).otherwise(None)
        collides = usable & (written.drop_nulls().n_unique().over(trimmed) > 1)
        outcome("A13(b)", code, n, count(collides))
        frames.append(_cell_rows(
            raw, column, collides, "A13(b)", "violation", "identifiers_equal_after_trimming",
            "identifiers that stay different after trimming; two different values here "
            "become one, so the identity of the record is ambiguous",
        ))

    if raw.file == "loans":
        # A7(a): the same identifier on more than one loan record.
        for code in ("RREL2", "RREL3"):
            column = present[code]
            if column is None:
                continue
            usable = _identifier(column.frame_column)
            group = pl.when(usable).then(pl.col(column.frame_column)).otherwise(None)
            duplicate = usable & (pl.len().over(group) > 1)
            outcome("A7(a)", code, count(usable), count(duplicate))
            frames.append(_cell_rows(
                raw, column, duplicate, "A7(a)", "violation", "identifier_on_several_records",
                "a value on one loan record only", note=_other_record(group),
            ))
        # A12: a value in one loan-identifier column equals a different
        # record's value in the other.
        original, new = present["RREL2"], present["RREL3"]
        if original is not None and new is not None:
            for mine, theirs in ((original, new), (new, original)):
                usable = _identifier(mine.frame_column)
                values = frame.filter(_identifier(theirs.frame_column)).select(
                    pl.col(theirs.frame_column).alias("_k"), pl.col("record").alias("_r")
                )
                hits = (
                    frame.filter(usable)
                    .select("record", pl.col(mine.frame_column).alias("_k"))
                    .join(values, on="_k")
                    .filter(pl.col("record") != pl.col("_r"))
                    .group_by("record").agg(pl.col("_r").min().alias("_other"))
                )
                records = hits["record"].to_list()
                outcome("A12", mine.field, count(usable), len(records))
                if records:
                    others = dict(zip(records, hits["_other"].to_list(), strict=True))
                    frames.append(_cell_rows(
                        raw, mine, pl.col("record").is_in(records), "A12", "violation",
                        "identifier_collision",
                        f"a value that is not the {theirs.field} of a different loan record; the "
                        "pipeline halts or mismatches on such a pool today (issue #45)",
                        note=pl.format(
                            "equals {} of record {}", pl.lit(theirs.field),
                            pl.col("record").replace_strict(others, return_dtype=pl.UInt32),
                        ),
                    ))
    else:
        # A8(a): the same (RREC2, RREC4) pair on more than one collateral record.
        loan, collateral = present["RREC2"], present["RREC4"]
        if loan is not None and collateral is not None:
            usable = _identifier(loan.frame_column) & _identifier(collateral.frame_column)
            pair = pl.when(usable).then(
                pl.struct(pl.col(loan.frame_column), pl.col(collateral.frame_column))
            ).otherwise(None)
            duplicate = usable & (pl.len().over(pair) > 1)
            outcome("A8(a)", "RREC4", count(usable), count(duplicate))
            frames.append(_cell_rows(
                raw, collateral, duplicate, "A8(a)", "violation", "pair_on_several_records",
                "a (RREC2, RREC4) pair on one collateral record only", note=_other_record(pair),
            ))

    # A7(b) / A8(b): whole-record duplicates. Candidates are found by a
    # hash of the record and confirmed cell by cell.
    rule_id = "A7(b)" if raw.file == "loans" else "A8(b)"
    body = [c.frame_column for c in raw.columns]
    if body:
        candidates = frame.filter(frame.select(body).hash_rows().is_duplicated())
        whole = pl.struct([pl.col(c).fill_null("") for c in body])
        duplicates = candidates.filter(pl.len().over(whole) > 1)
        outcome(rule_id, None, raw.n_records, duplicates.height)
        if duplicates.height:
            frames.append(
                duplicates.select(
                    pl.lit(raw.file).alias("file"), "record", "physical_line",
                    pl.lit(None, dtype=pl.UInt16).alias("column_position"),
                    pl.lit(None, dtype=pl.String).alias("header"),
                    pl.lit(None, dtype=pl.String).alias("field"),
                    pl.lit(rule_id).alias("rule_id"), pl.lit("violation").alias("result_kind"),
                    pl.lit(_severity(rule_id)).alias("severity"),
                    pl.lit(None, dtype=pl.String).alias("value_state"),
                    pl.lit(None, dtype=pl.String).alias("raw_value"),
                    pl.lit(None, dtype=pl.Date).alias("nd4_date"),
                    pl.lit("each record once; every cell of this record equals another record's").alias("expected"),
                    pl.lit("whole_record_duplicate").alias("reason"),
                    _other_record(whole).alias("note"),
                )
                .cast(RESULT_SCHEMA)  # type: ignore[arg-type]
            )
    return frames, outcomes


def _file_facts(
    raw: RawFile, pool: PoolFacts, modal: date | None
) -> tuple[FileFacts, list[pl.DataFrame], list[RuleOutcome]]:
    """The file's cut-off dates and unique identifiers, and A9's rows for
    the records of this file. `modal` is the loans file's most frequent
    cut-off date, given for the collateral file."""
    assert raw.frame is not None
    frame = raw.frame
    facts = FileFacts(raw.file)
    frames: list[pl.DataFrame] = []
    outcomes: list[RuleOutcome] = []

    sources: list[tuple[str, ColumnResolution]] = []
    if raw.file == "loans" and (column := raw.column_of("RREL6")) is not None:
        sources.append(("RREL6", column))
    if (column := raw.extension("pool_cutoff_date")) is not None:
        sources.append((f"{column.header} ({raw.file} file)", column))
    for name, column in sources:
        dates = frame.select(strict_date(pl.col(column.frame_column)).alias("d"))["d"].drop_nulls()
        facts.cutoffs[name] = Counter(dict(dates.value_counts().iter_rows()))

    cut, _ = cutoff_columns(raw, pool)
    if sources:
        facts.records_without_cutoff = cut["_cut"].null_count()
        held = cut["_cut"].drop_nulls().value_counts().sort(["count", "_cut"], descending=True)
        facts.modal_cutoff = held["_cut"][0] if held.height else None
        if modal is None:
            modal = facts.modal_cutoff
        with_cut = frame.with_columns(cut["_cut"], cut["_cut_reason"])
        column = sources[0][1]
        missing = pl.col("_cut").is_null()
        differs = (
            pl.col("_cut").is_not_null() & (pl.col("_cut") != modal) if modal is not None else pl.lit(False)
        )
        failed = int(with_cut.select((missing | differs).sum()).item())
        outcomes.append(RuleOutcome(
            "A9", raw.file, None, applicable=raw.n_records, evaluated=raw.n_records, failed=failed,
        ))
        raw_with = replace(raw, frame=with_cut)
        frames.append(_cell_rows(
            raw_with, column, missing, "A9", "violation", "cutoff_not_established_on_record",
            "one valid cut-off date on the record; the acceptance policy would stop on a pool "
            "without a single cut-off date",
            note=pl.col("_cut_reason").fill_null(CUTOFF_NOT_ESTABLISHED),
        ))
        if modal is not None:
            frames.append(_cell_rows(
                raw_with, column, differs, "A9", "violation", "cutoff_differs_from_the_pool",
                f"{modal.isoformat()}, the cut-off date most loan records hold; the acceptance "
                "policy would stop on a pool with more than one cut-off date",
            ))

    code = "RREL1" if raw.file == "loans" else "RREC1"
    if (column := raw.column_of(code)) is not None:
        values = frame.filter(_identifier(column.frame_column))[column.frame_column]
        facts.unique_identifiers = Counter(dict(values.value_counts().iter_rows()))
    if raw.file == "loans" and (column := raw.column_of("RREL23")) is not None:
        facts.earliest_origination = frame.select(strict_date(pl.col(column.frame_column)).min()).item()
    return facts, frames, outcomes


def file_rules(
    raw: RawFile, pool: PoolFacts, modal: date | None = None
) -> tuple[pl.DataFrame, list[RuleOutcome], FileFacts]:
    """Every family A rule that one readable file answers on its own."""
    rows, frames, outcomes = _header_rules(raw)
    outcome = RuleOutcome("A1(a)", raw.file, None, applicable=1, evaluated=1)
    if raw.n_records == 0:
        outcome.failed = 1
        rows.append(_column_row(
            raw.file, "A1(a)", "violation", "no_data_records",
            "at least one record after the header; the pipeline cannot process a file with none",
        ))
    outcomes.append(outcome)
    for more_frames, more_outcomes in (_loader_rule(raw), _repair_rule(raw), _identifier_rules(raw)):
        frames.extend(more_frames)
        outcomes.extend(more_outcomes)
    facts, fact_frames, fact_outcomes = _file_facts(raw, pool, modal)
    frames.extend(fact_frames)
    outcomes.extend(fact_outcomes)
    return concat_results([results_frame(rows), *frames]), outcomes, facts


def pool_rules(
    facts: Sequence[FileFacts], pool: PoolFacts, as_of: date
) -> tuple[pl.DataFrame, list[RuleOutcome]]:
    """A9 (the pool-level row), A10 and A11: facts about the two files
    together."""
    rows: list[dict[str, object]] = []
    outcomes: list[RuleOutcome] = []

    # A9: every distinct cut-off date with its source.
    by_date: dict[date, list[str]] = defaultdict(list)
    for fact in facts:
        for source, counts in fact.cutoffs.items():
            for day, n in counts.items():
                by_date[day].append(f"{source}: {n} record(s)")
    without = sum(f.records_without_cutoff for f in facts)
    outcome = RuleOutcome("A9", "pool", None, applicable=1, evaluated=1)
    if len(by_date) > 1 or without:
        outcome.failed = 1
        listed = "; ".join(f"{day.isoformat()} ({', '.join(src)})" for day, src in sorted(by_date.items()))
        rows.append(_column_row(
            "pool", "A9", "violation", "more_than_one_cutoff_or_missing",
            "one cut-off date across RREL6, the export's Pool_Cutoff_Date and the collateral "
            f"file; found {listed or 'none'}"
            + (f"; {without} record(s) without an established cut-off" if without else ""),
        ))
    outcomes.append(outcome)

    # A10(a): the cut-off against the assessment date. A10(b): against the
    # earliest origination date.
    outcome_a = RuleOutcome("A10(a)", "pool", "RREL6", applicable=len(by_date), evaluated=len(by_date))
    for day in sorted(by_date):
        if day > as_of:
            outcome_a.failed += 1
            rows.append(_column_row(
                "pool", "A10(a)", "violation", "cutoff_after_the_assessment_date",
                f"a cut-off date before {as_of.isoformat()}, the date of this assessment",
                field_code="RREL6", raw_value=day.isoformat(),
            ))
        elif day == as_of:
            outcome_a.other_kinds["question"] += 1
            rows.append(_column_row(
                "pool", "A10(a)", "question", "cutoff_equals_the_assessment_date",
                "a cut-off date in the past; this one is the date of the assessment itself, "
                "and the rule's wording does not settle whether the same day is in the past",
                field_code="RREL6", raw_value=day.isoformat(),
            ))
    outcomes.append(outcome_a)
    earliest = next((f.earliest_origination for f in facts if f.earliest_origination), None)
    outcome_b = RuleOutcome("A10(b)", "pool", "RREL6")
    if pool.loans_cutoff is not None and earliest is not None:
        outcome_b.applicable = outcome_b.evaluated = 1
        if pool.loans_cutoff < earliest:
            outcome_b.other_kinds["question"] += 1
            rows.append(_column_row(
                "pool", "A10(b)", "question", "cutoff_before_every_origination_date",
                f"a cut-off date not before the earliest origination date ({earliest.isoformat()})",
                field_code="RREL6", raw_value=pool.loans_cutoff.isoformat(),
            ))
    outcomes.append(outcome_b)

    # A11: one unique identifier in each file, and the same in both.
    values: dict[str, Counter[str]] = {}
    for fact in facts:
        code = "RREL1" if fact.file == "loans" else "RREC1"
        outcome = RuleOutcome("A11", fact.file, code)
        if fact.unique_identifiers is not None:
            values[code] = fact.unique_identifiers
            outcome.applicable = outcome.evaluated = 1
            if len(fact.unique_identifiers) > 1:
                outcome.failed = 1
                rows.append(_column_row(
                    fact.file, "A11", "violation", "unique_identifier_not_constant",
                    f"one value of {code} in the file; found {len(fact.unique_identifiers)} "
                    "different values",
                    field_code=code,
                ))
        outcomes.append(outcome)
    outcome = RuleOutcome("A11", "pool", None)
    if len(values) == 2 and all(values.values()):
        outcome.applicable = outcome.evaluated = 1
        if set(values["RREL1"]) != set(values["RREC1"]):
            outcome.failed = 1
            rows.append(_column_row(
                "pool", "A11", "violation", "unique_identifiers_differ_between_the_files",
                "RREC1 equal to RREL1; the two files hold different unique identifiers",
            ))
    outcomes.append(outcome)
    return results_frame(rows), outcomes


__all__ = ["FileFacts", "file_rules", "pool_rules", "unreadable_rows"]
