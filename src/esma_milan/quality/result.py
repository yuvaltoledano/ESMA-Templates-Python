"""Result rows and check outcomes of the data quality report.

Mirrors nothing in the R reference (specification 3.2, 3.4, 10.3).

A rule never returns Python objects per record. It yields, once per
distinct combination of the values it reads, `Finding`s and `Mark`s; the
engine joins them back to the records, so every affected record gets its
row and nothing is sampled. All rules share one row schema
(`RESULT_SCHEMA`). Basis, source and defect key are not repeated on the
rows: they come from the catalogue by `rule_id`.

`RuleOutcome` carries the four coverage counts of specification 3.4 for
one rule on one field: applicable, evaluated, failed and not evaluable
(with reasons), and the counts of the other result kinds.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import date

import polars as pl

RESULT_SCHEMA: dict[str, pl.DataType] = {
    "file": pl.Categorical(),
    "record": pl.UInt32(),
    "physical_line": pl.UInt32(),
    "column_position": pl.UInt16(),
    "header": pl.Categorical(),
    "field": pl.Categorical(),
    "rule_id": pl.Categorical(),
    "result_kind": pl.Categorical(),
    "severity": pl.Categorical(),
    "value_state": pl.Categorical(),
    "raw_value": pl.String(),
    "nd4_date": pl.Date(),
    "expected": pl.Categorical(),
    "reason": pl.Categorical(),
    "note": pl.Categorical(),
}
"""`record` and `column_position` are 1-based. `record` is null for a
result about a whole column or the pool; `column_position` is null for an
absent field. The columns that repeat are categorical, so a file with many
findings per record costs little more than its raw values."""


@dataclass(frozen=True)
class Finding:
    """One result a rule emits for the records sharing a set of values."""

    rule_id: str
    result_kind: str
    reason: str
    expected: str | None = None
    note: str | None = None
    severity: str | None = None
    """Overrides the rule's catalogue severity for this row (P3g, P8, P12)."""
    nd4_date: date | None = None


@dataclass(frozen=True)
class Mark:
    """The rule applies to these records; `not_evaluable` says why it
    could not be evaluated on them, if it could not."""

    rule_id: str
    not_evaluable: str | None = None


@dataclass
class RuleOutcome:
    """Coverage of one rule on one field of one file (specification 3.4)."""

    rule_id: str
    file: str
    field: str | None
    applicable: int = 0
    evaluated: int = 0
    failed: int = 0
    not_evaluable: int = 0
    not_evaluable_reasons: Counter[str] = dataclass_field(default_factory=Counter)
    other_kinds: Counter[str] = dataclass_field(default_factory=Counter)

    @property
    def label(self) -> str:
        return outcome_label(self.applicable, self.evaluated, self.failed, self.not_evaluable)

    def add(self, other: RuleOutcome) -> None:
        self.applicable += other.applicable
        self.evaluated += other.evaluated
        self.failed += other.failed
        self.not_evaluable += other.not_evaluable
        self.not_evaluable_reasons.update(other.not_evaluable_reasons)
        self.other_kinds.update(other.other_kinds)


def outcome_label(applicable: int, evaluated: int, failed: int, not_evaluable: int) -> str:
    """The one-word label derived from the four counts."""
    if applicable == 0:
        return "not applicable"
    if evaluated == 0:
        return "not evaluated"
    if failed and not_evaluable:
        return "findings and partly evaluated"
    if failed:
        return "findings"
    if not_evaluable:
        return "partly evaluated"
    return "passed"


def empty_results() -> pl.DataFrame:
    return pl.DataFrame(schema=RESULT_SCHEMA)


def results_frame(rows: Iterable[Mapping[str, object]]) -> pl.DataFrame:
    """Rows about a column or the pool (no record) as a result frame."""
    data = [{name: row.get(name) for name in RESULT_SCHEMA} for row in rows]
    if not data:
        return empty_results()
    plain = {name: (pl.String() if dtype == pl.Categorical else dtype) for name, dtype in RESULT_SCHEMA.items()}
    return pl.DataFrame(data, schema=plain).cast(RESULT_SCHEMA)  # type: ignore[arg-type]


def concat_results(frames: Iterable[pl.DataFrame]) -> pl.DataFrame:
    kept = [f for f in frames if f.height]
    return pl.concat(kept) if kept else empty_results()
