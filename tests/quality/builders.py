"""Test data for the data quality report, built in code.

`Pair()` is a small compliant pair of files: every one of the 107 fields,
a currency companion for every amount, the export's cut-off column, and
values that break no rule. Tests change named cells and assess the result.
Nothing here comes from a real pool.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from esma_milan.quality.audit_read import RawFile, read_file
from esma_milan.quality.catalogue import FIELDS, FieldEntry, FileKind, fields_of
from esma_milan.quality.column_rules import ColumnAssessment, assess_columns
from esma_milan.quality.column_rules.context import PoolFacts, pool_facts
from esma_milan.quality.refdata import load_reference_data

CUTOFF = "2024-06-30"
REGION = "NL329"  # a NUTS 2021 level 3 code


def make_lei(body: str) -> str:
    """An LEI with valid ISO 17442 check digits for an 18-character body."""
    assert len(body) == 18
    number = int("".join(str(int(ch, 36)) for ch in body + "00"))
    return f"{body}{98 - number % 97:02d}"


LEI = make_lei("5299000TESTBANK001")
UNIQUE_IDENTIFIER = f"{LEI}N202401"

_SPECIFIC: dict[str, str] = {
    "RREL1": UNIQUE_IDENTIFIER, "RREC1": UNIQUE_IDENTIFIER,
    "RREL6": CUTOFF, "RREL7": "2020-03-01", "RREL8": "ND5", "RREL9": "ND5",
    "RREL12": "2021", "RREL22": "HELP TO BUY", "RREL23": "2019-05-10",
    "RREL24": "2049-05-10", "RREL59": "10", "RREL63": "ND5", "RREL65": "ND5",
    "RREL66": "ND5", "RREL68": "0", "RREL69": "PERF", "RREL70": "ND5",
    "RREL71": "ND5", "RREL72": "ND5", "RREL73": "ND5", "RREL74": "ND5",
    "RREL79": "EXAMPLE ORIGINAL LENDER NV", "RREL82": "EXAMPLE BANK NV",
    "RREC5": "RBLD", "RREC8": "1", "RREC11": "EXAMPLE EPC REGISTER",
    "RREC15": "2023-11-20", "RREC19": "2019-04-02", "RREC20": "ND5", "RREC21": "ND5",
}


def _default(entry: FieldEntry) -> str:
    if entry.code in _SPECIFIC:
        return _SPECIFIC[entry.code]
    kind = entry.kind
    if kind == "MONETARY":
        return "1000.50"
    if kind == "PERCENTAGE":
        return "2.5"
    if kind == "INTEGER":
        return "12"
    if kind == "DATE":
        return "2026-01-01"
    if kind == "BOOLEAN":
        return "Y"
    if kind == "LIST":
        assert entry.codes is not None
        return sorted(entry.codes)[0]
    if kind == "NUTS":
        return REGION
    if kind == "COUNTRY":
        return "NL"
    if kind == "CURRENCY":
        return "EUR"
    if kind == "LEI":
        return LEI
    return "TEXT"


def _record(file: FileKind, index: int) -> dict[str, str]:
    row = {"Pool_Cutoff_Date": CUTOFF}
    for entry in fields_of(file):
        row[entry.code] = _default(entry)
        if entry.kind == "MONETARY":
            row[f"{entry.code}_CURRENCY"] = "EUR"
    if file == "loans":
        row.update(RREL2=f"LOAN00{index}", RREL3=f"LOAN00{index}", RREL4=f"OBLIGOR00{index}",
                   RREL5=f"OBLIGOR00{index}", RREL30=f"{150000 + index}.00")
    else:
        row.update(RREC2=f"LOAN00{index}", RREC3=f"PROPERTY00{index}", RREC4=f"PROPERTY00{index}",
                   RREC13=f"{250000 + index}.00")
    return row


@dataclass
class Pair:
    """A pair of files as lists of rows (header -> cell)."""

    loans: list[dict[str, str]] = field(default_factory=lambda: [_record("loans", i) for i in (1, 2)])
    collateral: list[dict[str, str]] = field(
        default_factory=lambda: [_record("collateral", i) for i in (1, 2)]
    )

    def rows(self, file: FileKind) -> list[dict[str, str]]:
        return self.loans if file == "loans" else self.collateral

    def set(self, header: str, value: str, *, record: int | None = None) -> Pair:
        """Set `header` on one record (1-based) or on every record. The
        file is taken from the header (RREL/RREC); other headers are set
        in both files."""
        files: tuple[FileKind, ...] = (
            ("loans",) if header.startswith("RREL")
            else ("collateral",) if header.startswith("RREC") or header.startswith("OPT")
            else ("loans", "collateral")
        )
        for file in files:
            rows = self.rows(file)
            for i, row in enumerate(rows, start=1):
                if record is None or record == i:
                    row[header] = value
        return self

    def drop(self, header: str, file: FileKind | None = None) -> Pair:
        kinds: tuple[FileKind, ...] = ("loans", "collateral")
        for kind in kinds:
            if file in (None, kind):
                for row in self.rows(kind):
                    row.pop(header, None)
        return self

    def write(self, folder: Path) -> tuple[Path, Path]:
        paths = []
        names: tuple[tuple[FileKind, str], ...] = (
            ("loans", "loans.csv"), ("collateral", "collaterals.csv"),
        )
        for file, name in names:
            rows = self.rows(file)
            path = folder / name
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh, lineterminator="\n")
                headers = list(rows[0])
                writer.writerow(headers)
                writer.writerows([row.get(h, "") for h in headers] for row in rows)
            paths.append(path)
        return paths[0], paths[1]


@dataclass
class Assessed:
    raw: RawFile
    pool: PoolFacts
    assessment: ColumnAssessment

    def found(
        self, rule_id: str, field_code: str | None = None, kind: str | None = None
    ) -> list[dict[str, object]]:
        """Result rows of one rule (optionally one field and one kind)."""
        frame = self.assessment.results.filter(pl.col("rule_id") == rule_id)
        if field_code is not None:
            frame = frame.filter(pl.col("field") == field_code)
        if kind is not None:
            frame = frame.filter(pl.col("result_kind") == kind)
        return frame.to_dicts()

    def rules_on(self, field_code: str) -> set[tuple[str, str]]:
        """(rule, result kind) of every row on one field, apart from the
        P9 context row of a constant column (the pair has two records, so
        most of its columns are constant)."""
        frame = self.assessment.results.filter(
            (pl.col("field") == field_code) & (pl.col("rule_id") != "P9")
        )
        return set(zip(frame["rule_id"].to_list(), frame["result_kind"].to_list(), strict=True))

    def rows_beyond_cardinality(self) -> list[dict[str, object]]:
        return self.assessment.results.filter(pl.col("rule_id") != "P9").to_dicts()

    def outcome(self, rule_id: str, field_code: str) -> tuple[int, int, int, int]:
        """(applicable, evaluated, failed, not evaluable) summed over the
        field's columns."""
        hits = [o for o in self.assessment.outcomes if o.rule_id == rule_id and o.field == field_code]
        return (
            sum(o.applicable for o in hits), sum(o.evaluated for o in hits),
            sum(o.failed for o in hits), sum(o.not_evaluable for o in hits),
        )


def assess(pair: Pair, folder: Path, file: FileKind) -> Assessed:
    """Read the pair and run the per-column rules on one of its files."""
    loans_path, collateral_path = pair.write(folder)
    loans = read_file(loans_path, "loans")
    pool = pool_facts(loans)
    raw = loans if file == "loans" else read_file(collateral_path, "collateral")
    return Assessed(raw, pool, assess_columns(raw, pool, ref=load_reference_data()))


__all__ = ["CUTOFF", "FIELDS", "LEI", "REGION", "UNIQUE_IDENTIFIER", "Assessed", "Pair", "assess", "make_lei"]
