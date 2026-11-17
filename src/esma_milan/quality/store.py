"""The on-disk store of the data quality report's row-level detail.

Mirrors nothing in the R reference (specification 10.3).

The detail (one row per finding and record) can run to millions of rows.
It is written to a working directory as Parquet parts as soon as each part
exists, so that no file's result rows are in memory while the pipeline
runs and no exporter holds the whole detail at once:

    results/<part>.parquet   result rows in the one schema of `result.py`,
                             sorted by record: one part per assessed column,
                             one for a file's column-level rows, one per
                             family of file-level rules, one for the pool
    index/<part>.parquet     the counts of that part (`COUNT_KEYS`: rows and
                             distinct records); the summary and the
                             inventories are built from these
    records/<file>.parquet   one narrow row per record (identifiers, labels)
    links.parquet            collateral record, loan record, convention, match
    repairs/<file>.parquet   cells whose bytes were repaired

The working directory is unique to the run. It holds a marker file with the
process id of the run that made it. A run removes another run's marked
directory only when that process is no longer running, and never touches a
directory without the marker. The store holds raw cell values of the
user's files; the command line removes it at the end of the run, also after
a failure and after Ctrl-C.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import polars as pl

from esma_milan.quality.catalogue import FileKind
from esma_milan.quality.result import RESULT_SCHEMA, empty_results

MARKER = ".esma-milan-dq-work"
PREFIX = ".dq-work-"

ROW_GROUP_ROWS = 50_000

COUNT_KEYS: tuple[str, ...] = ("file", "field", "rule_id", "result_kind", "severity", "reason")
COUNT_SCHEMA: dict[str, pl.DataType] = {
    **{key: pl.String() for key in COUNT_KEYS},
    "rows": pl.UInt32(),
    "records": pl.UInt32(),
    "column_level": pl.Boolean(),
}


def result_counts(rows: pl.DataFrame) -> pl.DataFrame:
    """Rows and distinct records per file, field, rule, result kind,
    severity and reason. `column_level` is true for results about a whole
    column or the pool (no record)."""
    if rows.height == 0:
        return pl.DataFrame(schema=COUNT_SCHEMA)
    return (
        rows.group_by([pl.col(key).cast(pl.String) for key in COUNT_KEYS], maintain_order=True)
        .agg(
            pl.len().cast(pl.UInt32).alias("rows"),
            pl.col("record").drop_nulls().n_unique().cast(pl.UInt32).alias("records"),
            pl.col("record").is_null().all().alias("column_level"),
        )
        .cast(COUNT_SCHEMA)  # type: ignore[arg-type]
    )


def _process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def _remove_stale(parent: Path) -> None:
    """Remove working directories a dead run left in `parent`: only those
    with this application's marker, and only when the process named in the
    marker is no longer running."""
    for candidate in parent.glob(f"{PREFIX}*"):
        marker = candidate / MARKER
        if not candidate.is_dir() or not marker.is_file():
            continue
        try:
            pid = int(marker.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        if not _process_is_running(pid):
            shutil.rmtree(candidate, ignore_errors=True)


class ResultStore:
    """Parquet parts in a working directory this run created."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @classmethod
    def create(cls, parent: Path) -> ResultStore:
        """A new, empty store in a directory unique to this run: inside
        `parent` (beside the files it becomes) or, if that cannot be
        written, in the system temporary directory. Raises OSError when
        neither can hold it."""
        try:
            parent.mkdir(parents=True, exist_ok=True)
            _remove_stale(parent)
            root = Path(tempfile.mkdtemp(prefix=PREFIX, dir=parent))
        except OSError:
            root = Path(tempfile.mkdtemp(prefix=PREFIX))
        (root / MARKER).write_text(f"{os.getpid()}\n", encoding="utf-8")
        for name in ("results", "index", "records", "repairs"):
            (root / name).mkdir()
        return cls(root)

    @classmethod
    def open(cls, root: Path) -> ResultStore:
        """A store another process of this run is writing or has written."""
        if not (root / MARKER).is_file():
            raise FileNotFoundError(f"not a data quality working directory: {root}")
        return cls(root)

    # -- writing ------------------------------------------------------------

    def write_results(self, part: str, rows: pl.DataFrame) -> None:
        """Write one part of result rows, sorted by record, and its counts.
        An empty part writes nothing, and removes an earlier part of that
        name."""
        if rows.height == 0:
            for folder in ("results", "index"):
                (self.root / folder / f"{part}.parquet").unlink(missing_ok=True)
            return
        # Small row groups, so that a reader asking for a range of records
        # (the exporters do) decodes the groups in that range and not the part.
        rows.sort("record", "column_position", "rule_id", nulls_last=True).write_parquet(
            self.root / "results" / f"{part}.parquet", row_group_size=ROW_GROUP_ROWS
        )
        result_counts(rows).write_parquet(self.root / "index" / f"{part}.parquet")

    def write_records(self, file: FileKind, frame: pl.DataFrame) -> None:
        frame.write_parquet(self.root / "records" / f"{file}.parquet")

    def write_links(self, frame: pl.DataFrame) -> None:
        frame.write_parquet(self.root / "links.parquet")

    def write_repairs(self, file: FileKind, frame: pl.DataFrame) -> None:
        frame.write_parquet(self.root / "repairs" / f"{file}.parquet")

    # -- reading ------------------------------------------------------------

    def result_parts(self) -> list[Path]:
        return sorted((self.root / "results").glob("*.parquet"))

    def scan_results(self) -> pl.LazyFrame:
        """Every result row, lazily. Callers aggregate or stream; nothing
        here loads the detail."""
        parts = self.result_parts()
        if not parts:
            return empty_results().lazy()
        return pl.concat([pl.scan_parquet(part) for part in parts]).cast(RESULT_SCHEMA)  # type: ignore[arg-type]

    @property
    def index(self) -> pl.DataFrame:
        """The counts of every part written so far."""
        parts = sorted((self.root / "index").glob("*.parquet"))
        if not parts:
            return pl.DataFrame(schema=COUNT_SCHEMA)
        return pl.concat([pl.read_parquet(part) for part in parts])

    def has_records(self, file: FileKind) -> bool:
        return (self.root / "records" / f"{file}.parquet").is_file()

    def scan_records(self, file: FileKind) -> pl.LazyFrame:
        return pl.scan_parquet(self.root / "records" / f"{file}.parquet")

    def has_links(self) -> bool:
        return (self.root / "links.parquet").is_file()

    def scan_links(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.root / "links.parquet")

    def scan_repairs(self, file: FileKind) -> pl.LazyFrame | None:
        path = self.root / "repairs" / f"{file}.parquet"
        return pl.scan_parquet(path) if path.is_file() else None

    # -- removal ------------------------------------------------------------

    def remove(self) -> None:
        """Remove the working directory and everything in it."""
        if (self.root / MARKER).is_file():
            shutil.rmtree(self.root, ignore_errors=True)
