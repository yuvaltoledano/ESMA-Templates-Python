"""The on-disk store of the row-level detail."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from esma_milan.quality import assess_files
from esma_milan.quality.store import MARKER, PREFIX, ResultStore, result_counts
from tests.quality.builders import Pair


def _poor_pair() -> Pair:
    pair = Pair().set("RREL43", "N/A").set("RREL30", "abc", record=1).set("RREC13", "-5", record=2)
    return pair.set("RREL82", "Lower Case Name")


def _sorted(frame: pl.DataFrame) -> pl.DataFrame:
    text = frame.with_columns(pl.col(pl.Categorical).cast(pl.String))
    return text.sort("file", "record", "column_position", "rule_id", "reason", nulls_last=True)


def test_rows_written_equal_rows_read_and_nothing_stays_in_memory(tmp_path: Path) -> None:
    loans, collaterals = _poor_pair().write(tmp_path)
    in_memory = assess_files(loans, collaterals)
    store = ResultStore.create(tmp_path / "out")
    stored = assess_files(loans, collaterals, store=store)

    assert stored.store_root == store.root
    # With a store the report object holds no result row and no per-record frame.
    assert stored.loans.results.height == 0 and stored.collateral.results.height == 0
    assert stored.pool_results.height == 0
    assert stored.loans.records is None and stored.collateral.records is None
    assert in_memory.loans.results.height > 0 and in_memory.loans.records is not None

    assert _sorted(stored.results).equals(_sorted(in_memory.results))
    assert stored.results.schema == in_memory.results.schema
    for ours, theirs in (
        (stored.records("loans"), in_memory.records("loans")),
        (stored.records("collateral"), in_memory.records("collateral")),
    ):
        assert ours is not None and theirs is not None and ours.equals(theirs)
    key = ["file", "field", "rule_id", "result_kind", "severity", "reason"]
    assert stored.counts.sort(key).equals(result_counts(in_memory.results).sort(key))
    assert stored.outcomes == in_memory.outcomes
    assert [p.states for p in stored.loans.profiles] == [p.states for p in in_memory.loans.profiles]


def test_one_part_per_column_and_the_index_counts_them(tmp_path: Path) -> None:
    loans, collaterals = _poor_pair().write(tmp_path)
    store = ResultStore.create(tmp_path / "out")
    assess_files(loans, collaterals, store=store)
    names = [p.stem for p in store.result_parts()]
    assert "loans-columns" in names and "collateral-columns" in names
    assert any(n.startswith("loans-0") for n in names)
    assert store.index["rows"].sum() == store.scan_results().select(pl.len()).collect().item()
    for part in store.result_parts():
        records = pl.read_parquet(part)["record"].drop_nulls().to_list()
        assert records == sorted(records)


def test_create_is_unique_to_the_run_and_two_stores_live_side_by_side(tmp_path: Path) -> None:
    first, second = ResultStore.create(tmp_path), ResultStore.create(tmp_path)
    assert first.root != second.root
    assert first.root.parent == second.root.parent == tmp_path
    assert first.root.name.startswith(PREFIX)
    assert (first.root / MARKER).read_text(encoding="utf-8").strip() == str(os.getpid())
    first.write_results("x", _result_rows())
    # Creating the second did not remove the first (this process is running).
    assert first.scan_results().select(pl.len()).collect().item() == 1
    second.remove()
    assert not second.root.exists() and first.root.exists()
    first.remove()
    assert list(tmp_path.iterdir()) == []


def _result_rows() -> pl.DataFrame:
    from esma_milan.quality.result import results_frame

    return results_frame([{"file": "loans", "field": "RREL1", "rule_id": "P1",
                           "result_kind": "violation", "severity": "Warn", "reason": "field_absent"}])


def _dead_process_id() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def test_a_stale_marked_directory_is_removed_and_an_unmarked_one_is_left_alone(tmp_path: Path) -> None:
    stale = tmp_path / f"{PREFIX}stale"
    stale.mkdir()
    (stale / MARKER).write_text(f"{_dead_process_id()}\n", encoding="utf-8")
    (stale / "results").mkdir()
    alive = tmp_path / f"{PREFIX}alive"
    alive.mkdir()
    (alive / MARKER).write_text(f"{os.getpid()}\n", encoding="utf-8")
    unmarked = tmp_path / f"{PREFIX}not-ours"
    unmarked.mkdir()
    (unmarked / "keep.txt").write_text("someone else's", encoding="utf-8")
    garbled = tmp_path / f"{PREFIX}garbled"
    garbled.mkdir()
    (garbled / MARKER).write_text("not a number", encoding="utf-8")

    store = ResultStore.create(tmp_path)
    assert not stale.exists()  # its process is gone
    assert alive.exists()  # its process is running
    assert (unmarked / "keep.txt").exists()  # never one without a marker
    assert garbled.exists()
    store.remove()


def test_falls_back_to_the_temporary_directory_and_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocked = tmp_path / "a-file"
    blocked.write_text("x", encoding="utf-8")  # a file where the directory should be
    store = ResultStore.create(blocked / "deal")
    assert blocked.read_text(encoding="utf-8") == "x"
    assert tmp_path not in store.root.parents
    assert (store.root / MARKER).is_file()
    store.remove()
    assert not store.root.exists()

    def no_room(*_: object, **__: object) -> str:
        raise OSError("no space")

    monkeypatch.setattr("esma_milan.quality.store.tempfile.mkdtemp", no_room)
    with pytest.raises(OSError, match="no space"):
        ResultStore.create(tmp_path / "deal")


def test_open_needs_the_marker_and_remove_only_removes_a_store(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ResultStore.open(tmp_path)
    (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
    ResultStore(tmp_path).remove()  # no marker: nothing happens
    assert (tmp_path / "keep.txt").exists()
    store = ResultStore.create(tmp_path / "deal")
    assert ResultStore.open(store.root).root == store.root
    store.remove()
