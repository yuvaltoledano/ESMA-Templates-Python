"""The audit read against the loader: the same records, in the same order."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from esma_milan.io_layer.read_csv import read_and_clean
from esma_milan.quality.audit_read import read_file
from esma_milan.runner import run_pipeline

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

READABLE: dict[str, tuple[bytes, int]] = {
    "plain": (b"a,b,c\n1,x,3\n2,y,4\n", 2),
    "blank line between records": (b"a,b,c\n1,x,3\n\n2,y,4\n", 3),
    "trailing blank lines": (b"a,b,c\n1,x,3\n2,y,4\n\n\n", 4),
    "no final newline": (b"a,b,c\n1,x,3\n2,y,4", 2),
    "quoted line break": (b'a,b,c\n1,"x\nz",3\n2,y,4\n', 2),
    "short row": (b"a,b,c\n1,x\n2,y,4\n", 2),
    "blank lines before the header": (b"\n\na,b,c\n1,x,3\n2,y,4\n", 2),
    "byte-order mark": (b"\xef\xbb\xbfa,b,c\n1,x,3\n2,y,4\n", 2),
    "CRLF with blank lines": (b"a,b,c\r\n1,x,3\r\n\r\n2,y,4\r\n\r\n", 4),
    "whitespace-only line": (b"a,b,c\n1,x,3\n   \n2,y,4\n", 3),
    "commas only": (b"a,b,c\n1,x,3\n,,\n2,y,4\n", 3),
    "repeated header": (b"a,b,a\n1,x,3\n2,y,4\n", 2),
    "quoted header with a line break": (b'a,"b\nB",c\n1,x,3\n2,y,4\n', 2),
    "latin-1 byte and a blank line": (b"a,b,c\n1,caf\xe9,3\n\n2,y,4\n", 3),
}


def _write(tmp_path: Path, raw: bytes) -> Path:
    path = tmp_path / "f.csv"
    path.write_bytes(raw)
    return path


def _text(values: list[object]) -> list[str | None]:
    return [None if v is None or v == "" else str(v) for v in values]


@pytest.mark.parametrize("name", sorted(READABLE))
def test_record_n_is_the_loaders_index_n_minus_1(tmp_path: Path, name: str) -> None:
    raw, n_records = READABLE[name]
    path = _write(tmp_path, raw)
    audit = read_file(path, "loans")
    loader = read_and_clean(path, {}, None, None)
    assert audit.unreadable is None and audit.frame is not None
    assert audit.n_records == loader.height == n_records
    assert audit.frame["record"].to_list() == list(range(1, n_records + 1))
    for position in range(1, len(audit.headers) + 1):
        ours = _text(audit.frame[f"c{position}"].to_list())
        theirs = _text(loader.to_series(position - 1).to_list())
        assert ours == theirs, (name, position)


def test_headers_are_kept_as_supplied(tmp_path: Path) -> None:
    audit = read_file(_write(tmp_path, b" a ,b,a\n1,x,3\n"), "loans")
    assert audit.headers == (" a ", "b", "a")
    assert [c.position for c in audit.columns] == [1, 2, 3]
    bom = read_file(_write(tmp_path, b"\xef\xbb\xbfa,b\n1,2\n"), "loans")
    assert bom.headers == ("a", "b")


def test_physical_line_counts_breaks_inside_cells(tmp_path: Path) -> None:
    raw = b'\na,b\n1,"x\ny\nz"\n2,q\n\n3,r\n'
    audit = read_file(_write(tmp_path, raw), "loans")
    assert audit.frame is not None
    assert audit.frame["record"].to_list() == [1, 2, 3, 4]
    # line 1 blank, 2 header, 3-5 record 1, 6 record 2, 7 blank record, 8 record 4
    assert audit.frame["physical_line"].to_list() == [3, 6, 7, 8]


def test_repaired_bytes_are_kept_and_text_equals_the_loaders(tmp_path: Path) -> None:
    path = _write(tmp_path, b"\nname,n\ncaf\xe9,1\nplain,2\n\x80uro,3\n")
    audit = read_file(path, "loans")
    loader = read_and_clean(path, {}, None, None)
    assert audit.frame is not None
    assert audit.frame["c1"].to_list() == loader["name"].to_list() == ["café", "plain", "€uro"]
    assert audit.encoding_repaired.rows() == [(1, 1, "636166e9"), (3, 1, "8075726f")]
    clean = read_file(_write(tmp_path, b"a\n1\n"), "loans")
    assert clean.encoding_repaired.height == 0


def test_every_long_row_is_listed(tmp_path: Path) -> None:
    raw = b'a,b,c\n1,x,3,9\n2,y,4\n3,"p\nq",5,6,7\n'
    path = _write(tmp_path, raw)
    with pytest.raises(pl.exceptions.ComputeError):
        read_and_clean(path, {}, None, None)
    audit = read_file(path, "loans")
    assert audit.frame is None and audit.n_records == 0 and audit.columns == ()
    assert audit.unreadable is not None
    assert audit.unreadable.reason == "rows_with_more_fields_than_the_header"
    assert [(r.record, r.physical_line, r.fields) for r in audit.unreadable.long_rows] == [
        (1, 2, 4), (3, 4, 5),
    ]
    assert (audit.unreadable.record, audit.unreadable.physical_line) == (1, 2)


def test_unterminated_quote(tmp_path: Path) -> None:
    audit = read_file(_write(tmp_path, b'a,b,c\n1,x,3\n2,"y,4\n3,z,5\n'), "loans")
    assert audit.unreadable is not None and audit.frame is None
    assert audit.unreadable.reason == "quote_not_closed_or_malformed"
    assert (audit.unreadable.record, audit.unreadable.physical_line) == (2, 3)


def test_empty_file_and_header_only(tmp_path: Path) -> None:
    empty = read_file(_write(tmp_path, b""), "loans")
    assert empty.unreadable is not None and empty.unreadable.reason == "no_header"
    header_only = read_file(_write(tmp_path, b"a,b,c\n"), "loans")
    assert header_only.unreadable is None and header_only.n_records == 0
    assert header_only.headers == ("a", "b", "c")


def test_lone_carriage_returns_are_not_splittable(tmp_path: Path) -> None:
    audit = read_file(_write(tmp_path, b"a,b,c\r1,x,3\r2,y,4\r"), "loans")
    assert audit.unreadable is not None
    assert audit.unreadable.reason == "records_split_differently"
    assert audit.n_records == 0


def test_file_identity(tmp_path: Path) -> None:
    audit = read_file(_write(tmp_path, b"a\n1\n"), "collateral")
    assert (audit.name, audit.size, audit.file) == ("f.csv", 4, "collateral")
    assert len(audit.sha256) == 64


@pytest.mark.parametrize(
    ("fixture", "loan_id", "collateral_id"),
    [
        ("synthetic", "new_underlying_exposure_identifier", "new_collateral_identifier"),
        ("real_anonymised", "RREL3", "RREC4"),
    ],
)
def test_record_numbers_against_the_pipelines_record_index(
    fixture: str, loan_id: str, collateral_id: str
) -> None:
    folder = FIXTURES / fixture
    result = run_pipeline(
        loans_file_path=folder / "loans.csv",
        collaterals_file_path=folder / "collaterals.csv",
        deal_name="T",
        dry_run=True,
        verbose=False,
    )
    assert result.stage3 is not None
    assert result.loan_record_index is not None and result.collateral_record_index is not None
    for audit_file, kind, header, index, stage_frame, stage_column in (
        ("loans.csv", "loans", loan_id, result.loan_record_index,
         result.stage3.loans, "new_underlying_exposure_identifier"),
        ("collaterals.csv", "collateral", collateral_id, result.collateral_record_index,
         result.stage3.properties, "new_collateral_identifier"),
    ):
        audit = read_file(folder / audit_file, kind)  # type: ignore[arg-type]
        assert audit.frame is not None
        column = f"c{audit.headers.index(header) + 1}"
        by_record = dict(zip(audit.frame["record"].to_list(), audit.frame[column].to_list(), strict=True))
        ours = [by_record[i + 1] for i in index.to_list()]
        assert ours == stage_frame[stage_column].to_list()
