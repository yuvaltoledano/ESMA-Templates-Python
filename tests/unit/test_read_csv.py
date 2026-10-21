"""Tests for esma_milan.io_layer.read_csv.read_and_clean.

Mirrors a subset of the R tests in test-utils.R::"read_and_clean..." and
exercises:

- successful read against the staged synthetic loans.csv / collaterals.csv
- ESMA NA tokens (ND, ND1..ND5, NA, "") are converted to nulls
- character_cols schema override forces ID columns to String
- the taxonomy rename is applied to source columns whose name is a key,
  and untouched columns pass through unchanged
- duplicate column names after rename produce a hard error
- a missing file raises FileNotFoundError
- an empty file (header only, no data) raises ValueError
- numeric_cols: taxonomy-numeric columns come out Float64 whatever
  Polars inferred, whitespace is trimmed as readr does, and a
  non-numeric value is a hard error naming file, field, row and value
- non-UTF-8 input: Windows-1252 / Latin-1 bytes are repaired per field
  (header included) before Polars sees the file, with one warning
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import structlog

from esma_milan.io_layer.read_csv import read_and_clean

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC = REPO_ROOT / "tests" / "fixtures" / "synthetic"


# ---------------------------------------------------------------------------
# Against the staged synthetic fixture
# ---------------------------------------------------------------------------


def test_reads_synthetic_loans() -> None:
    # Synthetic loans CSV uses already-snake_case headers (no taxonomy
    # rename needed); we still pass the taxonomy and exercise the
    # passthrough path.
    df = read_and_clean(
        SYNTHETIC / "loans.csv",
        taxonomy={},
        character_cols=("RREL2", "RREL3", "RREL4", "RREL5"),
    )
    assert df.height == 8
    # Spot-check a few well-known columns.
    assert "calc_loan_id" not in df.columns  # not yet generated
    assert "original_underlying_exposure_identifier" in df.columns
    assert "current_principal_balance" in df.columns
    assert "account_status" in df.columns
    # Account status is read as String (not categorical, no coercion).
    assert df.schema["account_status"] == pl.String


def test_reads_synthetic_collaterals() -> None:
    df = read_and_clean(
        SYNTHETIC / "collaterals.csv",
        taxonomy={},
        character_cols=("RREC2", "RREC3", "RREC4"),
    )
    assert df.height == 12
    assert "underlying_exposure_identifier" in df.columns
    assert "current_valuation_amount" in df.columns
    assert "property_type" in df.columns


# ---------------------------------------------------------------------------
# NA-token handling
# ---------------------------------------------------------------------------


def test_na_tokens_converted_to_null(tmp_path: Path) -> None:
    """Mirrors test-utils.R::"read_and_clean treats ND and ND3 as NA values".

    All seven configured NA tokens ("", NA, ND, ND1..ND5) plus the
    canonical empty cell must map to Polars null.
    """
    p = tmp_path / "loans.csv"
    p.write_text(
        "id,balance\n"
        "L1,100\n"     # control
        "L2,NA\n"      # NA token
        "L3,ND\n"
        "L4,ND1\n"
        "L5,ND2\n"
        "L6,ND3\n"
        "L7,ND4\n"
        "L8,ND5\n"
        "L9,\n"        # empty
    )
    df = read_and_clean(p, taxonomy={})
    balances = df["balance"].to_list()
    assert balances[0] == 100  # control
    assert all(b is None for b in balances[1:]), balances


# ---------------------------------------------------------------------------
# character_cols schema override
# ---------------------------------------------------------------------------


def test_character_cols_keeps_numeric_looking_ids_as_string(tmp_path: Path) -> None:
    p = tmp_path / "loans.csv"
    p.write_text("RREL2,RREL30\n12345,100\n67890,200\n")
    df = read_and_clean(
        p,
        taxonomy={"RREL2": "Original Underlying Exposure Identifier", "RREL30": "Current Principal Balance"},
        character_cols=("RREL2",),
    )
    # RREL2 was forced to String even though every value is integer-looking.
    assert df.schema["original_underlying_exposure_identifier"] == pl.String
    assert df["original_underlying_exposure_identifier"].to_list() == ["12345", "67890"]
    # RREL30 was inferred as Int64 by Polars' usual rules (no numeric_cols
    # passed, so whole-file inference stands).
    assert df.schema["current_principal_balance"] == pl.Int64
    assert df["current_principal_balance"].to_list() == [100, 200]


# ---------------------------------------------------------------------------
# Taxonomy rename + clean_names
# ---------------------------------------------------------------------------


def test_taxonomy_rename_and_clean_names_applied(tmp_path: Path) -> None:
    p = tmp_path / "loans.csv"
    p.write_text("RREL2,RREL30,SomeOther Column\nL1,100,xx\n")
    df = read_and_clean(
        p,
        taxonomy={
            "RREL2": "Original Underlying Exposure Identifier",
            "RREL30": "Current Principal Balance",
        },
    )
    # RREL2 / RREL30 renamed via taxonomy then cleaned.
    assert "original_underlying_exposure_identifier" in df.columns
    assert "current_principal_balance" in df.columns
    # Untaxonomy'd column passes through clean_names directly.
    assert "someother_column" in df.columns


def test_unmapped_columns_pass_through_clean_names(tmp_path: Path) -> None:
    p = tmp_path / "loans.csv"
    p.write_text("pool_cutoff_date,Some Column\n2024-01-01,x\n")
    df = read_and_clean(p, taxonomy={"DOES_NOT_APPLY": "x"})
    assert df.columns == ["pool_cutoff_date", "some_column"]


# ---------------------------------------------------------------------------
# Duplicate column names hard error
# ---------------------------------------------------------------------------


def test_duplicate_after_rename_raises(tmp_path: Path) -> None:
    """Mirrors r_reference/R/utils.R:104-112 - the taxonomy mapping has two
    source codes pointing at the same target name, producing two columns
    with the same name after rename. This is a critical data-quality
    issue and must halt the pipeline."""
    p = tmp_path / "loans.csv"
    p.write_text("RREL2,RREL3\nA,B\n")
    with pytest.raises(ValueError, match="Duplicated column names"):
        read_and_clean(
            p,
            taxonomy={
                "RREL2": "Loan Identifier",
                "RREL3": "Loan Identifier",  # collision
            },
        )


def test_duplicate_after_clean_names_raises(tmp_path: Path) -> None:
    """Two columns whose clean_names happen to collide (e.g. one already
    snake-cased, one Title Case that snake-cases to the same name)."""
    p = tmp_path / "loans.csv"
    p.write_text("Loan Identifier,loan_identifier\nA,B\n")
    with pytest.raises(ValueError, match="Duplicated column names"):
        read_and_clean(p, taxonomy={})


# ---------------------------------------------------------------------------
# File-level errors
# ---------------------------------------------------------------------------


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_and_clean(tmp_path / "missing.csv", taxonomy={})


def test_empty_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("a,b\n")  # header only
    with pytest.raises(ValueError, match="No data rows"):
        read_and_clean(p, taxonomy={})


# ---------------------------------------------------------------------------
# numeric_cols: taxonomy-typed numeric fields
# ---------------------------------------------------------------------------

_TAX = {"RREL2": "Original Underlying Exposure Identifier", "RREL30": "Current Principal Balance"}


def _csv_with_balances(tmp_path: Path, values: list[str]) -> Path:
    p = tmp_path / "loans.csv"
    def quote(v: str) -> str:
        return f'"{v}"' if "," in v else v

    p.write_text("RREL2,RREL30\n" + "".join(f"L{i},{quote(v)}\n" for i, v in enumerate(values, 1)))
    return p


def test_numeric_cols_widens_inferred_int64_to_float64(tmp_path: Path) -> None:
    p = _csv_with_balances(tmp_path, ["100", "200"])
    df = read_and_clean(p, _TAX, numeric_cols={"current_principal_balance"})
    assert df.schema["current_principal_balance"] == pl.Float64
    assert df["current_principal_balance"].to_list() == [100.0, 200.0]


def test_numeric_cols_all_nd_column_is_null_float64(tmp_path: Path) -> None:
    """An all-ND column infers as String; the cast makes it an all-null
    Float64 so its dtype does not depend on the data in this file."""
    p = _csv_with_balances(tmp_path, ["ND", "ND5", ""])
    df = read_and_clean(p, _TAX, numeric_cols={"current_principal_balance"})
    assert df.schema["current_principal_balance"] == pl.Float64
    assert df["current_principal_balance"].to_list() == [None, None, None]


def test_numeric_cols_trims_whitespace_like_readr(tmp_path: Path) -> None:
    """readr reads with trim_ws = TRUE, so " 100 " is 100 in R. Polars
    infers such a column as String; the cast trims before parsing."""
    p = _csv_with_balances(tmp_path, [" 100 ", "200", "\t+7.5"])
    df = read_and_clean(p, _TAX, numeric_cols={"current_principal_balance"})
    assert df["current_principal_balance"].to_list() == [100.0, 200.0, 7.5]


def test_numeric_cols_non_numeric_value_is_hard_error(tmp_path: Path) -> None:
    """Deliberate deviation from readr, which would null the value with a
    warning: a silently missing balance changes the pool."""
    p = _csv_with_balances(tmp_path, ["100", "200", "1.234,56", "ND"])
    with pytest.raises(ValueError) as excinfo:
        read_and_clean(p, _TAX, numeric_cols={"current_principal_balance"})
    msg = str(excinfo.value)
    first_line = msg.splitlines()[0]
    assert "loans.csv" in first_line
    assert "RREL30" in first_line and "current_principal_balance" in first_line
    assert "1 non-numeric value" in first_line
    assert "data row 3" in msg and "'1.234,56'" in msg


def test_numeric_cols_error_lists_at_most_ten_rows_and_total(tmp_path: Path) -> None:
    p = _csv_with_balances(tmp_path, ["x"] * 12)
    with pytest.raises(ValueError) as excinfo:
        read_and_clean(p, _TAX, numeric_cols={"current_principal_balance"})
    msg = str(excinfo.value)
    assert "12 non-numeric value(s)" in msg
    assert msg.count("data row ") == 10
    assert "... and 2 more" in msg


def test_numeric_cols_boolean_inferred_column_is_hard_error(tmp_path: Path) -> None:
    """Polars infers TRUE/FALSE as Boolean. A monetary column holding only
    TRUE/FALSE is data corruption; it must not be cast to 1.0 / 0.0."""
    p = _csv_with_balances(tmp_path, ["TRUE", "FALSE"])
    with pytest.raises(ValueError, match="2 non-numeric value") as excinfo:
        read_and_clean(p, _TAX, numeric_cols={"current_principal_balance"})
    # The samples show the file's spelling, not the Boolean Polars parsed.
    msg = str(excinfo.value)
    assert "data row 1: 'TRUE'" in msg and "data row 2: 'FALSE'" in msg
    assert "True" not in msg.replace("TRUE", "")


def test_numeric_cols_absent_columns_are_ignored(tmp_path: Path) -> None:
    p = _csv_with_balances(tmp_path, ["100"])
    df = read_and_clean(p, _TAX, numeric_cols={"not_in_file", "current_principal_balance"})
    assert df.columns == ["original_underlying_exposure_identifier", "current_principal_balance"]


def test_whole_file_inference_handles_late_decimal(tmp_path: Path) -> None:
    """Without numeric_cols, a column whose first 10,000 rows are whole
    numbers and whose last is a decimal is still Float64, because
    inference now scans the whole file rather than a 10,000-row sample."""
    p = _csv_with_balances(tmp_path, ["1"] * 10_000 + ["2.5"])
    df = read_and_clean(p, _TAX)
    assert df.schema["current_principal_balance"] == pl.Float64
    assert df["current_principal_balance"][-1] == 2.5


# ---------------------------------------------------------------------------
# Non-UTF-8 input
# ---------------------------------------------------------------------------


def test_valid_utf8_file_is_not_repaired_and_logs_no_warning(tmp_path: Path) -> None:
    p = tmp_path / "loans.csv"
    p.write_text("RREL2,name\nL1,Müller\n", encoding="utf-8")
    with structlog.testing.capture_logs() as logs:
        df = read_and_clean(p, _TAX)
    assert df["name"].to_list() == ["Müller"]
    assert not [e for e in logs if e["event"] == "csv_encoding_repaired"]


def test_windows_1252_file_is_repaired_with_one_warning(tmp_path: Path) -> None:
    """Mirrors R: readr reads the bytes, safe_for_excel's sanitize_utf8
    decodes each invalid string as Windows-1252. Here the repair happens
    before Polars, which would otherwise raise on the first invalid byte."""
    p = tmp_path / "loans.csv"
    p.write_bytes("RREL2,name,note,RREL30\nL1,Müller,€100 “ok” — fine,100\nL2,Plain,x,200\n".encode("cp1252"))
    with structlog.testing.capture_logs() as logs:
        df = read_and_clean(p, _TAX, numeric_cols={"current_principal_balance"})
    assert df["name"].to_list() == ["Müller", "Plain"]
    assert df["note"].to_list() == ["€100 “ok” — fine", "x"]
    assert df["current_principal_balance"].to_list() == [100.0, 200.0]
    events = [e for e in logs if e["event"] == "csv_encoding_repaired"]
    assert len(events) == 1
    assert events[0]["file"] == "loans.csv"
    assert events[0]["n_fields_repaired"] == 2


def test_latin1_only_byte_is_preserved_not_stripped(tmp_path: Path) -> None:
    """0x81 is undefined in Windows-1252; the cascade's Latin-1 pass keeps
    it as U+0081 (one code point), exactly as sanitize_utf8_value's unit
    test pins and as R's iconv(latin1) does. Nothing is stripped."""
    p = tmp_path / "loans.csv"
    p.write_bytes(b"RREL2,name\nL1,x\x81y\n")
    df = read_and_clean(p, _TAX)
    assert df["name"].to_list() == ["x\u0081y"]


def test_repair_preserves_quoted_fields_and_crlf(tmp_path: Path) -> None:
    """The csv-module round trip must be lossless for the dialect readr
    accepts: quoted commas, doubled quotes, embedded newlines, CRLF
    endings, no trailing newline."""
    p = tmp_path / "loans.csv"
    p.write_bytes(b'RREL2,text,RREL30\r\n"L,1","a ""q""\r\nb",100\r\nL2,M\xfcller,ND')
    df = read_and_clean(p, _TAX, numeric_cols={"current_principal_balance"})
    assert df["original_underlying_exposure_identifier"].to_list() == ["L,1", "L2"]
    assert df["text"].to_list() == ['a "q"\r\nb', "Müller"]
    assert df["current_principal_balance"].to_list() == [100.0, None]


def test_windows_1252_header_is_repaired_and_renamed_via_taxonomy(tmp_path: Path) -> None:
    """R sanitises column names in safe_for_excel too. A header cell with
    an accented character must come back as the intended text so the
    taxonomy rename (keyed on that text) still applies."""
    p = tmp_path / "loans.csv"
    p.write_bytes("RREL2,Währung\nL1,EUR\n".encode("cp1252"))
    df = read_and_clean(p, {"RREL2": "Original Underlying Exposure Identifier", "Währung": "Loan Currency"})
    assert df.columns == ["original_underlying_exposure_identifier", "loan_currency"]
    assert df["loan_currency"].to_list() == ["EUR"]
