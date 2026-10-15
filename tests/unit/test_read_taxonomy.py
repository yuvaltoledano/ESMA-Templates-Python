"""Tests for esma_milan.io_layer.read_taxonomy.load_taxonomy().

Covers:
- Successful load against the staged synthetic taxonomy fixture.
- Error when the file is missing.
- Error when FIELD CODE column is absent.
- Error when FIELD NAME column is absent.
- Rows with NA in either column are filtered out (matching R's
  filter(!is.na(`FIELD CODE`), !is.na(`FIELD NAME`))).
- Whitespace-only cells treated as missing.
- Spot-check of expected mappings (RREL2, RREC9 etc) against the
  staged taxonomy.
- load_taxonomy_formats() / numeric_field_names(): the FORMAT column
  read and the derived set of Float64-typed field names.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest

from esma_milan.io_layer.read_taxonomy import (
    load_taxonomy,
    load_taxonomy_formats,
    numeric_field_names,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_TAXONOMY = REPO_ROOT / "tests" / "fixtures" / "synthetic" / "taxonomy.xlsx"


def _make_taxonomy(path: Path, header: list[str | None], rows: list[list[object]]) -> None:
    wb = openpyxl.Workbook(write_only=True)
    ws = wb.create_sheet("Sheet1")
    ws.append(header)
    for r in rows:
        ws.append(r)
    wb.save(path)


def test_loads_synthetic_taxonomy() -> None:
    mapping = load_taxonomy(SYNTHETIC_TAXONOMY)
    # Spot-check a handful of well-known ESMA codes against their official
    # taxonomy names. Drift in any of these means the staged taxonomy has
    # changed (and the rest of the pipeline likely needs updating too).
    assert mapping["RREL2"] == "Original Underlying Exposure Identifier"
    assert mapping["RREL3"] == "New Underlying Exposure Identifier"
    assert mapping["RREL30"] == "Current Principal Balance"
    assert mapping["RREL69"] == "Account Status"
    assert mapping["RREC2"] == "Underlying Exposure Identifier"
    assert mapping["RREC9"] == "Property Type"
    assert mapping["RREC13"] == "Current Valuation Amount"
    assert mapping["RREC23"] == "Guarantor Type"


def test_taxonomy_size_matches_synthetic_fixture() -> None:
    mapping = load_taxonomy(SYNTHETIC_TAXONOMY)
    # Synthetic taxonomy has ~109 data rows; the metadata "Information
    # section" header rows have NA codes and are filtered out.
    assert 80 < len(mapping) < 200, f"unexpected taxonomy size: {len(mapping)}"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_taxonomy(tmp_path / "missing.xlsx")


def test_missing_field_code_column_raises(tmp_path: Path) -> None:
    p = tmp_path / "tax.xlsx"
    _make_taxonomy(
        p,
        header=["SECTION", "FIELD NAME"],  # no FIELD CODE
        rows=[["Underlying", "Original Underlying Exposure Identifier"]],
    )
    with pytest.raises(ValueError, match="FIELD CODE"):
        load_taxonomy(p)


def test_missing_field_name_column_raises(tmp_path: Path) -> None:
    p = tmp_path / "tax.xlsx"
    _make_taxonomy(
        p,
        header=["FIELD CODE", "SECTION"],  # no FIELD NAME
        rows=[["RREL2", "Underlying"]],
    )
    with pytest.raises(ValueError, match="FIELD NAME"):
        load_taxonomy(p)


def test_rows_with_na_field_code_are_filtered(tmp_path: Path) -> None:
    # Mirrors R's filter(!is.na(`FIELD CODE`), !is.na(`FIELD NAME`)).
    p = tmp_path / "tax.xlsx"
    _make_taxonomy(
        p,
        header=["FIELD CODE", "FIELD NAME", "SECTION"],
        rows=[
            ["RREL1", "Unique Identifier", "S"],
            [None, "Section Header With No Code", "S"],  # NA code -> drop
            ["RREL2", None, "S"],                         # NA name -> drop
            ["RREL3", "New Underlying Exposure Identifier", "S"],
        ],
    )
    mapping = load_taxonomy(p)
    assert mapping == {
        "RREL1": "Unique Identifier",
        "RREL3": "New Underlying Exposure Identifier",
    }


def test_whitespace_only_cells_are_treated_as_na(tmp_path: Path) -> None:
    p = tmp_path / "tax.xlsx"
    _make_taxonomy(
        p,
        header=["FIELD CODE", "FIELD NAME"],
        rows=[
            ["RREL1", "Unique Identifier"],
            ["   ", "Whitespace code"],   # drop
            ["RREL2", "   "],             # drop
            ["", "Empty code"],           # drop
        ],
    )
    mapping = load_taxonomy(p)
    assert mapping == {"RREL1": "Unique Identifier"}


def test_field_columns_can_be_in_any_order(tmp_path: Path) -> None:
    p = tmp_path / "tax.xlsx"
    _make_taxonomy(
        p,
        header=["FIELD NAME", "OTHER", "FIELD CODE"],  # reverse order
        rows=[["Name1", "x", "C1"]],
    )
    assert load_taxonomy(p) == {"C1": "Name1"}


def test_header_cells_are_stripped(tmp_path: Path) -> None:
    p = tmp_path / "tax.xlsx"
    _make_taxonomy(
        p,
        header=["FIELD CODE  ", "  FIELD NAME"],  # surrounding whitespace
        rows=[["RREL1", "Unique Identifier"]],
    )
    assert load_taxonomy(p) == {"RREL1": "Unique Identifier"}


def test_duplicate_codes_last_one_wins(tmp_path: Path) -> None:
    # Matches tibble::deframe() behaviour in R.
    p = tmp_path / "tax.xlsx"
    _make_taxonomy(
        p,
        header=["FIELD CODE", "FIELD NAME"],
        rows=[
            ["RREL1", "First"],
            ["RREL1", "Second"],  # overwrites
        ],
    )
    assert load_taxonomy(p) == {"RREL1": "Second"}


# ---------------------------------------------------------------------------
# load_taxonomy_formats / numeric_field_names
# ---------------------------------------------------------------------------


def test_loads_synthetic_taxonomy_formats() -> None:
    formats = load_taxonomy_formats(SYNTHETIC_TAXONOMY)
    # One of each numeric format, one date, one list - the FORMAT column
    # is the taxonomy's per-field schema that read_csv types numeric
    # fields from.
    assert formats["RREL30"] == "{MONETARY}"
    assert formats["RREL43"] == "{PERCENTAGE}"
    assert formats["RREL68"] == "{INTEGER-9999}"
    assert formats["RREL7"] == "{DATEFORMAT}"
    assert formats["RREL69"] == "{LIST}"
    assert set(formats) == set(load_taxonomy(SYNTHETIC_TAXONOMY))


def test_missing_format_column_raises(tmp_path: Path) -> None:
    """A taxonomy without FORMAT cannot type the numeric fields; silently
    falling back to type inference would reintroduce the sample-inference
    crash, so this is a hard error naming the column."""
    p = tmp_path / "tax.xlsx"
    _make_taxonomy(
        p,
        header=["FIELD CODE", "FIELD NAME"],
        rows=[["RREL30", "Current Principal Balance"]],
    )
    assert load_taxonomy(p) == {"RREL30": "Current Principal Balance"}  # still fine
    with pytest.raises(ValueError, match="FORMAT"):
        load_taxonomy_formats(p)


def test_rows_with_na_format_are_filtered(tmp_path: Path) -> None:
    p = tmp_path / "tax.xlsx"
    _make_taxonomy(
        p,
        header=["FIELD CODE", "FIELD NAME", "FORMAT"],
        rows=[
            ["RREL30", "Current Principal Balance", "{MONETARY}"],
            ["RREL69", "Account Status", None],  # NA format -> drop
            [None, "Section header", "{LIST}"],  # NA code -> drop
        ],
    )
    assert load_taxonomy_formats(p) == {"RREL30": "{MONETARY}"}


def test_numeric_field_names_selects_monetary_percentage_and_integer() -> None:
    taxonomy = {
        "RREL30": "Current Principal Balance",
        "RREL43": "Current Interest Rate",
        "RREL68": "Number Of Days In Arrears",
        "RREL69": "Account Status",
        "RREL7": "Origination Date",
        "RREL79": "Original Lender Name",
        "RREC8": "Lien",
    }
    formats = {
        "RREL30": "{MONETARY}",
        "RREL43": "{PERCENTAGE}",
        "RREL68": "{INTEGER-9999}",
        "RREL69": "{LIST}",
        "RREL7": "{DATEFORMAT}",
        "RREL79": "{ALPHANUM-100}",
        "RREC8": "{INTEGER-99}",  # a different bound still matches by prefix
        "RREX1": "{MONETARY}",  # format without a name -> no target column
    }
    assert numeric_field_names(taxonomy, formats) == frozenset({
        "current_principal_balance",
        "current_interest_rate",
        "number_of_days_in_arrears",
        "lien",
    })
