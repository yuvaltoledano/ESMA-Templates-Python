"""How a header becomes a field (specification 5.3)."""

from __future__ import annotations

from pathlib import Path

from esma_milan.io_layer.read_csv import read_and_clean
from esma_milan.io_layer.read_taxonomy import bundled_taxonomy_path, load_taxonomy
from esma_milan.quality.audit_read import resolve_headers


def _resolve(headers: list[str], file: str = "loans") -> list[tuple[str, str | None]]:
    return [(c.how, c.field) for c in resolve_headers(tuple(headers), file)]  # type: ignore[arg-type]


def test_three_header_styles_reach_one_field() -> None:
    assert _resolve(["RREL30", "Current Principal Balance", "current_principal_balance"]) == [
        ("esma_code", "RREL30"),
        ("taxonomy_field_name", "RREL30"),
        ("cleaned_name", "RREL30"),
    ]
    assert _resolve(["Geographic Region - Obligor", "CURRENT PRINCIPAL BALANCE"]) == [
        ("taxonomy_field_name", "RREL11"),
        ("cleaned_name", "RREL30"),
    ]


def test_a_code_is_matched_exactly_as_the_pipeline_matches_it() -> None:
    assert _resolve(["rrel30", "RREL30 ", " RREL30"]) == [("unmapped", None)] * 3


def test_unmapped_and_export_extensions() -> None:
    assert _resolve(["my_notes", "Sec_Id", "Pool_Cutoff_Date", "pool_cutoff_date"]) == [
        ("unmapped", None),
        ("export_extension", None),
        ("export_extension", None),
        ("export_extension", None),
    ]


def test_currency_companions() -> None:
    assert _resolve(["RREL30_CURRENCY", "rrel29_currency", "RREL43_CURRENCY", "RREC13_CURRENCY"]) == [
        ("currency_companion", "RREL30"),  # not in the pipeline's fixed list; kept by it
        ("currency_companion", "RREL29"),
        ("unmapped", None),  # RREL43 is a percentage
        ("unmapped", None),  # a collateral field in the loans file
    ]
    assert _resolve(["RREC13_CURRENCY"], "collateral") == [("currency_companion", "RREC13")]


def test_a_loans_header_that_matches_a_collateral_field_is_unmapped() -> None:
    columns = resolve_headers(("lien", "RREC5", "Unique Identifier"), "loans")
    assert [(c.how, c.field) for c in columns] == [
        ("unmapped", None),
        ("unmapped", None),
        ("taxonomy_field_name", "RREL1"),
    ]
    # The pipeline renames by the whole taxonomy, whatever the file.
    assert [c.pipeline_name for c in columns] == ["lien", "collateral_type", "unique_identifier"]
    assert _resolve(["lien", "RREC5", "Unique Identifier"], "collateral") == [
        ("cleaned_name", "RREC8"),
        ("esma_code", "RREC5"),
        ("taxonomy_field_name", "RREC1"),
    ]


def test_a_repeated_header_resolves_both_columns() -> None:
    columns = resolve_headers(("RREL3", "RREL3"), "loans")
    assert [(c.position, c.how, c.field) for c in columns] == [
        (1, "esma_code", "RREL3"),
        (2, "esma_code", "RREL3"),
    ]


def test_optional_elements_are_recognised_in_the_collateral_file_only() -> None:
    headers = [
        "Current_Gross_Annual_Rental_Income",
        "original_gross_annual_rental_income",
        "current_gross_annual_rental_income_currency",
    ]
    assert _resolve(headers, "collateral") == [
        ("optional_element", "OPT_RENT_CURRENT"),
        ("optional_element", "OPT_RENT_ORIGINAL"),
        ("currency_companion", "OPT_RENT_CURRENT"),
    ]
    assert _resolve(headers, "loans") == [("unmapped", None)] * 3


def test_pipeline_name_of_a_repeated_header_is_what_the_loader_produces(tmp_path: Path) -> None:
    path = tmp_path / "f.csv"
    path.write_text("RREL3,RREL3,note,RREL3,note\nA,B,x,C,y\n", encoding="utf-8")
    loaded = read_and_clean(path, load_taxonomy(bundled_taxonomy_path()), None, None)
    columns = resolve_headers(("RREL3", "RREL3", "note", "RREL3", "note"), "loans")
    assert [c.pipeline_name for c in columns] == loaded.columns
    assert loaded.columns == [
        "new_underlying_exposure_identifier", "rrel3_duplicated_0", "note",
        "rrel3_duplicated_1", "note_duplicated_0",
    ]
    # The audit still resolves every one of them to the field.
    assert [c.field for c in columns] == ["RREL3", "RREL3", None, "RREL3", None]
