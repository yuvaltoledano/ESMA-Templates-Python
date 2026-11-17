"""The catalogue is reviewed data; these tests keep it equal to its sources."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import openpyxl
import pytest

from esma_milan.io_layer.read_taxonomy import bundled_taxonomy_path
from esma_milan.quality.catalogue import (
    CODE_LISTS,
    FIELDS,
    RULES,
    FieldEntry,
    fields_of,
    mapping_tokens,
)

REPO = Path(__file__).resolve().parents[2]
MAPPING = REPO / "docs" / "data-quality" / "rre-source-mapping.csv"
SPECIFICATION = REPO / "docs" / "data-quality" / "specification.md"


def _mapping_rows() -> list[dict[str, str]]:
    with MAPPING.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _taxonomy_rows() -> dict[str, dict[str, object]]:
    wb = openpyxl.load_workbook(bundled_taxonomy_path(), read_only=True, data_only=True)
    try:
        ws = wb[wb.sheetnames[0]]
        ws.reset_dimensions()
        rows = list(ws.iter_rows(values_only=True))
    finally:
        wb.close()
    header = [str(h) for h in rows[0]]
    out: dict[str, dict[str, object]] = {}
    for row in rows[1:]:
        record = dict(zip(header, row, strict=False))
        if record["FIELD NAME"]:
            out[str(record["FIELD CODE"])] = record
    return out


def _entries_in_mapping_order() -> list[FieldEntry]:
    optional = iter(fields_of("collateral", optional=True))
    return [
        next(optional) if row["field"] == "(optional)" else FIELDS[row["field"]]
        for row in _mapping_rows()
    ]


def test_107_fields_and_two_optional_elements() -> None:
    assert len(fields_of("loans")) == 84
    assert len(fields_of("collateral")) == 23
    assert [f.code for f in fields_of("collateral", optional=True)] == [
        "OPT_RENT_CURRENT", "OPT_RENT_ORIGINAL",
    ]
    assert len(FIELDS) == 109


def test_fields_equal_the_bundled_taxonomy() -> None:
    taxonomy = _taxonomy_rows()
    template = [f for f in FIELDS.values() if not f.optional]
    assert [f.code for f in template] == list(taxonomy)
    for field in template:
        row = taxonomy[field.code]
        assert field.name == row["FIELD NAME"]
        assert "{" + field.taxonomy_format + "}" == row["FORMAT"]
        assert field.nd1_4_allowed is (row["ND1-ND4 allowed?"] == "YES")
        assert field.nd5_allowed is (row["ND5 allowed?"] == "YES")


def test_fields_equal_the_mapping_file() -> None:
    rows = _mapping_rows()
    assert len(rows) == 109
    for row, field in zip(rows, _entries_in_mapping_order(), strict=True):
        assert field.name == row["name"]
        assert field.taxonomy_format == row["taxonomy_format"]
        assert field.nd1_4_allowed is (row["nd1_4_allowed"] == "YES")
        assert field.nd5_allowed is (row["nd5_allowed"] == "YES")
        category = row["final_report_category_provisional"][:1]
        assert field.final_report_category == (int(category) if category.isdigit() else None)
        assert field.critical is row["rejection_effect"].startswith("critical field")


def test_code_lists_equal_the_codes_in_the_taxonomy_text() -> None:
    taxonomy = _taxonomy_rows()
    list_fields = [code for code, row in taxonomy.items() if row["FORMAT"] == "{LIST}"]
    assert sorted(CODE_LISTS) == sorted(list_fields)
    assert len(list_fields) == 24
    for code in list_fields:
        found = re.findall(r"\(([A-Z0-9]{4})\)", str(taxonomy[code]["CONTENT TO REPORT"]))
        assert len(found) == len(set(found)), code
        assert CODE_LISTS[code] == frozenset(found), code
        assert FIELDS[code].codes == CODE_LISTS[code]


def _cell_tokens(cell: str) -> set[str]:
    tokens: set[str] = set()
    for part in cell.split("|"):
        token = part.strip().split(" ")[0]
        if token == "P3a-P3e":
            tokens |= {"P3a", "P3b", "P3c", "P3d", "P3e"}
        else:
            tokens.add(token)
    return tokens


def test_mapping_file_checks_equal_the_catalogue_both_ways() -> None:
    for row, field in zip(_mapping_rows(), _entries_in_mapping_order(), strict=True):
        in_file = _cell_tokens(row["implementing_checks"])
        in_catalogue = set(mapping_tokens(field))
        assert in_file - in_catalogue == set(), (field.code, "only in the mapping file")
        assert in_catalogue - in_file == set(), (field.code, "only in the catalogue")


def test_every_rule_in_the_mapping_file_is_implemented_or_recorded() -> None:
    seen: set[str] = set()
    for row in _mapping_rows():
        seen |= _cell_tokens(row["implementing_checks"])
    for token in sorted(seen - {"O1"}):
        matching = [r for r in RULES.values() if r.rule_id == token or r.rule_id.startswith(token + "(")]
        assert matching, token
        for rule in matching:
            assert rule.status in ("implemented", "partly_implemented", "not_implemented")
            if rule.status != "implemented":
                assert rule.not_implemented_part
    assert RULES["A10(a)"].status == "implemented"  # the single owner of P6(e)'s defect
    assert RULES["P6(e)"].status == "not_implemented"


def test_every_rule_id_of_the_specification_is_in_the_catalogue() -> None:
    # Appendix A lists retired ids of earlier documents; it is not a rule table.
    text = SPECIFICATION.read_text(encoding="utf-8").split("## Appendix A")[0]
    ids = set()
    for line in text.splitlines():
        match = re.match(r"^\| ([A-Z][0-9]+[a-g]?) \| (.*)", line)
        if match and not match.group(2).startswith("Retired"):
            ids.add(match.group(1))
    assert {"P1", "P3g", "P13", "A17", "C47", "D3", "E19", "F10", "G8", "L1"} <= ids
    for rule_id in sorted(ids):
        assert any(k == rule_id or k.startswith(rule_id + "(") for k in RULES), rule_id


@pytest.mark.parametrize("rule_id", sorted(RULES))
def test_every_rule_declares_its_attributes(rule_id: str) -> None:
    rule = RULES[rule_id]
    assert rule.title and rule.applies_to and rule.defect_key and rule.result_kinds
    assert rule.basis in ("E", "L", "X", "A", "P")
    assert rule.severity in ("Stop", "Warn", "Info")
    if rule.severity == "Stop":
        assert rule.basis in ("L", "A")


def test_field_attributes_from_the_specification() -> None:
    assert sum(f.not_below_zero for f in FIELDS.values() if not f.optional) == 15
    assert not FIELDS["RREL30"].not_below_zero and not FIELDS["RREL73"].not_below_zero
    assert sum(f.not_after_cutoff for f in FIELDS.values()) == 11
    assert sum(f.critical for f in FIELDS.values()) == 27
    assert FIELDS["RREL29"].dedicated_currency == "RREL28"
    assert FIELDS["RREC17"].dedicated_currency is None
    assert FIELDS["RREL82"].max_length == 100
    optional = FIELDS["OPT_RENT_CURRENT"]
    assert optional.nd5_allowed and not optional.nd1_4_allowed and optional.not_below_zero
