"""Tests for clean_name / clean_names.

Pinned against ~25 real ESMA column headers (taxonomy + synthetic +
documentation examples). If any header would produce different output
from janitor::clean_names() we'll discover it here and add a more
sophisticated rule.
"""

from __future__ import annotations

import pytest

from esma_milan.io_layer.clean_names import clean_name, clean_names


@pytest.mark.parametrize(
    "raw,expected",
    [
        # -- ESMA taxonomy headers --
        ("TEMPLATE CATEGORY", "template_category"),
        ("FIELD CODE", "field_code"),
        ("FIELD NAME", "field_name"),
        ("CONTENT TO REPORT", "content_to_report"),
        ("ND1-ND4 allowed?", "nd1_nd4_allowed"),
        ("ND5 allowed?", "nd5_allowed"),
        ("FORMAT", "format"),
        (
            "For info: existing ECB or EBA NPL template field code",
            "for_info_existing_ecb_or_eba_npl_template_field_code",
        ),
        # -- Title Case taxonomy field names --
        ("Original Underlying Exposure Identifier", "original_underlying_exposure_identifier"),
        ("New Underlying Exposure Identifier", "new_underlying_exposure_identifier"),
        ("Original Obligor Identifier", "original_obligor_identifier"),
        ("Account Status", "account_status"),
        ("Current Principal Balance", "current_principal_balance"),
        ("Origination Date", "origination_date"),
        ("Maturity Date", "maturity_date"),
        ("Property Type", "property_type"),
        ("Current Valuation Amount", "current_valuation_amount"),
        ("Date Last in Arrears", "date_last_in_arrears"),
        ("Geographic Region - obligor", "geographic_region_obligor"),
        # -- Special characters --
        ("MIG %", "mig_percent"),
        ("Property Postal, Region code or Nuts code", "property_postal_region_code_or_nuts_code"),
        ("Some/Slash/Header", "some_slash_header"),
        ("(brackets)", "brackets"),
        ("Multi---Underscores", "multi_underscores"),
        ("Trailing whitespace   ", "trailing_whitespace"),
        ("   leading whitespace", "leading_whitespace"),
        # -- Already-clean snake_case (synthetic fixture style) --
        ("original_underlying_exposure_identifier", "original_underlying_exposure_identifier"),
        ("pool_cutoff_date", "pool_cutoff_date"),
        ("date_of_restructuring", "date_of_restructuring"),
        ("property_pool_cutoff_date", "property_pool_cutoff_date"),
        # -- ESMA codes --
        ("RREL2", "rrel2"),
        ("RREC15", "rrec15"),
        # -- # produces _number_ -> trims to "number" --
        ("#", "number"),
        ("Item #1", "item_number_1"),
        # -- Pure punctuation collapses to empty -> empty string --
        # (callers must not rely on this for real headers)
        ("---", ""),
    ],
)
def test_clean_name_table(raw: str, expected: str) -> None:
    assert clean_name(raw) == expected, f"input={raw!r}"


def test_clean_names_preserves_order() -> None:
    inputs = ["FIELD CODE", "FIELD NAME", "MIG %", "ND1-ND4 allowed?"]
    assert clean_names(inputs) == [
        "field_code",
        "field_name",
        "mig_percent",
        "nd1_nd4_allowed",
    ]


def test_clean_names_idempotent_on_already_clean() -> None:
    already_clean = [
        "original_underlying_exposure_identifier",
        "pool_cutoff_date",
        "rrel2",
    ]
    assert clean_names(already_clean) == already_clean
