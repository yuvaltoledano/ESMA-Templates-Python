"""Facts about the bundled reference lists that do not depend on the copy."""

from __future__ import annotations

from esma_milan.quality.refdata import NUTS_YEARS, load_reference_data


def test_countries() -> None:
    ref = load_reference_data()
    assert len(ref.countries) == 249
    assert {"NA", "GB", "GR", "NL"} <= ref.countries
    # Eurostat's prefixes are not ISO codes.
    assert "UK" not in ref.countries and "EL" not in ref.countries
    assert all(len(c) == 2 and c.isascii() and c.isupper() for c in ref.countries)


def test_currencies() -> None:
    ref = load_reference_data()
    assert "EUR" in ref.currencies_active and "GBP" in ref.currencies_active
    for withdrawn in ("DEM", "NLG"):
        assert withdrawn in ref.currencies_historic
        assert withdrawn not in ref.currencies_active
        assert withdrawn in ref.currencies
    assert all(len(c) == 3 and c.isascii() and c.isupper() for c in ref.currencies)


def test_nuts_years_and_recoding() -> None:
    ref = load_reference_data()
    assert tuple(ref.nuts3) == NUTS_YEARS == (2010, 2013, 2016, 2021, 2024)
    # DE915 was recoded between NUTS 2013 and NUTS 2016; Germany is in both.
    assert "DE915" in ref.nuts3[2013]
    assert "DE915" not in ref.nuts3[2016]
    assert "DE" in ref.nuts_countries[2013] and "DE" in ref.nuts_countries[2016]
    assert all(len(c) == 5 for codes in ref.nuts3.values() for c in codes)
    assert not any(c.endswith("ZZZ") for codes in ref.nuts3.values() for c in codes)


def test_nuts_country_coverage() -> None:
    ref = load_reference_data()
    assert "UK" in ref.nuts_countries[2021]
    assert "UK" not in ref.nuts_countries[2024]
    assert "EL" in ref.nuts_prefixes
    assert [len(ref.nuts_countries[y]) for y in NUTS_YEARS] == [35, 35, 37, 37, 38]


def test_manifest_matches_the_files() -> None:
    ref = load_reference_data()
    lists = ref.manifest["lists"]
    assert isinstance(lists, dict)
    assert lists["iso3166_1_alpha2"]["codes"] == len(ref.countries)
    assert lists["iso3166_1_alpha2"]["version"] == "4.20.1"
    assert lists["iso4217_active"]["codes"] == len(ref.currencies_active)
    assert lists["iso4217_historic"]["codes"] == len(ref.currencies_historic)
    for year in NUTS_YEARS:
        entry = lists[f"nuts3_{year}"]
        assert entry["codes"] == len(ref.nuts3[year])
        assert entry["countries"] == sorted(ref.nuts_countries[year])
    for entry in lists.values():
        assert entry["source"] and entry["address"] and entry["version"]
        assert len(entry["source_sha256"]) == 64
