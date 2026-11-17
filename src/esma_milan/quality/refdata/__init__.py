"""Reference lists bundled with the data quality report.

Mirrors nothing in the R reference. Each list is a text file of codes, one
per line, beside this module; `manifest.json` records where each came from,
its version or publication date, the SHA-256 of the source file, and for
the NUTS lists the countries each year covers. The lists are copies of
published standards and are replaced as a whole when a source republishes;
`config.DQ_RULESET_VERSION` is raised when that happens.

A classification year that is not bundled, or a country a bundled year
does not cover, is never treated as "invalid": the caller reports a
limited assessment (specification 6.2, P11).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property, lru_cache
from importlib.resources import files

NUTS_YEARS: tuple[int, ...] = (2010, 2013, 2016, 2021, 2024)


@dataclass(frozen=True)
class ReferenceData:
    countries: frozenset[str]
    """ISO 3166-1 alpha-2 codes."""
    currencies_active: frozenset[str]
    currencies_historic: frozenset[str]
    nuts3: Mapping[int, frozenset[str]]
    """NUTS level 3 codes by classification year."""
    nuts_countries: Mapping[int, frozenset[str]]
    """The two-letter prefixes each year's list covers. Eurostat's own
    prefixes (`EL`, `UK`), which are not ISO codes, are among them."""
    manifest: Mapping[str, object]

    @cached_property
    def currencies(self) -> frozenset[str]:
        """Active and historic codes: the ESMA currency type permits both."""
        return self.currencies_active | self.currencies_historic

    @cached_property
    def nuts_prefixes(self) -> frozenset[str]:
        """Every country prefix of any bundled NUTS year."""
        return frozenset().union(*self.nuts_countries.values())


def _codes(name: str) -> frozenset[str]:
    text = files(__package__).joinpath(name).read_text(encoding="utf-8")
    return frozenset(line for line in text.splitlines() if line)


@lru_cache(maxsize=1)
def load_reference_data() -> ReferenceData:
    nuts3 = {year: _codes(f"nuts3_{year}.txt") for year in NUTS_YEARS}
    manifest = json.loads(files(__package__).joinpath("manifest.json").read_text(encoding="utf-8"))
    return ReferenceData(
        countries=_codes("iso3166_1_alpha2.txt"),
        currencies_active=_codes("iso4217_active.txt"),
        currencies_historic=_codes("iso4217_historic.txt"),
        nuts3=nuts3,
        nuts_countries={y: frozenset(c[:2] for c in codes) for y, codes in nuts3.items()},
        manifest=manifest,
    )
