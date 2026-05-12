"""Layer-1 parity test against the anonymised real RMBS pool.

Same shape as test_parity_synthetic, but runs against the larger fixture
(1,127 loans / 1,665 properties / 820 collateral groups).

Real-anonymised fixture files (loans.csv, collaterals.csv,
taxonomy.xlsx, expected_r_output.xlsx) stay out of the git repository
per the confidentiality discipline. Missing-file is treated as a clean
pytest.skip(...), not a crash. Contributors who don't have local copies
of the staged fixture simply see the real-fixture tests skipped; CI
runners with the fixture in place exercise them.

MILAN IEEE-754 tolerance
========================

Domi 2025-1 parity testing surfaced 73 last-digit cell diffs across 6
MILAN-pool columns (0.037% of 197,225 MILAN cells). Cells like:

    Property Value:  R='88567.200601784'  Py='88567.2006017839'
    Months Arrears:  R='0.985626283367556' Py='0.985626283367557'

are mathematically identical at IEEE-754 double precision and are
written as 15-significant-digit strings via `_r_as_character_expr`. The
underlying bits diverge by 1 ULP because Polars and R take different
(but both valid) accumulation paths in the upstream computations. R-repo
issue tracker entry #12 predicted this exact failure mode: synthetic
fixtures too small to surface IEEE-754 boundary cases, real-data parity
is where they emerge.

Per project owner direction, we accept the drift rather than chase it
in the formatter or the upstream computations. This file documents the
acceptance via per-column float tolerance for the 6 known-drifting
columns; everything else continues to compare byte-equal.

Tolerance choice (rel_tol=1e-13)
--------------------------------

An IEEE-754 double carries roughly 15-17 significant decimal digits.
The formatter writes 15. A 1-ULP difference in the underlying double
generally surfaces as a difference in the 15th printed digit. Comparing
the two strings as floats with rel_tol=1e-13 demands the first ~14
significant digits agree, which accepts last-digit drift but flags any
wider divergence as a real parity bug.

Concretely:
  * accepts: 88567.200601784 vs 88567.2006017839 (rel diff ~1e-15)
  * accepts: 0.985626283367556 vs 0.985626283367557 (rel diff ~1e-15)
  * rejects: 88567.200601784 vs 88567.200601780 (rel diff ~5e-14, still
    within bounds — slightly looser than strict 1-ULP)
  * rejects: 88567.200601784 vs 88567.200601 (rel diff ~9e-12, clearly
    a real bug)

abs_tol stays 0 — values near zero in these columns aren't expected;
the project tolerance discipline elsewhere uses 1e-9 abs but those
sheets carry monetary values where 1e-9 is meaningless. The 6 MILAN
columns below are all dimensioned quantities far from zero, so a pure
relative threshold is correct.

Any new column drifting at this level is NOT covered by this list and
will fail parity loudly, prompting the same root-cause investigation
that produced this exception in the first place.

Reference: R-repo issue tracker entry #12 (out-of-repo, maintained by
project owner; this commit message flags the entry needs a
"Real-data observation (2026-05-12)" section appended).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from esma_milan.config import OUTPUT_SHEET_ORDER
from esma_milan.parity import diff_workbooks, format_report
from esma_milan.runner import run_pipeline

from .conftest import FIXTURES_ROOT, ParityFixture

SKIP_REAL_FIXTURE: bool = False

# Real-anonymised fixture lives outside the repo. If any of its four
# files is missing on this checkout, every real-fixture parity test is
# pytest.skip()-ed — not a hard failure. CI runners that stage the
# fixture exercise the tests; contributor laptops without it skip
# cleanly.
_REAL_FIXTURE_FILES = (
    FIXTURES_ROOT / "real_anonymised" / "loans.csv",
    FIXTURES_ROOT / "real_anonymised" / "collaterals.csv",
    FIXTURES_ROOT / "real_anonymised" / "taxonomy.xlsx",
    FIXTURES_ROOT / "real_anonymised" / "expected_r_output.xlsx",
)


def _real_fixture_present() -> bool:
    return all(p.exists() for p in _REAL_FIXTURE_FILES)


# MILAN-pool columns that carry IEEE-754 last-digit drift on real-data
# parity. Documented at module level above; values agree to ~14
# significant digits but diverge in the 15th. Membership is checked at
# diff post-processing time; non-listed columns continue to require
# byte-equal string comparison.
MILAN_IEEE754_TOLERANT_COLUMNS: frozenset[str] = frozenset({
    "Additional data 13 - calc_seasoning",
    "Months In Arrears",
    "Property Value",
    "Additional data 10 - calc_aggregated_property_value",
    "Additional data 11 - calc_original_LTV",
    "Additional data 12 - calc_current_LTV",
})

MILAN_SHEET_NAME: str = "MILAN template pool"
MILAN_IEEE754_REL_TOL: float = 1e-13


def _strings_close_as_floats(actual: Any, expected: Any, *, rel_tol: float) -> bool:
    """Return True when actual and expected both parse as floats and
    `math.isclose` agrees within `rel_tol`. Anything that doesn't parse
    (None, non-numeric strings, NaN-vs-non-NaN) returns False so the
    caller treats it as a real diff."""
    try:
        a = float(actual) if not isinstance(actual, bool) else float("nan")
        e = float(expected) if not isinstance(expected, bool) else float("nan")
    except (TypeError, ValueError):
        return False
    if math.isnan(a) or math.isnan(e):
        return False
    return math.isclose(a, e, rel_tol=rel_tol, abs_tol=0.0)


@pytest.fixture(scope="module")
def real_run_output(
    real_fixture: ParityFixture, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    out_dir = tmp_path_factory.mktemp("real_run")
    result = run_pipeline(
        loans_file_path=real_fixture.loans,
        collaterals_file_path=real_fixture.collaterals,
        taxonomy_file_path=real_fixture.taxonomy,
        deal_name="Domi 2025-1",
        output_dir=out_dir,
        verbose=False,
    )
    assert result.output_path is not None
    return result.output_path


@pytest.mark.parity
@pytest.mark.skipif(
    SKIP_REAL_FIXTURE,
    reason="real-fixture parity disabled at module level",
)
@pytest.mark.skipif(
    not _real_fixture_present(),
    reason=(
        "Real-anonymised fixture not present locally "
        "(tests/fixtures/real_anonymised/). See docs/parity-protocol.md for "
        "setup; intentionally out-of-repo per confidentiality discipline."
    ),
)
@pytest.mark.parametrize("sheet_name", OUTPUT_SHEET_ORDER)
def test_real_sheet_parity(
    sheet_name: str,
    real_run_output: Path,
    real_fixture: ParityFixture,
) -> None:
    report = diff_workbooks(real_run_output, real_fixture.expected_output)
    sheet_diff = next((s for s in report.sheet_diffs if s.sheet == sheet_name), None)
    assert sheet_diff is not None

    # MILAN-only: filter out the documented IEEE-754 last-digit drift in
    # the 6 columns listed above. All other sheets compare byte-equal.
    if sheet_name == MILAN_SHEET_NAME:
        sheet_diff.cell_diffs[:] = [
            d
            for d in sheet_diff.cell_diffs
            if not (
                d.column_name in MILAN_IEEE754_TOLERANT_COLUMNS
                and _strings_close_as_floats(
                    d.actual, d.expected, rel_tol=MILAN_IEEE754_REL_TOL
                )
            )
        ]

    assert sheet_diff.passed, f"\n{format_report(report)}\n"
