"""Layer-2 parity test: Python output vs a *live* R re-run, synthetic pool.

Layer 1 (test_parity_synthetic.py) diffs the Python output against the
pre-computed tests/fixtures/synthetic/expected_r_output.xlsx. That catches
Python-side regressions but is blind to R-side drift: if an upstream R-package
update changes R's output, the stale expected_r_output.xlsx still encodes the
*old* R behaviour and Layer 1 keeps passing.

Layer 2 closes that gap. It re-runs the R reference in CI against the same
synthetic fixture inputs, then diffs the Python output against the
freshly-generated R workbook. A divergence here means either Python drifted
or R drifted - either way it is a real parity break worth a human look.

This test is gated by `@pytest.mark.live_r` so only the nightly workflow
(`pytest -m "live_r"`) collects it. It additionally skips cleanly when
`Rscript` is not on PATH (developer laptops without R) or when the
`r_reference` submodule has not been checked out, so a stray invocation in
PR CI is a no-op rather than an error.

The R entry point is `r_reference/R/main.R`, invoked via `Rscript` from the
`r_reference/` working directory so its `here::here()` path resolution works.
Non-interactive mode requires `--loans`, `--collaterals` and `--deal`;
`--taxonomy` and `--output` are optional. R writes the workbook to
`<output>/<deal_name>/<pool_cutoff_date> <deal_name> Flattened loans and
collaterals.xlsx`, so the test globs for the produced .xlsx rather than
reconstructing that filename.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from esma_milan.parity import diff_workbooks, format_report
from esma_milan.runner import run_pipeline

from .conftest import REPO_ROOT, ParityFixture

# Deal name shared with the Layer 1 synthetic test so R and Python produce
# the same output filename / subdirectory.
DEAL_NAME: str = "SYNTHETIC_FIXTURE"

R_REFERENCE_ROOT: Path = REPO_ROOT / "r_reference"
R_MAIN_SCRIPT: Path = R_REFERENCE_ROOT / "R" / "main.R"

# Running R against the 8-loan synthetic fixture takes seconds; 5 minutes is
# a generous ceiling that still fails fast if R hangs.
R_TIMEOUT_SECONDS: int = 300

_RSCRIPT: str | None = shutil.which("Rscript")


@pytest.mark.live_r
@pytest.mark.skipif(
    _RSCRIPT is None,
    reason="R not available in this environment (Rscript not on PATH)",
)
def test_python_matches_live_r_on_synthetic_fixture(
    synthetic_fixture: ParityFixture,
    tmp_path: Path,
) -> None:
    """Re-run R against the synthetic fixture and diff Python's output
    against the freshly-generated R workbook.

    Catches R-side drift that the pre-computed expected output used by
    Layer 1 cannot detect. Synthetic is too small to surface the IEEE-754
    last-digit drift documented for the real_anonymised fixture, so this
    asserts strict cell-for-cell parity across all sheets - same bar as
    the Layer 1 synthetic test.
    """
    if not R_MAIN_SCRIPT.exists():
        pytest.skip(
            f"r_reference submodule not checked out (missing {R_MAIN_SCRIPT}); "
            "run `git submodule update --init --recursive`"
        )

    # --- Re-run R against the fixture inputs ------------------------------
    r_out_dir = tmp_path / "r_output"
    cmd = [
        _RSCRIPT or "Rscript",  # _RSCRIPT is non-None here (skipif guard).
        "R/main.R",
        f"--loans={synthetic_fixture.loans}",
        f"--collaterals={synthetic_fixture.collaterals}",
        f"--taxonomy={synthetic_fixture.taxonomy}",
        f"--deal={DEAL_NAME}",
        f"--output={r_out_dir}",
    ]
    proc = subprocess.run(
        cmd,
        cwd=R_REFERENCE_ROOT,
        capture_output=True,
        text=True,
        timeout=R_TIMEOUT_SECONDS,
        check=False,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"R re-run failed (exit {proc.returncode}).\n"
            f"command: {' '.join(cmd)}\n"
            f"--- R stdout ---\n{proc.stdout}\n"
            f"--- R stderr ---\n{proc.stderr}"
        )

    # R writes to <output>/<deal_name>/<...>.xlsx; glob rather than rebuild
    # the cutoff-date-derived filename. Exclude Excel lock artefacts.
    r_workbooks = sorted(
        p for p in r_out_dir.rglob("*.xlsx") if not p.name.startswith("~$")
    )
    assert len(r_workbooks) == 1, (
        f"expected exactly one R workbook under {r_out_dir}, found {r_workbooks}\n"
        f"--- R stdout ---\n{proc.stdout}\n--- R stderr ---\n{proc.stderr}"
    )
    r_workbook = r_workbooks[0]

    # --- Run the Python pipeline against the same inputs ------------------
    result = run_pipeline(
        loans_file_path=synthetic_fixture.loans,
        collaterals_file_path=synthetic_fixture.collaterals,
        taxonomy_file_path=synthetic_fixture.taxonomy,
        deal_name=DEAL_NAME,
        output_dir=tmp_path / "python_output",
        verbose=False,
    )
    assert result.output_path is not None, "run_pipeline returned None outside dry_run"

    # --- Compare via the existing parity harness --------------------------
    report = diff_workbooks(result.output_path, r_workbook)
    assert report.passed, f"\n{format_report(report)}\n"
