# Parity protocol

## What "parity" means

The Python pipeline at `src/esma_milan/` must produce output that is
**bit-for-bit identical** to the R reference at `r_reference/R/`, on every
input. "Bit-for-bit" is operationalised as the cell-by-cell comparison
implemented in `src/esma_milan/parity/diff.py`:

- **Numeric**: equal within `1e-9` absolute or `1e-12` relative tolerance.
- **String**: byte-equal after openpyxl read.
- **Date**: stored in the format the R writer emits per sheet (Excel
  serial number in `Cleaned ESMA loans`, `Cleaned ESMA properties`, and
  `Combined flattened pool`; ISO date string in `MILAN template pool`).
  See `config.SHEETS_WITH_EXCEL_SERIAL_DATES`.
- **Group IDs**: `collateral_group_id` and its `calc_*` aliases are
  arbitrary integer labels assigned by the connected-components algorithm
  (igraph in R, networkx in Python). They are canonicalised before
  comparison: each side relabels its groups by sorted-min `calc_loan_id`,
  starting at 1, and only the canonicalised labels are diffed.

## Three layers

### Layer 1 — Static fixture parity (default, no R required)

The fixtures under `tests/fixtures/synthetic/` and
`tests/fixtures/real_anonymised/` each include a pre-computed
`expected_r_output.xlsx`. Layer 1 runs the Python pipeline on the inputs
and asserts cell equality against this static file.

This is the layer that runs in normal CI and on every developer machine.
**No R installation required.** Driven by `tests/parity/test_parity_*.py`.

### Layer 2 — Live R execution (CI nightly)

Re-runs the R pipeline against the same inputs from a clean R install via
`r-lib/actions/setup-r@v2`, then compares both outputs. Catches drift if
the static `expected_r_output.xlsx` ever goes stale (e.g. submodule pointer
bumped without fixture regen). Local devs can run this if they happen to
have R installed, but it is **not required**: Layer 1 is the primary parity
contract and Layer 2 is a drift-detector.

### Layer 3 — Differential fuzzing (CI nightly + weekly)

Hypothesis-generated random pools, run through both R and Python, diffed.
Catches edge cases the curated fixtures miss. Requires a live R install
(CI only). Same setup as Layer 2.

## Cell-by-cell diff protocol

Implemented in `esma_milan.parity.diff_workbooks`:

1. Open both workbooks read-only via openpyxl.
2. Compare sheet names and order. Mismatch: report and stop comparing.
3. Per sheet:
   - Read header row and data rows.
   - Compare column order. Mismatch: report and skip cell-level diff.
   - Compare row counts. Mismatch: note it; diff the overlap.
   - Build group-id relabel maps for both sides (canonicalisation).
   - Compare each cell with the type-aware tolerance rules above.
4. Aggregate into a `DiffReport` with per-sheet, per-cell detail. The
   report is human-readable via `format_report()` and machine-readable via
   `report.summary()`.

## Running parity locally

```sh
# Run the full Python test suite (unit + parity Layer 1):
uv run pytest

# Diff one specific Python output against an expected workbook:
uv run python scripts/run_parity_check.py \
  path/to/python_output.xlsx \
  tests/fixtures/synthetic/expected_r_output.xlsx
```

The script exits 0 on parity, 1 on diff, 2 on file-level errors.

## Stage-by-stage progress tracking

`tests/parity/test_parity_synthetic.py::SHEET_STATUS` enumerates each of
the 10 sheets and marks it `pending` (xfail) or `passing`. As stages port
over, flip entries from `pending` to `passing`. CI requires all 10 sheets
in the synthetic fixture to be `passing` before the real-fixture parity
suite is enabled (see `SKIP_REAL_FIXTURE` in `test_parity_real.py`).

## Regenerating fixtures

The staged `expected_r_output.xlsx` files are the parity contract. They
should change only when:

1. The R reference (`r_reference/`) is updated and we deliberately bump
   the submodule pointer to a newer tagged version.
2. The synthetic dataset definition changes in
   `r_reference/rmarkdown/analyst-walkthrough.Rmd` and we re-export it.

Both cases require running the R pipeline against the fixture inputs in
a clean R environment and committing the new XLSX. The procedure lives
in `scripts/regenerate_fixtures.sh` (TODO: written when first regen is
needed). Do not regenerate fixtures from the Python pipeline — that
trivially makes parity self-referential and defeats the entire test.
