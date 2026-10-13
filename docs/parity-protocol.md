# Parity protocol

## What "parity" means

The Python pipeline at `src/esma_milan/` must produce output that is
cell-for-cell identical to the R reference at `r_reference/R/`, on every
input. "Identical" is operationalised as the comparison implemented in
`src/esma_milan/parity/diff.py`:

- **Numeric**: equal within `1e-9` absolute or `1e-12` relative tolerance.
- **String**: byte-equal after openpyxl read.
- **Date**: stored in the format the R writer emits per sheet (Excel
  serial number in `Cleaned ESMA loans`, `Cleaned ESMA properties`, and
  `Combined flattened pool`; ISO date string in `MILAN template pool`).
  See `config.SHEETS_WITH_EXCEL_SERIAL_DATES`. A serial on one side and
  an ISO string on the other is a diff.
- **Empty cells**: `None` and `""` compare equal.
- **Group IDs**: canonicalised before comparison (next section).

## Group-id canonicalisation

`collateral_group_id` and its aliases (`calc_collateral_group_id`,
`Additional data 4 - calc_collateral_group_id`) are arbitrary integer
labels assigned by the connected-components algorithm (igraph in R,
networkx in Python). Only the partition they induce matters, so the diff
relabels both sides before comparing:

1. **In-sheet relabel.** For sheets that carry both a row-unique
   identifier (`calc_loan_id`, `Loan Identifier`, `calc_property_id`, …,
   searched in a fixed priority order) and a group-id column, each side
   builds a `{gid -> 1..N}` map by sorting groups on their lex-smallest
   identifier, and group-id cells are compared through that map.
2. **Cross-sheet alignment.** `Group classifications` is one row per
   group with no loan or property column of its own. The diff derives a
   context from each workbook's `Cleaned ESMA loans` sheet
   (`calc_loan_id`, `collateral_group_id`), relabels the group-id cells
   through it and reorders the sheet's rows by each group's lex-smallest
   `calc_loan_id`, so the same partition produces the same row order
   regardless of the integer labels each side chose. The alignment only
   fires when both workbooks have a populated `Cleaned ESMA loans` sheet;
   otherwise rows are compared positionally.

## Layer 1: static fixture parity (no R required)

The fixtures under `tests/fixtures/synthetic/` and
`tests/fixtures/real_anonymised/` each include a pre-computed
`expected_r_output.xlsx`. Layer 1 runs the Python pipeline on the inputs
and asserts cell equality against this static file, sheet by sheet, for
all ten sheets. This is the parity contract. It runs in CI and on every
developer machine; no R installation is required. Driven by
`tests/parity/test_parity_synthetic.py` and `test_parity_real.py`.

There is no other parity layer; the staged workbooks are the only
reference.

### Per-column tolerance on the real fixture

On the real fixture, six MILAN-sheet columns show last-digit drift: the
values are written as 15-significant-digit strings via R's
`as.character()` semantics, and Polars and R reach the underlying double
by different (both valid) accumulation orders, so about 0.04% of MILAN
cells differ by one unit in the 15th digit. `test_parity_real.py`
accepts this for exactly these columns and nothing else:

```
Additional data 13 - calc_seasoning
Months In Arrears
Property Value
Additional data 10 - calc_aggregated_property_value
Additional data 11 - calc_original_LTV
Additional data 12 - calc_current_LTV
```

A cell in one of these columns passes when both sides parse as floats
and agree within `rel_tol = 1e-13` (about 14 significant digits), with
no absolute tolerance. Any other column, and any wider divergence in
these six, fails parity. The synthetic fixture is compared byte-equal
with no such tolerance.

## Cell-by-cell diff protocol

Implemented in `esma_milan.parity.diff_workbooks`:

1. Open both workbooks read-only via openpyxl.
2. Compare sheet names and order. A mismatch is reported and fails the
   report; the sheets are still diffed by name.
3. Per sheet:
   - Read header row and data rows (trailing all-empty rows stripped).
   - Compare column order. Mismatch: report and skip cell-level diff.
   - Compare row counts. Mismatch: note it; diff the overlap.
   - Build the group-id relabel maps (in-sheet or cross-sheet).
   - Compare each cell with the type-aware tolerance rules above, up to
     `max_cell_diffs_per_sheet` diffs per sheet (default 50).
4. Aggregate into a `DiffReport` with per-sheet, per-cell detail. The
   report is human-readable via `format_report()` and machine-readable
   via `report.summary()`.

## Running parity locally

```sh
# Full Python test suite (unit + integration + parity Layer 1):
uv run pytest

# Only the parity tests:
uv run pytest tests/parity -v

# Diff one Python output against an expected workbook:
uv run python scripts/run_parity_check.py \
  path/to/python_output.xlsx \
  tests/fixtures/synthetic/expected_r_output.xlsx
```

The script exits 0 on parity, 1 on diff, 2 on file-level errors. It is
also the behaviour-preservation gate for refactors: run the pipeline
before and after a change and diff the two Python outputs against each
other.

## Regenerating fixtures

The staged `expected_r_output.xlsx` files are the parity contract. They
change only when:

1. The R reference (`r_reference/`) is updated and the submodule pointer
   is deliberately bumped to a newer tagged version.
2. The synthetic dataset definition changes in
   `r_reference/rmarkdown/analyst-walkthrough.Rmd` and is re-exported.

The procedure is manual and requires an R installation with the packages
listed in `r_reference/DESCRIPTION`:

1. From the `r_reference` checkout, run the R pipeline on the fixture
   inputs, for example:

   ```sh
   Rscript R/main.R \
     --loans=../tests/fixtures/synthetic/loans.csv \
     --collaterals=../tests/fixtures/synthetic/collaterals.csv \
     --taxonomy=../tests/fixtures/synthetic/taxonomy.xlsx \
     --deal=SYNTHETIC_FIXTURE \
     --output=/tmp/r_out
   ```

   (`main.R` reads `--loans`, `--collaterals`, `--taxonomy`, `--deal` and
   `--output`; see `r_reference/R/cli_helpers.R`.)
2. Copy the produced `<cutoff> <deal> Flattened loans and collaterals.xlsx`
   to the fixture directory as `expected_r_output.xlsx`.
3. Tag the R repository, update the submodule pointer to that tag and
   commit the new workbook, the pointer and the tag name in `README.md`
   together.

Never regenerate fixtures from the Python pipeline: that makes parity
self-referential and defeats the test.
