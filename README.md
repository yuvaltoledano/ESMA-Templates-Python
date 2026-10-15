# ESMA-Templates-Python

Python port of the ESMA → MILAN pipeline for European RMBS reporting:
it reads ESMA loan-level and collateral templates, reconciles the
many-to-many loan/property structure, flattens the pool to one row per
loan and maps it to Moody's 175-column MILAN template. The R reference
implementation at `r_reference/` is the contract; the Python output must
match it cell for cell.

## Status

The port is complete. All ten workbook sheets pass Layer-1 parity against
the staged R output for both fixtures (`tests/fixtures/synthetic`, 8 loans;
`tests/fixtures/real_anonymised`, 1,126 active loans). The CLI, the HTTP
API and the local web frontend all run the same `run_pipeline` function.

Known gaps (one line each):

- The workbook writer emits values only: no header styling, column widths
  or autofilter, unlike R's openxlsx output. Parity compares values.
- The "real" fixture is committed to the repo; identifiers are perturbed
  but originator/issuer legal names and LEIs are present
  (see `tests/fixtures/README.md`).
- The deal name is validated only at the API layer; the CLI passes it
  straight into the output path.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv installs it).

```sh
git clone --recurse-submodules <repo-url> ESMA-Templates-Python
cd ESMA-Templates-Python
uv python install 3.12
uv sync --frozen --extra dev

# Full test suite (unit + integration + Layer-1 parity on both fixtures)
uv run pytest

# Run the pipeline on the synthetic fixture
uv run esma-milan run \
  --loans tests/fixtures/synthetic/loans.csv \
  --collaterals tests/fixtures/synthetic/collaterals.csv \
  --taxonomy tests/fixtures/synthetic/taxonomy.xlsx \
  --deal SYNTHETIC_FIXTURE \
  --output /tmp/out

# Diff that output against the staged R workbook (prints PASS / FAIL)
uv run python scripts/run_parity_check.py \
  "/tmp/out/SYNTHETIC_FIXTURE/2024-06-30 SYNTHETIC_FIXTURE Flattened loans and collaterals.xlsx" \
  tests/fixtures/synthetic/expected_r_output.xlsx
```

The output file is named `<pool_cutoff_date> <deal> Flattened loans and
collaterals.xlsx` under `<output>/<deal>/`. `--output` defaults to
`data/clean` (git-ignored).

`--taxonomy` is required unless the current directory is the repo root
with the submodule checked out, in which case the default is
`r_reference/inputs/ESMA template taxonomy.xlsx`.

Other flags on `esma-milan run`:

| Flag | Meaning |
|---|---|
| `--aggregation auto\|by_loan\|by_group` | Property-value convention. `auto` (default) lets Stage 6 detect it; an ambiguous detection falls back to `by_loan` with a warning. |
| `--min-coverage 0..1` | Minimum coverage of the chosen loan-ID column against the collaterals file. Default 0.85; below it the run errors, between it and 1.0 it warns. |
| `--dry-run` | Run every stage but write no workbook. |
| `--verbose` / `--quiet` | Toggle INFO logging (default verbose). |

### API server and frontend

```sh
uv run esma-milan-server          # FastAPI on 0.0.0.0:8000, 4 workers

cd frontend
npm ci
npm run dev                       # Vite dev server on http://localhost:5173,
                                  # proxies /api/* to localhost:8000
```

Endpoints, form fields and response shapes are documented in
[docs/api.md](docs/api.md); the analysis view's stratifications in
[docs/analysis.md](docs/analysis.md); the frontend in
[frontend/README.md](frontend/README.md).

## Architecture

```
src/esma_milan/
├── config.py              # constants mirrored from r_reference/R/utils.R
├── nd.py                  # ESMA "no data" (ND/NA/blank) detection
├── io_layer/
│   ├── read_taxonomy.py   # taxonomy XLSX -> {field code: field name}
│   ├── read_csv.py        # CSV read, taxonomy rename, snake_case headers
│   ├── clean_names.py     # janitor::clean_names port
│   ├── date_parsing.py    # ISO / Excel-serial date parsing
│   ├── sanitize_utf8.py   # UTF-8 sanitiser port (repairs non-UTF-8 CSVs)
│   └── write_workbook.py  # 10-sheet openpyxl writer (values only)
├── pipeline/
│   ├── stage1.py          # read + clean + date normalisation
│   ├── filters.py         # Stage 2: active loans, RRE properties
│   ├── identifiers.py     # Stage 3: calc_loan_id / borrower / property IDs
│   ├── graph.py           # Stage 4: bipartite graph, collateral groups
│   ├── classification.py  # Stage 5: structure types 1-5
│   ├── enriched.py        # loans_enriched / properties_enriched joins
│   ├── valuation.py       # Stage 6 detection + valuation selection
│   ├── flatten.py         # Stage 7/8: one row per loan, LTVs, seasoning
│   ├── mapping_tables.py  # the four ID-mapping sheets
│   ├── milan_map.py       # Stage 9: 175-column MILAN template
│   └── exec_summary.py    # Stage 10: Execution Summary sheet
├── analysis/              # pool stratifications for the GUI (11 cuts)
├── parity/                # cell-by-cell workbook diff library
├── api/                   # FastAPI service (server, handlers, schemas)
├── runner.py              # run_pipeline(): single entry point for CLI + API
└── cli.py                 # Click CLI (`esma-milan run`)

frontend/                  # React + Vite + Tailwind single-page GUI
scripts/run_parity_check.py  # CLI wrapper around the parity diff library
r_reference/               # READ-ONLY git submodule, pinned at v1.1-r-reference
docs/                      # parity protocol, API and analysis references
tests/
├── fixtures/              # synthetic + real_anonymised inputs and R outputs
├── unit/                  # per-module unit tests
├── integration/           # per-stage tests on the synthetic fixture
└── parity/                # Layer-1 cell-by-cell parity, both fixtures
```

Every module that mirrors R code cites the R file and line range it
follows (`Mirrors r_reference/R/<file>:<lines>`); the citations are
checked against the pinned submodule tag.

## Parity

Parity is Layer 1 only: the Python pipeline runs on each fixture's inputs
and its workbook is diffed cell by cell against the committed
`expected_r_output.xlsx` produced by the R reference. No R installation
is needed. Tolerances, group-id canonicalisation and the per-column
IEEE-754 tolerance applied to six MILAN columns on the real fixture are
described in [docs/parity-protocol.md](docs/parity-protocol.md).

**Never modify `r_reference/`.** If something there looks wrong, raise it
against the R repository; the fix lands upstream and the submodule is
re-pinned.

## Domain reference

The R repository carries the analyst-facing documentation of the
pipeline's business logic. Read it there rather than in this repo:

- `r_reference/docs/esma-milan-guide.html` and
  `r_reference/docs/esma-milan-reference-updated.html`: analyst guide and
  field reference.
- `r_reference/docs/ESMA_PROCESSING_DOCUMENTATION.md`: code-aligned
  stage-by-stage description.
- `r_reference/rmarkdown/analyst-walkthrough.Rmd`: the walkthrough that
  defines the synthetic fixture.
- `r_reference/mapping/MILAN_mapping_implementation_guide.md`: the
  ESMA → MILAN field mapping.

## Development

```sh
uv run pytest                          # full suite
uv run pytest tests/unit -v            # unit only
uv run pytest tests/integration -v     # stage tests on the synthetic fixture
uv run pytest tests/parity -v          # Layer-1 parity only
uv run pytest -k "synthetic" -v        # one fixture
uv run ruff check src tests scripts    # lint
uv run mypy                            # strict type-check
cd frontend && npm run lint && npm run build
```

CI (`.github/workflows/ci.yml`) runs the Python checks with the locked
dependency set (`uv sync --frozen`) and the frontend lint and build on
Node 22.

## License

See [LICENSE](LICENSE) (Apache 2.0).
