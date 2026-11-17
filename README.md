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
`tests/fixtures/real_anonymised`, 1,126 active loans). It is used in three
ways: the CLI; the synchronous API (`POST /api/process`) for pools up to
about 30,000 loans; and the web page, which submits every pool as a
background job run by a separate `esma-milan-worker` process.
[docs/deploy.md](docs/deploy.md) runs all of it on one server with Docker
Compose and Caddy. All paths call the same `run_pipeline` function.

Known gaps (one line each):

- The workbook writer emits values only: no header styling, column widths
  or autofilter, unlike R's openxlsx output. Parity compares values.
- The "real" fixture is committed to the repo; identifiers are perturbed
  but the original lender's and originator's legal names, LEIs and
  countries, and the EPC provider name, are real
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

Beside it the command writes the data quality report on the two input
files, as four files named after the workbook (`<stem>` is the workbook's
name without `.xlsx`):

| File | Content |
|---|---|
| `<stem> data quality summary.json` | the summary: every rule's outcome, the finding groups with a sentence each, the two inventories, the population of every loan record, the waterfall from the file to the MILAN output, the links between the files, the provenance, and the state of the other three files |
| `<stem> data quality detail.csv` | every finding row: the rule, the record and column, the value as written, and the record's identifiers |
| `<stem> dropped loans.csv` | every loan that does not reach the MILAN output for its balance or its collateral, with the waterfall line it is assigned to and every other cause seen |
| `<stem> data quality report.xlsx` | the same report as a workbook: `DQ Summary`, `DQ Fields`, `DQ Columns`, `DQ Checks`, `DQ Findings`, `DQ Rows`, `DQ Dropped loans` |

The rules are in `docs/data-quality/specification.md`: the per-column
rules, the rules about the files themselves (headers, duplicates,
identifiers, cut-off dates, currencies), and the reconciliation of the
pool with the pipeline's own stage results. The files are written on a
`--dry-run` and when the pipeline fails as well; with no cut-off date
established, `unknown-cutoff` takes its place in the names. Both CSVs
start with a schema line and write every value of the input exactly, with
one escape rule for backslashes and control characters (specification
10.3). The workbook's `DQ Rows` sheet holds at most 500,000 finding rows
and says so when it is cut; the detail CSV always holds every row.

The report is additive: the MILAN workbook, the exit status and the
messages are what they are without it, and it states no accept or reject
decision. A rule shown with severity Stop has no effect on the run. The
files are assessed in a process of their own before the conversion, and
the report is written in another after it, so the conversion's memory use
is unchanged; on a 500,000-loan pool the report adds between one and two
minutes to the run. The checks across columns and records, the acceptance rules and the
web page arrive in later releases; the API does not produce the report
yet.

The ESMA taxonomy that renames and types the CSV columns is not an
input. It is a fixed ESMA artefact bundled with the package
(`src/esma_milan/data/esma_rre_taxonomy.xlsx`), so the CLI runs from any
directory. There is no `--taxonomy` flag; R's `main.R` has one, and this
is a documented deviation (see `cli.py`).

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

The page submits every pool as a background job (`POST /api/jobs`) and
polls for progress; a separate process, `uv run esma-milan-worker`, runs
the jobs one at a time from a directory on disk (`data/jobs` by default).
Pools up to about 30,000 loans can also be run synchronously with
`POST /api/process`. See [docs/api.md](docs/api.md).

To run the service and frontend on a server, over HTTPS and behind a
password, follow [docs/deploy.md](docs/deploy.md) (Docker Compose +
Caddy; the `Dockerfile`, `docker-compose.yml`, `deploy/Caddyfile` and
`.env.example` at the repo root are what it uses).

## Architecture

```
src/esma_milan/
├── config.py              # constants mirrored from r_reference/R/utils.R
├── nd.py                  # ESMA "no data" (ND/NA/blank) detection
├── errors.py              # PipelineDataError (typed data faults) and the catalogue of fault sites
├── data/
│   └── esma_rre_taxonomy.xlsx  # the ESMA RRE taxonomy, identical to the R reference's copy
├── io_layer/
│   ├── read_taxonomy.py   # bundled taxonomy XLSX -> {field code: field name}, {field code: FORMAT}
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
├── quality/               # data quality report on the two uploaded files (additive; see docs/data-quality/)
│   ├── catalogue.py       # the 107 fields and every rule, as reviewed data
│   ├── refdata/           # ISO 3166-1, ISO 4217, NUTS level 3 lists and their manifest
│   ├── audit_read.py      # the audit's own read of each file; header resolution
│   ├── states.py          # value states (empty, ND options, markers, present) and parsing
│   ├── column_rules/      # per-column rules P1 to P13 and the engine that runs them
│   ├── family_a.py        # rules about the files themselves: A1 to A16
│   ├── populations.py     # population, identity-confidence and lifecycle labels per record
│   ├── links.py           # links between collateral records and loan records
│   ├── waterfall.py       # the waterfall, the dropped-loans list, G8 and A17
│   ├── inventories.py     # expected-field and actual-column inventories
│   ├── result.py          # result rows and coverage counts
│   ├── store.py           # the row-level detail on disk, in a working directory of the run
│   ├── child.py           # the assessment and the report writing, each in a process of its own
│   ├── assess.py          # assess_files(), complete_report() and the report object
│   ├── export.py          # the summary document
│   ├── detail_csv.py      # the detail CSV and the dropped-loans CSV
│   ├── report_workbook.py # the report workbook
│   └── report_files.py    # the four files written beside the MILAN workbook
├── analysis/              # pool stratifications for the GUI (11 cuts)
├── parity/                # cell-by-cell workbook diff library
├── api/
│   ├── server.py          # FastAPI endpoints: /api/health, /api/process, /api/jobs
│   ├── handlers.py        # upload validation, pipeline calls, error classification
│   ├── schemas.py         # Pydantic response models and error codes
│   ├── jobs.py            # on-disk background-job store (one directory per job)
│   └── worker.py          # the `esma-milan-worker` process that runs queued jobs
├── runner.py              # run_pipeline(): single entry point for CLI, API and worker
└── cli.py                 # Click CLI (`esma-milan run`)

frontend/                  # React + Vite + Tailwind single-page GUI
├── public/mapping-reference.html  # analyst-facing mapping reference, served as-is by the web UI
scripts/run_parity_check.py  # CLI wrapper around the parity diff library
r_reference/               # READ-ONLY git submodule, pinned at v1.1-r-reference
docs/                      # parity protocol, API, analysis, deployment and decisions references
Dockerfile                 # app (Python) and web (Caddy + built frontend) images
docker-compose.yml         # app + worker + web containers for the server deployment
deploy/Caddyfile           # HTTPS, password, static frontend, /api proxy
.env.example               # per-server settings (copy to .env on the server)
tests/
├── fixtures/              # synthetic + real_anonymised inputs and R outputs
├── unit/                  # per-module unit tests
├── integration/           # per-stage tests on the synthetic fixture
├── quality/               # data quality report: rules on constructed data, reviewed results on both fixtures
└── parity/                # Layer-1 cell-by-cell parity, both fixtures
```

Every module that mirrors R code cites the R file and line range it
follows (`Mirrors r_reference/R/<file>:<lines>`); the citations were
verified by hand against `v1.1-r-reference` in the documentation pass
of #24.

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

[`frontend/public/mapping-reference.html`](frontend/public/mapping-reference.html)
is the analyst-facing description of what this tool does to a pool: the
nine steps from the two ESMA files to the flattened pool, every one of the
175 MILAN columns with its ESMA source, rule and missing-value behaviour
(searchable), the code maps, the Execution Summary metrics and a
data-quality checklist, with worked examples from the synthetic fixture.
It is hand-written from the Python pipeline, carries a version line, and
is served by the web UI ("Mapping reference" in the top bar, or
`/mapping-reference.html`). Open the file directly in a browser to read it
without the service. A change to the mapping is not complete until this
document and its version line are updated (`CLAUDE.md`).

The R repository carries the documentation of the original pipeline:

- `r_reference/docs/esma-milan-guide.html` and
  `r_reference/docs/esma-milan-reference-updated.html`: the R analyst
  guide and field reference, which the document above supersedes for
  this tool.
- `r_reference/docs/ESMA_PROCESSING_DOCUMENTATION.md`: code-aligned
  stage-by-stage description.
- `r_reference/rmarkdown/analyst-walkthrough.Rmd`: the walkthrough that
  defines the synthetic fixture.
- `r_reference/mapping/MILAN_mapping_implementation_guide.md`: the
  ESMA → MILAN field mapping.
- `r_reference/inputs/ESMA template taxonomy.xlsx`: the ESMA RRE
  taxonomy. The package bundles the same file as
  `src/esma_milan/data/esma_rre_taxonomy.xlsx`; a test keeps the two
  byte-identical, so an ESMA republication is a submodule bump plus a
  copy.

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
uv run esma-milan-worker               # background-job worker (needed for the page's jobs)
docker build --target app . && docker build --target web .   # the two deployment images
```

CI (`.github/workflows/ci.yml`) runs the Python checks with the locked
dependency set (`uv sync --frozen`), the frontend lint and build on
Node 22, and builds the `app` and `web` Docker images.

Working rules for Claude Code sessions are in `CLAUDE.md`; the reasons
behind the current design are in `docs/decisions.md`.

## License

See [LICENSE](LICENSE) (Apache 2.0).
