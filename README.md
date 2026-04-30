# ESMA-Templates-Python

Python implementation of the ESMA → MILAN pipeline for European RMBS
reporting. Bit-for-bit parity with the R reference at `r_reference/`.

## Status

🚧 Under active development. Tracking the §9 deliverable order from the
project brief; current state is "skeleton + parity harness against a
no-op stub". Stage progress per sheet is tracked in
`tests/parity/test_parity_synthetic.py::SHEET_STATUS`.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```sh
git clone --recurse-submodules <repo-url> ESMA-Templates-Python
cd ESMA-Templates-Python
uv python install 3.12
uv sync --extra dev

# Run the test suite (unit + Layer-1 parity)
uv run pytest

# Run the pipeline (currently a no-op stub)
uv run esma-milan run \
  --loans tests/fixtures/synthetic/loans.csv \
  --collaterals tests/fixtures/synthetic/collaterals.csv \
  --taxonomy tests/fixtures/synthetic/taxonomy.xlsx \
  --deal SYNTHETIC_DEMO \
  --output /tmp/out

# Diff a Python output against an expected R workbook
uv run python scripts/run_parity_check.py \
  /tmp/out/SYNTHETIC_DEMO/STUB\ SYNTHETIC_DEMO\ Flattened\ loans\ and\ collaterals.xlsx \
  tests/fixtures/synthetic/expected_r_output.xlsx
```

## Architecture

```
src/esma_milan/
├── config.py            # constants from r_reference/R/utils.R
├── io_layer/            # CSV reading, taxonomy, dates, workbook write
├── pipeline/            # filters, identifiers, graph, classification,
│                        # valuation, flatten, MILAN map (175 fields)
├── parity/              # cell-by-cell diff library (used by tests + CLI)
├── api/                 # FastAPI service (lands after pipeline parity)
├── runner.py            # single library entry point, used by CLI + API
└── cli.py               # Click CLI

r_reference/             # READ-ONLY git submodule pinned at v1.0
tests/
├── fixtures/            # synthetic + real_anonymised pre-staged with
│                        # canonical R output workbooks
├── unit/                # per-stage unit tests
└── parity/              # Layer-1 cell-by-cell parity tests
```

## Parity protocol

See [docs/parity-protocol.md](docs/parity-protocol.md). Three layers:

- **Layer 1** (default, no R required): Python output vs staged R
  expected output. Runs in normal CI and on every dev machine.
- **Layer 2** (CI nightly): Live R re-execution as drift detector.
- **Layer 3** (CI nightly + weekly): Hypothesis differential fuzzing.

The R reference is the contract. **Never modify `r_reference/`** — if
something there looks wrong, flag it in writing per §10 of the brief and
the reference is updated upstream and re-pinned.

## Development

```sh
uv run pytest                         # full suite
uv run pytest tests/unit -v           # unit only
uv run pytest tests/parity -v         # Layer-1 parity only
uv run pytest -k "synthetic" -v       # one fixture
uv run ruff check                     # lint
uv run mypy                           # strict type-check
```

## License

See [LICENSE](LICENSE).
