# Parity-test fixtures

These fixtures are used to verify bit-for-bit parity between the Python
pipeline and the R reference implementation pinned at `r_reference/`.

- `synthetic/` — 8-loan synthetic dataset from `r_reference/rmarkdown/analyst-walkthrough.Rmd`.
- `real_anonymised/` — anonymised real RMBS pool, identifiers perturbed,
  counterparty names redacted.

Each subfolder contains:
- `loans.csv`, `collaterals.csv`, `taxonomy.xlsx` — pipeline inputs
- `expected_r_output.xlsx` — canonical R output, used as the parity baseline

Do not regenerate these fixtures without coordinating: regeneration changes
the parity baseline and must match a re-tagged `r_reference/` version.
