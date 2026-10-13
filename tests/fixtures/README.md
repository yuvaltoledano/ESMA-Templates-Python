# Parity-test fixtures

These fixtures verify cell-for-cell parity between the Python pipeline
and the R reference implementation pinned at `r_reference/`.

- `synthetic/`: 8-loan, 12-collateral synthetic dataset defined in
  `r_reference/rmarkdown/analyst-walkthrough.Rmd`, built so each of the
  five structure types appears exactly once. Headers are already
  snake_case.
- `real_anonymised/`: a real RMBS pool (1,295 loan rows, 1,952 collateral
  rows; 1,126 active loans / 1,664 properties / 819 collateral groups
  after filtering) with ESMA `RREL*`/`RREC*` headers.

Each subfolder contains:

- `loans.csv`, `collaterals.csv`, `taxonomy.xlsx`: pipeline inputs
- `expected_r_output.xlsx`: the R output for those inputs, used as the
  parity baseline

## What is and is not anonymised in `real_anonymised/`

The fixture is committed to this repository. Loan, borrower and
collateral identifiers are perturbed and the deal name used in tests
("Domi 2025-1") is the one in the data. The following fields are **not**
redacted and carry their original values:

- `RREL79` originator legal name, `RREL80` originator LEI, `RREL81`
  originator country;
- `RREL82` issuer/SSPE legal name, `RREL83` its LEI, `RREL84` its country;
- `RREC11` guarantor legal name on the collateral file.

Treat the directory accordingly when sharing the repository.

Do not regenerate these fixtures without coordinating: regeneration
changes the parity baseline and must match a re-tagged `r_reference/`
version (see `docs/parity-protocol.md`, "Regenerating fixtures").
