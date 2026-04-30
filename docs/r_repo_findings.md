# R-repo findings

Tracker of issues found in `r_reference/` while porting to Python. None block
Python parity (the port mirrors R's actual behaviour, bugs included), but
each is worth eventual upstream review and a possible re-tag of the R
reference.

| #  | Title                                                                  | Severity                          | Discovered in    | Status |
|----|------------------------------------------------------------------------|-----------------------------------|------------------|--------|
| 1  | `test-parse-iso-or-excel-date.R`: serial 42642 expectation off by 2    | Test bug — assertion incorrect    | Stage 1          | Open   |
| 2  | `ESMA_PROCESSING_DOCUMENTATION.md`: NA-token list missing `ND`/`ND3`   | Doc bug — text incorrect          | Stage 1          | Open   |
| 3  | Project brief: `EXCLUDED_COLLATERAL_TYPES` count miscount (19 vs 20)   | Doc nit                           | Stage 1          | Resolved (Python matches code) |
| 4  | `clean_names()` camelCase splitting: ESMA headers don't exercise it    | Documented limitation             | Stage 1          | Resolved (Python omits, with test pinning the boundary) |
| 5  | R `main.R` CLI surface narrower than Python `cli.py`                   | Deliberate documented deviation   | Stage 1          | Resolved (Python is strict superset) |
| 6  | `pipeline.R:458-464` `tryCatch(...)` silently swallows join errors     | Code bug — silent failure mode    | Stage 8.5        | Open   |
| 7  | Naming asymmetry: `loan_file_ids.loan_id` vs `loans_enriched.calc_loan_id` | Code quality — naming inconsistency | Stage 8.5      | Open   |

---

## #1 — `test-parse-iso-or-excel-date.R`: serial 42642 expectation off by 2

**Location:** `r_reference/tests/testthat/test-parse-iso-or-excel-date.R::"parses Excel serials from the user's sample"`

**Issue:** The test claims `parse_iso_or_excel_date("42642") == as.Date("2016-10-01")`. R's actual `as.Date(42642, origin="1899-12-30")` arithmetic produces `2016-09-29` (verified independently with Python's identical proleptic-Gregorian arithmetic). The test as written should be failing its assertion. The pair `(42642, "2016-10-01")` is off by 2 days; the correct R-arithmetic value is `2016-09-29`.

**Suggested upstream fix:** either correct the expected value in the test, or correct the input value (Excel-DATE(2016,10,1) is serial 42644, not 42642). Either fix lands cleanly without touching `parse_iso_or_excel_date()` itself.

**Python parity impact:** none. The Python port matches R's actual arithmetic, which is what the live pipeline relies on (synthetic fixture round-trips serial 43539 ↔ 2019-03-15 correctly). The Python test pins the arithmetically-correct values.

---

## #2 — `ESMA_PROCESSING_DOCUMENTATION.md`: NA-token list missing `ND`/`ND3`

**Location:** `r_reference/docs/ESMA_PROCESSING_DOCUMENTATION.md` §3.3.

**Issue:** The doc lists CSV NA tokens as `("", "NA", "ND1", "ND2", "ND4", "ND5")`, omitting `"ND"` and `"ND3"`. The actual code in `r_reference/R/utils.R:75-82` lists all seven: `("", "NA", "ND", "ND1", "ND2", "ND3", "ND4", "ND5")`. The MD doc is stale.

**Suggested upstream fix:** update the doc to list all seven tokens.

**Python parity impact:** none. The Python port matches the code, not the doc.

---

## #3 — Project brief: `EXCLUDED_COLLATERAL_TYPES` count miscount (19 vs 20)

**Location:** Internal project brief (not in `r_reference/`).

**Issue:** Brief said "20-code list"; actual `r_reference/R/utils.R:22-33` defines 19 codes. The Python port matches the constant.

**Status:** Resolved — flagged at Stage 1 sign-off, brief amended verbally.

---

## #4 — `clean_names()` camelCase splitting

**Location:** `clean_names()` is part of the `janitor` R package; ESMA headers don't exercise camelCase splitting.

**Issue:** The Python port's custom `clean_names_polars()` (Stage 1) implements a subset of `janitor::clean_names()` sufficient for ESMA inputs. CamelCase splitting (e.g. `RREL30Currency` → `rrel30_currency`) is NOT implemented, since no ESMA header uses camelCase.

**Status:** Resolved — Python helper documents the omission, with a unit test pinning the assumption (`test_clean_names_table` covers the actual ESMA shapes; a future header that introduces camelCase will fail this test, prompting the one-line addition).

**Python parity impact:** none for ESMA inputs.

---

## #5 — R `main.R` CLI surface narrower than Python `cli.py`

**Location:** `r_reference/R/main.R::run_main()` vs `src/esma_milan/cli.py`.

**Issue:** R exposes only `--loans / --collaterals / --taxonomy / --deal / --output` as CLI flags. `aggregation_method`, `min_coverage`, `dry_run`, `verbose` are programmatic-only.

**Status:** Resolved — Python CLI is a strict superset per project brief §8 (deliberate, documented deviation).

---

## #6 — `pipeline.R:458-464` `tryCatch(...)` silently swallows join errors

**Location:** `r_reference/R/pipeline.R:458-464`:

```r
loans_to_properties <- tryCatch(
  {
    safe_left_join(loans_to_properties, loans_group_classifications, by = c("loan_id" = "calc_loan_id"))
  },
  error = function(e) loans_to_properties
)
```

**Issue:** The classification-tail join on `loans_to_properties` is wrapped in `tryCatch(...)` that silently falls back to the unannotated form on any error. The classification tail (`collateral_group_id`, `cross_collateralized_set`, `full_set`, `structure_type`) is a **contract** on the workbook output, not an optional enhancement. Fail-quiet on a contract join is the wrong default — if the join unexpectedly fails on real data (column name change upstream, type mismatch, missing classification frame, etc.), the sheet emits without classification columns and nobody notices until a downstream consumer trips over the absent columns.

**Suggested upstream fix:** replace the `tryCatch` with explicit error logging via `logger::log_error()` and re-raise. Or remove the wrapper entirely so the error propagates to the runner-level `tryCatch` that handles top-level errors.

**Python parity impact:** none for normal inputs (the join always succeeds when upstream stages have produced their expected outputs). Stage 8.5's Python port joins unconditionally — a real failure would surface as a clear exception rather than a silently-degraded sheet.

---

## #7 — Naming asymmetry: `loan_file_ids.loan_id` vs `loans_enriched.calc_loan_id`

**Location:** `r_reference/R/pipeline.R:335-345` (rename `calc_loan_id` → `loan_id` for the mapping-table base frames) and the join at `:460` (which has to bridge the two names via `c("loan_id" = "calc_loan_id")`).

**Issue:** Mid-pipeline, R renames `calc_loan_id` → `loan_id` to drive the mapping-table pivots, but `loans_enriched` (used elsewhere in the pipeline) keeps `calc_loan_id`. The classification join at line 460 has to bridge the two via `by = c("loan_id" = "calc_loan_id")`. The asymmetry is harmless given correct join specs but is the kind of thing that bites during refactors when somebody assumes one frame's column name applies to the other.

**Suggested upstream fix:** standardise on a single name. Either keep `calc_loan_id` everywhere (drop the rename in `loan_file_ids` definition) and emit it as the sheet header; or rename `loans_enriched.calc_loan_id` → `loan_id` earlier and propagate. The first option is cleaner since `calc_loan_id` is the canonical name everywhere else.

**Python parity impact:** none. Stage 8.5's Python port preserves R's renaming (loan_file_ids uses `loan_id`) so the workbook output is byte-equal.
