# Analysis view

`POST /api/process` with `analysis_only=true`, or a background job
submitted with the same flag (served by `GET /api/jobs/{id}/result`),
returns pool stratifications and the Execution Summary for the GUI's
**Analyze pool** view. The code
lives in `src/esma_milan/analysis/`: `stratifications.py` holds one
function per cut, `labels.py` the ESMA code → label maps, `__init__.py`
the ordered registry the API walks.

All cuts are computed from the Stage-7 `combined_flattened` frame: one
row per loan, with the **main property's** attributes joined on (the
property `select_main_property` picks per collateral group). For a loan
backed by several properties, the whole loan balance is attributed to
the main property's region, type, occupancy and valuation method. That is
the same convention the workbook's Execution Summary uses.

## The eleven stratifications

Keys are the JSON keys in `stratifications`; order is the tile order in
the GUI.

| Key | Title | Source column | Type / chart | Rows |
|---|---|---|---|---|
| `interest_rate_type` | Interest Rate Type | `interest_rate_type` (RREL42) | categorical / pie | 13 ESMA codes |
| `amortisation_type` | Amortisation Type | `amortisation_type` (RREL35) | categorical / pie | 5 codes |
| `current_interest_rate` | Current Interest Rate | `current_interest_rate` (RREL43, percent units) | bucketed / bar | 17 buckets, 0.5% steps to 8%, then `8%+` |
| `seasoning` | Seasoning (months) | `calc_seasoning` × 12 | bucketed / bar | `0-12`, `12-24`, `24-36`, `36-60`, `60-120`, `120+ months` |
| `current_ltv` | Current LTV | `calc_current_LTV` (decimal) | bucketed / bar | `<=50%`, `50-70%`, `70-80%`, `80-90%`, `90-100%`, `>100%` |
| `geographic` | Geographic Distribution | `geographic_region_collateral` | categorical / bar | top 10 regions by balance, then `Other` |
| `property_type` | Property Type | `property_type` (RREC9) | categorical / pie | 9 codes |
| `property_valuation_type` | Property Valuation Type | `final_valuation_method` (RREC14/RREC18 code list) | categorical / pie | 9 codes |
| `loan_purpose` | Loan Purpose | `purpose` (RREL27) | categorical / pie | 13 codes |
| `occupancy` | Occupancy | `occupancy_type` (RREC7) | categorical / pie | 5 codes |
| `employment_type` | Employment Type | `employment_status` (RREL13) | categorical / pie | 9 codes |

Conventions:

- **ESMA-coded categoricals** show every published code as a row, in
  taxonomy order, even at count 0, labelled `CODE — Description` verbatim
  from the taxonomy's "CONTENT TO REPORT" column (the bundled
  `src/esma_milan/data/esma_rre_taxonomy.xlsx`, identical to the R
  reference's copy; two trims: the "underlying exposure" qualifier and
  the FOWN/POWN "i.e. …" clauses). Nulls and codes outside the taxonomy
  go to a `UNK — Unknown / not in ESMA taxonomy` row that appears only
  when its count is above zero.
- **Geographic** collapses regions beyond the top ten into `Other`; null
  regions are also counted in `Other`.
- **Bucketed numerics** use right-closed intervals `(a, b]`; the first
  bucket includes 0. Rows whose value is null or non-finite are excluded
  from the buckets and reported in `note` ("N loans excluded from buckets
  due to missing <column>."), while `total` still counts the whole pool.
- `count_pct` and `balance_pct` are decimals in [0, 1] against the table
  total (the bucketed total for numerics, so the table sums to 100%).
- `weighted_average` is the balance-weighted mean of the raw loan-level
  values (not bucket midpoints) and is populated only for the three
  bucketed numerics: months for seasoning, decimal for LTV, percent units
  for the rate. It is `null` for every categorical and for geography.
- If a cut cannot be computed (typically a missing column on this
  dataset) its entry carries `error` and empty rows; the other cuts are
  unaffected.
- The display labels in `labels.py` are a GUI concern and may change; the
  parity-sensitive ESMA → MILAN numeric code maps live separately in
  `pipeline/milan_map.py`.

## Execution Summary

`execution_summary` is Sheet 1 of the workbook as a list of
`{label, value}` rows: 38 base metrics (deal, cut-off date, aggregation
method, balances, borrower and loan counts, averages, concentration,
WA interest rate, reset and maturity metrics, WA and threshold LTVs,
interest-rate-type, payment-type, purpose, employment and arrears shares)
followed by one `Loan Count: <structure type>` row per structure type
present. Values are the strings the workbook writes (currencies as
`1,234,567.89`, percentages as `12.34%`, dates as ISO, missing as
`<not available>`), so the frontend renders them verbatim.
