# ESMA-MILAN GUI

A minimal web frontend for the ESMA-MILAN pipeline API. One screen, one
form: upload a loans CSV and a collaterals CSV (plus an optional taxonomy
XLSX), enter a deal name, then either

- **Process Pool & Download**: submit a background job (`POST /api/jobs`),
  show its progress, then offer **Download workbook** (a link to
  `GET /api/jobs/{id}/result`, the ten-sheet workbook) and **Delete from
  server**, or
- **Analyze Pool**: submit the same job with `analysis_only=true`, then
  fetch the result JSON and render it inline, with **Delete from server**
  underneath.

This is intentionally minimal: no dry-run preview, no job history (the
page remembers only the job it started, via `sessionStorage`, so a
reload of the same tab resumes polling), no auth of its own
(`../docs/deploy.md` puts a password in front of it on a server). It
exists so the API can be exercised through real use rather than `curl`.

## Job flow

Both buttons go through the same states: idle, uploading, polling, then
success or error. While polling, the page asks `GET /api/jobs/{id}` every
2 seconds and shows "Queued, waiting for the worker to start..." or
"Stage n of 11: ..." with a progress bar and the elapsed time. **Stop
watching** leaves the job running on the server; reloading the same tab
picks it up again. A job that was deleted or has expired on the server
ends polling with the API's `404 job_not_found`. After five consecutive
failed polls (no response at all) the page reports "Cannot reach API".
The result's retention period, shown under the download link, comes from
`GET /api/health`.

## Analyze Pool view

Below the form the view shows:

- a header with loan, property and group counts, total current balance,
  the aggregation method the pipeline chose and any pipeline warnings;
- the **Execution Summary** table (the workbook's Sheet 1: 38 metric rows
  plus one per structure type), values rendered verbatim as the server
  pre-formats them;
- eleven **stratification tiles** in a two-column grid, in the order the
  API returns them (interest rate type, amortisation type, current
  interest rate, seasoning, current LTV, geography, property type,
  property valuation type, loan purpose, occupancy, employment type).
  Each tile has a pie or bar chart (recharts) and a table of bucket,
  count, % count, balance, % balance with a total row; pie tiles carry a
  colour swatch per row in place of a legend. The three bucketed numeric
  tiles (seasoning, current LTV, current interest rate) show the pool
  balance-weighted average under the title. A tile whose cut could not be
  computed shows the API's `error` text instead of the chart; a `note`
  (e.g. loans excluded for missing values) is shown under the table.

See `../docs/analysis.md` for the semantics of each cut.

## Validation, errors and health

- Client-side checks mirror the API: both CSVs and a deal name are
  required, and the deal name must match `[A-Za-z0-9 ._-]{1,100}`. Files
  over the per-file cap that `GET /api/health` reports (`max_upload_mb`:
  50 on a laptop, 600 in the server deployment) get a warning; the
  server enforces the limit.
- Every API failure arrives as the structured `ErrorResponse`
  (`error`, `message`, optional `details`) and is shown with its code and
  an expandable details block. A non-JSON failure, or no response at all,
  is reported as "Cannot reach API at localhost:8000".
- The footer polls `GET /api/health` on load and every 30 seconds and
  shows healthy / unhealthy / unreachable.

## Stack

React 19 + Vite 8 + Tailwind CSS v3 + recharts 3, in plain JavaScript
(no TypeScript).

## Running it

The API must be running on `localhost:8000`, and the background-job
worker must be running too, or every job stays "Queued". In one terminal,
from the repo root:

```sh
uv run esma-milan-server
```

In a second terminal, also from the repo root (jobs are stored under
`data/jobs`):

```sh
uv run esma-milan-worker
```

In a third terminal:

```sh
cd frontend
npm ci
npm run dev
```

Then open <http://localhost:5173>.

Requests to `/api/*` are proxied to `http://localhost:8000` by the Vite
dev server (see `vite.config.js`), so the browser only ever makes
same-origin requests and the API needs no CORS configuration.

## Scripts

- `npm run dev`: dev server with HMR
- `npm run build`: production build into `dist/` (emits a chunk-size
  warning for the recharts bundle; expected)
- `npm run lint`: ESLint
- `npm run preview`: serve the production build locally
