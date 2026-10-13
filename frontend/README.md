# ESMA-MILAN GUI

A minimal local web frontend for the ESMA-MILAN pipeline API. One
screen, one form: upload a loans CSV and a collaterals CSV (plus an
optional taxonomy XLSX), enter a deal name, then either

- **Process Pool & Download**: `POST /api/process` and save the returned
  ten-sheet workbook in the browser, or
- **Analyze Pool**: `POST /api/process` with `analysis_only=true` and
  render the result inline.

This is intentionally minimal: no dry-run preview, no history, no auth,
no deployment story. It exists so the API can be exercised through real
use rather than `curl`.

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
  over 50 MB get a warning; the server enforces the limit.
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

The API must be running on `localhost:8000`. In one terminal, from the
repo root:

```sh
uv run esma-milan-server
```

In another terminal:

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
