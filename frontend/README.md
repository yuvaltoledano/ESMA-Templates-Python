# ESMA-MILAN GUI

A single-page web frontend for the ESMA-MILAN pipeline API, in the
"Dashboard" look: a navy top bar, one upload row, a job status bar, and
for an analysed pool a KPI strip, the Execution Summary and eleven
stratification tiles. One form: upload a loans CSV and a collaterals
CSV, enter a deal name, then either

- **Process & download**: submit a background job (`POST /api/jobs`),
  show its progress, then offer **Download workbook** (a link to
  `GET /api/jobs/{id}/result`, the ten-sheet workbook) and **Delete from
  server**, or
- **Analyze pool**: submit the same job with `analysis_only=true`, then
  fetch the result JSON and render it inline, with **Delete from server**
  in the status bar.

This is intentionally minimal: no dry-run preview, no job history (the
page remembers only the job it started, via `sessionStorage`, so a
reload of the same tab resumes polling), no auth of its own
(`../docs/deploy.md` puts a password in front of it on a server). It
exists so the API can be exercised through real use rather than `curl`.

## Code layout

```
src/
  main.jsx                 bundled IBM Plex fonts (@fontsource, latin subsets), mounts <App/>
  App.jsx                  composition only: shell + page sections, wires the hooks; no fetch calls
  index.css                Tailwind directives and the page's base colours and type size
  lib/
    tokens.js              every colour, radius and shadow; read by tailwind.config.js and the charts
    api.js                 every fetch: createJob, getJobStatus, fetchAnalysis, deleteJob, getHealth;
                           parses ErrorResponse; typed ApiError / NetworkError; the "Cannot reach API" string
    format.js              formatElapsed, formatCurrency, formatPercent, formatEurAxis, formatBucketTick,
                           formatWeightedAverage, formatFileSize, formatTime (pure functions)
    chartPalette.js        the 15-colour palette, the muted "Other" colour, the bar colour
    storage.js             the sessionStorage slot for the current job (guarded; never throws)
  hooks/
    useHealth.js           30 s poll of /api/health -> {status, maxUploadMb, retentionHours, lastChecked}
    useJob.js              submit / poll / stop / delete / resume; owns the 2 s interval and the failure cutoff
    useAnalysis.js         one GET of the result URL for a done analysis job, kept until the job changes
  components/
    shell/                 TopBar (wordmark, nav slot, HealthStatus), Footer
    upload/                UploadCard (the one-row form), FileField
    job/                   JobStatusBar, StatePill, ErrorPanel
    analysis/              AnalysisHeader, KpiStrip, KpiCard, ExecutionSummaryCard, StratificationGrid,
                           StratificationTile, StratTable, charts/ (DonutChart, BalanceBarChart, ChartTooltip)
    ui/                    Button, Card, Disclosure, Spinner (the shared primitives; no component library)
```

The one rule: components receive data and callbacks as props; only the
hooks and `lib/api.js` talk to the network or to storage. Every
component carries a short JSDoc comment stating what it renders and its
props. Design tokens live in `src/lib/tokens.js` and reach the markup as
Tailwind classes (`bg-brand-navy`, `text-ink-muted`, `rounded-card`,
...); no component spells a hex. Adding a page to the top bar is one
entry in `TopBar.jsx`'s `NAV_ITEMS`; an entry with `external: true`
opens in a new tab.

`public/mapping-reference.html` is the analyst-facing mapping reference
(see the repository `README.md`, "Domain reference"). Vite copies
`public/` into `dist/` unchanged, so the page is served next to the app at
`/mapping-reference.html` with no build step and no React; the "Mapping
reference" item in the top bar opens it. It is self-contained (inline
CSS and script, no fonts or scripts fetched from any host) and is edited
by hand.

## Job flow

Both buttons go through the same states: idle, uploading, polling, then
success or error. While polling, the page asks `GET /api/jobs/{id}` every
2 seconds and the status bar shows a Queued or Running pill with "Queued,
waiting for the worker to start..." or "Stage n of 11: ..." plus a
progress bar and the elapsed time. **Stop watching** leaves the job
running on the server; reloading the same tab picks it up again. A job
that was deleted or has expired on the server ends polling with the
API's `404 job_not_found`. After five consecutive failed polls (no
response at all) the page reports "Cannot reach API". A finished job
shows a Done pill with the actions for its mode; a failed job shows a
Failed pill and the error panel (message, code, expandable details,
Delete from server). The result's retention period, shown under the
status bar, comes from `GET /api/health`.

## Analyze pool view

Below the status bar the view shows:

- a header line with the deal name, the aggregation method the pipeline
  chose, the group count and any pipeline warnings;
- six KPI cards: Current balance, Loans and Properties from the
  response's `summary`, and Borrowers, WA LTV and WA rate verbatim from
  the Execution Summary rows "Number Of Borrowers", "WA Current LTV" and
  "WA Interest Rate" ("—" when a row is missing);
- the **Execution summary** card (the workbook's Sheet 1: 38 metric rows
  plus one per structure type), values rendered verbatim as the server
  pre-formats them, in a scrolling box with a sticky header;
- eleven **stratification tiles** in a three-column grid (two below
  1280px, one on a phone), in the order the API returns them. Each tile
  has a donut (`pie`) or bar (`bar`) chart (recharts) and, behind a
  **Show table** disclosure, a table of bucket, count, % count, balance,
  % balance with a total row; donut tiles carry a colour swatch per row
  in place of a legend. The three bucketed numeric tiles (seasoning,
  current LTV, current interest rate) show the pool balance-weighted
  average under the title. A tile whose cut could not be computed shows
  the API's `error` text instead of the chart; a `note` (e.g. loans
  excluded for missing values) is shown under the table.

See `../docs/analysis.md` for the semantics of each cut.

## Validation, errors and health

- Client-side checks mirror the API: both CSVs and a deal name are
  required, and the deal name must match `[A-Za-z0-9 ._-]{1,100}`. A file
  over the per-file cap that `GET /api/health` reports (`max_upload_mb`:
  50 on a laptop, 600 in the server deployment) gets a warning under its
  field; the server enforces the limit.
- Every API failure arrives as the structured `ErrorResponse`
  (`error`, `message`, optional `details`) and is shown with its code and
  an expandable details block. A non-JSON failure, or no response at all,
  is reported as "Cannot reach API at localhost:8000".
- The top bar polls `GET /api/health` on load and every 30 seconds and
  shows API healthy / unhealthy / unreachable; the footer shows the time
  of the last check. The worker is not reported: the API has no endpoint
  for it.

## Stack

React 19 + Vite 8 + Tailwind CSS v3 + recharts 3, in plain JavaScript
(no TypeScript). IBM Plex Sans and IBM Plex Mono are bundled from
`@fontsource` at build time; the page requests nothing from a font CDN.

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

## Follow-ups

- Tests: `vitest` would cover `lib/format.js`, `lib/storage.js` and the
  `useJob` state machine without a browser; not added in the redesign PR.
- A Jobs page (list, resume, delete) would need a job-list endpoint the
  API does not have today.
