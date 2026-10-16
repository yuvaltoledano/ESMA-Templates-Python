# HTTP API

The FastAPI service in `src/esma_milan/api/` is a thin wrapper around
`runner.run_pipeline`. Start it with `uv run esma-milan-server`
(uvicorn on `0.0.0.0:8000`, 4 worker processes by default; see "Limits"
for the environment variables that change them). There is no
authentication in the app; it is meant to run locally behind the Vite dev
proxy or behind a reverse proxy you control. `docs/deploy.md` describes the
supported server deployment (Caddy: HTTPS and a password in front).

## Endpoints

### `GET /api/health`

Liveness check plus the two deployment settings the page needs at load:

```json
{"status": "ok", "max_upload_mb": 50, "job_retention_hours": 12.0}
```

`max_upload_mb` is the per-file cap for `POST /api/jobs`
(`ESMA_MILAN_MAX_UPLOAD_MB`); `job_retention_hours` is how long a finished
job's result stays available (`ESMA_MILAN_JOB_RETENTION_HOURS`).

### `POST /api/process`

Multipart form upload. Runs the pipeline on the uploaded files and
returns one of three bodies depending on the flags.

| Field | Type | Required | Notes |
|---|---|---|---|
| `loans` | file | yes | ESMA loans CSV. Content-Type `text/csv` or `application/csv`. |
| `collaterals` | file | yes | ESMA collaterals CSV. Same content types. |
| `taxonomy` | file | no | Taxonomy XLSX (`application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`). Defaults to `r_reference/inputs/ESMA template taxonomy.xlsx` relative to the server's working directory; a 500 is returned if that file is absent and none was uploaded. |
| `deal_name` | string | yes | 1 to 100 characters from `[A-Za-z0-9 ._-]`. Flows into the output filename and the `Content-Disposition` header. |
| `aggregation` | `auto` \| `by_loan` \| `by_group` | no | Default `auto`. |
| `min_coverage` | float in [0, 1] | no | Default `0.85` (`config.DEFAULT_MIN_LOAN_ID_COVERAGE`). |
| `dry_run` | bool | no | Default `false`. Run the pipeline, write no workbook, return counts. |
| `analysis_only` | bool | no | Default `false`. Run the pipeline, return stratifications + Execution Summary as JSON. Mutually exclusive with `dry_run`. |

### Background jobs: `POST /api/jobs`, `GET /api/jobs/{id}`, `GET /api/jobs/{id}/result`, `DELETE /api/jobs/{id}`

`POST /api/process` runs the whole pipeline inside one request and
refuses pools above about 30,000 loans. Larger pools (the writer alone
takes about 16 minutes at 300,000 loans) go through a job: the request
returns at once with an id, a separate worker process
(`esma-milan-worker`) runs the pipeline, and the client polls for
progress and fetches the result when it is done. The frontend uses jobs
for every pool.

| Method | Path | Success | Errors |
|---|---|---|---|
| `POST` | `/api/jobs` | `202 {"job_id", "status_url", "result_url"}` | 400 `missing_field` / `invalid_deal_name` / `invalid_content_type`, 413 `file_too_large`, 422 `validation_error`, 429 |
| `GET` | `/api/jobs/{id}` | `200` `JobStatus` (below) | 404 `job_not_found`, 429 |
| `GET` | `/api/jobs/{id}/result` | `200` the workbook (same `Content-Disposition` as `/api/process`) or the `AnalysisResponse` JSON | 409 `job_not_ready` while queued or running (`details.state`), 409 carrying the job's own `error` when it failed, 404 `job_not_found`, 429 |
| `DELETE` | `/api/jobs/{id}` | `204` | 409 `job_not_ready` while queued or running, 404 `job_not_found`, 429 |

**Request.** The same multipart form as `/api/process` minus `dry_run`:
`loans`, `collaterals`, optional `taxonomy`, `deal_name`, `aggregation`,
`min_coverage`, and `analysis_only` (which selects the mode: `workbook`
or `analysis`). Validation is the same except that there is no
loan-count limit and the per-file cap is `ESMA_MILAN_MAX_UPLOAD_MB`
(default 50, so a laptop behaves as before; the server deployment sets
600). Uploads are streamed to the job's directory, never held in memory.

**Ids** are 32 lowercase hex characters (`uuid4`). Anything else is
answered `404 job_not_found` before the filesystem is touched.

**State machine.** `queued` -> `running` -> `done` or `failed`. A
`JobStatus` body:

```json
{
  "job_id": "c2b83a7325bd4e47bf55443ddb07a1c5",
  "state": "running",
  "mode": "workbook",
  "deal_name": "DEAL_2025-1",
  "created_at": "2026-10-16T12:35:59+00:00",
  "started_at": "2026-10-16T12:35:59+00:00",
  "finished_at": null,
  "stage": {"index": 7, "total": 11, "label": "Stage 7: flatten and derived fields"},
  "error": null,
  "result": null,
  "retention_hours": 12.0
}
```

`stage` advances through the pipeline's eleven boundaries ("Stage 1:
read and clean" ... "Mapping tables", "Writing workbook"); an analysis
job stops at nine, since no workbook is written. When `done`, `result`
is `{"filename", "size_bytes"}` and the result endpoint serves that file.
When `failed`, `error` is the usual `{error, message, details}` with the
same code `/api/process` would have returned for the same input
(`invalid_csv`, `internal_error`, ...), or one of the job-specific codes:

- `output_too_large`: a sheet would exceed Excel's limit of 1,048,576
  rows; nothing was written. `details` names the sheet and row count.
  (The synchronous endpoint returns the same code as a 413.)
- `interrupted`: the worker was restarted while the job was running (a
  redeploy, or the kernel killed it for memory). The job is not retried;
  submit the pool again.

**Retention.** A job's input files are deleted the moment it finishes.
The result stays until `DELETE` or until `ESMA_MILAN_JOB_RETENTION_HOURS`
after `finished_at` (jobs that never finished count from `created_at`),
when the worker's sweep removes the whole job directory.

**Storage.** One directory per job under `ESMA_MILAN_JOBS_DIR` (default
`data/jobs`, relative to the working directory; `/jobs` in the container,
a volume shared by the API and the worker).

## Response shapes

**Workbook download** (default). `200` with the `.xlsx` bytes,
`Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`
and `Content-Disposition: attachment; filename="<cutoff> <deal> Flattened loans and collaterals.xlsx"`.
The workbook never touches disk on the server.

**Dry run** (`dry_run=true`). `200` JSON:

```json
{
  "deal_name": "SYNTHETIC_FIXTURE",
  "chosen_aggregation": "by_loan",
  "loan_count": 8,
  "property_count": 12,
  "group_count": 5,
  "warnings": []
}
```

`loan_count` is the row count of the combined flattened pool,
`property_count` the properties surviving Stage 3, `group_count` the
number of collateral groups. `warnings` carries a note when `auto`
aggregation detection was ambiguous and fell back to `by_loan`.

**Analysis** (`analysis_only=true`). `200` JSON:

```json
{
  "deal_name": "...",
  "summary": {
    "loan_count": 8, "property_count": 12, "group_count": 5,
    "total_current_balance": 1550000.0,
    "chosen_aggregation": "by_loan", "warnings": []
  },
  "stratifications": {
    "<key>": {
      "title": "Interest Rate Type",
      "type": "categorical | bucketed",
      "chart_type": "pie | bar",
      "rows": [{"label": "...", "count": 3, "count_pct": 0.375,
                "balance": 350000.0, "balance_pct": 0.2258}],
      "total": {"count": 8, "balance": 1550000.0},
      "error": null, "note": null, "weighted_average": null
    }
  },
  "execution_summary": [{"label": "Deal Name", "value": "..."}]
}
```

The eleven stratification keys and their semantics are in
[analysis.md](analysis.md). `execution_summary` is the Execution Summary
sheet (38 base rows plus one per structure type), values pre-formatted
exactly as they appear in the workbook.

**Errors.** Every non-2xx response has the same body:

```json
{"error": "<code>", "message": "<client-safe summary>", "details": {...} | null}
```

| Status | `error` | When |
|---|---|---|
| 400 | `missing_field` | a required upload or form field is absent |
| 400 | `invalid_content_type` | an upload's Content-Type is not in the allowlist |
| 400 | `invalid_deal_name` | `deal_name` empty, over 100 chars, or has a disallowed character |
| 400 | `invalid_csv` | the pipeline rejected the data (unparseable CSV, missing or duplicate columns, missing `pool_cutoff_date`, …); `details.reason` holds the first line of the pipeline error |
| 400 | `validation_error` | `analysis_only` and `dry_run` both true |
| 422 | `validation_error` | a field is present but malformed (e.g. `min_coverage` out of range, or an unknown `aggregation` value) |
| 413 | `file_too_large` | a single upload exceeds the per-file cap: 50 MiB for `/api/process`, `ESMA_MILAN_MAX_UPLOAD_MB` (default 50) for `/api/jobs` |
| 413 | `size_limit_exceeded` | the loans CSV has more than ~30,000 data rows (counted by newlines before parsing); use `POST /api/jobs` |
| 413 | `output_too_large` | a sheet would exceed Excel's 1,048,576-row limit; nothing was written |
| 429 | `rate_limit_exceeded` | per-IP rate limit tripped |
| 500 | `internal_error` | anything unexpected; details stay in the server log |
| 404 | `job_not_found` | no such job: malformed id, unknown, expired or deleted |
| 409 | `job_not_ready` | the job has not finished (result or delete requested too early) |
| 409 | the job's own code | `GET .../result` on a failed job; also `interrupted` when the worker restarted mid-job |

## Limits

- Per-file upload cap: 50 MiB for `/api/process`, checked before the
  bytes are read; `ESMA_MILAN_MAX_UPLOAD_MB` (default 50) for
  `/api/jobs`.
- Pool-size guardrail: ~30,000 loan rows for the synchronous endpoint;
  larger pools go through `POST /api/jobs`, which has no row limit (a
  sheet may still not exceed Excel's 1,048,576 rows).
- Rate limits (slowapi, per client IP, in-memory counters per worker
  process, so the effective ceiling is the configured value times the
  number of workers): `10/minute` for `/api/process` and `POST /api/jobs`,
  `60/minute` for `/api/health`, `120/minute` for the job status, result
  and delete endpoints. Override with the environment variables
  `ESMA_MILAN_RATE_LIMIT_PROCESS`, `ESMA_MILAN_RATE_LIMIT_HEALTH` and
  `ESMA_MILAN_RATE_LIMIT_JOBS` (slowapi syntax).
- Behind a reverse proxy, set `ESMA_MILAN_TRUST_PROXY=1` so the
  rate-limit key is the first address in `X-Forwarded-For` rather than
  the proxy's own IP. Only do this when nothing but the proxy can reach
  the app (the Compose deployment in `docs/deploy.md` publishes no app
  port); with the flag unset, the peer IP is the key.
- Host, port and worker count: `ESMA_MILAN_HOST` (default `0.0.0.0`),
  `ESMA_MILAN_PORT` (default `8000`), `ESMA_MILAN_WORKERS` (default `4`).

For `/api/process`, uploads are written to a per-request temporary
directory that is removed before the response is sent; the pipeline's
input interface is path-based. Job uploads live in the job's `inputs/`
directory until the job finishes (see Retention above).
