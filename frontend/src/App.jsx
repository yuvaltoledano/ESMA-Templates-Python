// ESMA-MILAN GUI: a single-page client for the pipeline API.
//
// Two modes share one upload form (loans CSV, collaterals CSV, optional
// taxonomy XLSX, deal name). Both submit a background job:
//   POST /api/jobs            -> 202 {job_id, status_url, result_url}
//   GET  /api/jobs/{id}       -> polled every 2 s: queued | running (with a
//                                stage label) | done | failed
//   GET  /api/jobs/{id}/result-> download mode: a plain link the browser
//                                follows (stored credentials apply);
//                                analysis mode: AnalysisResponse JSON
//                                rendered inline as the Analyze Pool view
//   DELETE /api/jobs/{id}     -> "Delete from server" once done or failed
// The current job id is kept in sessionStorage so a reload resumes polling.
// Every non-2xx response is the API's ErrorResponse ({error, message,
// details}); a non-JSON failure is treated as "API unreachable". The
// footer polls GET /api/health, which also reports the per-file upload
// cap the page warns about. All requests use relative /api URLs, proxied
// to localhost:8000 by the Vite dev server (vite.config.js).
// See docs/api.md and docs/analysis.md for the contract.
import { useCallback, useEffect, useState } from 'react'
import {
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

// Mirrors the API's deal_name rule (handlers.py::validate_deal_name):
// letters, digits, space, hyphen, underscore, period; length 1-100.
// The server stays the source of truth - this is just fast-fail UX.
const DEAL_NAME_PATTERN = /^[A-Za-z0-9 ._-]{1,100}$/

const NETWORK_ERROR_MESSAGE =
  'Cannot reach API at localhost:8000. Is `uv run esma-milan-server` running?'

// Background-job polling. The server's status rate limit (120/minute by
// default) comfortably covers one poll every 2 s; a short run of failed
// polls (a proxy hiccup, a redeploy) is tolerated before giving up.
const JOB_STORAGE_KEY = 'esma-milan-job'
const POLL_INTERVAL_MS = 2000
const MAX_POLL_FAILURES = 5

// sessionStorage is per tab and may be unavailable (private mode); every
// access is guarded so the page works without it.
function readStoredJob() {
  try {
    const raw = sessionStorage.getItem(JOB_STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    return parsed && typeof parsed.id === 'string' ? parsed : null
  } catch {
    return null
  }
}

function storeJob(job) {
  try {
    if (job) sessionStorage.setItem(JOB_STORAGE_KEY, JSON.stringify(job))
    else sessionStorage.removeItem(JOB_STORAGE_KEY)
  } catch {
    // Storage unavailable: a reload will not resume, nothing else changes.
  }
}

function errorFromBody(body) {
  return {
    type: 'api',
    code: body.error,
    message: body.message || 'The request failed.',
    details: body.details ?? null,
  }
}

function networkError() {
  return { type: 'network', message: NETWORK_ERROR_MESSAGE }
}

function formatElapsed(ms) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return minutes > 0 ? `${minutes} min ${seconds} s` : `${seconds} s`
}

// Categorical pie-chart palette. Indexed by spec-row position, so the
// biggest categoricals (interest rate type and loan purpose, 13 ESMA
// codes each, plus a possible UNK row) each get a distinct colour without
// cycling. The first 11 entries are the original palette; the last four
// extend it. The bar charts use only CHART_COLORS[0].
const CHART_COLORS = [
  '#0f766e', '#1d4ed8', '#b45309', '#7c3aed',
  '#be123c', '#0369a1', '#15803d', '#c2410c',
  '#475569', '#a16207', '#9f1239', '#4d7c0f',
  '#6d28d9', '#0e7490', '#b91c1c',
]

// EUR formatter without decimals - pool balances are large enough that
// cent-level precision in a stratification table is noise.
const EUR_FORMAT = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'EUR',
  maximumFractionDigits: 0,
})
const PCT_FORMAT = new Intl.NumberFormat('en-US', {
  style: 'percent',
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
})

// Compact-EUR formatter for bar-chart Y-axis ticks. Real pools push
// into the hundreds of millions; the full euro string overflows the
// axis gutter. Convention is standard structured-finance:
//   < €1K   -> exact (€500)
//   €1K-€1M -> €1K, €500K (no decimals)
//   €1M-€1B -> €1.5M, €500M (one decimal)
//   >= €1B  -> €1.5B, €2.3B (one decimal)
// The chart tooltip still uses the full EUR formatter so hover-over a
// bar shows the precise number.
function formatEurAxis(value) {
  if (value == null || Number.isNaN(value)) return ''
  const abs = Math.abs(value)
  const sign = value < 0 ? '-' : ''
  if (abs < 1_000) return `${sign}€${Math.round(abs)}`
  if (abs < 1_000_000) return `${sign}€${Math.round(abs / 1_000)}K`
  if (abs < 1_000_000_000) return `${sign}€${(abs / 1_000_000).toFixed(1)}M`
  return `${sign}€${(abs / 1_000_000_000).toFixed(1)}B`
}

// X-axis tick formatter for bucketed bar charts. Strips the unit
// suffix from the bucketed labels (seasoning's " months", LTV's "%")
// since the chart's title already conveys the unit and the suffix
// just crowds the tick text. Tables keep the full label. Geography
// bar-chart labels (NL-NH, NL-UT, ...) carry neither suffix, so the
// formatter passes them through unchanged.
function formatBucketTick(value) {
  return String(value).replace(/ months$/, '').replace(/%$/, '')
}

function HealthIndicator({ status }) {
  const config = {
    healthy: { color: 'bg-green-500', label: 'healthy' },
    down: { color: 'bg-red-500', label: 'unhealthy' },
    unknown: { color: 'bg-slate-400', label: 'unreachable' },
  }[status]
  return (
    <span className="inline-flex items-center gap-2 text-sm text-slate-500">
      API status:
      <span className={`h-2.5 w-2.5 rounded-full ${config.color}`} />
      {config.label}
    </span>
  )
}

function App() {
  const [loansFile, setLoansFile] = useState(null)
  const [collateralsFile, setCollateralsFile] = useState(null)
  const [taxonomyFile, setTaxonomyFile] = useState(null)
  const [dealName, setDealName] = useState('')

  // A job started in this tab (or restored from sessionStorage after a
  // reload): { id, mode: 'download'|'analysis', statusUrl, resultUrl }.
  const [job, setJob] = useState(() => readStoredJob())
  // 'idle' | 'uploading' | 'polling' | 'stopped' | 'success' | 'error' | 'deleted'
  const [phase, setPhase] = useState(() => (readStoredJob() ? 'polling' : 'idle'))
  // Last GET /api/jobs/{id} body, for the progress panel.
  const [jobStatus, setJobStatus] = useState(null)
  const [now, setNow] = useState(() => Date.now())
  // The download mode produces { filename, url }; analysis mode the full
  // AnalysisResponse JSON. One state holder, distinguished by `mode`.
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null) // { type, message, code?, details? }
  const [formError, setFormError] = useState(null) // client-side validation

  const [health, setHealth] = useState('unknown')
  // From GET /api/health: { maxUploadMb, retentionHours }. Null until the
  // first health response; the page cannot read the server's env itself.
  const [serverLimits, setServerLimits] = useState(null)

  // Poll /api/health on load and every 30s so a server that died
  // unexpectedly is visible without a manual refresh.
  useEffect(() => {
    let cancelled = false
    async function check() {
      try {
        const res = await fetch('/api/health')
        if (cancelled) return
        if (!res.ok) {
          setHealth('down')
          return
        }
        const body = await res.json()
        if (!cancelled) {
          setHealth(body && body.status === 'ok' ? 'healthy' : 'down')
          if (body && typeof body.max_upload_mb === 'number') {
            setServerLimits({
              maxUploadMb: body.max_upload_mb,
              retentionHours: body.job_retention_hours,
            })
          }
        }
      } catch {
        if (!cancelled) setHealth('unknown')
      }
    }
    check()
    const id = setInterval(check, 30000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  // Poll the job while the page is watching it. Ends on done/failed, on
  // "Stop watching", or when the job disappears (404 after expiry/delete).
  useEffect(() => {
    if (phase !== 'polling' || !job) return undefined
    let cancelled = false
    let failures = 0

    async function finishAnalysis() {
      const res = await fetch(job.resultUrl)
      const data = await res.json().catch(() => null)
      if (cancelled) return
      if (!res.ok || !data) {
        setError(data && typeof data.error === 'string' ? errorFromBody(data) : networkError())
        setPhase('error')
        return
      }
      setResult({ mode: 'analysis', data })
      setPhase('success')
    }

    async function tick() {
      let res
      try {
        res = await fetch(job.statusUrl)
      } catch {
        failures += 1
        if (failures >= MAX_POLL_FAILURES && !cancelled) {
          setError(networkError())
          setPhase('error')
        }
        return
      }
      if (cancelled) return
      failures = 0
      if (res.status === 429) return // rate limited: skip this tick, keep polling
      const body = await res.json().catch(() => null)
      if (cancelled) return
      if (!res.ok || !body) {
        if (res.status === 404) storeJob(null)
        setError(body && typeof body.error === 'string' ? errorFromBody(body) : networkError())
        setPhase('error')
        return
      }
      setJobStatus(body)
      setNow(Date.now())
      if (body.state === 'done') {
        if (job.mode === 'analysis') {
          try {
            await finishAnalysis()
          } catch {
            if (!cancelled) {
              setError(networkError())
              setPhase('error')
            }
          }
        } else {
          setResult({
            mode: 'download',
            filename: body.result?.filename ?? 'pool.xlsx',
            url: job.resultUrl,
          })
          setPhase('success')
        }
      } else if (body.state === 'failed') {
        setError(
          body.error && typeof body.error.error === 'string'
            ? errorFromBody(body.error)
            : { type: 'api', code: 'internal_error', message: 'The job failed.' },
        )
        setPhase('error')
      }
    }

    tick()
    const id = setInterval(tick, POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [phase, job])

  // Any change to the form is a fresh attempt - drop stale results/errors.
  const resetStatus = useCallback(() => {
    setFormError(null)
    setError(null)
    setPhase((p) => (['success', 'error', 'stopped', 'deleted'].includes(p) ? 'idle' : p))
    setResult(null)
  }, [])

  const onFileChange = (setter) => (event) => {
    setter(event.target.files[0] ?? null)
    resetStatus()
  }

  const onDealNameChange = (event) => {
    setDealName(event.target.value)
    resetStatus()
  }

  // Warn (the server enforces) when a file exceeds the per-file cap the
  // server reported; silent until /api/health has answered.
  const maxUploadBytes = serverLimits ? serverLimits.maxUploadMb * 1024 * 1024 : null
  const oversizedFiles = [
    ['loans', loansFile],
    ['collaterals', collateralsFile],
    ['taxonomy', taxonomyFile],
  ].filter(([, file]) => file && maxUploadBytes !== null && file.size > maxUploadBytes)

  // Both buttons share validation + the POST; only the result handling
  // diverges. The intent is signalled by the `mode` argument, which maps
  // directly onto the API's `analysis_only` form field.
  async function submit(mode) {
    if (!loansFile || !collateralsFile) {
      setFormError('Both a loans CSV and a collaterals CSV are required.')
      return
    }
    if (dealName.trim() === '') {
      setFormError('Deal name is required.')
      return
    }
    if (!DEAL_NAME_PATTERN.test(dealName)) {
      setFormError(
        'Deal name may only contain letters, digits, spaces, hyphens, ' +
          'underscores and periods, and must be 1-100 characters long.',
      )
      return
    }

    setFormError(null)
    setError(null)
    setResult(null)
    setJobStatus(null)
    setPhase('uploading')

    const formData = new FormData()
    formData.append('loans', loansFile)
    formData.append('collaterals', collateralsFile)
    formData.append('deal_name', dealName)
    if (taxonomyFile) formData.append('taxonomy', taxonomyFile)
    if (mode === 'analysis') formData.append('analysis_only', 'true')

    try {
      const response = await fetch('/api/jobs', {
        method: 'POST',
        body: formData,
      })

      // Every API failure path returns the ErrorResponse JSON shape. A
      // non-JSON body means the request never reached the API (e.g. the
      // dev proxy could not connect) - treat that as a network error.
      let body = null
      try {
        body = await response.json()
      } catch {
        body = null
      }
      if (!response.ok) {
        setError(body && typeof body.error === 'string' ? errorFromBody(body) : networkError())
        setPhase('error')
        return
      }
      if (!body || typeof body.job_id !== 'string') {
        setError(networkError())
        setPhase('error')
        return
      }
      const newJob = {
        id: body.job_id,
        mode,
        statusUrl: body.status_url,
        resultUrl: body.result_url,
      }
      storeJob(newJob)
      setJob(newJob)
      setNow(Date.now())
      setPhase('polling')
    } catch {
      setError(networkError())
      setPhase('error')
    }
  }

  function stopWatching() {
    // The job keeps running on the server; the stored id lets a reload
    // pick it up again.
    setPhase('stopped')
  }

  async function deleteFromServer() {
    if (!job) return
    try {
      const response = await fetch(job.statusUrl, { method: 'DELETE' })
      if (response.status === 204 || response.status === 404) {
        storeJob(null)
        setJob(null)
        setJobStatus(null)
        setResult(null)
        setPhase('deleted')
        return
      }
      let body = null
      try {
        body = await response.json()
      } catch {
        body = null
      }
      setError(body && typeof body.error === 'string' ? errorFromBody(body) : networkError())
      setPhase('error')
    } catch {
      setError(networkError())
      setPhase('error')
    }
  }

  const busy = phase === 'uploading' || phase === 'polling'
  const stage = jobStatus?.stage ?? null
  const progressPct =
    stage && stage.total > 0 ? Math.round((stage.index / stage.total) * 100) : 0
  const startedMs = jobStatus
    ? Date.parse(jobStatus.started_at ?? jobStatus.created_at)
    : null
  const elapsed = startedMs ? formatElapsed(now - startedMs) : null
  const retentionText = serverLimits
    ? `Kept on the server for ${serverLimits.retentionHours} hours, then deleted automatically.`
    : 'Kept on the server for a limited time, then deleted automatically.'

  return (
    <div className="flex min-h-screen flex-col bg-slate-50 text-slate-700">
      <header className="border-b border-slate-200 bg-white px-6 py-4">
        <h1 className="text-lg font-semibold text-slate-800">
          ESMA-MILAN Pipeline
        </h1>
      </header>

      <main className="mx-auto w-full max-w-5xl flex-1 p-6">
        <form
          onSubmit={(e) => {
            e.preventDefault()
            submit('download')
          }}
          className="space-y-5 rounded-lg border border-slate-200 bg-white p-6"
        >
          <FileField
            id="loans"
            label="Loans CSV"
            accept=".csv,text/csv"
            file={loansFile}
            onChange={onFileChange(setLoansFile)}
            disabled={busy}
            required
          />
          <FileField
            id="collaterals"
            label="Collaterals CSV"
            accept=".csv,text/csv"
            file={collateralsFile}
            onChange={onFileChange(setCollateralsFile)}
            disabled={busy}
            required
          />
          <FileField
            id="taxonomy"
            label="Taxonomy (optional)"
            accept=".xlsx"
            file={taxonomyFile}
            onChange={onFileChange(setTaxonomyFile)}
            disabled={busy}
          />

          <div>
            <label
              htmlFor="deal-name"
              className="mb-1 block text-sm font-medium text-slate-700"
            >
              Deal Name
            </label>
            <input
              id="deal-name"
              type="text"
              value={dealName}
              onChange={onDealNameChange}
              disabled={busy}
              placeholder="e.g. DEAL_2025-1"
              className="w-full rounded border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none disabled:bg-slate-100"
            />
          </div>

          {oversizedFiles.length > 0 && (
            <p className="text-sm text-amber-700">
              Notice: {oversizedFiles.map(([name]) => name).join(', ')} exceed
              the server&apos;s upload limit of {serverLimits.maxUploadMb} MB
              per file and will be rejected.
            </p>
          )}

          <div className="flex flex-col gap-3 sm:flex-row">
            <button
              type="submit"
              disabled={busy}
              className="flex-1 rounded bg-slate-800 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-400"
            >
              Process Pool &amp; Download
            </button>
            <button
              type="button"
              onClick={() => submit('analysis')}
              disabled={busy}
              className="flex-1 rounded border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Analyze Pool
            </button>
          </div>
        </form>

        <div className="mt-5 space-y-3">
          {formError && (
            <p className="rounded border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-800">
              {formError}
            </p>
          )}

          {phase === 'uploading' && (
            <div className="flex items-center gap-3 rounded border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600">
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-slate-600" />
              Uploading files... large pools can take a few minutes to upload.
            </div>
          )}

          {phase === 'polling' && job && (
            <JobProgress
              job={job}
              status={jobStatus}
              progressPct={progressPct}
              elapsed={elapsed}
              onStop={stopWatching}
            />
          )}

          {phase === 'stopped' && job && (
            <p className="rounded border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600">
              Stopped watching job <code className="text-xs">{job.id}</code>. It keeps
              running on the server; reload this page to pick it up again, or
              submit a new pool.
            </p>
          )}

          {phase === 'success' && result?.mode === 'download' && (
            <div className="rounded border border-green-300 bg-green-50 px-4 py-3 text-sm text-green-800">
              <p className="font-medium">Workbook ready: {result.filename}</p>
              <p className="mt-1 text-xs text-green-700">{retentionText}</p>
              <div className="mt-3 flex flex-col gap-2 sm:flex-row">
                <a
                  href={result.url}
                  download={result.filename}
                  className="rounded bg-green-700 px-4 py-2 text-center text-sm font-medium text-white hover:bg-green-600"
                >
                  Download workbook
                </a>
                <button
                  type="button"
                  onClick={deleteFromServer}
                  className="rounded border border-green-700 bg-white px-4 py-2 text-sm font-medium text-green-800 hover:bg-green-100"
                >
                  Delete from server
                </button>
              </div>
            </div>
          )}

          {phase === 'deleted' && (
            <p className="rounded border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600">
              Deleted from the server.
            </p>
          )}

          {phase === 'error' && error && (
            <div className="rounded border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800">
              <p className="font-medium">{error.message}</p>
              {error.type === 'api' && error.code && (
                <p className="mt-1 text-xs text-red-600">Code: {error.code}</p>
              )}
              {error.type === 'api' && error.details && (
                <details className="mt-2 text-xs text-red-700">
                  <summary className="cursor-pointer">Show details</summary>
                  <pre className="mt-1 overflow-x-auto whitespace-pre-wrap">
                    {JSON.stringify(error.details, null, 2)}
                  </pre>
                </details>
              )}
              {job && jobStatus && ['done', 'failed'].includes(jobStatus.state) && (
                <button
                  type="button"
                  onClick={deleteFromServer}
                  className="mt-3 rounded border border-red-400 bg-white px-3 py-1.5 text-xs font-medium text-red-800 hover:bg-red-100"
                >
                  Delete from server
                </button>
              )}
            </div>
          )}
        </div>

        {phase === 'success' && result?.mode === 'analysis' && (
          <>
            <AnalysisResults analysis={result.data} />
            <div className="mt-4 flex items-center gap-3 text-xs text-slate-500">
              <span>{retentionText}</span>
              <button
                type="button"
                onClick={deleteFromServer}
                className="rounded border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50"
              >
                Delete from server
              </button>
            </div>
          </>
        )}
      </main>

      <footer className="border-t border-slate-200 bg-white px-6 py-3">
        <HealthIndicator status={health} />
      </footer>
    </div>
  )
}

function JobProgress({ job, status, progressPct, elapsed, onStop }) {
  const state = status?.state ?? 'queued'
  const stage = status?.stage ?? null
  let headline
  if (!status) headline = 'Checking job status...'
  else if (state === 'queued') headline = 'Queued, waiting for the worker to start...'
  else if (stage) headline = `Stage ${stage.index} of ${stage.total}: ${stage.label.replace(/^Stage \d+: /, '')}`
  else headline = 'Running...'
  return (
    <div className="rounded border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-slate-600" />
          <span>{headline}</span>
        </div>
        <button
          type="button"
          onClick={onStop}
          className="rounded border border-slate-300 bg-white px-3 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50"
        >
          Stop watching
        </button>
      </div>
      <div className="mt-3 h-2 w-full overflow-hidden rounded bg-slate-200">
        <div
          className="h-2 rounded bg-slate-700 transition-all"
          style={{ width: `${state === 'queued' ? 0 : Math.max(progressPct, 3)}%` }}
        />
      </div>
      <p className="mt-2 text-xs text-slate-500">
        {job.mode === 'analysis' ? 'Analysis' : 'Workbook'} job{' '}
        <code>{job.id}</code>
        {elapsed ? ` - ${elapsed} elapsed` : ''}. The job keeps running on the
        server if you stop watching or close this tab.
      </p>
    </div>
  )
}

function AnalysisResults({ analysis }) {
  const {
    deal_name: dealName,
    summary,
    stratifications,
    execution_summary: executionSummary,
  } = analysis
  return (
    <section className="mt-6 space-y-4">
      <header className="rounded-lg border border-slate-200 bg-white p-4">
        <h2 className="text-base font-semibold text-slate-800">
          Pool analysis: {dealName}
        </h2>
        <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-sm text-slate-600 sm:grid-cols-4">
          <SummaryStat label="Loans" value={summary.loan_count.toLocaleString()} />
          <SummaryStat label="Properties" value={summary.property_count.toLocaleString()} />
          <SummaryStat label="Groups" value={summary.group_count.toLocaleString()} />
          <SummaryStat
            label="Total balance"
            value={EUR_FORMAT.format(summary.total_current_balance)}
          />
        </dl>
        <p className="mt-2 text-xs text-slate-500">
          Aggregation: {summary.chosen_aggregation}
        </p>
        {summary.warnings && summary.warnings.length > 0 && (
          <ul className="mt-2 list-disc pl-5 text-xs text-amber-700">
            {summary.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        )}
      </header>

      {executionSummary && executionSummary.length > 0 && (
        <ExecutionSummary rows={executionSummary} />
      )}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {Object.entries(stratifications).map(([key, strat]) => (
          <StratificationTile key={key} stratKey={key} strat={strat} />
        ))}
      </div>
    </section>
  )
}

function ExecutionSummary({ rows }) {
  // The Execution Summary mirrors Sheet 1 of the workbook: ~43 rows
  // of pre-formatted metric/value pairs (38 base metrics + per-
  // structure-type breakdown rows). Values are byte-equal to what the
  // workbook writes - the server pre-formats with R-faithful helpers
  // (`_fmt_comma`, `_fmt_pct`) so the frontend renders them verbatim.
  return (
    <article className="rounded-lg border border-slate-200 bg-white p-4">
      <h3 className="text-sm font-semibold text-slate-800">Execution Summary</h3>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
              <th className="py-1.5 pr-2">Metric</th>
              <th className="py-1.5 pr-2 text-right">Value</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={`${row.label}-${i}`} className="border-b border-slate-100">
                <td className="py-1.5 pr-2 align-top text-slate-700 whitespace-normal break-words">
                  {row.label}
                </td>
                <td className="py-1.5 pr-2 text-right align-top tabular-nums text-slate-800">
                  {row.value}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </article>
  )
}

function SummaryStat({ label, value }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-slate-500">{label}</dt>
      <dd className="text-sm font-medium text-slate-800">{value}</dd>
    </div>
  )
}

// Pool-level weighted-average formatter, keyed on the stratification's
// registry key. seasoning's WA is in months (zero decimals, "months"
// suffix); current_ltv's is a decimal 0..1+ (one decimal, percent);
// current_interest_rate's is already in percent units (two decimals,
// i.e. basis-point resolution). Other strats have
// `weighted_average: null` so this is only called for the three
// numeric ones.
function formatWeightedAverage(stratKey, value) {
  if (stratKey === 'seasoning') {
    return `WA: ${Math.round(value).toLocaleString()} months`
  }
  if (stratKey === 'current_ltv') {
    return `WA: ${(value * 100).toFixed(1)}%`
  }
  if (stratKey === 'current_interest_rate') {
    return `WA: ${value.toFixed(2)}%`
  }
  return `WA: ${value}`
}

function StratificationTile({ stratKey, strat }) {
  const {
    title,
    chart_type: chartType,
    rows,
    total,
    error,
    note,
    weighted_average: weightedAverage,
  } = strat

  if (error) {
    return (
      <article className="rounded-lg border border-slate-200 bg-white p-4">
        <h3 className="text-sm font-semibold text-slate-800">{title}</h3>
        <p className="mt-2 text-sm text-slate-500">Not available: {error}</p>
      </article>
    )
  }

  // Stable color per spec-row index. Keyed off the row's position in
  // `rows` (not in the filtered chart data), so the same ESMA code
  // always gets the same colour across pools: a "FLIF" slice is the
  // same colour in every run, even when other codes drop in and out
  // of being non-zero. The table swatch and the pie slice read from
  // the same array, so the two stay in sync.
  const rowColors = rows.map((_, i) => CHART_COLORS[i % CHART_COLORS.length])

  // Swatches replace the pie chart's removed legend - they're the
  // bridge between slice colour and bucket name. Bar charts use a
  // single colour for all bars and already carry the x-axis label
  // under each bar, so no swatch is needed there (and a different
  // swatch colour next to identically-coloured bars would mislead).
  const showSwatches = chartType === 'pie'

  return (
    <article className="rounded-lg border border-slate-200 bg-white p-4">
      <h3 className="text-sm font-semibold text-slate-800">{title}</h3>
      {weightedAverage != null && (
        <p className="mt-0.5 text-xs font-medium text-slate-600">
          {formatWeightedAverage(stratKey, weightedAverage)}
        </p>
      )}
      <div className="mt-3 h-[220px]">
        <StratificationChart
          chartType={chartType}
          rows={rows}
          rowColors={rowColors}
          stratKey={stratKey}
        />
      </div>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
              <th className="py-1.5 pr-2">Bucket</th>
              <th className="py-1.5 pr-2 text-right">Count</th>
              <th className="py-1.5 pr-2 text-right">% Count</th>
              <th className="py-1.5 pr-2 text-right">Balance</th>
              <th className="py-1.5 text-right">% Balance</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={row.label} className="border-b border-slate-100">
                <td className="py-1.5 pr-2 align-top text-slate-700 whitespace-normal break-words">
                  <div className="flex items-start gap-2">
                    {showSwatches && (
                      <span
                        className="mt-1 inline-block h-3 w-3 shrink-0 rounded-sm"
                        style={{ backgroundColor: rowColors[i] }}
                        aria-hidden="true"
                      />
                    )}
                    <span>{row.label}</span>
                  </div>
                </td>
                <td className="py-1.5 pr-2 text-right tabular-nums">
                  {row.count.toLocaleString()}
                </td>
                <td className="py-1.5 pr-2 text-right tabular-nums text-slate-500">
                  {PCT_FORMAT.format(row.count_pct)}
                </td>
                <td className="py-1.5 pr-2 text-right tabular-nums">
                  {EUR_FORMAT.format(row.balance)}
                </td>
                <td className="py-1.5 text-right tabular-nums text-slate-500">
                  {PCT_FORMAT.format(row.balance_pct)}
                </td>
              </tr>
            ))}
            <tr className="bg-slate-50 font-medium text-slate-800">
              <td className="py-1.5 pr-2">Total</td>
              <td className="py-1.5 pr-2 text-right tabular-nums">
                {total.count.toLocaleString()}
              </td>
              <td />
              <td className="py-1.5 pr-2 text-right tabular-nums">
                {EUR_FORMAT.format(total.balance)}
              </td>
              <td />
            </tr>
          </tbody>
        </table>
      </div>
      {note && <p className="mt-2 text-xs text-slate-500">{note}</p>}
    </article>
  )
}

// Custom tooltip shared by pie + bar charts. Recharts 3.x routes the
// hovered datum through `payload[0].payload`; we carry count, count_pct,
// balance and balance_pct on every data point so a single hover surfaces
// all four metrics that show in the table below.
function ChartTooltip({ active, payload }) {
  if (!active || !payload || payload.length === 0) return null
  const datum = payload[0].payload
  if (!datum) return null
  return (
    <div className="rounded-md bg-slate-900/95 px-3 py-2 text-xs text-white shadow-lg">
      <div className="mb-1 font-medium">{datum.name}</div>
      <div className="tabular-nums">
        Count: {datum.count.toLocaleString()} (
        {PCT_FORMAT.format(datum.count_pct)})
      </div>
      <div className="tabular-nums">
        Balance: {EUR_FORMAT.format(datum.balance)} (
        {PCT_FORMAT.format(datum.balance_pct)})
      </div>
    </div>
  )
}

function StratificationChart({ chartType, rows, rowColors, stratKey }) {
  // Carry each row's stable colour into the chart data: filtering out
  // zero-balance pie slices would otherwise re-shift index-based
  // colour assignment, breaking the swatch <-> slice match.
  // count / count_pct / balance_pct ride along on every point so the
  // ChartTooltip can show all four metrics on hover without a second
  // data structure or a lookup back into `rows`.
  const data = rows
    .map((row, i) => ({
      name: row.label,
      value: row.balance,
      color: rowColors[i],
      count: row.count,
      count_pct: row.count_pct,
      balance: row.balance,
      balance_pct: row.balance_pct,
    }))
    .filter((row) => (chartType === 'pie' ? row.value > 0 : true))

  if (data.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-slate-400">
        No data to chart
      </div>
    )
  }

  if (chartType === 'pie') {
    // No <Legend>: faithful ESMA labels are long enough that Recharts'
    // default legend layout wraps and overflows onto the table below.
    // The colour swatches in the table row labels are the legend's
    // replacement.
    return (
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            innerRadius="35%"
            outerRadius="80%"
          >
            {data.map((entry) => (
              <Cell key={entry.name} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip content={<ChartTooltip />} />
        </PieChart>
      </ResponsiveContainer>
    )
  }

  // Dense bucket axes (current_interest_rate: 17 buckets) need steeper
  // rotation and more vertical room so the long labels don't overlap;
  // geographic's 10-or-fewer regions read fine at a shallow tilt; all
  // other bucketed charts have short enough labels to stay horizontal.
  const denseAxis = stratKey === 'current_interest_rate'
  const tiltedAxis = stratKey === 'geographic'
  const axisAngle = denseAxis ? -45 : tiltedAxis ? -30 : 0
  const axisAnchor = denseAxis || tiltedAxis ? 'end' : 'middle'
  const axisHeight = denseAxis ? 60 : 40

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 5, right: 10, left: 5, bottom: 20 }}>
        <XAxis
          dataKey="name"
          tick={{ fontSize: 10 }}
          angle={axisAngle}
          textAnchor={axisAnchor}
          height={axisHeight}
          interval={0}
          tickFormatter={formatBucketTick}
        />
        <YAxis tick={{ fontSize: 10 }} width={60} tickFormatter={formatEurAxis} />
        <Tooltip content={<ChartTooltip />} cursor={{ fill: 'rgba(15,23,42,0.05)' }} />
        <Bar dataKey="value" fill={CHART_COLORS[0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

function FileField({ id, label, accept, file, onChange, disabled, required }) {
  return (
    <div>
      <label
        htmlFor={id}
        className="mb-1 block text-sm font-medium text-slate-700"
      >
        {label}
        {required && <span className="text-red-500"> *</span>}
      </label>
      <input
        id={id}
        type="file"
        accept={accept}
        onChange={onChange}
        disabled={disabled}
        className="block w-full text-sm text-slate-600 file:mr-3 file:rounded file:border-0 file:bg-slate-100 file:px-3 file:py-2 file:text-sm file:font-medium file:text-slate-700 hover:file:bg-slate-200 disabled:opacity-50"
      />
      {file && (
        <p className="mt-1 text-xs text-slate-500">
          {file.name} ({(file.size / 1024).toFixed(1)} KB)
        </p>
      )}
    </div>
  )
}

export default App
