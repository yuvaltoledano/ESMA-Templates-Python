// ESMA-MILAN GUI: a single-page client for the pipeline API.
//
// Two modes share one upload form (loans CSV, collaterals CSV, deal
// name). Both submit a background job:
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
//
// The network, storage and formatting code lives in lib/ and hooks/; this
// file holds the markup.
import { useState } from 'react'
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

import { useAnalysis } from './hooks/useAnalysis.js'
import { useHealth } from './hooks/useHealth.js'
import { useJob } from './hooks/useJob.js'
import { BAR_COLOR, colorForRow } from './lib/chartPalette.js'
import {
  formatBucketTick,
  formatCount,
  formatCurrency,
  formatEurAxis,
  formatFileSize,
  formatPercent,
  formatWeightedAverage,
} from './lib/format.js'

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
  const [dealName, setDealName] = useState('')

  const health = useHealth()
  const { job, phase, status: jobStatus, result, error: jobError, formError, elapsed, progressPct, busy, actions } =
    useJob()
  const analysis = useAnalysis(job, phase)
  const error = jobError ?? analysis.error

  const onFileChange = (setter) => (event) => {
    setter(event.target.files[0] ?? null)
    actions.reset()
  }

  const onDealNameChange = (event) => {
    setDealName(event.target.value)
    actions.reset()
  }

  // Warn (the server enforces) when a file exceeds the per-file cap the
  // server reported; silent until /api/health has answered.
  const maxUploadBytes = health.maxUploadMb !== null ? health.maxUploadMb * 1024 * 1024 : null
  const oversizedFiles = [
    ['loans', loansFile],
    ['collaterals', collateralsFile],
  ].filter(([, file]) => file && maxUploadBytes !== null && file.size > maxUploadBytes)

  const submit = (mode) => actions.submit(mode, { loansFile, collateralsFile, dealName })
  const stopWatching = actions.stopWatching
  const deleteFromServer = actions.deleteFromServer

  const retentionText =
    health.retentionHours !== null
      ? `Kept on the server for ${health.retentionHours} hours, then deleted automatically.`
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
              the server&apos;s upload limit of {health.maxUploadMb} MB
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

          {error && (
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

        {phase === 'success' && result?.mode === 'analysis' && analysis.data && (
          <>
            <AnalysisResults analysis={analysis.data} />
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
        <HealthIndicator status={health.status} />
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
          <SummaryStat label="Loans" value={formatCount(summary.loan_count)} />
          <SummaryStat label="Properties" value={formatCount(summary.property_count)} />
          <SummaryStat label="Groups" value={formatCount(summary.group_count)} />
          <SummaryStat
            label="Total balance"
            value={formatCurrency(summary.total_current_balance)}
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
  const rowColors = rows.map((row, i) => colorForRow(row.label, i))

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
                  {formatCount(row.count)}
                </td>
                <td className="py-1.5 pr-2 text-right tabular-nums text-slate-500">
                  {formatPercent(row.count_pct)}
                </td>
                <td className="py-1.5 pr-2 text-right tabular-nums">
                  {formatCurrency(row.balance)}
                </td>
                <td className="py-1.5 text-right tabular-nums text-slate-500">
                  {formatPercent(row.balance_pct)}
                </td>
              </tr>
            ))}
            <tr className="bg-slate-50 font-medium text-slate-800">
              <td className="py-1.5 pr-2">Total</td>
              <td className="py-1.5 pr-2 text-right tabular-nums">
                {formatCount(total.count)}
              </td>
              <td />
              <td className="py-1.5 pr-2 text-right tabular-nums">
                {formatCurrency(total.balance)}
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
        Count: {formatCount(datum.count)} (
        {formatPercent(datum.count_pct)})
      </div>
      <div className="tabular-nums">
        Balance: {formatCurrency(datum.balance)} (
        {formatPercent(datum.balance_pct)})
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
        <Bar dataKey="value" fill={BAR_COLOR} />
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
          {file.name} ({formatFileSize(file.size)})
        </p>
      )}
    </div>
  )
}

export default App
