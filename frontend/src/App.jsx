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
//                                rendered inline as the Analyze pool view
//   DELETE /api/jobs/{id}     -> "Delete from server" once done or failed
// The current job id is kept in sessionStorage so a reload resumes polling.
// Every non-2xx response is the API's ErrorResponse ({error, message,
// details}); a non-JSON failure is treated as "API unreachable". The top
// bar polls GET /api/health, which also reports the per-file upload cap
// the page warns about. All requests use relative /api URLs, proxied to
// localhost:8000 by the Vite dev server (vite.config.js).
// See docs/api.md and docs/analysis.md for the contract.
//
// This file only composes the page. Network and storage live in lib/ and
// hooks/; every component under components/ receives data and callbacks
// as props.
import { useState } from 'react'

import { AnalysisHeader } from './components/analysis/AnalysisHeader.jsx'
import { ExecutionSummaryCard } from './components/analysis/ExecutionSummaryCard.jsx'
import { KpiStrip } from './components/analysis/KpiStrip.jsx'
import { StratificationGrid } from './components/analysis/StratificationGrid.jsx'
import { ErrorPanel } from './components/job/ErrorPanel.jsx'
import { JobStatusBar } from './components/job/JobStatusBar.jsx'
import { Footer } from './components/shell/Footer.jsx'
import { TopBar } from './components/shell/TopBar.jsx'
import { UploadCard } from './components/upload/UploadCard.jsx'
import { useAnalysis } from './hooks/useAnalysis.js'
import { useHealth } from './hooks/useHealth.js'
import { useJob } from './hooks/useJob.js'

function App() {
  const [loansFile, setLoansFile] = useState(null)
  const [collateralsFile, setCollateralsFile] = useState(null)
  const [dealName, setDealName] = useState('')

  const health = useHealth()
  const { job, phase, status, result, error: jobError, formError, elapsed, progressPct, busy, actions } = useJob()
  const analysis = useAnalysis(job, phase)
  const error = jobError ?? analysis.error

  // Any change to the form is a fresh attempt - drop stale results/errors.
  const onFileChange = (setter) => (event) => {
    setter(event.target.files[0] ?? null)
    actions.reset()
  }
  const onDealNameChange = (event) => {
    setDealName(event.target.value)
    actions.reset()
  }
  const submit = (mode) => actions.submit(mode, { loansFile, collateralsFile, dealName })

  const retentionText =
    health.retentionHours !== null
      ? `Kept on the server for ${health.retentionHours} hours, then deleted automatically.`
      : 'Kept on the server for a limited time, then deleted automatically.'

  // The status bar shows while this tab is watching, has stopped watching,
  // or has a finished job to act on. An error from a rejected submission
  // has no status body and shows as the panel alone.
  const showJobBar =
    Boolean(job) && (phase === 'polling' || (Boolean(status) && ['stopped', 'success', 'error'].includes(phase)))
  const canDelete = Boolean(job) && Boolean(status) && ['done', 'failed'].includes(status.state)
  const showAnalysis = phase === 'success' && result?.mode === 'analysis' && analysis.data

  return (
    <div className="flex min-h-screen flex-col overflow-x-hidden">
      <TopBar health={health.status} />

      <main className="mx-auto flex w-full max-w-content flex-1 flex-col gap-5 p-6">
        <UploadCard
          loansFile={loansFile}
          collateralsFile={collateralsFile}
          dealName={dealName}
          onLoansChange={onFileChange(setLoansFile)}
          onCollateralsChange={onFileChange(setCollateralsFile)}
          onDealNameChange={onDealNameChange}
          onSubmit={submit}
          busy={busy}
          uploading={phase === 'uploading'}
          formError={formError}
          maxUploadMb={health.maxUploadMb}
        />

        {showJobBar && (
          <JobStatusBar
            job={job}
            phase={phase}
            status={status}
            result={result}
            elapsed={elapsed}
            progressPct={progressPct}
            retentionText={retentionText}
            analysisLoading={analysis.loading}
            onStop={actions.stopWatching}
            onDelete={actions.deleteFromServer}
          />
        )}

        {error && <ErrorPanel error={error} canDelete={canDelete} onDelete={actions.deleteFromServer} />}

        {phase === 'deleted' && <p className="text-base text-ink-muted">Deleted from the server.</p>}

        {showAnalysis && (
          <>
            <AnalysisHeader dealName={analysis.data.deal_name} summary={analysis.data.summary} />
            <KpiStrip summary={analysis.data.summary} executionSummary={analysis.data.execution_summary} />
            <div className="flex min-w-0 flex-col gap-5 lg:flex-row lg:items-start">
              {analysis.data.execution_summary && analysis.data.execution_summary.length > 0 && (
                <ExecutionSummaryCard rows={analysis.data.execution_summary} />
              )}
              <StratificationGrid stratifications={analysis.data.stratifications} />
            </div>
          </>
        )}
      </main>

      <Footer lastChecked={health.lastChecked} />
    </div>
  )
}

export default App
