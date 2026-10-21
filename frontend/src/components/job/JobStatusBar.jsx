import { Button } from '../ui/Button.jsx'
import { Card } from '../ui/Card.jsx'
import { Spinner } from '../ui/Spinner.jsx'
import { StatePill } from './StatePill.jsx'

function headlineFor(phase, status) {
  if (phase === 'stopped') return 'Stopped watching'
  if (!status) return 'Checking job status...'
  if (status.state === 'queued') return 'Queued, waiting for the worker to start...'
  if (status.state === 'done' || status.state === 'failed') return null // the pill says it
  if (status.stage) {
    return `Stage ${status.stage.index} of ${status.stage.total}: ${status.stage.label.replace(/^Stage \d+: /, '')}`
  }
  return 'Running...'
}

/**
 * The job status bar, shown while this tab has a job: state pill, deal
 * name, headline ("Stage n of N: label", "Queued, waiting for the worker
 * to start...", ...), elapsed time, a thin progress bar, and the actions
 * for the state: Stop watching while queued or running; Download workbook
 * and Delete from server when done in download mode; Delete from server
 * when done in analysis mode; nothing when stopped (reload to resume).
 * Props: `job`, `phase`, `status` (last JobStatus body), `result`,
 * `elapsed`, `progressPct`, `retentionText`, `analysisLoading`, `onStop`,
 * `onDelete`.
 */
export function JobStatusBar({
  job,
  phase,
  status,
  result,
  elapsed,
  progressPct,
  retentionText,
  analysisLoading,
  onStop,
  onDelete,
}) {
  const state = status?.state ?? 'queued'
  const headline = headlineFor(phase, status)
  const polling = phase === 'polling'
  const done = phase === 'success'
  const width = state === 'queued' ? 0 : state === 'done' ? 100 : Math.max(progressPct, 3)
  const modeLabel = job.mode === 'analysis' ? 'Analysis' : 'Workbook'

  return (
    <Card className="px-4 py-3">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <StatePill state={state} />
        {status?.deal_name && <span className="font-medium text-ink">{status.deal_name}</span>}
        {headline && (
          <span className="flex items-center gap-2 text-base text-ink-muted">
            {polling && <Spinner />}
            {headline}
          </span>
        )}
        {elapsed && <span className="font-mono text-xs text-ink-faint">{elapsed} elapsed</span>}
        <div className="ml-auto flex flex-wrap items-center gap-2">
          {polling && (
            <Button variant="ghost" onClick={onStop}>
              Stop watching
            </Button>
          )}
          {done && result?.mode === 'download' && (
            <Button variant="primary" href={result.url} download={result.filename}>
              Download workbook
            </Button>
          )}
          {done && (
            <Button variant="ghost" onClick={onDelete}>
              Delete from server
            </Button>
          )}
        </div>
      </div>
      <div className="mt-3 h-1 w-full overflow-hidden rounded-pill bg-line-soft">
        <div className="h-1 rounded-pill bg-accent transition-all" style={{ width: `${width}%` }} />
      </div>
      <p className="mt-2 text-xs text-ink-faint">
        {phase === 'stopped' ? (
          <>
            Stopped watching job <code className="font-mono">{job.id}</code>. It keeps running on the
            server; reload this page to pick it up again, or submit a new pool.
          </>
        ) : done && result?.mode === 'download' ? (
          <>
            Workbook ready: {result.filename}. {retentionText}
          </>
        ) : done ? (
          <>
            {analysisLoading ? 'Loading the analysis...' : 'Analysis ready.'} {retentionText}
          </>
        ) : (
          <>
            {modeLabel} job <code className="font-mono">{job.id}</code>. The job keeps running on the
            server if you stop watching or close this tab.
          </>
        )}
      </p>
    </Card>
  )
}
