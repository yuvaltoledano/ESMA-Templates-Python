import { useCallback, useEffect, useState } from 'react'

import {
  NetworkError,
  createJob,
  deleteJob,
  errorState,
  errorStateFromJobStatus,
  getJobStatus,
} from '../lib/api.js'
import { formatElapsed } from '../lib/format.js'
import { clearStoredJob, readStoredJob, storeJob } from '../lib/storage.js'

// Mirrors the API's deal_name rule (handlers.py::validate_deal_name):
// letters, digits, space, hyphen, underscore, period; length 1-100.
// The server stays the source of truth - this is just fast-fail UX.
const DEAL_NAME_PATTERN = /^[A-Za-z0-9 ._-]{1,100}$/

// Background-job polling. The server's status rate limit (120/minute by
// default) comfortably covers one poll every 2 s; a short run of failed
// polls (a proxy hiccup, a redeploy) is tolerated before giving up.
const POLL_INTERVAL_MS = 2000
const MAX_POLL_FAILURES = 5

const RESETTABLE_PHASES = ['success', 'error', 'stopped', 'deleted']

/**
 * The page's job state machine: submit, poll, stop watching, delete, and
 * resume after a reload. Owns the 2 s poll, the consecutive-failure
 * cutoff and the sessionStorage slot; nothing else touches them.
 *
 * `phase` is one of 'idle' | 'uploading' | 'polling' | 'stopped' |
 * 'success' | 'error' | 'deleted'. `job` is {id, mode: 'download' |
 * 'analysis', statusUrl, resultUrl} or null; `status` is the last
 * GET /api/jobs/{id} body; `result` is {mode: 'download', filename, url}
 * or {mode: 'analysis'} (the analysis JSON itself is fetched by
 * useAnalysis); `error` and `formError` are what the panels render.
 */
export function useJob() {
  const [job, setJob] = useState(() => readStoredJob())
  const [phase, setPhase] = useState(() => (readStoredJob() ? 'polling' : 'idle'))
  const [status, setStatus] = useState(null)
  const [now, setNow] = useState(() => Date.now())
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [formError, setFormError] = useState(null)

  // Poll the job while the page is watching it. Ends on done/failed, on
  // "Stop watching", or when the job disappears (404 after expiry/delete).
  useEffect(() => {
    if (phase !== 'polling' || !job) return undefined
    let cancelled = false
    let failures = 0

    async function tick() {
      const outcome = await getJobStatus(job.statusUrl)
      if (cancelled) return
      if (outcome.kind === 'unreachable') {
        failures += 1
        if (failures >= MAX_POLL_FAILURES) {
          setError(errorState(new NetworkError()))
          setPhase('error')
        }
        return
      }
      failures = 0
      if (outcome.kind === 'rate_limited') return
      if (outcome.kind === 'error') {
        if (outcome.status === 404) clearStoredJob()
        setError(errorState(outcome.error))
        setPhase('error')
        return
      }
      const body = outcome.body
      setStatus(body)
      setNow(Date.now())
      if (body.state === 'done') {
        if (job.mode === 'analysis') {
          setResult({ mode: 'analysis' })
        } else {
          setResult({
            mode: 'download',
            filename: body.result?.filename ?? 'pool.xlsx',
            url: job.resultUrl,
          })
        }
        setPhase('success')
      } else if (body.state === 'failed') {
        setError(errorStateFromJobStatus(body))
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
  // The job itself (and its storage slot) is kept, so a reload still
  // resumes it.
  const reset = useCallback(() => {
    setFormError(null)
    setError(null)
    setPhase((p) => (RESETTABLE_PHASES.includes(p) ? 'idle' : p))
    setResult(null)
  }, [])

  // Both buttons share validation + the POST; only the result handling
  // diverges. `mode` maps onto the API's `analysis_only` form field.
  const submit = useCallback(async (mode, { loansFile, collateralsFile, dealName }) => {
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
    setStatus(null)
    setPhase('uploading')

    const formData = new FormData()
    formData.append('loans', loansFile)
    formData.append('collaterals', collateralsFile)
    formData.append('deal_name', dealName)
    if (mode === 'analysis') formData.append('analysis_only', 'true')

    try {
      const created = await createJob(formData)
      const newJob = { ...created, mode }
      storeJob(newJob)
      setJob(newJob)
      setNow(Date.now())
      setPhase('polling')
    } catch (err) {
      setError(errorState(err))
      setPhase('error')
    }
  }, [])

  // The job keeps running on the server; the stored id lets a reload
  // pick it up again.
  const stopWatching = useCallback(() => {
    setPhase('stopped')
  }, [])

  const deleteFromServer = useCallback(async () => {
    if (!job) return
    try {
      await deleteJob(job.statusUrl)
      clearStoredJob()
      setJob(null)
      setStatus(null)
      setResult(null)
      setPhase('deleted')
    } catch (err) {
      setError(errorState(err))
      setPhase('error')
    }
  }, [job])

  const stage = status?.stage ?? null
  const progressPct = stage && stage.total > 0 ? Math.round((stage.index / stage.total) * 100) : 0
  const startedMs = status ? Date.parse(status.started_at ?? status.created_at) : null
  const elapsed = startedMs ? formatElapsed(now - startedMs) : null
  const busy = phase === 'uploading' || phase === 'polling'

  return {
    job,
    phase,
    status,
    result,
    error,
    formError,
    elapsed,
    progressPct,
    busy,
    actions: { submit, stopWatching, deleteFromServer, reset },
  }
}
