import { useEffect, useState } from 'react'

import { errorState, fetchAnalysis } from '../lib/api.js'

/**
 * Fetches the AnalysisResponse of a done analysis job, once per job, and
 * keeps it until the job changes. Runs when `phase` is 'success' and the
 * job's mode is 'analysis'; that is the moment the poller used to fetch
 * it inline, so the page still makes exactly one GET of the result URL.
 * A failure is reported as the same {type, message, code, details} the
 * job errors use.
 *
 * @returns {{data: object | null, loading: boolean, error: object | null}}
 */
export function useAnalysis(job, phase) {
  const wanted = phase === 'success' && job?.mode === 'analysis' ? job : null
  const [state, setState] = useState({ jobId: null, data: null, error: null })

  useEffect(() => {
    if (!wanted) return undefined
    let cancelled = false
    fetchAnalysis(wanted.resultUrl).then(
      (data) => {
        if (!cancelled) setState({ jobId: wanted.id, data, error: null })
      },
      (err) => {
        if (!cancelled) setState({ jobId: wanted.id, data: null, error: errorState(err) })
      },
    )
    return () => {
      cancelled = true
    }
  }, [wanted])

  if (!wanted) return { data: null, loading: false, error: null }
  if (state.jobId !== wanted.id) return { data: null, loading: true, error: null }
  return { data: state.data, loading: false, error: state.error }
}
