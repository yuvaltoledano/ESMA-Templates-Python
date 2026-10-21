import { useEffect, useState } from 'react'

import { getHealth } from '../lib/api.js'

// GET /api/health on load and every 30 s, so a server that died
// unexpectedly is visible without a manual refresh. The response also
// carries the per-file upload cap and the job retention the page shows.
const HEALTH_INTERVAL_MS = 30000

/**
 * @returns {{
 *   status: 'healthy' | 'down' | 'unknown',
 *   maxUploadMb: number | null,
 *   retentionHours: number | null,
 *   lastChecked: number | null,
 * }}
 * `status` is 'unknown' until the first answer and whenever the fetch
 * fails (unreachable), 'down' for a non-2xx or non-"ok" answer. The
 * limits are kept from the last answer that carried a numeric
 * `max_upload_mb`, so a later outage does not blank them.
 */
export function useHealth() {
  const [state, setState] = useState({
    status: 'unknown',
    maxUploadMb: null,
    retentionHours: null,
    lastChecked: null,
  })

  useEffect(() => {
    let cancelled = false

    async function check() {
      const result = await getHealth()
      if (cancelled) return
      const checkedAt = Date.now()
      setState((prev) => {
        if (!result.reachable) return { ...prev, status: 'unknown', lastChecked: checkedAt }
        const body = result.body
        const healthy = result.ok && Boolean(body) && body.status === 'ok'
        const next = { ...prev, status: healthy ? 'healthy' : 'down', lastChecked: checkedAt }
        if (body && typeof body.max_upload_mb === 'number') {
          next.maxUploadMb = body.max_upload_mb
          next.retentionHours = body.job_retention_hours
        }
        return next
      })
    }

    check()
    const id = setInterval(check, HEALTH_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  return state
}
