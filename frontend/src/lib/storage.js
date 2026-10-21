// The current job, kept in sessionStorage so a reload of the same tab
// resumes polling. sessionStorage is per tab and may be unavailable
// (private mode, blocked storage); every access is guarded so the page
// works without it. Nothing here throws.

const JOB_STORAGE_KEY = 'esma-milan-job'

/** @returns {{id: string, mode: string, statusUrl: string, resultUrl: string} | null} */
export function readStoredJob() {
  try {
    const raw = sessionStorage.getItem(JOB_STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    return parsed && typeof parsed.id === 'string' ? parsed : null
  } catch {
    return null
  }
}

/** Store `job`, or clear the slot when `job` is null. */
export function storeJob(job) {
  try {
    if (job) sessionStorage.setItem(JOB_STORAGE_KEY, JSON.stringify(job))
    else sessionStorage.removeItem(JOB_STORAGE_KEY)
  } catch {
    // Storage unavailable: a reload will not resume, nothing else changes.
  }
}

export function clearStoredJob() {
  storeJob(null)
}
