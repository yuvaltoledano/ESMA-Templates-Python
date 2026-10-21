// Every request the page makes, in one place. All URLs are relative: the
// Vite dev server proxies /api to localhost:8000 and Caddy does the same in
// the deployment. Functions here parse the API's ErrorResponse
// ({error, message, details}) and either return parsed data or throw a
// typed error; the hooks decide what each outcome means for the page.
//
// Documented message: a response that is not the API's JSON (the proxy
// could not connect, Caddy answered with its own page, no response at
// all) is reported as the network error below; the string is quoted in
// frontend/README.md and docs/deploy.md's troubleshooting section.

export const NETWORK_ERROR_MESSAGE =
  'Cannot reach API at localhost:8000. Is `uv run esma-milan-server` running?'

/** A non-2xx answer carrying the API's ErrorResponse body. */
export class ApiError extends Error {
  constructor(body, status) {
    super(body.message || 'The request failed.')
    this.name = 'ApiError'
    this.type = 'api'
    this.code = body.error
    this.details = body.details ?? null
    this.status = status
  }
}

/** No usable answer from the API: a failed fetch or a non-JSON body. */
export class NetworkError extends Error {
  constructor() {
    super(NETWORK_ERROR_MESSAGE)
    this.name = 'NetworkError'
    this.type = 'network'
    this.code = undefined
    this.details = null
  }
}

/** The `{type, message, code, details}` shape the error panel renders. */
export function errorState(err) {
  return { type: err.type ?? 'network', message: err.message, code: err.code, details: err.details ?? null }
}

/** Error state for a failed job's status body (`body.error` is an ErrorResponse). */
export function errorStateFromJobStatus(body) {
  const inner = body && body.error
  if (inner && typeof inner.error === 'string') {
    return { type: 'api', code: inner.error, message: inner.message || 'The request failed.', details: inner.details ?? null }
  }
  return { type: 'api', code: 'internal_error', message: 'The job failed.', details: null }
}

async function readJson(response) {
  try {
    return await response.json()
  } catch {
    return null
  }
}

function isErrorBody(body) {
  return Boolean(body) && typeof body.error === 'string'
}

function errorFor(response, body) {
  return isErrorBody(body) ? new ApiError(body, response.status) : new NetworkError()
}

/**
 * POST /api/jobs. Resolves to {id, statusUrl, resultUrl}; throws ApiError
 * on a rejected request and NetworkError when nothing usable came back.
 */
export async function createJob(formData) {
  let response
  try {
    response = await fetch('/api/jobs', { method: 'POST', body: formData })
  } catch {
    throw new NetworkError()
  }
  const body = await readJson(response)
  if (!response.ok) throw errorFor(response, body)
  if (!body || typeof body.job_id !== 'string') throw new NetworkError()
  return { id: body.job_id, statusUrl: body.status_url, resultUrl: body.result_url }
}

/**
 * One GET of a job's status URL, classified for the poller:
 *   {kind: 'unreachable'}            the fetch itself threw
 *   {kind: 'rate_limited'}           429: skip this tick, keep polling
 *   {kind: 'error', error, status}   any other non-2xx, or a non-JSON body
 *   {kind: 'ok', body}               the JobStatus body
 */
export async function getJobStatus(statusUrl) {
  let response
  try {
    response = await fetch(statusUrl)
  } catch {
    return { kind: 'unreachable' }
  }
  if (response.status === 429) return { kind: 'rate_limited' }
  const body = await readJson(response)
  if (!response.ok || !body) {
    return { kind: 'error', error: errorFor(response, body), status: response.status }
  }
  return { kind: 'ok', body }
}

/** GET a done analysis job's result URL; resolves to the AnalysisResponse. */
export async function fetchAnalysis(resultUrl) {
  let response
  try {
    response = await fetch(resultUrl)
  } catch {
    throw new NetworkError()
  }
  const body = await readJson(response)
  if (!response.ok || !body) throw errorFor(response, body)
  return body
}

/** DELETE a job. 204 and 404 both count as deleted; anything else throws. */
export async function deleteJob(statusUrl) {
  let response
  try {
    response = await fetch(statusUrl, { method: 'DELETE' })
  } catch {
    throw new NetworkError()
  }
  if (response.status === 204 || response.status === 404) return
  const body = await readJson(response)
  throw errorFor(response, body)
}

/**
 * GET /api/health, classified for the health poller:
 *   {reachable: false}                      the fetch threw
 *   {reachable: true, ok: boolean, body}    body is null when not JSON
 */
export async function getHealth() {
  let response
  try {
    response = await fetch('/api/health')
  } catch {
    return { reachable: false }
  }
  if (!response.ok) return { reachable: true, ok: false, body: null }
  return { reachable: true, ok: true, body: await readJson(response) }
}
