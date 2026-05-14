import { useCallback, useEffect, useState } from 'react'

// Mirrors the API's deal_name rule (handlers.py::validate_deal_name):
// letters, digits, space, hyphen, underscore, period; length 1-100.
// The server stays the source of truth - this is just fast-fail UX.
const DEAL_NAME_PATTERN = /^[A-Za-z0-9 ._-]{1,100}$/

// Matches the API's per-file transport cap (handlers.py::MAX_UPLOAD_FILE_SIZE).
// Client-side this only warns - the server enforces it.
const MAX_FILE_SIZE = 50 * 1024 * 1024

const NETWORK_ERROR_MESSAGE =
  'Cannot reach API at localhost:8000. Is `uv run esma-milan-server` running?'

// Pull the download filename out of a Content-Disposition header. The API
// produces a simple `attachment; filename="<name>"`, so a basic extraction
// covers it; fall back to a generic name if the header is absent.
function parseFilename(contentDisposition, fallback) {
  if (!contentDisposition) return fallback
  const match = /filename="?([^";]+)"?/.exec(contentDisposition)
  return match ? match[1] : fallback
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
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

  // 'idle' | 'processing' | 'success' | 'error'
  const [phase, setPhase] = useState('idle')
  const [result, setResult] = useState(null) // { filename }
  const [error, setError] = useState(null) // { type, message, code?, details? }
  const [formError, setFormError] = useState(null) // client-side validation

  const [health, setHealth] = useState('unknown')

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

  // Any change to the form is a fresh attempt - drop stale results/errors.
  const resetStatus = useCallback(() => {
    setFormError(null)
    setError(null)
    setPhase((p) => (p === 'success' || p === 'error' ? 'idle' : p))
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

  const oversizedFiles = [
    ['loans', loansFile],
    ['collaterals', collateralsFile],
    ['taxonomy', taxonomyFile],
  ].filter(([, file]) => file && file.size > MAX_FILE_SIZE)

  async function handleSubmit(event) {
    event.preventDefault()

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
    setPhase('processing')

    const formData = new FormData()
    formData.append('loans', loansFile)
    formData.append('collaterals', collateralsFile)
    formData.append('deal_name', dealName)
    if (taxonomyFile) formData.append('taxonomy', taxonomyFile)

    try {
      const response = await fetch('/api/process', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        // Every API failure path returns the ErrorResponse JSON shape. A
        // non-JSON body means the request never reached the API (e.g. the
        // dev proxy could not connect) - treat that as a network error.
        let body = null
        try {
          body = await response.json()
        } catch {
          body = null
        }
        if (body && typeof body.error === 'string') {
          setError({
            type: 'api',
            code: body.error,
            message: body.message || 'The request failed.',
            details: body.details ?? null,
          })
        } else {
          setError({ type: 'network', message: NETWORK_ERROR_MESSAGE })
        }
        setPhase('error')
        return
      }

      const blob = await response.blob()
      const filename = parseFilename(
        response.headers.get('Content-Disposition'),
        'pool.xlsx',
      )
      triggerDownload(blob, filename)
      setResult({ filename })
      setPhase('success')
    } catch {
      setError({ type: 'network', message: NETWORK_ERROR_MESSAGE })
      setPhase('error')
    }
  }

  const busy = phase === 'processing'

  return (
    <div className="flex min-h-screen flex-col bg-slate-50 text-slate-700">
      <header className="border-b border-slate-200 bg-white px-6 py-4">
        <h1 className="text-lg font-semibold text-slate-800">
          ESMA-MILAN Pipeline
        </h1>
      </header>

      <main className="mx-auto w-full max-w-2xl flex-1 p-6">
        <form
          onSubmit={handleSubmit}
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
              placeholder="e.g. Domi 2025-1"
              className="w-full rounded border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none disabled:bg-slate-100"
            />
          </div>

          {oversizedFiles.length > 0 && (
            <p className="text-sm text-amber-700">
              Notice: {oversizedFiles.map(([name]) => name).join(', ')} exceed
              50 MB. The server may reject files this large.
            </p>
          )}

          <button
            type="submit"
            disabled={busy}
            className="w-full rounded bg-slate-800 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            Process Pool
          </button>
        </form>

        <div className="mt-5">
          {formError && (
            <p className="rounded border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-800">
              {formError}
            </p>
          )}

          {busy && (
            <div className="flex items-center gap-3 rounded border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600">
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-slate-600" />
              Running... please wait.
            </div>
          )}

          {phase === 'success' && result && (
            <p className="rounded border border-green-300 bg-green-50 px-4 py-3 text-sm text-green-800">
              Downloaded: {result.filename}
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
            </div>
          )}
        </div>
      </main>

      <footer className="border-t border-slate-200 bg-white px-6 py-3">
        <HealthIndicator status={health} />
      </footer>
    </div>
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
