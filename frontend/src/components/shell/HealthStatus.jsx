const STATES = {
  healthy: { dot: 'bg-accent-glow', label: 'API healthy' },
  down: { dot: 'bg-danger', label: 'API unhealthy' },
  unknown: { dot: 'bg-brand-navyMuted', label: 'API unreachable' },
}

/**
 * The top bar's health line: a dot and "API healthy" / "API unhealthy" /
 * "API unreachable". The worker is not reported; the API has no endpoint
 * for it. Props: `status` ('healthy' | 'down' | 'unknown').
 */
export function HealthStatus({ status }) {
  const config = STATES[status] ?? STATES.unknown
  return (
    <span className="inline-flex items-center gap-2 text-xs font-medium text-brand-navyText">
      <span className={`h-2 w-2 rounded-pill ${config.dot}`} aria-hidden="true" />
      {config.label}
    </span>
  )
}
