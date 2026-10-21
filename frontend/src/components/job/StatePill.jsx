const STYLES = {
  queued: { classes: 'bg-line-soft text-ink-muted', label: 'Queued' },
  running: { classes: 'bg-line-soft text-ink', label: 'Running' },
  done: { classes: 'bg-accent-soft text-accent-softText', label: 'Done' },
  failed: { classes: 'bg-danger-soft text-danger', label: 'Failed' },
}

/**
 * The job state as a pill: Queued, Running, Done or Failed in its soft
 * colour. Props: `state` (a JobStatus `state` value).
 */
export function StatePill({ state }) {
  const style = STYLES[state] ?? STYLES.queued
  return (
    <span
      className={`inline-flex items-center rounded-pill px-2.5 py-0.5 text-xs font-semibold uppercase tracking-wider ${style.classes}`}
    >
      {style.label}
    </span>
  )
}
