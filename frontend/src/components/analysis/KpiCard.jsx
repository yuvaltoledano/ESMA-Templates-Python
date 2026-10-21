import { Card } from '../ui/Card.jsx'

/**
 * One KPI: an uppercase label over a 20px mono value. Props: `label`,
 * `value` (a string; "—" when the source row is missing).
 */
export function KpiCard({ label, value }) {
  return (
    <Card className="min-w-0 px-4 py-3">
      <p className="text-xs font-medium uppercase tracking-wider text-ink-muted">{label}</p>
      <p className="mt-1 truncate font-mono text-kpi font-medium tabular-nums text-ink" title={value}>
        {value}
      </p>
    </Card>
  )
}
