import { formatCount, formatCurrency, formatPercent } from '../../../lib/format.js'

/**
 * Tooltip shared by the donut and bar charts. recharts hands the hovered
 * datum through `payload[0].payload`; every data point carries count,
 * count_pct, balance and balance_pct so one hover shows all four metrics
 * from the table. Props: recharts' `active` and `payload`.
 */
export function ChartTooltip({ active, payload }) {
  if (!active || !payload || payload.length === 0) return null
  const datum = payload[0].payload
  if (!datum) return null
  return (
    <div className="rounded-control border border-line bg-surface px-3 py-2 text-xs text-ink shadow-card">
      <div className="mb-1 font-medium">{datum.name}</div>
      <div className="font-mono text-ink-muted">
        Count: {formatCount(datum.count)} ({formatPercent(datum.count_pct)})
      </div>
      <div className="font-mono text-ink-muted">
        Balance: {formatCurrency(datum.balance)} ({formatPercent(datum.balance_pct)})
      </div>
    </div>
  )
}
