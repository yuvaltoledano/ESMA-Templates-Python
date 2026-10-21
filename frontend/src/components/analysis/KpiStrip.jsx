import { formatCount, formatCurrency } from '../../lib/format.js'
import { KpiCard } from './KpiCard.jsx'

const MISSING = '—'

/** The value of the Execution Summary row with `label`, or "—". */
function rowValue(executionSummary, label) {
  const row = executionSummary.find((entry) => entry.label === label)
  return row && row.value != null && row.value !== '' ? row.value : MISSING
}

/**
 * Six KPI cards in one row: Current balance, Loans and Properties from
 * `summary`; Borrowers, WA LTV and WA rate verbatim from the Execution
 * Summary rows "Number Of Borrowers", "WA Current LTV" and "WA Interest
 * Rate". Two columns on a phone, three from 768px, six from 1280px.
 * Props: `summary` (AnalysisSummary), `executionSummary` (rows).
 */
export function KpiStrip({ summary, executionSummary }) {
  const rows = executionSummary ?? []
  const kpis = [
    ['Current balance', formatCurrency(summary.total_current_balance)],
    ['Loans', formatCount(summary.loan_count)],
    ['Borrowers', rowValue(rows, 'Number Of Borrowers')],
    ['Properties', formatCount(summary.property_count)],
    ['WA LTV', rowValue(rows, 'WA Current LTV')],
    ['WA rate', rowValue(rows, 'WA Interest Rate')],
  ]
  return (
    <div className="grid grid-cols-2 gap-5 md:grid-cols-3 xl:grid-cols-6">
      {kpis.map(([label, value]) => (
        <KpiCard key={label} label={label} value={value} />
      ))}
    </div>
  )
}
