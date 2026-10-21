import { colorForRow } from '../../lib/chartPalette.js'
import { formatWeightedAverage } from '../../lib/format.js'
import { Card } from '../ui/Card.jsx'
import { Disclosure } from '../ui/Disclosure.jsx'
import { StratTable } from './StratTable.jsx'
import { BalanceBarChart } from './charts/BalanceBarChart.jsx'
import { DonutChart } from './charts/DonutChart.jsx'

/**
 * One stratification: title, the weighted-average line where the API
 * gives one, a donut (`chart_type: "pie"`) or bars (`"bar"`), the row
 * table behind a "Show table" disclosure, and the API's `note` under it.
 * A stratification with `error` shows "Not available: {error}" in place
 * of the chart. Props: `stratKey` (registry key), `strat` (the
 * Stratification object).
 */
export function StratificationTile({ stratKey, strat }) {
  const { title, chart_type: chartType, rows, total, error, note, weighted_average: weightedAverage } = strat

  if (error) {
    return (
      <Card title={title} className="min-w-0">
        <p className="px-4 pb-4 pt-2 text-base text-ink-muted">Not available: {error}</p>
      </Card>
    )
  }

  // Colours are assigned by row position so the same ESMA code keeps its
  // colour across pools; the chart and the table swatches share the list.
  const colors = rows.map((row, i) => colorForRow(row.label, i))
  const isPie = chartType === 'pie'

  return (
    <Card title={title} className="min-w-0">
      {weightedAverage != null && (
        <p className="px-4 pt-0.5 font-mono text-xs font-medium text-ink-muted">
          {formatWeightedAverage(stratKey, weightedAverage)}
        </p>
      )}
      <div className="mt-2 h-[220px] px-2">
        {isPie ? (
          <DonutChart rows={rows} colors={colors} />
        ) : (
          <BalanceBarChart rows={rows} colors={colors} stratKey={stratKey} />
        )}
      </div>
      <div className="px-4 pb-3">
        <Disclosure closedLabel="Show table" openLabel="Hide table">
          <div className="mt-1">
            <StratTable rows={rows} total={total} colors={colors} showSwatches={isPie} />
          </div>
        </Disclosure>
        {note && <p className="mt-2 text-xs text-ink-faint">{note}</p>}
      </div>
    </Card>
  )
}
