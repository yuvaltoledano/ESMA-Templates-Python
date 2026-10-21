import { formatCount, formatCurrency, formatPercent } from '../../lib/format.js'

/**
 * The rows of one stratification: Bucket, Count, % Count, Balance,
 * % Balance and a Total line. Pie tiles show a colour swatch per row in
 * place of a chart legend. Props: `rows`, `total`, `colors` (one per
 * row), `showSwatches`.
 */
export function StratTable({ rows, total, colors, showSwatches }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[420px] text-left text-base">
        <thead>
          <tr className="border-b border-line-soft text-xs uppercase tracking-wider text-ink-muted">
            <th className="py-1.5 pr-2 font-medium">Bucket</th>
            <th className="py-1.5 pr-2 text-right font-medium">Count</th>
            <th className="py-1.5 pr-2 text-right font-medium">% Count</th>
            <th className="py-1.5 pr-2 text-right font-medium">Balance</th>
            <th className="py-1.5 text-right font-medium">% Balance</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={row.label} className="border-b border-line-hair">
              <td className="py-1.5 pr-2 align-top text-ink">
                <div className="flex items-start gap-2">
                  {showSwatches && (
                    <span
                      className="mt-1 inline-block h-3 w-3 shrink-0 rounded-sm"
                      style={{ backgroundColor: colors[i] }}
                      aria-hidden="true"
                    />
                  )}
                  <span>{row.label}</span>
                </div>
              </td>
              <td className="py-1.5 pr-2 text-right font-mono tabular-nums text-ink">{formatCount(row.count)}</td>
              <td className="py-1.5 pr-2 text-right font-mono tabular-nums text-ink-muted">
                {formatPercent(row.count_pct)}
              </td>
              <td className="py-1.5 pr-2 text-right font-mono tabular-nums text-ink">
                {formatCurrency(row.balance)}
              </td>
              <td className="py-1.5 text-right font-mono tabular-nums text-ink-muted">
                {formatPercent(row.balance_pct)}
              </td>
            </tr>
          ))}
          <tr className="bg-surface-alt font-medium text-ink">
            <td className="py-1.5 pr-2">Total</td>
            <td className="py-1.5 pr-2 text-right font-mono tabular-nums">{formatCount(total.count)}</td>
            <td />
            <td className="py-1.5 pr-2 text-right font-mono tabular-nums">{formatCurrency(total.balance)}</td>
            <td />
          </tr>
        </tbody>
      </table>
    </div>
  )
}
