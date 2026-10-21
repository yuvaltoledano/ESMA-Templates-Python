import { Card } from '../ui/Card.jsx'

/**
 * The workbook's Sheet 1 as a two-column table: every row the API sent
 * (38 base metrics plus one per structure type), label in muted ink and
 * the pre-formatted value right-aligned in mono, in a scrolling box with
 * a sticky header. Props: `rows` (list of {label, value}).
 */
export function ExecutionSummaryCard({ rows }) {
  return (
    <Card title="Execution summary" className="w-full shrink-0 lg:max-w-summary">
      <div className="mt-3 max-h-[640px] overflow-auto border-t border-line-soft">
        <table className="w-full text-left text-base">
          <thead className="sticky top-0 bg-surface">
            <tr className="text-xs uppercase tracking-wider text-ink-muted">
              <th className="px-4 py-2 font-medium">Metric</th>
              <th className="px-4 py-2 text-right font-medium">Value</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={`${row.label}-${i}`} className="border-t border-line-hair">
                <td className="px-4 py-1.5 align-top text-ink-muted">{row.label}</td>
                <td className="whitespace-nowrap px-4 py-1.5 text-right align-top font-mono tabular-nums text-ink">
                  {row.value}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}
