import { formatCount } from '../../lib/format.js'

/**
 * One line above the KPI strip: the deal name, the aggregation method
 * the pipeline chose and the group count, plus any pipeline warnings.
 * Props: `dealName`, `summary` (AnalysisSummary).
 */
export function AnalysisHeader({ dealName, summary }) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
      <h2 className="text-base font-semibold text-ink">Pool analysis: {dealName}</h2>
      <p className="text-xs text-ink-muted">
        Aggregation: <span className="font-mono">{summary.chosen_aggregation}</span>
        {' · '}Groups: <span className="font-mono">{formatCount(summary.group_count)}</span>
      </p>
      {summary.warnings && summary.warnings.length > 0 && (
        <ul className="w-full list-disc pl-5 text-xs text-warn">
          {summary.warnings.map((warning, i) => (
            <li key={i}>{warning}</li>
          ))}
        </ul>
      )}
    </div>
  )
}
