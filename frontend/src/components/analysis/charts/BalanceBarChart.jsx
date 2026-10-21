import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import { BAR_COLOR } from '../../../lib/chartPalette.js'
import { formatBucketTick, formatEurAxis } from '../../../lib/format.js'
import { tokens } from '../../../lib/tokens.js'
import { ChartTooltip } from './ChartTooltip.jsx'
import { toChartData } from './chartData.js'
import { EmptyChart } from './EmptyChart.jsx'

const TICK_STYLE = { fontSize: 10, fill: tokens.ink.muted, fontFamily: 'IBM Plex Mono, ui-monospace, monospace' }

/**
 * Vertical bars of balance by bucket for a bucketed or geographic
 * stratification, all in the accent colour. Labels are tilted -30° so
 * they stay legible in a three-column tile (about 245px wide at 1280px);
 * the dense interest-rate axis (17 buckets) tilts -45°.
 * Props: `rows`, `colors` (unused for fill, kept for the tooltip data),
 * `stratKey` (the registry key, for the axis rules).
 */
export function BalanceBarChart({ rows, colors, stratKey }) {
  const data = toChartData(rows, colors)
  if (data.length === 0) return <EmptyChart />

  const denseAxis = stratKey === 'current_interest_rate'
  const axisAngle = denseAxis ? -45 : -30
  const axisAnchor = 'end'
  const axisHeight = denseAxis ? 60 : 50

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 5, right: 10, left: 5, bottom: 20 }}>
        <XAxis
          dataKey="name"
          tick={TICK_STYLE}
          angle={axisAngle}
          textAnchor={axisAnchor}
          height={axisHeight}
          interval={0}
          tickFormatter={formatBucketTick}
          stroke={tokens.line.DEFAULT}
        />
        <YAxis tick={TICK_STYLE} width={60} tickFormatter={formatEurAxis} stroke={tokens.line.DEFAULT} />
        <Tooltip content={<ChartTooltip />} cursor={{ fill: tokens.line.hair }} />
        <Bar dataKey="value" fill={BAR_COLOR} radius={[2, 2, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}
