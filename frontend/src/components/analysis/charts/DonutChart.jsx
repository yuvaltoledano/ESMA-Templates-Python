import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts'

import { ChartTooltip } from './ChartTooltip.jsx'
import { toChartData } from './chartData.js'
import { EmptyChart } from './EmptyChart.jsx'

/**
 * Donut of balance by row for a categorical stratification. Rows with a
 * zero balance are left out (their colour slot is kept, so the table
 * swatches still match). No legend: the table rows are the legend. Props:
 * `rows` (stratification rows), `colors` (one per row).
 */
export function DonutChart({ rows, colors }) {
  const data = toChartData(rows, colors).filter((point) => point.value > 0)
  if (data.length === 0) return <EmptyChart />
  return (
    <ResponsiveContainer width="100%" height="100%">
      <PieChart>
        <Pie data={data} dataKey="value" nameKey="name" innerRadius="35%" outerRadius="80%" stroke="none">
          {data.map((entry) => (
            <Cell key={entry.name} fill={entry.color} />
          ))}
        </Pie>
        <Tooltip content={<ChartTooltip />} />
      </PieChart>
    </ResponsiveContainer>
  )
}
