/**
 * Chart data points for a stratification's rows. Each point carries the
 * row's stable colour (assigned by row position, so a filtered chart
 * still matches the table swatches) and every metric the tooltip shows.
 */
export function toChartData(rows, colors) {
  return rows.map((row, i) => ({
    name: row.label,
    value: row.balance,
    color: colors[i],
    count: row.count,
    count_pct: row.count_pct,
    balance: row.balance,
    balance_pct: row.balance_pct,
  }))
}
