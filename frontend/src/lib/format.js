// Pure formatting helpers. No React, no network.

const EUR_FORMAT = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'EUR',
  maximumFractionDigits: 0,
})

const PCT_FORMAT = new Intl.NumberFormat('en-US', {
  style: 'percent',
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
})

/** Milliseconds -> "3 min 7 s" or "42 s". */
export function formatElapsed(ms) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return minutes > 0 ? `${minutes} min ${seconds} s` : `${seconds} s`
}

/** EUR without decimals: pool balances are large enough that cents are noise. */
export function formatCurrency(value) {
  return EUR_FORMAT.format(value)
}

/** A decimal in [0, 1] -> "12.3%". */
export function formatPercent(value) {
  return PCT_FORMAT.format(value)
}

/** An integer count with thousands separators. */
export function formatCount(value) {
  return value.toLocaleString()
}

// Compact-EUR formatter for bar-chart Y-axis ticks. Real pools push
// into the hundreds of millions; the full euro string overflows the
// axis gutter. Convention is standard structured-finance:
//   < €1K   -> exact (€500)
//   €1K-€1M -> €1K, €500K (no decimals)
//   €1M-€1B -> €1.5M, €500M (one decimal)
//   >= €1B  -> €1.5B, €2.3B (one decimal)
// The chart tooltip still uses the full formatter so a hover shows the
// precise number.
export function formatEurAxis(value) {
  if (value == null || Number.isNaN(value)) return ''
  const abs = Math.abs(value)
  const sign = value < 0 ? '-' : ''
  if (abs < 1_000) return `${sign}€${Math.round(abs)}`
  if (abs < 1_000_000) return `${sign}€${Math.round(abs / 1_000)}K`
  if (abs < 1_000_000_000) return `${sign}€${(abs / 1_000_000).toFixed(1)}M`
  return `${sign}€${(abs / 1_000_000_000).toFixed(1)}B`
}

// X-axis tick formatter for bucketed bar charts. Strips the unit suffix
// from the bucketed labels (seasoning's " months", LTV's "%") since the
// chart's title already conveys the unit. Tables keep the full label.
// Geography labels (NL-NH, NL-UT, ...) carry neither suffix and pass
// through unchanged.
export function formatBucketTick(value) {
  return String(value).replace(/ months$/, '').replace(/%$/, '')
}

// Pool-level weighted-average line, keyed on the stratification's
// registry key. seasoning's WA is in months (zero decimals, "months"
// suffix); current_ltv's is a decimal 0..1+ (one decimal, percent);
// current_interest_rate's is already in percent units (two decimals).
// Other stratifications have `weighted_average: null`.
export function formatWeightedAverage(stratKey, value) {
  if (stratKey === 'seasoning') {
    return `WA: ${Math.round(value).toLocaleString()} months`
  }
  if (stratKey === 'current_ltv') {
    return `WA: ${(value * 100).toFixed(1)}%`
  }
  if (stratKey === 'current_interest_rate') {
    return `WA: ${value.toFixed(2)}%`
  }
  return `WA: ${value}`
}

/** Bytes -> "123.4 KB", the file line under an upload field. */
export function formatFileSize(bytes) {
  return `${(bytes / 1024).toFixed(1)} KB`
}

/** A timestamp (ms) -> "14:32:05" in the browser's locale. */
export function formatTime(ms) {
  return new Date(ms).toLocaleTimeString()
}
