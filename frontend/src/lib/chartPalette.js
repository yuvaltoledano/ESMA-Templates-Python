import { tokens } from './tokens.js'

// Categorical chart palette, indexed by spec-row position so the biggest
// categoricals (interest rate type and loan purpose, 13 ESMA codes each,
// plus a possible UNK row) each get a distinct colour without cycling.
// Adjacent entries differ in lightness as well as hue, so neighbouring
// slices stay distinguishable for colour-blind readers; keep that property
// when adjusting. The bar charts use the accent colour only.
export const CHART_PALETTE = [
  '#0e7490', '#f59e0b', '#6366f1', '#0891b2', '#d97706',
  '#4f46e5', '#14b8a6', '#ea580c', '#8b5cf6', '#0d9488',
  '#c2410c', '#7c3aed', '#2dd4bf', '#fb923c', '#a78bfa',
]

/** The catch-all "Other" row (geographic stratification) is drawn muted. */
export const OTHER_COLOR = tokens.ink.faint

/** The single colour of every bar in a bucketed chart. */
export const BAR_COLOR = tokens.accent.DEFAULT

/** Colour for the row at `index` with label `label`. */
export function colorForRow(label, index) {
  if (label === 'Other') return OTHER_COLOR
  return CHART_PALETTE[index % CHART_PALETTE.length]
}
