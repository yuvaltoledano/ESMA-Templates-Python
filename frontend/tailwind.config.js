import { radius, shadow, tokens } from './src/lib/tokens.js'

// Semantic theme built from src/lib/tokens.js. Components use the class
// names generated here (bg-brand-navy, text-ink-muted, rounded-card, ...)
// and never a raw hex; the charts import the same tokens module.
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: tokens,
      borderRadius: radius,
      boxShadow: shadow,
      fontFamily: {
        sans: ['"IBM Plex Sans"', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'monospace'],
      },
      fontSize: {
        // Base body size for the page; labels use the default `text-xs`
        // (12px) and KPI values `text-kpi`.
        base: ['14px', { lineHeight: '1.45' }],
        kpi: ['20px', { lineHeight: '1.2' }],
      },
      maxWidth: {
        content: '1440px',
        summary: '420px',
      },
    },
  },
  plugins: [],
}
