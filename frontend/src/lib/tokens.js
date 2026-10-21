// Design tokens for the "Dashboard" look. This is the single home of every
// colour: tailwind.config.js reads it to build the utility classes, and the
// chart components read it for the recharts `fill` props that cannot take a
// class name. Components never spell a hex literal themselves.
//
// Contrast (WCAG, text on background): ink.muted and ink on every surface
// pass 4.5:1 comfortably; ink.faint passes on `surface` (4.76) and
// `surface.alt` (4.55) but not on `ground` (4.23), so muted text that sits
// directly on the page background uses ink.muted.
export const tokens = {
  brand: { navy: '#0b2a3a', navyText: '#e2ecf2', navyMuted: '#9fb3c0' },
  accent: {
    DEFAULT: '#0e7490',
    hover: '#155e75',
    soft: '#ccfbf1',
    softText: '#115e59',
    glow: '#5eead4',
  },
  ground: '#eef2f6',
  surface: { DEFAULT: '#ffffff', alt: '#f8fafc' },
  ink: { DEFAULT: '#0f172a', muted: '#475569', faint: '#64748b' },
  line: { DEFAULT: '#cbd5e1', soft: '#e2e8f0', hair: '#f1f5f9' },
  ok: '#15803d',
  warn: { DEFAULT: '#b45309', soft: '#fef3c7' },
  danger: { DEFAULT: '#b91c1c', soft: '#fee2e2' },
}

export const radius = { card: '10px', control: '8px', pill: '999px' }

export const shadow = { card: '0 1px 2px rgba(15,23,42,0.06)' }
