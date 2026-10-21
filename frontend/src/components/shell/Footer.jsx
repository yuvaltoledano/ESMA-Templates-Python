import { formatTime } from '../../lib/format.js'

/**
 * One muted line with the time of the last GET /api/health. Sits on the
 * page background, so it uses ink.muted (ink.faint falls short of 4.5:1
 * there). Props: `lastChecked` (ms timestamp or null).
 */
export function Footer({ lastChecked }) {
  return (
    <footer className="mx-auto w-full max-w-content px-6 py-4 text-xs text-ink-muted">
      {lastChecked ? `Last health check ${formatTime(lastChecked)}` : 'Health not checked yet'}
    </footer>
  )
}
