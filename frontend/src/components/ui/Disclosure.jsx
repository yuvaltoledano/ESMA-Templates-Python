import { useState } from 'react'

/**
 * A native `<details>` with a styled summary whose text changes with the
 * open state. Props: `closedLabel`, `openLabel`, `children`.
 */
export function Disclosure({ closedLabel, openLabel, children }) {
  const [open, setOpen] = useState(false)
  return (
    <details onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary className="inline-flex min-h-[44px] cursor-pointer select-none items-center text-xs font-medium uppercase tracking-wider text-ink-muted hover:text-ink md:min-h-[32px]">
        {open ? openLabel : closedLabel}
      </summary>
      {children}
    </details>
  )
}
