/** A small rotating ring in the accent colour. Props: `className`. */
export function Spinner({ className = '' }) {
  return (
    <span
      aria-hidden="true"
      className={`inline-block h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-line border-t-accent ${className}`}
    />
  )
}
