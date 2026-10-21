/**
 * A white surface with the design's radius, hairline border and shadow.
 * Props: `title` (optional heading), `actions` (optional node at the
 * heading's right), `className`, `children`.
 */
export function Card({ title, actions, className = '', children }) {
  return (
    <section className={`rounded-card border border-line-soft bg-surface shadow-card ${className}`}>
      {(title || actions) && (
        <header className="flex items-center justify-between gap-3 px-4 pt-4">
          {title && <h2 className="text-base font-semibold text-ink">{title}</h2>}
          {actions}
        </header>
      )}
      {children}
    </section>
  )
}
