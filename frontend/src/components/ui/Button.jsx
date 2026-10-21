const VARIANTS = {
  primary: 'bg-accent text-white hover:bg-accent-hover disabled:bg-line disabled:text-ink-faint',
  secondary: 'border border-accent bg-surface text-accent hover:bg-accent-soft disabled:border-line disabled:text-ink-faint',
  ghost: 'border border-line bg-surface text-ink-muted hover:bg-surface-alt disabled:text-ink-faint',
  danger: 'border border-danger bg-surface text-danger hover:bg-danger-soft',
}

/**
 * The page's one button style. Renders an `<a>` when `href` is given
 * (the workbook download link), otherwise a `<button>`. Props: `variant`
 * ('primary' | 'secondary' | 'ghost' | 'danger'), `type`, `disabled`,
 * `onClick`, `href`, `download`, `className`, `children`. Controls are at
 * least 44px tall on a phone and 36px from the md breakpoint up.
 */
export function Button({
  variant = 'primary',
  type = 'button',
  disabled = false,
  onClick,
  href,
  download,
  className = '',
  children,
}) {
  const classes =
    'inline-flex min-h-[44px] items-center justify-center rounded-control px-4 text-base font-medium ' +
    'transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-glow ' +
    'disabled:cursor-not-allowed md:min-h-[36px] ' +
    `${VARIANTS[variant]} ${className}`
  if (href) {
    return (
      <a href={href} download={download} className={classes}>
        {children}
      </a>
    )
  }
  return (
    <button type={type} disabled={disabled} onClick={onClick} className={classes}>
      {children}
    </button>
  )
}
