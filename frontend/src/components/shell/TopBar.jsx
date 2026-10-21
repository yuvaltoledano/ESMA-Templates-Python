import { HealthStatus } from './HealthStatus.jsx'

// The nav slot. Adding a page later is one more entry here; `current`
// marks the item underlined in the accent glow.
const NAV_ITEMS = [{ label: 'Process', href: '#', current: true }]

function LineChartIcon() {
  return (
    <svg
      viewBox="0 0 20 20"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M2 17h16" />
      <path d="M3 13l4-5 4 3 6-7" />
    </svg>
  )
}

/**
 * The 56px navy bar: wordmark with a line-chart icon, the nav items and
 * the API health line. Wraps on narrow screens. Props: `health` (the
 * status string from useHealth).
 */
export function TopBar({ health }) {
  return (
    <header className="bg-brand-navy text-brand-navyText">
      <div className="mx-auto flex min-h-[56px] w-full max-w-content flex-wrap items-center gap-x-6 gap-y-2 px-6 py-2">
        <div className="flex items-center gap-2 text-base font-semibold">
          <span className="text-accent-glow">
            <LineChartIcon />
          </span>
          <span>ESMA → MILAN</span>
        </div>
        <nav aria-label="Pages" className="flex items-center gap-4">
          {NAV_ITEMS.map((item) => (
            <a
              key={item.label}
              href={item.href}
              aria-current={item.current ? 'page' : undefined}
              className={`inline-flex min-h-[44px] items-center border-b-2 text-base md:min-h-[40px] ${
                item.current
                  ? 'border-accent-glow text-brand-navyText'
                  : 'border-transparent text-brand-navyMuted hover:text-brand-navyText'
              }`}
            >
              {item.label}
            </a>
          ))}
        </nav>
        <div className="ml-auto">
          <HealthStatus status={health} />
        </div>
      </div>
    </header>
  )
}
