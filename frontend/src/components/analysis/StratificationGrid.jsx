import { StratificationTile } from './StratificationTile.jsx'

/**
 * The stratification tiles, one per key of the response's
 * `stratifications` in the order the API returns them (the registry
 * order): three columns from 1280px, two from 768px, one below. Props:
 * `stratifications` (the keyed object from AnalysisResponse).
 */
export function StratificationGrid({ stratifications }) {
  return (
    <div className="grid min-w-0 flex-1 grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-3">
      {Object.entries(stratifications).map(([key, strat]) => (
        <StratificationTile key={key} stratKey={key} strat={strat} />
      ))}
    </div>
  )
}
