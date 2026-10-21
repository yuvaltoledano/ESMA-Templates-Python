import { Button } from '../ui/Button.jsx'

/**
 * A failure in the danger soft colour: the message, the API error code
 * when there is one, the expandable JSON details, and "Delete from
 * server" when the job it belongs to is done or failed. Props: `error`
 * ({type, message, code?, details?}), `canDelete`, `onDelete`.
 */
export function ErrorPanel({ error, canDelete, onDelete }) {
  return (
    <div className="rounded-card border border-danger/20 bg-danger-soft px-4 py-3 text-base text-danger">
      <p className="font-medium">{error.message}</p>
      {error.type === 'api' && error.code && <p className="mt-1 font-mono text-xs">Code: {error.code}</p>}
      {error.type === 'api' && error.details && (
        <details className="mt-2 text-xs">
          <summary className="cursor-pointer">Show details</summary>
          <pre className="mt-1 overflow-x-auto whitespace-pre-wrap font-mono">
            {JSON.stringify(error.details, null, 2)}
          </pre>
        </details>
      )}
      {canDelete && (
        <div className="mt-3">
          <Button variant="danger" onClick={onDelete}>
            Delete from server
          </Button>
        </div>
      )}
    </div>
  )
}
