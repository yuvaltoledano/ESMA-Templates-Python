import { Button } from '../ui/Button.jsx'
import { Card } from '../ui/Card.jsx'
import { Spinner } from '../ui/Spinner.jsx'
import { FileField } from './FileField.jsx'

/**
 * The upload form: loans CSV, collaterals CSV, deal name and the two
 * actions in one wrapping row. Enter submits in download mode. Props:
 * `loansFile`, `collateralsFile`, `dealName`, `onLoansChange`,
 * `onCollateralsChange`, `onDealNameChange`, `onSubmit(mode)` with mode
 * 'download' | 'analysis', `busy` (controls disabled), `uploading`
 * (shows the upload line), `formError` (client-side validation text),
 * `maxUploadMb` (for the per-field size warning).
 */
export function UploadCard({
  loansFile,
  collateralsFile,
  dealName,
  onLoansChange,
  onCollateralsChange,
  onDealNameChange,
  onSubmit,
  busy,
  uploading,
  formError,
  maxUploadMb,
}) {
  return (
    <Card>
      <form
        onSubmit={(event) => {
          event.preventDefault()
          onSubmit('download')
        }}
        className="flex flex-wrap items-start gap-4 p-4"
      >
        <FileField
          id="loans"
          label="Loans CSV"
          accept=".csv,text/csv"
          file={loansFile}
          onChange={onLoansChange}
          disabled={busy}
          required
          maxUploadMb={maxUploadMb}
        />
        <FileField
          id="collaterals"
          label="Collaterals CSV"
          accept=".csv,text/csv"
          file={collateralsFile}
          onChange={onCollateralsChange}
          disabled={busy}
          required
          maxUploadMb={maxUploadMb}
        />
        <div className="min-w-0 flex-1 basis-56">
          <label
            htmlFor="deal-name"
            className="mb-1 block text-xs font-medium uppercase tracking-wider text-ink-muted"
          >
            Deal Name<span className="text-danger"> *</span>
          </label>
          <input
            id="deal-name"
            type="text"
            value={dealName}
            onChange={onDealNameChange}
            disabled={busy}
            placeholder="e.g. DEAL_2025-1"
            className="block min-h-[44px] w-full rounded-control border border-line bg-surface px-3 text-base text-ink placeholder:text-ink-faint focus:border-accent focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-glow disabled:bg-surface-alt md:min-h-[40px]"
          />
        </div>
        <div className="flex w-full flex-col gap-2 sm:w-auto sm:flex-row sm:self-end">
          <Button type="submit" variant="primary" disabled={busy} className="sm:min-h-[40px]">
            Process &amp; download
          </Button>
          <Button variant="secondary" disabled={busy} onClick={() => onSubmit('analysis')} className="sm:min-h-[40px]">
            Analyze pool
          </Button>
        </div>
        {formError && (
          <p className="w-full rounded-control bg-warn-soft px-3 py-2 text-base text-warn">{formError}</p>
        )}
        {uploading && (
          <p className="flex w-full items-center gap-3 text-base text-ink-muted">
            <Spinner />
            Uploading files... large pools can take a few minutes to upload.
          </p>
        )}
      </form>
    </Card>
  )
}
