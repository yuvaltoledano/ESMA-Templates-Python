import { formatFileSize } from '../../lib/format.js'

/**
 * One CSV upload field: a label, the native file input styled as a drop
 * zone, the chosen file's name and size, and the inline warning when the
 * file exceeds the per-file cap the server reported. Props: `id`,
 * `label`, `accept`, `file`, `onChange`, `disabled`, `required`,
 * `maxUploadMb` (null until /api/health has answered).
 */
export function FileField({ id, label, accept, file, onChange, disabled, required, maxUploadMb }) {
  const oversized = Boolean(file) && maxUploadMb !== null && file.size > maxUploadMb * 1024 * 1024
  return (
    <div className="min-w-0 flex-1 basis-64">
      <label htmlFor={id} className="mb-1 block text-xs font-medium uppercase tracking-wider text-ink-muted">
        {label}
        {required && <span className="text-danger"> *</span>}
      </label>
      <input
        id={id}
        type="file"
        accept={accept}
        onChange={onChange}
        disabled={disabled}
        className="block min-h-[44px] w-full rounded-control border border-dashed border-line bg-surface-alt px-2 py-1.5 text-base text-ink-muted file:mr-3 file:rounded-control file:border-0 file:bg-surface file:px-3 file:py-1.5 file:text-base file:font-medium file:text-ink hover:file:bg-line-hair disabled:opacity-50 md:min-h-[40px]"
      />
      {file && (
        <p className="mt-1 truncate text-xs text-ink-faint" title={file.name}>
          {file.name} ({formatFileSize(file.size)})
        </p>
      )}
      {oversized && (
        <p className="mt-1 text-xs text-warn">
          Exceeds the server&apos;s upload limit of {maxUploadMb} MB per file and will be rejected.
        </p>
      )}
    </div>
  )
}
