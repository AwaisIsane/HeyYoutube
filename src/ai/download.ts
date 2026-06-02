/**
 * Build a `monitor` create-option that forwards download progress (0..1), or
 * `undefined` when no handler was given (so callers can assign it conditionally).
 */
export function downloadMonitor(
  onDownload?: (progress: number) => void,
): CreateMonitorCallback | undefined {
  if (!onDownload) return undefined;
  return (m) =>
    m.addEventListener('downloadprogress', (e) => onDownload(e.loaded));
}
