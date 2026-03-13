/**
 * Extract a filesystem path from a `file://` URL or a web serve URL.
 * Returns `null` when the URL cannot be converted to a path.
 *
 * Supported formats:
 *   file:///Users/x        → /Users/x
 *   file:///C:/foo          → C:/foo  (Windows)
 *   /api/files/serve/foo    → /data/foo  (web/Docker serve URLs)
 */
export function fileUrlToPath(url: string): string | null {
  // Web serve URL: /api/files/serve/<relative> → /data/<relative>
  const SERVE_PREFIX = '/api/files/serve/'
  if (url.startsWith(SERVE_PREFIX)) {
    const relative = decodeURIComponent(url.slice(SERVE_PREFIX.length))
    return '/data/' + relative
  }

  if (url.startsWith('file://')) {
    let p = decodeURIComponent(url.slice(7)) // file:///Users/x -> /Users/x
    if (/^\/[A-Za-z]:/.test(p)) p = p.slice(1)
    return p
  }
  return null
}
