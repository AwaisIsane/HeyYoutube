// Small helpers for reasoning about YouTube watch URLs, shared by the background
// worker (which gates the side panel to watch pages) and the side panel (which
// re-syncs to the active tab).

/** True for a YouTube watch page URL, e.g. https://www.youtube.com/watch?v=… */
export function isWatchUrl(url: string | undefined | null): boolean {
  if (!url) return false;
  try {
    const u = new URL(url);
    return (
      (u.hostname === 'www.youtube.com' || u.hostname === 'youtube.com') &&
      u.pathname === '/watch'
    );
  } catch {
    return false;
  }
}

/** True for any page on the YouTube site (not necessarily a watch page). */
export function isYouTubeHost(url: string | undefined | null): boolean {
  if (!url) return false;
  try {
    const { hostname } = new URL(url);
    return hostname === 'www.youtube.com' || hostname === 'youtube.com';
  } catch {
    return false;
  }
}

/** Format a millisecond offset as `m:ss` (or `h:mm:ss` past an hour). */
export function formatTimestamp(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const s = total % 60;
  const m = Math.floor(total / 60) % 60;
  const h = Math.floor(total / 3600);
  const pad = (n: number) => String(n).padStart(2, '0');
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

/** Matches `m:ss` / `mm:ss` / `h:mm:ss` timestamps for linkifying answer text. */
export const TIMESTAMP_RE = /\b\d{1,2}:[0-5]\d(?::[0-5]\d)?\b/g;

/** Parse a `m:ss` / `h:mm:ss` timestamp into whole seconds. */
export function parseTimestampToSeconds(text: string): number {
  const parts = text.split(':').map((p) => parseInt(p, 10));
  if (parts.some((n) => Number.isNaN(n))) return 0;
  return parts.reduce((acc, p) => acc * 60 + p, 0);
}
