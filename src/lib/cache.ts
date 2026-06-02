// Lightweight per-videoId cache backed by chrome.storage.session (cleared when
// the browser closes). Stores cheap-to-serialize artifacts — cleaned transcript,
// generated summary, generated quiz — so switching modes or reopening the side
// panel doesn't recompute. Embedding vectors are persisted separately (and
// across browser restarts) for the last few videos in retrieval/index-cache.

const NS = 'yt-ai';

function keyFor(videoId: string): string {
  return `${NS}:${videoId}`;
}

type VideoCache = Record<string, unknown>;

async function readVideo(videoId: string): Promise<VideoCache> {
  const key = keyFor(videoId);
  const stored = await chrome.storage.session.get(key);
  return (stored[key] as VideoCache | undefined) ?? {};
}

export async function cacheGet<T>(
  videoId: string,
  field: string,
): Promise<T | undefined> {
  const entry = await readVideo(videoId);
  return entry[field] as T | undefined;
}

export async function cacheSet<T>(
  videoId: string,
  field: string,
  value: T,
): Promise<void> {
  const key = keyFor(videoId);
  const entry = await readVideo(videoId);
  entry[field] = value;
  await chrome.storage.session.set({ [key]: entry });
}
