// Persistent LRU cache of built retrieval indexes, keyed by videoId. Stores the
// cleaned transcript plus the serialized embedding vectors for the most recent
// few videos in chrome.storage.local (survives browser restarts), so reopening
// the Query mode on a recently-asked video skips the expensive embedding pass.
//
// Bounded to MAX_VIDEOS entries (most-recent-first) to keep well under the
// storage quota — each entry is on the order of (chunks * 384) floats.

import type { SerializedStore } from './vector-store';


const KEY = 'yt-ai:index-cache:v1';
const MAX_VIDEOS = 3;

export interface CachedIndex {
  videoId: string;
  transcript: string;
  store: SerializedStore;
  savedAt: number;
}

async function readAll(): Promise<CachedIndex[]> {
  const got = await chrome.storage.local.get(KEY);
  return (got[KEY] as CachedIndex[] | undefined) ?? [];
}

export async function loadIndex(videoId: string): Promise<CachedIndex | undefined> {
  const all = await readAll();
  const entry = all.find((e) => e.videoId === videoId);
  if (!entry) return undefined;
  // Touch: move to front so it survives eviction as the "most recent".
  await writeAll([entry, ...all.filter((e) => e.videoId !== videoId)]);
  return entry;
}

export async function saveIndex(
  videoId: string,
  transcript: string,
  store: SerializedStore,
): Promise<void> {
  const all = await readAll();
  const rest = all.filter((e) => e.videoId !== videoId);
  const next: CachedIndex[] = [{ videoId, transcript, store, savedAt: Date.now() }, ...rest];
  await writeAll(next.slice(0, MAX_VIDEOS));
}

async function writeAll(entries: CachedIndex[]): Promise<void> {
  await chrome.storage.local.set({ [KEY]: entries.slice(0, MAX_VIDEOS) });
}
