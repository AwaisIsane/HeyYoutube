// Typed message contract shared between the content script, background worker,
// and side panel.

export interface TranscriptSegment {
  /** Cleaned text of the segment. */
  text: string;
  /** Start time of the segment in milliseconds. */
  tStartMs: number;
}

export type TranscriptStatus = 'ok' | 'no-captions' | 'unavailable' | 'error';

export interface TranscriptResult {
  videoId: string;
  title: string;
  status: TranscriptStatus;
  /** Full cleaned transcript text (timestamps/metadata stripped). */
  text: string;
  /** Per-cue text + start time, used to timestamp retrieval chunks for seeking. */
  segments?: TranscriptSegment[];
  lang?: string;
  error?: string;
}

export type Message =
  /** Side panel -> content script: request the current transcript. */
  | { type: 'GET_TRANSCRIPT' }
  /** Content script -> side panel (response): the requested transcript. */
  | { type: 'TRANSCRIPT_RESULT'; payload: TranscriptResult }
  /** Content script -> runtime: pushed when the watch page loads or navigates. */
  | { type: 'TRANSCRIPT_UPDATED'; payload: TranscriptResult }
  /** Side panel -> content script: seek the watch-page video to a time. */
  | { type: 'SEEK'; payload: { seconds: number } };

const MESSAGE_TYPES = new Set<string>([
  'GET_TRANSCRIPT',
  'TRANSCRIPT_RESULT',
  'TRANSCRIPT_UPDATED',
  'SEEK',
]);

export function isMessage(value: unknown): value is Message {
  return (
    typeof value === 'object' &&
    value !== null &&
    MESSAGE_TYPES.has((value as { type?: unknown }).type as string)
  );
}

/** Promise wrapper around chrome.tabs.sendMessage with graceful failure. */
export async function sendToTab<T = unknown>(
  tabId: number,
  message: Message,
): Promise<T | undefined> {
  try {
    return (await chrome.tabs.sendMessage(tabId, message)) as T;
  } catch {
    // No content script in that tab (not a watch page, or not yet injected).
    return undefined;
  }
}
