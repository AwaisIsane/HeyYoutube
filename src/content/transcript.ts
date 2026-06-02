// Content script for YouTube watch pages.
// Resolves the video's caption track via the InnerTube `player` endpoint
// (ANDROID client), fetches the timedtext, cleans it into plain text, and serves
// it to the side panel — both on demand (GET_TRANSCRIPT) and pushed on SPA
// navigation (TRANSCRIPT_UPDATED).
//
// Why InnerTube instead of scraping the watch page: as of 2025 the caption
// `baseUrl` embedded in the page's `ytInitialPlayerResponse` is gated behind a
// Proof-of-Origin Token (PoToken) and returns `200 OK` with an *empty body*
// without one. The ANDROID InnerTube client returns caption `baseUrl`s that are
// not PoToken-gated and serve the full timedtext, so we go through it instead.

import {
  isMessage,
  type Message,
  type TranscriptResult,
  type TranscriptSegment,
} from '@/lib/messages';

const DEBUG = false; // flip to true to trace transcript extraction in the console
function log(...args: unknown[]): void {
  if (DEBUG) console.log('[yt-transcript]', ...args);
}

// InnerTube ANDROID client. The caption baseUrls it returns are not PoToken-gated.
const INNERTUBE_PLAYER_URL =
  'https://www.youtube.com/youtubei/v1/player?prettyPrint=false';
const INNERTUBE_CLIENT_VERSION = '20.10.38';
const INNERTUBE_CONTEXT = {
  client: {
    clientName: 'ANDROID',
    clientVersion: INNERTUBE_CLIENT_VERSION,
    hl: 'en',
  },
};

interface CaptionTrack {
  baseUrl: string;
  languageCode?: string;
  kind?: string; // 'asr' for auto-generated
  name?: { simpleText?: string; runs?: { text: string }[] };
}

function isAllowedCaptionUrl(url: string): boolean {
  try {
    const { hostname } = new URL(url);
    return hostname === 'www.youtube.com' || hostname.endsWith('.googlevideo.com');
  } catch {
    return false;
  }
}

// Cache the last result for the current video so repeat requests are instant.
let current: TranscriptResult | null = null;
let inFlight: Promise<TranscriptResult> | null = null;
let currentVideoId: string | null = null;

function getVideoId(): string | null {
  return new URL(location.href).searchParams.get('v');
}

/**
 * Resolve the video's player response via the InnerTube ANDROID client. Runs
 * same-origin (we're on www.youtube.com), so no CORS or API key is required —
 * the client context in the body is enough.
 */
async function fetchPlayerResponse(videoId: string): Promise<any | null> {
  const res = await fetch(INNERTUBE_PLAYER_URL, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ context: INNERTUBE_CONTEXT, videoId }),
  });
  log('innertube player fetch', { ok: res.ok, status: res.status });
  if (!res.ok) return null;
  try {
    return await res.json();
  } catch (err) {
    log('failed to parse innertube player JSON', err);
    return null;
  }
}

function pickTrack(tracks: CaptionTrack[]): CaptionTrack | null {
  if (tracks.length === 0) return null;
  const pageLang = (navigator.language || 'en').slice(0, 2);
  const manual = tracks.filter((t) => t.kind !== 'asr');
  const pool = manual.length > 0 ? manual : tracks;
  return (
    pool.find((t) => t.languageCode?.startsWith(pageLang)) ??
    pool.find((t) => t.languageCode?.startsWith('en')) ??
    pool[0]!
  );
}

function cleanSegmentText(raw: string): string {
  return raw
    .replace(/\s*\n\s*/g, ' ') // newlines inside a cue -> space
    .replace(/\[[^\]]*\]/g, '') // [Music], [Applause], ...
    .replace(/\s{2,}/g, ' ')
    .trim();
}

/**
 * Parse a timedtext document into segments. Handles both the srv3 format the
 * ANDROID client serves (`<p t="ms" d="ms"><s>word</s>…</p>`, where each `<s>`
 * carries its own leading space so textContent joins cleanly) and the legacy
 * format (`<text start="s" dur="s">…</text>`). Empty rolling-ASR spacer cues
 * and exact repeats of the previous line are dropped.
 */
function parseTimedText(doc: Document): TranscriptSegment[] {
  const ps = Array.from(doc.querySelectorAll('p'));
  const isSrv3 = ps.length > 0;
  const nodes = isSrv3 ? ps : Array.from(doc.querySelectorAll('text'));

  const segments: TranscriptSegment[] = [];
  let prev = '';
  for (const node of nodes) {
    const text = cleanSegmentText(node.textContent ?? '');
    if (!text || text === prev) continue; // drop empties + ASR repeats
    prev = text;
    const tStartMs = isSrv3
      ? Math.round(parseFloat(node.getAttribute('t') ?? '0'))
      : Math.round(parseFloat(node.getAttribute('start') ?? '0') * 1000);
    segments.push({ text, tStartMs });
  }
  return segments;
}

/** Parse the json3 caption format as a fallback, should a track serve it. */
function parseJson3(body: string): TranscriptSegment[] {
  interface Json3Event {
    tStartMs?: number;
    segs?: { utf8?: string }[];
  }
  const events = (JSON.parse(body) as { events?: Json3Event[] }).events ?? [];
  const segments: TranscriptSegment[] = [];
  let prev = '';
  for (const ev of events) {
    if (!ev.segs) continue;
    const text = cleanSegmentText(ev.segs.map((s) => s.utf8 ?? '').join(''));
    if (!text || text === prev) continue;
    prev = text;
    segments.push({ text, tStartMs: ev.tStartMs ?? 0 });
  }
  return segments;
}

/**
 * Fetch and parse a caption track. The ANDROID baseUrl serves srv3 XML; we sniff
 * the body so a json3 response is handled too. An empty/garbage body yields no
 * segments rather than throwing.
 */
async function fetchTranscriptSegments(
  track: CaptionTrack,
): Promise<TranscriptSegment[]> {
  if (!isAllowedCaptionUrl(track.baseUrl)) {
    log('caption baseUrl has unexpected hostname, skipping');
    return [];
  }
  const res = await fetch(track.baseUrl, { credentials: 'include' });
  const body = res.ok ? await res.text() : '';
  log('caption fetch', { ok: res.ok, status: res.status, bodyLength: body.length });
  const trimmed = body.trim();
  if (!trimmed) return [];

  try {
    if (trimmed.startsWith('{')) return parseJson3(trimmed);
    const doc = new DOMParser().parseFromString(body, 'text/xml');
    return parseTimedText(doc);
  } catch (err) {
    log('caption parse failed', err);
    return [];
  }
}

function getTitle(player: any): string {
  return (
    player?.videoDetails?.title ||
    document.title.replace(/\s*-\s*YouTube\s*$/, '') ||
    'YouTube video'
  );
}

async function extract(videoId: string): Promise<TranscriptResult> {
  try {
    const player = await fetchPlayerResponse(videoId);
    const title = getTitle(player);

    const playability = player?.playabilityStatus?.status;
    log('playabilityStatus', playability, '| hasPlayer', !!player);
    if (playability && playability !== 'OK') {
      return { videoId, title, status: 'unavailable', text: '' };
    }

    const tracks: CaptionTrack[] =
      player?.captions?.playerCaptionsTracklistRenderer?.captionTracks ?? [];
    log(
      'captionTracks found',
      tracks.length,
      tracks.map((t) => ({ lang: t.languageCode, kind: t.kind })),
    );
    const track = pickTrack(tracks);
    if (!track) {
      return { videoId, title, status: 'no-captions', text: '' };
    }
    log('picked track', { lang: track.languageCode, kind: track.kind });

    const segments = await fetchTranscriptSegments(track);
    const text = segments
      .map((s) => s.text)
      .join(' ')
      .replace(/\s{2,}/g, ' ')
      .trim();
    log('extracted text length', text.length, '| segments', segments.length);
    if (!text) {
      return { videoId, title, status: 'no-captions', text: '' };
    }

    return {
      videoId,
      title,
      status: 'ok',
      text,
      segments,
      lang: track.languageCode,
    };
  } catch (err) {
    return {
      videoId,
      title: document.title.replace(/\s*-\s*YouTube\s*$/, ''),
      status: 'error',
      text: '',
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

async function getTranscript(): Promise<TranscriptResult> {
  const videoId = getVideoId();
  if (!videoId) {
    return { videoId: '', title: '', status: 'unavailable', text: '' };
  }
  if (current && current.videoId === videoId && current.status === 'ok') {
    return current;
  }
  if (inFlight && currentVideoId === videoId) return inFlight;

  currentVideoId = videoId;
  inFlight = extract(videoId).then((result) => {
    current = result;
    inFlight = null;
    return result;
  });
  return inFlight;
}

// Respond to side-panel requests.
chrome.runtime.onMessage.addListener((msg: unknown, _sender, sendResponse) => {
  if (!isMessage(msg)) return;
  if (msg.type === 'GET_TRANSCRIPT') {
    getTranscript().then((payload) => {
      const response: Message = { type: 'TRANSCRIPT_RESULT', payload };
      sendResponse(response);
    });
    return true; // async response
  }
  if (msg.type === 'SEEK') {
    const video = document.querySelector('video');
    if (video) {
      const seconds = msg.payload.seconds;
      if (Number.isFinite(seconds) && seconds >= 0) {
        video.currentTime = seconds;
        void video.play().catch(() => {}); // autoplay may be blocked; ignore
      }
    }
    return undefined; // no response needed
  }
  return undefined;
});

// Re-extract on YouTube SPA navigation and push to any open side panel.
function onNavigated(): void {
  const videoId = getVideoId();
  if (!videoId || videoId === currentVideoId) return;
  current = null;
  getTranscript().then((payload) => {
    // An orphaned content script (extension reloaded/updated while this page
    // stayed open) loses chrome.runtime, so reading .sendMessage would throw
    // synchronously past the .catch below.
    if (!chrome.runtime?.id) return;
    const message: Message = { type: 'TRANSCRIPT_UPDATED', payload };
    chrome.runtime.sendMessage(message).catch(() => {
      // No side panel listening; ignore.
    });
  });
}

window.addEventListener('yt-navigate-finish', onNavigated);
// Prime the cache on initial load.
void getTranscript();
