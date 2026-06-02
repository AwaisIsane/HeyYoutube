// Side panel controller. A Chrome side panel is one shared document per window
// that does NOT reload on tab switch, so each watch tab keeps its own TabView
// (see ./tabs) — its own panel DOM, transcript, sessions, and in-flight runs.
// main.ts owns the shared chrome (now-watching strip, tab nav, global model
// download) and routes user actions to the active tab's view; the three feature
// modes (./modes) operate on whichever view they're handed.

import {
  isMessage,
  sendToTab,
  type Message,
  type TranscriptResult,
} from '@/lib/messages';
import { isWatchUrl, isYouTubeHost, formatTimestamp } from '@/lib/youtube';
import { languageModelAvailability, createSession } from '@/ai/prompt';
import { summarizerAvailability } from '@/ai/summarizer';
import { $, delay, setStatus, showBanner, clearBanner, errorMessage } from './dom';
import type { PanelContext } from './context';
import * as tabs from './tabs';
import type { TabView, Mode } from './tabs';
import { runSummarize, resetSummarize, copySummary } from './modes/summarize';
import { runQuizMode, resetQuiz } from './modes/quiz';
import { runQuery, clearQuery, resetQuery } from './modes/query';

// Shared now-watching strip (one per window, re-painted from the active view).
const titleEl = $('video-title');
const thumbEl = $<HTMLImageElement>('video-thumb');
const durEl = $('video-dur');

let summarizerReady = false;
let currentTabId: number | null = null;
let panelWindowId: number | null = null;

// ---------- Model availability (one global process) ----------
// The model download runs once in the browser process and is shared across every
// tab. We drive it with a single create() monitor, share one in-flight promise so
// concurrent mode starts don't each open a banner, and keep the banner up across
// tab switches (downloadActive) so its progress never vanishes when you switch.
let downloadPromise: Promise<void> | null = null;
let downloadActive = false;

async function downloadModel(): Promise<void> {
  downloadActive = true;
  // First-time setup note: dropped once progress starts so it doesn't repeat.
  let note = ' This is a one-time setup and may take a few minutes.';
  showBanner(`Downloading on-device model… 0%${note}`, 'warn');
  try {
    const session = await createSession({
      onDownload: (p) => {
        showBanner(`Downloading on-device model… ${Math.round(p * 100)}%${note}`, 'warn');
      },
    });
    session.destroy(); // warm-up only; the per-mode sessions create their own
  } finally {
    downloadActive = false;
  }
  clearBanner();
}

async function ensureModelReady(): Promise<boolean> {
  const avail: Availability = await languageModelAvailability();
  if (avail === 'unavailable') {
    showBanner(
      'Gemini Nano is unavailable. Requires Chrome 138+ with the on-device model enabled.',
      'err',
    );
    return false;
  }
  if (avail === 'available') return true;

  // downloadable | downloading — fetch the model once, reporting progress.
  if (!downloadPromise) downloadPromise = downloadModel();
  try {
    await downloadPromise;
    return true;
  } catch (err) {
    downloadPromise = null; // let the next attempt retry
    showBanner(`Model download failed: ${errorMessage(err)}`, 'err');
    return false;
  }
}

// Clear the shared banner unless a model download is showing its progress there.
function clearBannerIfIdle(): void {
  if (!downloadActive) clearBanner();
}

// ---------- Now-watching strip ----------
// Paint the shared strip from a view's stored values (called on activate and when
// the active view's transcript loads).
function renderWatching(view: TabView): void {
  const w = view.watching;
  titleEl.textContent = w.title;
  if (w.videoId) {
    thumbEl.src = `https://i.ytimg.com/vi/${w.videoId}/mqdefault.jpg`;
    thumbEl.hidden = false;
  } else {
    thumbEl.hidden = true;
    thumbEl.removeAttribute('src');
  }
  durEl.hidden = w.durationMs <= 0;
  if (w.durationMs > 0) durEl.textContent = formatTimestamp(w.durationMs);
  setStatus(w.statusText, w.statusCls);
}

// Store a status line on the view and paint it only if that view is on screen, so
// a background tab's progress updates its own state without touching the display.
function setViewStatus(view: TabView, text: string, cls = ''): void {
  view.watching.statusText = text;
  view.watching.statusCls = cls;
  if (tabs.isActive(view)) setStatus(text, cls);
}

function onTranscript(view: TabView, result: TranscriptResult): void {
  if (view.transcript && view.transcript.videoId !== result.videoId) resetViewModes(view);
  view.transcript = result;

  const w = view.watching;
  w.title = result.title || 'YouTube video';
  w.videoId = result.videoId;
  const segs = result.segments;
  w.durationMs = segs?.length ? segs[segs.length - 1]!.tStartMs : 0;

  switch (result.status) {
    case 'ok':
      w.statusText = 'Transcript loaded';
      w.statusCls = 'ok';
      break;
    case 'no-captions':
      w.statusText = 'No captions available for this video';
      w.statusCls = 'err';
      break;
    case 'unavailable':
      w.statusText = 'Transcript unavailable';
      w.statusCls = 'err';
      break;
    default:
      w.statusText = result.error ? `Error: ${result.error}` : 'Could not read transcript';
      w.statusCls = 'err';
  }

  if (tabs.isActive(view)) {
    renderWatching(view);
    if (result.status === 'ok') clearBannerIfIdle();
  }
}

// ---------- Reset ----------
// Clear a view's three panels back to their empty/intro state (DOM only; runs and
// sessions are torn down by tabs.reset).
function resetViewModes(view: TabView): void {
  resetSummarize(view);
  resetQuiz(view);
  resetQuery(view);
}

// ---------- Transcript loading (per view) ----------
async function getActiveTab(): Promise<chrome.tabs.Tab | undefined> {
  if (panelWindowId == null) return undefined;
  const [tab] = await chrome.tabs.query({ active: true, windowId: panelWindowId });
  return tab;
}

// Manifest content scripts are only injected into pages that load AFTER the
// extension is installed/reloaded — never into already-open tabs. So a watch page
// opened earlier has no content script and GET_TRANSCRIPT goes unanswered. Inject
// it on demand so the user doesn't have to reload the video.
async function injectContentScript(tabId: number): Promise<boolean> {
  const files = chrome.runtime.getManifest().content_scripts?.[0]?.js ?? [];
  if (files.length === 0) return false;
  try {
    await chrome.scripting.executeScript({ target: { tabId }, files });
    return true;
  } catch {
    return false;
  }
}

async function loadTranscriptForTab(view: TabView): Promise<void> {
  const tabId = view.tabId;
  let res = await sendToTab<Message>(tabId, { type: 'GET_TRANSCRIPT' });
  // undefined means no content script received the message — almost always
  // because the tab was open before the extension loaded.
  if (res === undefined && (await injectContentScript(tabId))) {
    // The CRXJS loader registers its onMessage listener via an async import()
    // that resolves shortly AFTER executeScript, so poll rather than retry once.
    // We keep polling even if the user switches away — the result lands in this
    // view and is shown when they return.
    for (let i = 0; i < 20 && res === undefined; i++) {
      await delay(100);
      res = await sendToTab<Message>(tabId, { type: 'GET_TRANSCRIPT' });
    }
  }
  if (res && res.type === 'TRANSCRIPT_RESULT') onTranscript(view, res.payload);
  else setViewStatus(view, 'Transcript not ready — try reloading the video');
}

function transcriptText(view: TabView): string | null {
  const t = view.transcript;
  if (!t || t.status !== 'ok' || !t.text) {
    showBanner('No usable transcript for this video.', 'warn');
    return null;
  }
  clearBannerIfIdle();
  return t.text;
}

// Load the transcript on demand, called by every action so the fetch (and its
// status) only happens once the user asks. Returns usable text, or null after
// surfacing why it can't be used.
async function ensureTranscript(view: TabView): Promise<string | null> {
  if (!view.transcript || view.transcript.status !== 'ok') {
    setViewStatus(view, 'Loading transcript…');
    await loadTranscriptForTab(view);
  }
  return transcriptText(view);
}

// ---------- Mode switching ----------
// Slide the accent underline to sit beneath the active tab (8px inset each side,
// matching the design). Runs on switch, load, resize, and once fonts settle.
function moveInk(tab: HTMLElement): void {
  const ink = $('tab-ink');
  const tabsNav = $('tabs');
  const r = tab.getBoundingClientRect();
  const pr = tabsNav.getBoundingClientRect();
  ink.style.width = `${r.width - 16}px`;
  ink.style.left = `${r.left - pr.left + 8}px`;
}

// Reflect a view's selected mode onto the shared tab nav (the panels themselves
// are toggled by tabs.setMode / tabs.activate).
function reflectMode(view: TabView): void {
  document.querySelectorAll<HTMLElement>('.tab').forEach((t) => {
    const on = t.dataset.mode === view.mode;
    t.classList.toggle('active', on);
    if (on) moveInk(t);
  });
}

function selectMode(view: TabView, mode: Mode): void {
  tabs.setMode(view, mode);
  reflectMode(view);
}

function placeInk(): void {
  const active = document.querySelector<HTMLElement>('.tab.active');
  if (active) moveInk(active);
}

// ---------- Shared state handed to the mode controllers ----------
// Ask the tab's content script to seek its <video>. Fire-and-forget.
function seekVideo(view: TabView, seconds: number): void {
  void sendToTab(view.tabId, { type: 'SEEK', payload: { seconds } });
}

const ctx: PanelContext = {
  ensureTranscript,
  ensureModelReady,
  summarizerAvailable: () => summarizerReady,
  seekVideo,
};

// ---------- Theme ----------
// The design ships light + dark; follow the OS rather than adding a toggle.
function applyTheme(dark: boolean): void {
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
}

// ---------- Composer (chat) ----------
const field = (view: TabView) =>
  view.panels.query.querySelector<HTMLTextAreaElement>('#query-input')!;
const sendBtn = (view: TabView) =>
  view.panels.query.querySelector<HTMLButtonElement>('#query-send')!;

function autoGrow(el: HTMLTextAreaElement): void {
  el.style.height = 'auto';
  el.style.height = `${Math.min(el.scrollHeight, 96)}px`;
}

function resetComposer(view: TabView): void {
  const el = field(view);
  el.value = '';
  el.style.height = 'auto';
  sendBtn(view).disabled = true;
}

function submitQuery(view: TabView): void {
  const q = field(view).value.trim();
  if (!q) return;
  resetComposer(view);
  void runQuery(view, ctx, q);
}

// Bind the panel to whatever tab is active in its window. No-op if already showing
// that tab; otherwise mount the new tab's view (creating it if needed). State is
// never wiped — each tab keeps its own work running in the background.
async function syncToActiveTab(): Promise<void> {
  const tab = await getActiveTab();
  const tabId = tab?.id ?? null;
  if (tabId === currentTabId) return;
  currentTabId = tabId;
  if (tabId == null) return;

  const view = tabs.activate(tabId);
  reflectMode(view);

  // For tabs we haven't loaded a transcript into yet, set an inviting status.
  if (view.transcript == null) {
    view.watching.statusText = !isWatchUrl(tab?.url)
      ? isYouTubeHost(tab?.url)
        ? 'Open a video to begin'
        : 'Not a YouTube page — open a video to begin'
      : 'Pick Summary, Chat, or Quiz to begin';
    view.watching.statusCls = '';
  }
  renderWatching(view);
  clearBannerIfIdle();
}

// ---------- Wiring ----------
function wire(): void {
  document
    .querySelectorAll<HTMLElement>('.tab')
    .forEach((tab) =>
      tab.addEventListener('click', () => {
        const view = tabs.activeView();
        if (view) selectMode(view, tab.dataset.mode as Mode);
      }),
    );

  // Position the tab underline once layout/fonts settle (it's 0-width otherwise).
  requestAnimationFrame(() => requestAnimationFrame(placeInk));
  window.addEventListener('load', placeInk);
  window.addEventListener('resize', placeInk);
  if (document.fonts?.ready) void document.fonts.ready.then(placeInk);

  // Buttons + composer live inside per-tab panel clones, so wire via event
  // delegation on the static #content mount point and dispatch to the active view.
  const content = $('content');
  content.addEventListener('click', (e) => {
    const view = tabs.activeView();
    if (!view) return;
    const t = e.target as HTMLElement;
    if (t.closest('#run-summary')) void runSummarize(view, ctx);
    else if (t.closest('#regen-summary')) void runSummarize(view, ctx, { force: true });
    else if (t.closest('#copy-summary')) void copySummary(view);
    else if (t.closest('#run-quiz')) void runQuizMode(view, ctx);
    else if (t.closest('#query-clear')) {
      void clearQuery(view);
      resetComposer(view);
    }
  });
  content.addEventListener('submit', (e) => {
    if (!(e.target as HTMLElement).closest('#query-form')) return;
    e.preventDefault();
    const view = tabs.activeView();
    if (view) submitQuery(view);
  });
  content.addEventListener('input', (e) => {
    const ta = (e.target as HTMLElement).closest<HTMLTextAreaElement>('#query-input');
    if (!ta) return;
    const send = ta.closest('.panel')!.querySelector<HTMLButtonElement>('#query-send')!;
    send.disabled = ta.value.trim() === '';
    autoGrow(ta);
  });
  content.addEventListener('keydown', (e) => {
    const ke = e as KeyboardEvent;
    if (ke.key !== 'Enter' || ke.shiftKey) return;
    if (!(e.target as HTMLElement).closest('#query-input')) return;
    e.preventDefault();
    const view = tabs.activeView();
    if (view) submitQuery(view);
  });

  // Transcript pushes from SPA navigation. Reset the specific tab that navigated
  // — it may be a background tab — not just the active one.
  chrome.runtime.onMessage.addListener(
    (msg: unknown, sender: chrome.runtime.MessageSender) => {
      if (!isMessage(msg) || msg.type !== 'TRANSCRIPT_UPDATED') return;
      if (sender.id !== chrome.runtime.id) return;
      const tabId = sender.tab?.id;
      if (tabId == null) return;
      const view = tabs.get(tabId);
      if (!view) return; // never bound; loads lazily when the tab is activated
      // New video: drop the old state and return to neutral. The transcript
      // reloads lazily on the next action rather than auto-loading here.
      resetViewModes(view);
      tabs.reset(view);
      setViewStatus(view, 'Pick Summary, Chat, or Quiz to begin');
      if (tabs.isActive(view)) renderWatching(view);
    },
  );

  chrome.tabs.onActivated.addListener((info) => {
    if (info.windowId === panelWindowId) void syncToActiveTab();
  });

  // When a tab navigates away from a watch page (e.g. back to the home feed) the
  // content script stops pushing, so reset that tab's view here.
  chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.url == null) return;
    const view = tabs.get(tabId);
    if (!view || isWatchUrl(tab.url)) return;
    resetViewModes(view);
    tabs.reset(view);
    setViewStatus(
      view,
      isYouTubeHost(tab.url)
        ? 'Open a video to begin'
        : 'Not a YouTube page — open a video to begin',
    );
    if (tabs.isActive(view)) renderWatching(view);
  });

  // Closing a tab tears down its panel DOM and Gemini sessions.
  chrome.tabs.onRemoved.addListener((tabId) => tabs.destroy(tabId));
}

async function init(): Promise<void> {
  const themeQuery = window.matchMedia('(prefers-color-scheme: dark)');
  applyTheme(themeQuery.matches);
  themeQuery.addEventListener('change', (e) => applyTheme(e.matches));

  panelWindowId = (await chrome.windows.getCurrent()).id ?? null;
  wire();
  summarizerReady = (await summarizerAvailability()) !== 'unavailable';
  await syncToActiveTab();
}

void init();
