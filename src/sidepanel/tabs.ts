// Per-tab side-panel state. A Chrome side panel is one shared document per
// window that does NOT reload on tab switch, so without this every switch would
// clobber one video's work with another's. Each watch tab instead owns its own
// live panel DOM (kept alive while detached, so listeners + scroll survive) plus
// its Query sessions and in-flight run control. Background summaries/quizzes/chats
// keep running and write into their own (possibly off-screen) nodes, so returning
// to a tab shows the exact same view.

import type { TranscriptResult } from '@/lib/messages';
import type { VectorStore } from '@/retrieval/vector-store';
import type { QuerySession } from '@/ai/query-flow';

export type Mode = 'summarize' | 'review' | 'query';
export const MODES: Mode[] = ['summarize', 'review', 'query'];

// One in-flight action per tab+mode. Starting a new run aborts the previous one
// and bumps `seq`; stale callbacks compare against the captured seq and no-op, so
// only the newest run owns the panel. This replaces the old videoId-based live()
// guard — switching tabs no longer cancels anything; only a new run does.
interface RunControl {
  seq: number;
  ctrl: AbortController | null;
}

export interface Watching {
  title: string;
  videoId: string | null;
  durationMs: number;
  statusText: string;
  statusCls: string;
}

export interface TabView {
  tabId: number;
  transcript: TranscriptResult | null;
  mode: Mode;
  /** The three <section class="panel"> clones for this tab. Mounted into #content
   *  only while the tab is active; detached (but referenced) otherwise. */
  panels: Record<Mode, HTMLElement>;
  /** Shared now-watching strip values, re-painted on activate. */
  watching: Watching;
  /** Query mode keeps its index + Gemini session alive across switches. */
  query: { vectorStore: VectorStore | null; session: QuerySession | null };
  runs: Record<Mode, RunControl>;
}

export interface RunHandle {
  /** True while this is still the newest run for the tab+mode (else it was
   *  superseded or the tab's state was reset — callers stop writing the panel). */
  live: () => boolean;
  signal: AbortSignal;
}

const views = new Map<number, TabView>();
let activeTabId: number | null = null;

function freshWatching(): Watching {
  return { title: 'Open a YouTube video', videoId: null, durationMs: 0, statusText: '', statusCls: '' };
}

function panelTemplate(): HTMLTemplateElement {
  const tpl = document.getElementById('panel-tpl');
  if (!(tpl instanceof HTMLTemplateElement))
    throw new Error('Side panel: missing <template id="panel-tpl">');
  return tpl;
}

function buildPanels(): Record<Mode, HTMLElement> {
  const frag = panelTemplate().content.cloneNode(true) as DocumentFragment;
  const pick = (mode: Mode): HTMLElement => {
    const el = frag.querySelector<HTMLElement>(`.panel[data-panel="${mode}"]`);
    if (!el) throw new Error(`Side panel: template missing panel "${mode}"`);
    return el;
  };
  return { summarize: pick('summarize'), review: pick('review'), query: pick('query') };
}

export function getOrCreate(tabId: number): TabView {
  const existing = views.get(tabId);
  if (existing) return existing;
  const view: TabView = {
    tabId,
    transcript: null,
    mode: 'summarize',
    panels: buildPanels(),
    watching: freshWatching(),
    query: { vectorStore: null, session: null },
    runs: {
      summarize: { seq: 0, ctrl: null },
      review: { seq: 0, ctrl: null },
      query: { seq: 0, ctrl: null },
    },
  };
  views.set(tabId, view);
  return view;
}

export function get(tabId: number): TabView | undefined {
  return views.get(tabId);
}

export function isActive(view: TabView): boolean {
  return view.tabId === activeTabId;
}

export function activeView(): TabView | null {
  return activeTabId == null ? null : views.get(activeTabId) ?? null;
}

function applyMode(view: TabView): void {
  for (const m of MODES) view.panels[m].hidden = m !== view.mode;
}

// Detaching and re-attaching a node resets its scroll position, so we stash the
// scrollable regions' offsets when leaving a tab and restore them on return.
const SCROLLERS = '.scroll, #chat';

function eachScroller(view: TabView, fn: (el: HTMLElement) => void): void {
  for (const m of MODES) view.panels[m].querySelectorAll<HTMLElement>(SCROLLERS).forEach(fn);
}

// Mount this tab's panels into #content (replacing the previous tab's, which stay
// alive inside their TabView) and make it the active tab.
export function activate(tabId: number): TabView {
  const leaving = activeView();
  if (leaving && leaving.tabId !== tabId)
    eachScroller(leaving, (el) => (el.dataset.scrollTop = String(el.scrollTop)));

  const view = getOrCreate(tabId);
  activeTabId = tabId;
  const content = document.getElementById('content')!;
  content.replaceChildren(view.panels.summarize, view.panels.review, view.panels.query);
  applyMode(view);
  eachScroller(view, (el) => {
    if (el.dataset.scrollTop) el.scrollTop = Number(el.dataset.scrollTop);
  });
  return view;
}

export function setMode(view: TabView, mode: Mode): void {
  view.mode = mode;
  if (isActive(view)) applyMode(view);
}

// Begin a run for a tab+mode: abort any previous run, take a fresh AbortController,
// and hand back a live() guard tied to this run's sequence number.
export function beginRun(view: TabView, mode: Mode): RunHandle {
  const r = view.runs[mode];
  r.ctrl?.abort();
  r.ctrl = new AbortController();
  const seq = ++r.seq;
  return { live: () => r.seq === seq, signal: r.ctrl.signal };
}

// Abort every in-flight run for a tab and tear down its Query session. Called on
// new-video navigation and tab close. Panel DOM is cleared by the mode reset
// helpers; this handles the non-DOM state.
export function reset(view: TabView): void {
  for (const m of MODES) {
    view.runs[m].ctrl?.abort();
    view.runs[m].ctrl = null;
    view.runs[m].seq++;
  }
  view.query.session?.destroy();
  view.query = { vectorStore: null, session: null };
  view.transcript = null;
  view.watching = freshWatching();
}

export function destroy(tabId: number): void {
  const view = views.get(tabId);
  if (!view) return;
  reset(view);
  views.delete(tabId);
  if (activeTabId === tabId) activeTabId = null;
}
