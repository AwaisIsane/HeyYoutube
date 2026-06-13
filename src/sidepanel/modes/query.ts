// Query mode: build (or load) a per-video vector index, then answer questions
// against it via the RAG QuerySession. The index + session live on the TabView
// (see sidepanel/tabs.ts) so they survive tab switches and keep streaming into
// this tab's chat even while another tab is on screen.

import { VectorStore } from '@/retrieval/vector-store';
import { QuerySession } from '@/ai/query-flow';
import { loadIndex, saveIndex } from '@/retrieval/index-cache';
import { showBanner, errorMessage } from '../dom';
import { formatTimestamp, parseTimestampToSeconds, TIMESTAMP_RE } from '@/lib/youtube';
import type { PanelContext } from '../context';
import { beginRun, type TabView } from '../tabs';

const MODE = 'query' as const;
const chatEl = (view: TabView): HTMLElement =>
  view.panels[MODE].querySelector<HTMLElement>('#chat')!;

const GREETING =
  "I've read the full transcript of this video. Ask me anything — I'll point you " +
  'to the exact moments I\'m drawing from.';

/** Reset the chat back to just the AI intro bubble. */
function restoreIntro(view: TabView): void {
  const chat = chatEl(view);
  chat.innerHTML = '';
  const intro = document.createElement('div');
  intro.className = 'intro';
  const av = document.createElement('div');
  av.className = 'av';
  const txt = document.createElement('div');
  txt.className = 'txt';
  txt.textContent = GREETING;
  intro.append(av, txt);
  chat.appendChild(intro);
}

/** Clear the chat back to the intro (called on new video). The session itself is
 *  torn down by tabs.reset(). */
export function resetQuery(view: TabView): void {
  restoreIntro(view);
}

// Append a message row (AI spark avatar or "You" avatar) and return its bubble
// so callers can stream text or render structured content into it.
function addMsg(view: TabView, role: 'user' | 'assistant'): HTMLElement {
  const m = document.createElement('div');
  m.className = `msg ${role}`;
  const av = document.createElement('div');
  av.className = 'av';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  m.append(av, bubble);
  const chat = chatEl(view);
  chat.appendChild(m);
  chat.scrollTop = chat.scrollHeight;
  return bubble;
}

/** A clickable timestamp that seeks the watch-page video when clicked. */
function seekButton(
  label: string,
  seconds: number,
  cls: string,
  view: TabView,
  ctx: PanelContext,
): HTMLButtonElement {
  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = cls;
  btn.textContent = label;
  btn.title = `Jump to ${label}`;
  btn.addEventListener('click', () => ctx.seekVideo(view, seconds));
  return btn;
}

/**
 * Render an answer into `el`: linkify any inline m:ss timestamps the model cited
 * (so "it's discussed around 4:12" is clickable), then append a "Jump to" row of
 * the source passages' start times. Built with DOM nodes — never innerHTML — so
 * model output can't inject markup.
 */
function renderAnswer(
  el: HTMLElement,
  answer: string,
  sources: number[],
  view: TabView,
  ctx: PanelContext,
): void {
  el.textContent = '';

  let last = 0;
  for (const m of answer.matchAll(TIMESTAMP_RE)) {
    const ts = m[0]!;
    const idx = m.index ?? 0;
    if (idx > last) el.appendChild(document.createTextNode(answer.slice(last, idx)));
    el.appendChild(seekButton(ts, parseTimestampToSeconds(ts), 'ts-link', view, ctx));
    last = idx + ts.length;
  }
  if (last < answer.length) el.appendChild(document.createTextNode(answer.slice(last)));

  if (sources.length === 0) return;
  const row = document.createElement('div');
  row.className = 'sources';
  row.appendChild(document.createTextNode('Jump to: '));
  for (const ms of sources) {
    row.appendChild(
      seekButton(formatTimestamp(ms), Math.floor(ms / 1000), 'source-chip', view, ctx),
    );
  }
  el.appendChild(row);
}

async function ensureQuerySession(
  view: TabView,
  ctx: PanelContext,
): Promise<QuerySession | null> {
  if (view.query.session) return view.query.session;
  const text = await ctx.ensureTranscript(view);
  if (!text) return null;
  if (!(await ctx.ensureModelReady())) return null;
  const transcript = view.transcript;
  if (!transcript) return null;

  const thinking = addMsg(view, 'assistant');
  thinking.textContent = 'Indexing transcript…';
  try {
    const cached = await loadIndex(transcript.videoId);
    if (cached) {
      thinking.textContent = 'Loading saved index…';
      view.query.vectorStore = VectorStore.deserialize(cached.store);
    } else {
      const store = new VectorStore();
      await store.build(text, transcript.segments, (done, total) => {
        thinking.textContent = `Indexing transcript… ${done}/${total}`;
      });
      await saveIndex(transcript.videoId, text, store.serialize());
      view.query.vectorStore = store;
    }
    // Download progress is already handled by ensureModelReady above, which owns
    // the single shared download banner; the model is ready by the time we're here.
    view.query.session = await QuerySession.create(
      view.query.vectorStore,
      transcript.title,
    );
    thinking.parentElement?.remove();
    return view.query.session;
  } catch (err) {
    thinking.parentElement?.remove();
    showBanner(`Could not build the index: ${errorMessage(err)}`, 'err');
    return null;
  }
}

export async function runQuery(
  view: TabView,
  ctx: PanelContext,
  question: string,
): Promise<void> {
  addMsg(view, 'user').textContent = question;
  const session = await ensureQuerySession(view, ctx);
  if (!session) return;

  // A new question aborts any still-streaming previous one for this tab.
  const { live, signal } = beginRun(view, MODE);
  const pending = addMsg(view, 'assistant');
  pending.innerHTML = '<div class="typing"><i></i><i></i><i></i></div>';
  const chat = chatEl(view);
  try {
    const result = await session.ask(question, {
      signal,
      onChunk: (text) => {
        if (!live()) return;
        pending.textContent = text;
        chat.scrollTop = chat.scrollHeight;
      },
    });
    if (!live()) return;
    renderAnswer(pending, result.answer, result.sources, view, ctx);
    chat.scrollTop = chat.scrollHeight;
  } catch (err) {
    if (live()) pending.textContent = `Error: ${errorMessage(err)}`;
    else pending.parentElement?.remove(); // superseded by a newer question
  }
}

// Clear the conversation: wipe the chat back to the intro, drop the running
// history (so prior Q&A no longer feeds follow-up rewriting). The vector index
// is kept, since rebuilding it is the expensive part.
export async function clearQuery(view: TabView): Promise<void> {
  await view.query.session?.reset();
  restoreIntro(view);
}
