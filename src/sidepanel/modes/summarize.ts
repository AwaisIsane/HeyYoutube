// Summarize mode: run the map-reduce summary flow and render it into the design's
// summary card, caching the result per video so re-entering the tab is instant.
// All DOM lives inside the owning tab's panel clone (see sidepanel/tabs.ts), so a
// run started here keeps updating its tab even while another tab is on screen.

import { renderMarkdown } from '@/lib/markdown';
import { cacheGet, cacheSet } from '@/lib/cache';
import { runSummary } from '@/ai/summarize-flow';
import {
  showBanner,
  showToast,
  errorMessage,
  setPanelState,
  setStatusStep,
} from '../dom';
import type { PanelContext } from '../context';
import { beginRun, type TabView } from '../tabs';

const MODE = 'summarize' as const;
const panel = (view: TabView): HTMLElement => view.panels[MODE];
const q = <T extends HTMLElement>(view: TabView, sel: string): T =>
  panel(view).querySelector<T>(sel)!;

/** Clear summary state + output (called on new video). */
export function resetSummarize(view: TabView): void {
  q(view, '#summary-output').innerHTML = '';
  q(view, '#summary-chiplet').textContent = '';
  setPanelState(panel(view), 'empty');
}

// Render markdown into the card: the model returns a one-line overview followed
// by key points, so the first paragraph becomes the TL;DR box and bullets the
// chiplet count.
function renderSummary(view: TabView, md: string): void {
  const out = q<HTMLElement>(view, '#summary-output');
  out.innerHTML = renderMarkdown(md);
  const first = out.querySelector('p');
  if (first) {
    first.classList.add('tldr');
    first.innerHTML = `<strong>TL;DR&nbsp;—</strong> ${first.innerHTML}`;
  }
  const points = out.querySelectorAll('li').length;
  q(view, '#summary-chiplet').textContent = points ? `${points} key points` : '';
  setPanelState(panel(view), 'done');
}

export async function runSummarize(
  view: TabView,
  ctx: PanelContext,
  opts: { force?: boolean } = {},
): Promise<void> {
  const text = await ctx.ensureTranscript(view);
  const transcript = view.transcript;
  if (!text || !transcript) return;

  const cached = await cacheGet<string>(transcript.videoId, 'summary');
  if (cached && !opts.force) {
    renderSummary(view, cached);
    return;
  }

  // Preserve an existing summary if a regenerate fails (we show 'loading' while
  // it runs but never cleared #summary-output, so 'done' restores the old one).
  const wasDone = panel(view).classList.contains('is-done');
  const { live, signal } = beginRun(view, MODE);
  setPanelState(panel(view), 'loading');
  setStatusStep(panel(view), 'Loading model…', 'Warming up the summarizer', 0, 3);

  try {
    if (!(await ctx.ensureModelReady())) {
      if (live()) setPanelState(panel(view), wasDone ? 'done' : 'empty');
      return;
    }
    if (live())
      setStatusStep(panel(view), 'Reading transcript…', 'Parsing the dialogue', 1, 3);

    const segments = transcript.segments ?? [{ text, tStartMs: 0 }];
    const summary = await runSummary(segments, {
      signal,
      useSummarizer: ctx.summarizerAvailable(),
      lang: transcript.lang,
      onProgress: (s) => {
        if (!live()) return;
        if (s.stage === 'finalizing') {
          setStatusStep(panel(view), 'Summarizing…', 'Distilling the key points', 2, 3);
        } else {
          const sub =
            s.stage === 'mapping'
              ? `Section ${(s.done ?? 0) + 1}/${s.total}`
              : 'Condensing notes';
          setStatusStep(panel(view), 'Reading transcript…', sub, 1, 3);
        }
      },
    });
    await cacheSet(transcript.videoId, 'summary', summary);
    if (live()) renderSummary(view, summary);
  } catch (err) {
    if (live()) {
      setPanelState(panel(view), wasDone ? 'done' : 'empty');
      showBanner(`Summary failed: ${errorMessage(err)}`, 'err');
    }
  }
}

/** Copy the rendered summary to the clipboard and flash the toast. */
export async function copySummary(view: TabView): Promise<void> {
  const text = q<HTMLElement>(view, '#summary-output').innerText.trim();
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    showToast();
  } catch {
    showBanner('Could not copy to clipboard.', 'warn');
  }
}
