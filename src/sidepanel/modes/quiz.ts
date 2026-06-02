// Review mode: generate a multiple-choice quiz from the transcript and render an
// interactive, self-scoring card list in the design's style. Cached per video.
// All DOM lives inside the owning tab's panel clone (see sidepanel/tabs.ts), so a
// run keeps updating its tab even while another tab is on screen.

import { cacheGet, cacheSet } from '@/lib/cache';
import { runQuiz, type QuizQuestion } from '@/ai/quiz-flow';
import { formatTimestamp } from '@/lib/youtube';
import {
  showBanner,
  errorMessage,
  setPanelState,
  setStatusStep,
} from '../dom';
import type { PanelContext } from '../context';
import { beginRun, type TabView } from '../tabs';

const MODE = 'review' as const;
const panel = (view: TabView): HTMLElement => view.panels[MODE];
const q = <T extends HTMLElement>(view: TabView, sel: string): T =>
  panel(view).querySelector<T>(sel)!;

/** Clear quiz state + output (called on new video). */
export function resetQuiz(view: TabView): void {
  q(view, '#quiz-output').innerHTML = '';
  setPanelState(panel(view), 'empty');
}

function renderQuiz(
  view: TabView,
  ctx: PanelContext,
  questions: QuizQuestion[],
): void {
  const out = q<HTMLElement>(view, '#quiz-output');
  out.innerHTML = '';
  const scoreTxt = q(view, '#quiz-score-txt');
  const scoreBar = q<HTMLElement>(view, '#quiz-score-bar');
  let answered = 0;
  scoreTxt.textContent = `0 / ${questions.length} answered`;
  scoreBar.style.width = '0%';

  questions.forEach((quizQ, qi) => {
    const card = document.createElement('div');
    card.className = 'quiz-q';

    const meta = document.createElement('div');
    meta.className = 'q-meta';
    const qn = document.createElement('span');
    qn.className = 'qn';
    qn.textContent = `Question ${qi + 1}`;
    const right = document.createElement('div');
    right.className = 'q-meta-right';
    // A clickable timestamp seeks the watch-page video to where this question's
    // section is discussed; absent when the transcript had no timing.
    if (quizQ.startMs != null) {
      const ms = quizQ.startMs;
      const label = formatTimestamp(ms);
      const jump = document.createElement('button');
      jump.type = 'button';
      jump.className = 'ts-link';
      jump.textContent = label;
      jump.title = `Jump to ${label}`;
      jump.addEventListener('click', () => ctx.seekVideo(view, Math.floor(ms / 1000)));
      right.appendChild(jump);
    }
    const qt = document.createElement('span');
    qt.className = 'qt';
    qt.textContent = 'multiple choice';
    right.appendChild(qt);
    meta.append(qn, right);
    card.appendChild(meta);

    const qText = document.createElement('div');
    qText.className = 'q-text';
    qText.textContent = quizQ.question;
    card.appendChild(qText);

    const buttons: HTMLButtonElement[] = [];
    quizQ.options.forEach((opt, oi) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'quiz-opt';
      const radio = document.createElement('span');
      radio.className = 'radio';
      const lab = document.createElement('span');
      lab.className = 'lab';
      lab.textContent = opt;
      b.append(radio, lab);
      b.addEventListener('click', () => {
        buttons.forEach((x) => (x.disabled = true));
        buttons[quizQ.answerIndex]!.classList.add('correct');
        if (oi !== quizQ.answerIndex) b.classList.add('wrong');
        answered++;

        const explain = document.createElement('div');
        explain.className = 'quiz-explain';
        explain.innerHTML = '<b>Correct&nbsp;—</b> ';
        explain.append(document.createTextNode(quizQ.explanation));
        card.appendChild(explain);

        scoreTxt.textContent = `${answered} / ${questions.length} answered`;
        scoreBar.style.width = `${(answered / questions.length) * 100}%`;
      });
      buttons.push(b);
      card.appendChild(b);
    });
    out.appendChild(card);
  });

  setPanelState(panel(view), 'done');
}

export async function runQuizMode(view: TabView, ctx: PanelContext): Promise<void> {
  const text = await ctx.ensureTranscript(view);
  const transcript = view.transcript;
  if (!text || !transcript) return;

  const cached = await cacheGet<QuizQuestion[]>(transcript.videoId, 'quiz');
  if (cached) {
    renderQuiz(view, ctx, cached);
    return;
  }

  const wasDone = panel(view).classList.contains('is-done');
  const { live, signal } = beginRun(view, MODE);
  setPanelState(panel(view), 'loading');
  setStatusStep(
    panel(view),
    'Generating questions…',
    'Pulling key concepts from the transcript',
    0,
    2,
  );

  try {
    if (!(await ctx.ensureModelReady())) {
      if (live()) setPanelState(panel(view), wasDone ? 'done' : 'empty');
      return;
    }
    const segments = transcript.segments ?? [{ text, tStartMs: 0 }];
    const questions = await runQuiz(segments, {
      signal,
      onProgress: (s) => {
        if (!live()) return;
        setStatusStep(
          panel(view),
          'Generating questions…',
          `Question ${(s.done ?? 0) + 1}/${s.total}`,
          0,
          2,
        );
      },
    });
    if (questions.length === 0) {
      if (live()) {
        setPanelState(panel(view), 'empty');
        showBanner('Could not generate a quiz from this transcript.', 'warn');
      }
      return;
    }
    await cacheSet(transcript.videoId, 'quiz', questions);
    if (live()) {
      setStatusStep(panel(view), 'Building quiz…', 'Writing answers and explanations', 1, 2);
      renderQuiz(view, ctx, questions);
    }
  } catch (err) {
    if (live()) {
      setPanelState(panel(view), wasDone ? 'done' : 'empty');
      showBanner(`Quiz failed: ${errorMessage(err)}`, 'err');
    }
  }
}
