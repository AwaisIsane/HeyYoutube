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

/** Build one interactive quiz card. `onAnswered` fires the first (and only)
 *  time the user picks an option, so the caller can update the running score. */
function buildQuizCard(
  view: TabView,
  ctx: PanelContext,
  quizQ: QuizQuestion,
  index: number,
  onAnswered: () => void,
): HTMLElement {
  const card = document.createElement('div');
  card.className = 'quiz-q';

  const meta = document.createElement('div');
  meta.className = 'q-meta';
  const qn = document.createElement('span');
  qn.className = 'qn';
  qn.textContent = `Question ${index + 1}`;
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
      const isCorrect = oi === quizQ.answerIndex;
      if (!isCorrect) b.classList.add('wrong');

      const explain = document.createElement('div');
      explain.className = isCorrect ? 'quiz-explain' : 'quiz-explain wrong';
      explain.innerHTML = isCorrect
        ? '<b>Correct&nbsp;—</b> '
        : '<b>Wrong&nbsp;—</b> ';
      explain.append(document.createTextNode(quizQ.explanation));
      card.appendChild(explain);

      onAnswered();
    });
    buttons.push(b);
    card.appendChild(b);
  });

  return card;
}

/**
 * Start rendering a quiz into the (cleared) output and return an `add` function
 * that appends one question at a time. The score denominator grows as questions
 * arrive, so it works whether they're fed in all at once (cache hit) or stream
 * in one by one as the model generates them.
 */
function quizRenderer(view: TabView, ctx: PanelContext): (quizQ: QuizQuestion) => void {
  const out = q<HTMLElement>(view, '#quiz-output');
  out.innerHTML = '';
  const scoreTxt = q(view, '#quiz-score-txt');
  const scoreBar = q<HTMLElement>(view, '#quiz-score-bar');
  let answered = 0;
  let total = 0;
  const refresh = () => {
    scoreTxt.textContent = `${answered} / ${total} answered`;
    scoreBar.style.width = total ? `${(answered / total) * 100}%` : '0%';
  };
  refresh();

  return (quizQ) => {
    const card = buildQuizCard(view, ctx, quizQ, total, () => {
      answered++;
      refresh();
    });
    total++;
    out.appendChild(card);
    refresh();
  };
}

function renderQuiz(view: TabView, ctx: PanelContext, questions: QuizQuestion[]): void {
  const add = quizRenderer(view, ctx);
  questions.forEach(add);
  setPanelState(panel(view), 'done');
}

/** Append (or move to the bottom) a "more questions coming" hint while the quiz
 *  is still streaming in; `clearGenHint` removes it once generation finishes. */
function showGenHint(view: TabView): void {
  const out = q<HTMLElement>(view, '#quiz-output');
  let hint = out.querySelector<HTMLElement>('.quiz-gen');
  if (!hint) {
    hint = document.createElement('div');
    hint.className = 'quiz-gen';
    const spark = document.createElement('span');
    spark.className = 'spark';
    const txt = document.createElement('span');
    txt.textContent = 'Generating more questions…';
    hint.append(spark, txt);
  }
  out.appendChild(hint); // keep it pinned below the latest card
}

function clearGenHint(view: TabView): void {
  q<HTMLElement>(view, '#quiz-output').querySelector('.quiz-gen')?.remove();
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

  // Created lazily on the first question (also when we flip from the loading
  // state to the results view); declared out here so the catch below can tell
  // whether questions are already on screen and must be kept there.
  let add: ((q: QuizQuestion) => void) | null = null;
  try {
    if (!(await ctx.ensureModelReady())) {
      if (live()) setPanelState(panel(view), wasDone ? 'done' : 'empty');
      return;
    }
    const segments = transcript.segments ?? [{ text, tStartMs: 0 }];
    // Render each question as it's generated rather than waiting for the whole
    // quiz.
    const questions = await runQuiz(segments, {
      signal,
      title: transcript.title,
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
      onQuestion: (quizQ) => {
        if (!live()) return;
        if (!add) {
          setPanelState(panel(view), 'done');
          add = quizRenderer(view, ctx);
        }
        add(quizQ);
        showGenHint(view);
      },
    });
    if (live()) clearGenHint(view);
    if (questions.length === 0) {
      if (live()) {
        setPanelState(panel(view), 'empty');
        showBanner('Could not generate a quiz from this transcript.', 'warn');
      }
      return;
    }
    await cacheSet(transcript.videoId, 'quiz', questions);
  } catch (err) {
    if (live()) {
      clearGenHint(view);
      // Keep any questions already rendered rather than wiping them.
      setPanelState(panel(view), add || wasDone ? 'done' : 'empty');
      showBanner(`Quiz failed: ${errorMessage(err)}`, 'err');
    }
  }
}
