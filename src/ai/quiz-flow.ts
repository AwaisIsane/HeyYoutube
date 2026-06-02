// Quiz generation: split the transcript into contiguous sections and, for each,
// draw one random part to write a multiple-choice question from. No summarization —
// questions are generated straight from the source text, which keeps it fast and
// avoids the truncated-output failures the fact-extraction step caused.

import { BigPrompt, type ReduceChunks } from './big-prompt';
import type { Chunk } from './chunking';
import type { TranscriptSegment } from '@/lib/messages';

// Grounding lives in the system prompt (not buried in each user prompt) and the
// session runs at low temperature — the API default of 1 is "random" and makes
// Gemini Nano invent facts. This mirrors the fix that grounded Query/"ask" mode.
// The task instruction is folded in here too, so each chunk is prompted as plain
// text and the input budget already accounts for the wrapper.
const QUIZ_SYSTEM =
  'You write a multiple-choice quiz question from one section of a video ' +
  'transcript. Use ONLY facts stated in the given section — never outside ' +
  'knowledge, and never invent names, numbers, or claims. The question and the ' +
  'correct option must be directly supported by the section. The three wrong ' +
  'options must be plausible but clearly unsupported by the section. The ' +
  'explanation must cite only what the section says. Write exactly one question ' +
  'with exactly 4 options testing a key fact from the section.';

// Factual generation wants determinism, not the API's default temperature of 1.
const QUIZ_TEMPERATURE = 0.3;
const QUIZ_TOP_K = 3;

export interface QuizQuestion {
  question: string;
  options: string[];
  answerIndex: number;
  explanation: string;
  /** Start time (ms) of the transcript section this question was drawn from, for
   *  a "jump to" link. Absent when the transcript had no timing. */
  startMs?: number;
}

export interface QuizProgress {
  stage: 'generating';
  done?: number;
  total?: number;
}

export interface QuizDeps {
  onProgress?: (p: QuizProgress) => void;
  signal?: AbortSignal;
}

const oneQuestionSchema = {
  type: 'object',
  properties: {
    question: { type: 'string' },
    options: {
      type: 'array',
      items: { type: 'string' },
      minItems: 4,
      maxItems: 4,
    },
    answerIndex: { type: 'integer', minimum: 0, maximum: 3 },
    explanation: { type: 'string' },
  },
  required: ['question', 'options', 'answerIndex', 'explanation'],
} as const;

// Question count scales with how much material the video has: roughly one
// question per TOKENS_PER_QUESTION of transcript, clamped to [MIN, MAX]. Tiny
// videos get a handful; a typical video lands around the ideal of ~10.
const MIN_QUESTIONS = 1;
const MAX_QUESTIONS = 15;
const TOKENS_PER_QUESTION = 400;

// Chunk the transcript into small parts (not full budget-sized chunks) so there
// are enough distinct parts to carve into sections and sample one question each.
const PART_TOKENS = 500;

function targetQuestionCount(totalTokens: number): number {
  const n = Math.round(totalTokens / TOKENS_PER_QUESTION);
  return Math.max(MIN_QUESTIONS, Math.min(MAX_QUESTIONS, n));
}

/**
 * Split `items` into `n` contiguous, roughly equal groups in order. Used to carve
 * the transcript into exactly the number of sections we want questions for, so
 * every part of the video lands in some section (none are dropped).
 */
function partitionInto<T>(items: T[], n: number): T[][] {
  const groups: T[][] = [];
  const size = items.length / n;
  for (let i = 0; i < n; i++) {
    groups.push(items.slice(Math.round(i * size), Math.round((i + 1) * size)));
  }
  return groups.filter((g) => g.length > 0);
}

export async function runQuiz(
  segments: TranscriptSegment[],
  deps: QuizDeps,
): Promise<QuizQuestion[]> {
  // One question per contiguous section, drawn from a random part within it, so
  // the quiz covers the video end to end without summarizing.
  let bp: BigPrompt;
  const reduceChunks: ReduceChunks = async (chunks: Chunk[]) => {
    const sections = partitionInto(chunks, bp.noOfChunks);
    return sections.map((s) => s[Math.floor(Math.random() * s.length)]!);
  };

  bp = new BigPrompt({
    systemPrompt: QUIZ_SYSTEM,
    temperature: QUIZ_TEMPERATURE,
    topK: QUIZ_TOP_K,
    signal: deps.signal,
    noOfChunks: MIN_QUESTIONS,
    reduceChunks,
  });

  try {
    // Aim for a question count proportional to the content.
    const totalTokens = await bp.measure(segments.map((s) => s.text).join(' '));
    bp.noOfChunks = targetQuestionCount(totalTokens);

    const raw = await bp.runOnChunks(segments, {
      responseConstraint: oneQuestionSchema,
      chunkTokens: PART_TOKENS,
      signal: deps.signal,
      onProgress: (done, total) =>
        deps.onProgress?.({ stage: 'generating', done, total }),
    });

    const questions: QuizQuestion[] = [];
    for (const { output, startMs } of raw) {
      let q: QuizQuestion;
      try {
        q = JSON.parse(output) as QuizQuestion;
      } catch {
        continue;
      }
      if (
        Array.isArray(q.options) &&
        q.options.length === 4 &&
        q.answerIndex >= 0 &&
        q.answerIndex < 4
      ) {
        questions.push({ ...q, startMs });
      }
    }
    return questions;
  } finally {
    bp.destroy();
  }
}
