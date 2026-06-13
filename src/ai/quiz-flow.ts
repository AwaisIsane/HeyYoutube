// Quiz generation: split the transcript into contiguous sections and, for each,
// pick the most *representative* part (by embedding similarity to the section
// centroid) to write a multiple-choice question from. No summarization —
// questions are generated straight from the source text, which keeps it fast and
// avoids the truncated-output failures the fact-extraction step caused.

import { BigPrompt, type ReduceChunks } from './big-prompt';
import type { Chunk } from './chunking';
import { embedLong } from '@/retrieval/embeddings';
import type { TranscriptSegment } from '@/lib/messages';

// Grounding lives in the system prompt (not buried in each user prompt) and the
// session runs at low temperature — the API default of 1 is "random" and makes
// Gemini Nano invent facts. This mirrors the fix that grounded Query/"ask" mode.
// The task instruction is folded in here too, so each chunk is prompted as plain
// text and the input budget already accounts for the wrapper.
const quizSystem = (title: string) =>
  'You write a multiple-choice quiz question from one section of a video ' +
  `transcript${title ? ` (the video is titled "${title}")` : ''}. Use ONLY ` +
  'facts stated in the given section — never outside knowledge, and never ' +
  'invent names, numbers, or claims. The question must be self-contained and ' +
  'understandable on its own — never refer to "this section", "the transcript", ' +
  'or "the passage". The question and the correct option must be directly ' +
  'supported by the section. The three wrong options must be plausible but ' +
  'clearly unsupported by the section. The explanation must cite only what the ' +
  'section says, in at most two short sentences. Write exactly one question ' +
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
  /** Video title, folded into the system prompt for self-contained questions. */
  title?: string;
  onProgress?: (p: QuizProgress) => void;
  /** Called with each question the moment it's generated, validated, and found
   *  not to duplicate an earlier one — so the UI can render questions one by one
   *  instead of waiting for the whole quiz. */
  onQuestion?: (q: QuizQuestion) => void;
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

function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i]! * b[i]!;
  return sum;
}

/**
 * The part most representative of its section: the one whose embedding lies
 * closest to the section centroid. Beats random sampling, which regularly lands
 * on filler (intros, sponsor reads, tangents) and yields throwaway questions.
 * Vectors are unit-length, so dot with the (unnormalized) centroid ranks by
 * cosine to the centroid direction.
 */
async function pickRepresentative(section: Chunk[]): Promise<Chunk> {
  if (section.length === 1) return section[0]!;
  const vectors = await Promise.all(section.map((c) => embedLong(c.text)));
  const dim = vectors[0]!.length;
  const centroid = new Float32Array(dim);
  for (const v of vectors) for (let i = 0; i < dim; i++) centroid[i]! += v[i]!;
  let best = 0;
  let bestScore = -Infinity;
  for (let i = 0; i < vectors.length; i++) {
    const s = dot(vectors[i]!, centroid);
    if (s > bestScore) {
      bestScore = s;
      best = i;
    }
  }
  return section[best]!;
}

// Two questions on the same fact embed nearly identically even when worded
// differently; above this cosine the later one is dropped.
const DUPLICATE_COSINE = 0.85;

/** A well-formed question: exactly 4 distinct options and an in-range answer. */
function isValidQuestion(q: QuizQuestion): boolean {
  return (
    Array.isArray(q.options) &&
    q.options.length === 4 &&
    q.answerIndex >= 0 &&
    q.answerIndex < 4 &&
    new Set(q.options.map((o) => o.trim().toLowerCase())).size === 4
  );
}

export async function runQuiz(
  segments: TranscriptSegment[],
  deps: QuizDeps,
): Promise<QuizQuestion[]> {
  // One question per contiguous section, drawn from the most representative
  // part within it, so the quiz covers the video end to end without summarizing.
  let bp: BigPrompt;
  const reduceChunks: ReduceChunks = async (chunks: Chunk[]) => {
    const sections = partitionInto(chunks, bp.noOfChunks);
    try {
      return await Promise.all(sections.map(pickRepresentative));
    } catch {
      // Embedding selection is an enhancement; fall back to random sampling.
      return sections.map((s) => s[Math.floor(Math.random() * s.length)]!);
    }
  };

  bp = new BigPrompt({
    systemPrompt: quizSystem(deps.title?.trim().slice(0, 120) ?? ''),
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

    // Validate, dedup, and surface each question the moment it's generated, so
    // the UI renders questions one by one. Dedup is incremental: a new question
    // is dropped only when it near-duplicates one we've already kept (and thus
    // already handed to the UI) — same result as the old end-of-run batch pass,
    // which also only compared each question against the earlier ones.
    const kept: QuizQuestion[] = [];
    const keptVectors: Float32Array[] = [];

    await bp.runOnChunks(segments, {
      responseConstraint: oneQuestionSchema,
      chunkTokens: PART_TOKENS,
      // A section that fails to generate (e.g. Nano overruns its output limit)
      // costs one question, not the whole quiz.
      skipFailedChunks: true,
      signal: deps.signal,
      onProgress: (done, total) =>
        deps.onProgress?.({ stage: 'generating', done, total }),
      onChunk: async ({ output, startMs }) => {
        let q: QuizQuestion;
        try {
          q = JSON.parse(output) as QuizQuestion;
        } catch {
          return;
        }
        if (!isValidQuestion(q)) return;
        q.startMs = startMs;

        // Best-effort embedding for dedup; if it fails, keep the question
        // rather than dropping it.
        let vec: Float32Array | undefined;
        try {
          vec = await embedLong(`${q.question} ${q.options[q.answerIndex]}`);
        } catch {
          /* dedup is best-effort; never drop a question over it */
        }
        if (vec) {
          const target = vec;
          if (keptVectors.some((v) => dot(target, v) > DUPLICATE_COSINE)) return;
          keptVectors.push(target);
        }

        kept.push(q);
        deps.onQuestion?.(q);
      },
    });

    return kept;
  } finally {
    bp.destroy();
  }
}
