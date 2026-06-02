// Sentence-aware chunking with two profiles:
//  - big-prompt chunks: large, to fit the model input minus the output reserve
//    and the system prompt (the budget is computed from the live session).
//  - retrieval chunks: small (MiniLM max seq is 256 tokens) with sentence overlap.
//
// Both profiles size chunks with a real tokenizer injected by the caller
// (session.measureInputUsage for the LLM, countTokens for MiniLM) — never a
// char/length heuristic.
//
// Chunks can be built straight from the transcript's timed cues so each chunk
// remembers the timestamp where it starts (see timedChunks /
// retrievalChunksFromSegments), which powers seek-to-time and "where in the
// video" answers.

import type { TranscriptSegment } from "@/lib/messages";

export interface Chunk {
  text: string;
  index: number;
  /** Token count from the relevant tokenizer (Nano for big-prompt, MiniLM for retrieval). */
  approxTokens: number;
  /** Start time (ms) of this chunk in the video; absent for untimed text. */
  startMs?: number;
  /**
   * Per-sentence start times within `text` (char offset -> ms), at caption-cue
   * granularity. Lets retrieval cite an exact moment inside a chunk instead of
   * only the chunk's opening time. Absent for untimed text.
   */
  times?: { offset: number; tStartMs: number }[];
}

// A sentence-ending run (with optional closing quote/bracket), or a final
// unpunctuated tail. Global so both .match and .matchAll can use it.
const SENTENCE_RE = /[^.!?]+[.!?]+(?:["')\]]+)?|\S[^.!?]*$/g;

/** A sentence to be chunked, optionally carrying the time it starts at. */
interface TimedSentence {
  text: string;
  tStartMs?: number;
}

/** Split text into sentences, keeping terminal punctuation. */
function splitSentences(text: string): string[] {
  const parts = text.match(SENTENCE_RE);
  return (parts ?? [text]).map((s) => s.trim()).filter(Boolean);
}
/**
 * Split the transcript into sentences, each stamped with the start time of the
 * cue its first character falls in. We rebuild the joined text the same way the
 * content script does (cues joined by a single space) and keep a char-offset ->
 * time index alongside, so a sentence's offset maps back to a cue timestamp.
 */
function timedSentences(segments: TranscriptSegment[]): TimedSentence[] {
  const marks: { offset: number; tStartMs: number }[] = [];
  const parts: string[] = [];
  let offset = 0;
  for (const seg of segments) {
    marks.push({ offset, tStartMs: seg.tStartMs });
    parts.push(seg.text);
    offset += seg.text.length + 1; // +1 for the joining space
  }
  const text = parts.join(" ");

  // Last cue whose offset is <= the given char offset.
  const timeAt = (at: number): number => {
    let lo = 0;
    let hi = marks.length - 1;
    let ans = 0;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (marks[mid]!.offset <= at) {
        ans = marks[mid]!.tStartMs;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return ans;
  };

  const sentences: TimedSentence[] = [];
  for (const m of text.matchAll(SENTENCE_RE)) {
    const raw = m[0]!;
    const lead = raw.length - raw.trimStart().length; // skip leading space
    const trimmed = raw.trim();
    if (trimmed)
      sentences.push({
        text: trimmed,
        tStartMs: timeAt((m.index ?? 0) + lead),
      });
  }
  return sentences;
}

interface ChunkOptions {
  /** Target tokens per chunk. */
  targetTokens: number;
  /** Number of trailing sentences to repeat at the start of the next chunk. */
  overlapSentences?: number;
}

/**
 * Hard-split one sentence that exceeds targetTokens into target-sized word
 * windows, so no chunk overflows. We estimate words-per-budget from a single
 * measured probe, then slice fixed windows front-to-back. A tiny trailing
 * remainder (less than half the budget — barely a fragment) is dropped.
 */
async function splitToBudget(
  text: string,
  targetTokens: number,
  countTokens: (text: string) => Promise<number>,
): Promise<{ text: string; approxTokens: number }[]> {
  const words = text.split(/\s+/).filter(Boolean);

  // Estimate how many words fit the budget from one probe measurement.
  const probeWords = Math.min(words.length, 32);
  const probeTokens = Math.max(
    1,
    await countTokens(words.slice(0, probeWords).join(" ")),
  );
  const span = Math.max(1, Math.round((probeWords * targetTokens) / probeTokens));

  const pieces: { text: string; approxTokens: number }[] = [];
  for (let i = 0; i < words.length; i += span) {
    const slice = words.slice(i, i + span).join(" ");
    const approxTokens = await countTokens(slice);
    // Drop a too-small trailing fragment rather than keep it as its own chunk.
    if (i + span >= words.length && approxTokens < targetTokens / 2) break;
    pieces.push({ text: slice, approxTokens });
  }
  return pieces;
}

/**
 * Group sentences into ~targetTokens chunks with optional sentence overlap. Each
 * chunk inherits the start time of its first sentence (undefined when untimed).
 */
async function chunkSentences(
  sentences: TimedSentence[],
  opts: ChunkOptions,
  countTokens: (text: string) => Promise<number>,
): Promise<Chunk[]> {
  const { targetTokens, overlapSentences = 0 } = opts;
  const chunks: Chunk[] = [];

  let buf: TimedSentence[] = [];
  let bufTokens = 0;

  const flush = async () => {
    if (buf.length === 0) return;
    const body = buf.map((s) => s.text).join(" ");
    // Record each sentence's char offset within `body` so retrieval can cite the
    // exact cue, not just the chunk start. Offsets follow the same join(" ").
    const times: { offset: number; tStartMs: number }[] = [];
    let off = 0;
    for (const s of buf) {
      if (s.tStartMs != null) times.push({ offset: off, tStartMs: s.tStartMs });
      off += s.text.length + 1;
    }
    chunks.push({
      text: body,
      index: chunks.length,
      approxTokens: await countTokens(body),
      startMs: buf[0]!.tStartMs,
      times: times.length ? times : undefined,
    });
    buf =
      overlapSentences > 0
        ? buf.slice(Math.max(0, buf.length - overlapSentences))
        : [];
    let sum = 0;
    for (const s of buf) sum += await countTokens(s.text);
    bufTokens = sum;
  };

  for (const sentence of sentences) {
    const target_estimate = await countTokens(sentence.text);
    // A sentence that alone exceeds the budget is hard-split into word-based
    // pieces that each fit, so no chunk can overflow the budget.
    if (target_estimate > targetTokens) {
      await flush();
      for (const piece of await splitToBudget(
        sentence.text,
        targetTokens,
        countTokens,
      )) {
        chunks.push({
          text: piece.text,
          index: chunks.length,
          approxTokens: piece.approxTokens,
          startMs: sentence.tStartMs,
          times:
            sentence.tStartMs != null
              ? [{ offset: 0, tStartMs: sentence.tStartMs }]
              : undefined,
        });
      }
      continue;
    }
    if (bufTokens + target_estimate > targetTokens) await flush();
    buf.push(sentence);
    bufTokens += target_estimate;
  }
  await flush();

  return chunks;
}

export async function chunkText(
  text: string,
  opts: ChunkOptions,
  countTokens: (text: string) => Promise<number>
): Promise<Chunk[]> {
  return chunkSentences(
    splitSentences(text).map((s) => ({ text: s })),
    opts,
    countTokens
  );
}

/**
 * Large timed chunks for BigPrompt. Each chunk fits `targetTokens` (the per-chunk
 * input budget the caller derives from the live session) and carries the start
 * time of its first cue, so downstream prompts stay within the model's input limit.
 */
export async function timedChunks(
  segments: TranscriptSegment[],
  opts: ChunkOptions,
  countTokens: (text: string) => Promise<number>,
): Promise<Chunk[]> {
  return chunkSentences(timedSentences(segments), opts, countTokens);
}

const RETRIEVAL_OPTS: ChunkOptions = { targetTokens: 200, overlapSentences: 1 };

/**
 * Small overlapping chunks for embedding-based retrieval (no timestamps). Pass
 * `countTokens` from the embeddings module so chunks are sized with MiniLM's
 * tokenizer.
 */
export function retrievalChunks(
  text: string,
  countTokens: (text: string) => Promise<number>,
): Promise<Chunk[]> {
  return chunkText(text, RETRIEVAL_OPTS, countTokens);
}

/** Retrieval chunks built from timed cues, so each carries a start time. */
export async function retrievalChunksFromSegments(
  segments: TranscriptSegment[],
  countTokens: (text: string) => Promise<number>,
): Promise<Chunk[]> {
  return chunkSentences(timedSentences(segments), RETRIEVAL_OPTS, countTokens);
}
