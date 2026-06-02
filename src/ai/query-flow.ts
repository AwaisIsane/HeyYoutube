// RAG query session tuned for Gemini Nano — a small on-device model that excels
// at rewriting and short-context Q&A but is weak at long-context reasoning. The
// strategy keeps what the model sees small, clean, and contiguous:
//   1. Rewrite each (possibly elliptical) question into a standalone search
//      query using the recent exchanges, so follow-ups like "what about the
//      second one?" still retrieve the right passages. (Rewriting is a Nano
//      strength.)
//   2. Retrieve top chunks, widen to neighbours, and merge into a few contiguous
//      passages — transcripts are sequential, so isolated chunks lose context.
//   3. Answer from a fresh, low-temperature session each turn (deterministic and
//      free of context bloat), carrying only a compact running history for
//      continuity. This removes the old "context window full" failure mode.

import { createSession } from './prompt';
import { availableInput } from '@/lib/tokens';
import { formatTimestamp } from '@/lib/youtube';
import type { VectorStore, Passage } from '@/retrieval/vector-store';

const ANSWER_SYSTEM =
  'You answer questions about a YouTube video using ONLY the transcript ' +
  'passages given with each question. Each passage starts with "[n]" and carries ' +
  'inline "(m:ss)" timestamps marking when each sentence is spoken. Ground every ' +
  'statement in those passages. When the user asks where or when something is ' +
  'discussed, answer with the (m:ss) timestamp that immediately precedes the ' +
  'relevant sentence. ' +
  "If they do not contain the answer, reply exactly: \"The transcript doesn't " +
  'cover that." Be concise and specific; do not pad or speculate.';

const REWRITE_SYSTEM =
  'You turn a user\'s latest message into one standalone search query for ' +
  'retrieving transcript passages. Resolve pronouns and references using the ' +
  'conversation. Reply with ONLY the query text — no quotes, no explanation.';

// Factual Q&A wants determinism, not the API's default temperature of 1.
const ANSWER_TEMPERATURE = 0.3;
const ANSWER_TOP_K = 3;

// Retrieval seeds (each widened to its neighbours and merged) and how many
// recent exchanges to carry for follow-up continuity.
const SEED_K = 4;
const NEIGHBOR_RADIUS = 1;
const HISTORY_TURNS = 2;
const HISTORY_ANSWER_CHARS = 280;

interface Exchange {
  question: string;
  answer: string;
}

export interface AskResult {
  answer: string;
  /** Start times (ms, ascending) of the passages the answer drew on, for
   *  "jump to" links. Empty when the transcript had no timing. */
  sources: number[];
}

export interface AskOptions {
  signal?: AbortSignal;
  /** Called with the cumulative answer text as it streams in. */
  onChunk?: (text: string) => void;
}

export class QuerySession {
  private history: Exchange[] = [];

  private constructor(
    private answerBase: LanguageModel,
    private rewriteBase: LanguageModel,
    private store: VectorStore,
  ) {}

  static async create(store: VectorStore): Promise<QuerySession> {
    const answerBase = await createSession({
      systemPrompt: ANSWER_SYSTEM,
      temperature: ANSWER_TEMPERATURE,
      topK: ANSWER_TOP_K,
    });
    const rewriteBase = await createSession({
      systemPrompt: REWRITE_SYSTEM,
      temperature: ANSWER_TEMPERATURE,
      topK: ANSWER_TOP_K,
    });
    return new QuerySession(answerBase, rewriteBase, store);
  }

  /** A compact transcript of the last few exchanges for follow-up continuity. */
  private recentHistory(): string {
    return this.history
      .slice(-HISTORY_TURNS)
      .map(
        (e) =>
          `Q: ${e.question}\nA: ${e.answer.slice(0, HISTORY_ANSWER_CHARS)}`,
      )
      .join('\n\n');
  }

  /**
   * Turn the raw question into a standalone retrieval query. Skipped on the
   * first turn (nothing to resolve) and falls back to the raw question if the
   * rewrite is empty or implausibly long.
   */
  private async rewriteQuery(question: string, signal?: AbortSignal): Promise<string> {
    if (this.history.length === 0) return question;
    const clone = await this.rewriteBase.clone();
    try {
      const prompt =
        `CONVERSATION:\n${this.recentHistory()}\n\n` +
        `LATEST MESSAGE: ${question}\n\nSearch query:`;
      const out = (await clone.prompt(prompt, { signal })).trim();
      return out && out.length <= question.length + 120 ? out : question;
    } catch {
      return question; // rewriting is best-effort; never block the answer on it
    } finally {
      clone.destroy();
    }
  }

  /**
   * Render a passage as a numbered block with inline (m:ss) markers at each
   * sentence boundary, so the model can cite the exact moment a point is made.
   * Falls back to a single opening timestamp when fine-grained timing is absent.
   */
  private static label(p: Passage, i: number): string {
    if (!p.times || p.times.length === 0) {
      const ts = p.startMs != null ? ` (${formatTimestamp(p.startMs)})` : '';
      return `[${i + 1}]${ts} ${p.text}`;
    }
    const marks = [...p.times].sort((a, b) => a.offset - b.offset);
    let out = `[${i + 1}]`;
    let cursor = 0;
    for (const t of marks) {
      const o = Math.min(Math.max(t.offset, 0), p.text.length);
      if (o > cursor) out += p.text.slice(cursor, o);
      out += ` (${formatTimestamp(t.tStartMs)}) `;
      cursor = o;
    }
    out += p.text.slice(cursor);
    return out;
  }

  /**
   * Assemble the answer prompt, trimming passages to fit the input budget.
   * Returns the prompt plus the passages that actually made it in, so the caller
   * can report which timestamps the answer is grounded in.
   */
  private async buildPrompt(
    session: LanguageModel,
    question: string,
    passages: Passage[],
  ): Promise<{ prompt: string; kept: Passage[] }> {
    const head = this.history.length
      ? `CONVERSATION SO FAR:\n${this.recentHistory()}\n\n`
      : '';
    const tail = `\n\nQUESTION: ${question}\nANSWER:`;
    const budget = availableInput(session);

    const kept: Passage[] = [];
    for (const p of passages) {
      const body = [...kept, p].map((t, i) => QuerySession.label(t, i)).join('\n\n');
      const prompt = `${head}TRANSCRIPT PASSAGES:\n${body}${tail}`;
      const promptEstimateContextUsage =await session.measureInputUsage(prompt);
      if (promptEstimateContextUsage > budget && kept.length > 0) break;
      kept.push(p);
    }

    const body = kept.map((t, i) => QuerySession.label(t, i)).join('\n\n');
    return { prompt: `${head}TRANSCRIPT PASSAGES:\n${body}${tail}`, kept };
  }

  async ask(question: string, opts: AskOptions = {}): Promise<AskResult> {
    const query = await this.rewriteQuery(question, opts.signal);
    const passages = await this.store.searchMerged(query, SEED_K, NEIGHBOR_RADIUS);

    // A fresh clone per turn: starts from just the system prompt, so context
    // never accumulates across questions.
    const session = await this.answerBase.clone();
    try {
      const { prompt, kept } = await this.buildPrompt(session, question, passages);

      let answer = '';
      if (opts.onChunk) {
        const reader = session
          .promptStreaming(prompt, { signal: opts.signal })
          .getReader();
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          answer += value;
          opts.onChunk(answer);
        }
      } else {
        answer = await session.prompt(prompt, { signal: opts.signal });
      }
      answer = answer.trim();

      this.history.push({ question, answer });
      const sources = [
        ...new Set(
          kept
            .map((p) => p.startMs)
            .filter((ms): ms is number => ms != null),
        ),
      ].sort((a, b) => a - b);
      return { answer, sources };
    } finally {
      session.destroy();
    }
  }

  /** Clear the running history (the vector index is kept). */
  async reset(): Promise<void> {
    this.history = [];
  }

  destroy(): void {
    this.answerBase.destroy();
    this.rewriteBase.destroy();
  }
}
