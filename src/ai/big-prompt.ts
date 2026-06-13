// A Prompter for inputs larger than Gemini Nano's context window. Given a timed
// transcript it splits it into budget-fitting chunks, hands them to a
// constructor-injected `reduceChunks` strategy (quiz samples a subset; summary
// map-reduces down to one), then runs each resulting chunk on a fresh session.

import { Prompter, type CreateOpts, type RunOpts } from './prompt';
import { timedChunks, type Chunk } from './chunking';
import type { TranscriptSegment } from '@/lib/messages';

/** Reduce the budget-fitting chunks down to the ones that should be prompted. */
export type ReduceChunks = (chunks: Chunk[]) => Promise<Chunk[]>;

export interface BigPromptOpts extends CreateOpts {
  /** Desired number of reduced output chunks (quiz: questions; summary: 1). */
  noOfChunks: number;
  reduceChunks: ReduceChunks;
}

export interface RunChunksOpts extends RunOpts {
  /** Cap the per-chunk size below the input budget, for finer sampling
   *  granularity (quiz). Defaults to the full input budget (summary). */
  chunkTokens?: number;
  /** Called as each reduced chunk is run (done index, total). */
  onProgress?: (done: number, total: number) => void;
  /** Skip a chunk whose run fails instead of aborting the whole batch — for
   *  callers where one bad chunk should cost one output, not the run (quiz).
   *  Aborts via `signal` still propagate. */
  skipFailedChunks?: boolean;
  /** Called with each chunk's raw output the moment it finishes, so callers can
   *  render incrementally instead of waiting for the whole batch. Awaited, so
   *  chunks stay strictly ordered. */
  onChunk?: (result: { output: string; startMs?: number }) => void | Promise<void>;
}

export class BigPrompt extends Prompter {
  noOfChunks: number;
  private reduceChunks: ReduceChunks;

  constructor(opts: BigPromptOpts) {
    super(opts);
    this.noOfChunks = opts.noOfChunks;
    this.reduceChunks = opts.reduceChunks;
  }

  /**
   * Split the transcript into chunks that each fit the per-chunk input budget,
   * reduce them via the injected strategy, then run every reduced chunk on its
   * own fresh session (so context never accumulates). Returns the raw outputs.
   */
  async runOnChunks(
    segments: TranscriptSegment[],
    opts: RunChunksOpts = {},
  ): Promise<{ output: string; startMs?: number }[]> {
    const budget = await this.inputBudget();
    const targetTokens = Math.min(budget, opts.chunkTokens ?? budget);
    const measure = (text: string) => this.measure(text);
    const chunks = await timedChunks(segments, { targetTokens }, measure);

    const reduced = await this.reduceChunks(chunks);

    const outputs: { output: string; startMs?: number }[] = [];
    for (const [i, chunk] of reduced.entries()) {
      opts.onProgress?.(i, reduced.length);
      const p = new Prompter({
        systemPrompt: this.systemPrompt,
        temperature: this.temperature,
        topK: this.topK,
        signal: opts.signal,
      });
      p.query = chunk.text;
      let output: string;
      try {
        output = await p.run({
          responseConstraint: opts.responseConstraint,
          signal: opts.signal,
        });
      } catch (err) {
        if (!opts.skipFailedChunks || opts.signal?.aborted) throw err;
        continue;
      } finally {
        p.destroy();
      }
      // Carry the source chunk's start time so callers can link a result back
      // to where in the video it came from (quiz "jump to").
      const result = { output, startMs: chunk.startMs };
      outputs.push(result);
      // Hand the result off only when a listener is registered. The session is
      // already destroyed above, so rendering never holds an idle session open.
      if (opts.onChunk) await opts.onChunk(result);
    }
    return outputs;
  }
}
