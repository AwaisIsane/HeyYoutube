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
      try {
        const output = await p.run({
          responseConstraint: opts.responseConstraint,
          signal: opts.signal,
        });
        // Carry the source chunk's start time so callers can link a result back
        // to where in the video it came from (quiz "jump to").
        outputs.push({ output, startMs: chunk.startMs });
      } finally {
        p.destroy();
      }
    }
    return outputs;
  }
}
