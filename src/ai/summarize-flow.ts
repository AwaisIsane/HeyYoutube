// Hierarchical map-reduce summarization tuned to maximize topic coverage
// (there are no follow-ups in summarize mode, so spend the budget on breadth).
//
// The transcript is split into budget-fitting chunks by BigPrompt; reduceChunks
// maps each chunk into key-point notes, collapses them until they fit one pass,
// and hands back a single combined chunk that BigPrompt synthesizes into the
// final markdown summary.

import { Prompter } from './prompt';
import { BigPrompt, type ReduceChunks } from './big-prompt';
import { chunkText, type Chunk } from './chunking';
import { createSummarizer, toOutputLanguage } from './summarizer';
import type { TranscriptSegment } from '@/lib/messages';

export interface SummaryProgress {
  stage: 'mapping' | 'reducing' | 'finalizing';
  done?: number;
  total?: number;
}

export interface SummaryDeps {
  /** Use the Summarizer API for per-unit summaries when true. */
  useSummarizer: boolean;
  /** Transcript language code, used to pick the summary output language. */
  lang?: string;
  /** Video title, given as context so notes resolve "the speaker"/topic. */
  title?: string;
  onProgress?: (p: SummaryProgress) => void;
  signal?: AbortSignal;
}

const mapSystem = (title: string) =>
  'Summarize the key points of the given section of a video transcript' +
  `${title ? ` (the video is titled "${title}")` : ''} as concise markdown ` +
  'bullet points. Capture every distinct fact, claim, name, and topic. Do not ' +
  'add commentary.';

const finalSystem = (title: string) =>
  'You are given bullet-point notes extracted from the transcript of a video' +
  `${title ? ` titled "${title}"` : ''}. Synthesize them into a clean, ` +
  'well-organized summary in markdown: a one-line overview followed by grouped ' +
  'key points. Preserve all distinct information; do not invent anything.';

/**
 * Repeatedly chunk `text` and re-summarize each chunk until the whole thing fits
 * `budget` tokens. Stops early if the text can no longer be split (a single
 * chunk) or stops shrinking between rounds, so a verbose model that fails to
 * compress can't spin forever — it just hands back the smallest form it reached.
 */
async function reduceToFit(
  text: string,
  budget: number,
  summarize: (chunk: string) => Promise<string>,
  onRound: () => void,
  measure: (text: string) => Promise<number>,
): Promise<string> {
  let combined = text;
  let usage = await measure(combined);

  while (usage > budget) {
    onRound();
    const chunks = await chunkText(combined, { targetTokens: budget }, measure);
    if (chunks.length <= 1) break; // can't split further; hand it off as-is

    const reduced: string[] = [];
    for (const c of chunks) reduced.push(await summarize(c.text));
    const next = reduced.join('\n\n');

    const nextUsage = await measure(next);
    if (nextUsage >= usage) break; // not actually compressing; stop
    combined = next;
    usage = nextUsage;
  }

  return combined;
}

export async function runSummary(
  segments: TranscriptSegment[],
  deps: SummaryDeps,
): Promise<string> {
  const title = deps.title?.trim().slice(0, 120) ?? '';

  // Map step: a key-points session (Summarizer API when available, else a
  // Prompter). The Summarizer is created once and reused across every chunk and
  // reduce round — per-chunk create/destroy was the map stage's main overhead.
  const mapPrompter = new Prompter({ systemPrompt: mapSystem(title), signal: deps.signal });
  let summarizer: Summarizer | undefined;
  const summarizeUnit = async (text: string): Promise<string> => {
    if (deps.useSummarizer) {
      summarizer ??= await createSummarizer({
        type: 'key-points',
        length: 'long',
        outputLanguage: toOutputLanguage(deps.lang),
        sharedContext: title
          ? `Sections of the transcript of the video "${title}".`
          : undefined,
      });
      return summarizer.summarize(text);
    }
    mapPrompter.query = text;
    return mapPrompter.run({ signal: deps.signal });
  };

  let bp: BigPrompt;
  const reduceChunks: ReduceChunks = async (chunks: Chunk[]) => {
    const budget = await bp.inputBudget();
    const measure = (t: string) => bp.measure(t);

    // Map: summarize each section into key points.
    const partials: string[] = [];
    for (let i = 0; i < chunks.length; i++) {
      deps.onProgress?.({ stage: 'mapping', done: i, total: chunks.length });
      partials.push(await summarizeUnit(chunks[i]!.text));
    }

    // Reduce: collapse partials until they fit a single synthesis pass.
    const combined = await reduceToFit(
      partials.join('\n\n'),
      budget,
      summarizeUnit,
      () => deps.onProgress?.({ stage: 'reducing' }),
      measure,
    );
    return [{ text: combined, index: 0, approxTokens: await measure(combined) }];
  };

  bp = new BigPrompt({
    systemPrompt: finalSystem(title),
    signal: deps.signal,
    noOfChunks: 1,
    reduceChunks,
  });

  try {
    const [first] = await bp.runOnChunks(segments, {
      signal: deps.signal,
      onProgress: () => deps.onProgress?.({ stage: 'finalizing' }),
    });
    return first?.output ?? '';
  } finally {
    bp.destroy();
    mapPrompter.destroy();
    summarizer?.destroy();
  }
}
