// Wrapper around Chrome's purpose-built Summarizer API, with graceful absence
// handling. The summarize flow prefers this when available and falls back to the
// Prompt API otherwise.

import { downloadMonitor } from './download';

// Output languages the Summarizer API supports; anything else falls back to en.
const SUPPORTED_OUTPUT_LANGUAGES = new Set(['en', 'es', 'ja']);

/** Map a transcript language code (e.g. "en-US") to a supported output language. */
export function toOutputLanguage(lang?: string): string {
  const base = lang?.slice(0, 2).toLowerCase();
  return base && SUPPORTED_OUTPUT_LANGUAGES.has(base) ? base : 'en';
}

export async function summarizerAvailability(): Promise<Availability> {
  if (typeof Summarizer === 'undefined') return 'unavailable';
  // Declare the output language here too: availability() counts as a Summarizer
  // request, and Chrome warns if it's omitted.
  return (await Summarizer.availability({ outputLanguage: 'en' })) as Availability;
}

export interface SummarizerOpts {
  type?: 'key-points' | 'tldr' | 'teaser' | 'headline';
  length?: 'short' | 'medium' | 'long';
  sharedContext?: string;
  /** Output language code (supported by the API: en, es, ja). Defaults to en. */
  outputLanguage?: string;
  onDownload?: (progress: number) => void;
}

export async function createSummarizer(opts: SummarizerOpts = {}): Promise<Summarizer> {
  const options: SummarizerCreateOptions = {
    type: opts.type ?? 'key-points',
    format: 'markdown',
    length: opts.length ?? 'long',
    outputLanguage: opts.outputLanguage ?? 'en',
  };
  if (opts.sharedContext) options.sharedContext = opts.sharedContext;
  const monitor = downloadMonitor(opts.onDownload);
  if (monitor) options.monitor = monitor;
  return Summarizer.create(options);
}
