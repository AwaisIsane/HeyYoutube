// Wrapper around Chrome's Prompt API (LanguageModel / Gemini Nano).

import { downloadMonitor } from './download';
import { availableInput, OUTPUT_RESERVE } from '@/lib/tokens';

export interface CreateOpts {
  systemPrompt?: string;
  /** Called with download progress (0..1) while the model downloads. */
  onDownload?: (progress: number) => void;
  signal?: AbortSignal;
  /** Sampling temperature. Lower = more deterministic (good for factual Q&A). */
  temperature?: number;
  /** Nucleus size; the API default is 3, max 8. */
  topK?: number;
}

export async function languageModelAvailability(): Promise<Availability> {
  if (typeof LanguageModel === 'undefined') return 'unavailable';
  return (await LanguageModel.availability());
}

function buildCreateOptions(opts: CreateOpts): LanguageModelCreateOptions {
  const options: LanguageModelCreateOptions = {};
  if (opts.systemPrompt) {
    options.initialPrompts = [{ role: 'system', content: opts.systemPrompt }];
  }
  if (opts.signal) options.signal = opts.signal;
  // temperature and topK must be set together or the API rejects them.
  options.temperature = opts.temperature ?? 1;
  options.topK = opts.topK ?? 3;
  const monitor = downloadMonitor(opts.onDownload);
  if (monitor) options.monitor = monitor;
  return options;
}

// TODOR refactor instead  this use new Prompter 
export async function createSession(opts: CreateOpts = {}): Promise<LanguageModel> {
  return LanguageModel.create(buildCreateOptions(opts));
}

export interface RunOpts {
  responseConstraint?: object;
  signal?: AbortSignal;
}

/**
 * Owns one Nano session plus its config. The system prompt, temperature and topK
 * are fixed at session-create time by the Prompt API, so changing them marks the
 * session dirty and it is lazily recreated on next use; the query is plain state
 * passed at prompt time.
 *
 * `run()` is stateless: it clones the base session (resetting the conversation to
 * just the system prompt), runs one prompt, then destroys the clone — so each
 * call starts from a clean, predictable context. An over-budget query is
 * truncated from the end to fit the model's input limit rather than throwing.
 */
export class Prompter {
  private base?: LanguageModel;
  private dirty = true;
  private _systemPrompt?: string;
  private _query = '';
  private _temperature: number;
  private _topK: number;
  private onDownload?: (progress: number) => void;
  private signal?: AbortSignal;

  constructor(opts: CreateOpts = {}) {
    this._systemPrompt = opts.systemPrompt;
    this._temperature = opts.temperature ?? 1;
    this._topK = opts.topK ?? 3;
    this.onDownload = opts.onDownload;
    this.signal = opts.signal;
  }

  get query(): string {
    return this._query;
  }
  set query(value: string) {
    this._query = value;
  }

  get systemPrompt(): string | undefined {
    return this._systemPrompt;
  }
  set systemPrompt(value: string | undefined) {
    if (value === this._systemPrompt) return;
    this._systemPrompt = value;
    this.dirty = true;
  }

  get temperature(): number {
    return this._temperature;
  }
  set temperature(value: number) {
    if (value === this._temperature) return;
    this._temperature = value;
    this.dirty = true;
  }

  get topK(): number {
    return this._topK;
  }
  set topK(value: number) {
    if (value === this._topK) return;
    this._topK = value;
    this.dirty = true;
  }

  /** Lazily (re)create the base session whenever its config has changed. */
  private async session(): Promise<LanguageModel> {
    if (this.dirty || !this.base) {
      this.base?.destroy();
      this.base = await createSession({
        systemPrompt: this._systemPrompt,
        temperature: this._temperature,
        topK: this._topK,
        onDownload: this.onDownload,
        signal: this.signal,
      });
      this.dirty = false;
    }
    return this.base;
  }

  /** Token count `text` would add as input on this session. */
  async measure(text: string): Promise<number> {
    return (await this.session()).measureInputUsage(text);
  }

  /** Tokens free for new input after the system prompt, output reserve and margin. */
  async inputBudget(): Promise<number> {
    return availableInput(await this.session());
  }

  /**
   * Total tokens the next run will consume: the system prompt (already loaded
   * into the session context), the query, and the reserved output window.
   */
  async getEstimatedUsage(query = this._query): Promise<number> {
    const session = await this.session();
    const systemUsage = (session as any).contextUsage as number ?? session.inputUsage;
    return systemUsage + (await session.measureInputUsage(query)) + OUTPUT_RESERVE;
  }

  async run(opts: RunOpts = {}): Promise<string> {
    const session = await this.session();
    const query = await this.fitQuery(session, this._query);
    const clone = await session.clone();
    try {
      return await clone.prompt(query, {
        responseConstraint: opts.responseConstraint as
          | Record<string, unknown>
          | undefined,
        signal: opts.signal ?? this.signal,
      });
    } finally {
      clone.destroy();
    }
  }

  /**
   * Trim the end of the query until the system prompt + query fits the input
   * budget. Shrinks proportionally by the measured overflow and re-measures,
   * the same approach splitToBudget uses for oversized sentences.
   */
  private async fitQuery(session: LanguageModel, query: string): Promise<string> {
    const budget = availableInput(session);
    let used = await session.measureInputUsage(query);
    let text = query;
    while (text.length > 0 && used > budget) {
      const keep = Math.max(1, Math.floor(text.length * (budget / used)) - 1);
      if (keep >= text.length) break;
      text = text.slice(0, keep);
      used = await session.measureInputUsage(text);
    }
    return text;
  }

  destroy(): void {
    this.base?.destroy();
  }
}
