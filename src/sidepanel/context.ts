// The slice of side-panel state each feature mode needs that is NOT tab-specific.
// Per-tab state (transcript, panel DOM, sessions, run control) lives on the
// TabView passed to each mode entry point; this interface covers the shared,
// cross-tab concerns main.ts implements. Keeping it an interface (rather than
// importing main.ts) avoids a circular dependency.

import type { TabView } from './tabs';

export interface PanelContext {
  /** Load the given tab's transcript on demand; returns usable text, or null
   *  after surfacing why it can't be used. */
  ensureTranscript(view: TabView): Promise<string | null>;
  /** Whether the on-device model can be created — modes check this before
   *  building their own session. The download is one shared global process. */
  ensureModelReady(): Promise<boolean>;
  /** Whether the purpose-built Summarizer API is usable (Summarize prefers it). */
  summarizerAvailable(): boolean;
  /** Seek the tab's video to a time (seconds). No-op off a watch tab. */
  seekVideo(view: TabView, seconds: number): void;
}
