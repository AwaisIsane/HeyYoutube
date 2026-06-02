// Stateless DOM helpers for the side panel: element lookup, the header status
// line, and the shared error/warning banner. No app or video state lives here —
// just thin wrappers over elements that always exist in index.html.

/**
 * Look up an element by id, throwing if it's missing. A typo'd id is a wiring
 * bug; failing loudly here beats returning null and crashing later somewhere
 * less obvious.
 */
export function $<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Side panel: missing element #${id}`);
  return el as T;
}

export const delay = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

// ---------- Header status line ----------
export function setStatus(text: string, cls = ''): void {
  const el = $('transcript-status');
  el.textContent = text;
  el.className = `status ${cls}`;
}

// ---------- Per-panel view state ----------
// Each feature panel shows exactly one of three states; CSS keys off the class.
// Helpers take the panel element directly, since each tab owns its own clone of
// the panels (see sidepanel/tabs.ts) — there is no single global panel per mode.
export type PanelState = 'empty' | 'loading' | 'done';

export function setPanelState(panel: HTMLElement, state: PanelState): void {
  panel.classList.remove('is-empty', 'is-loading', 'is-done');
  panel.classList.add(`is-${state}`);
}

// Drive the loading status card: title + sublabel, the counter, and the
// segmented progress track (segments before `step` are done, `step` is filling).
export function setStatusStep(
  panel: HTMLElement,
  title: string,
  sub: string,
  step: number,
  total: number,
): void {
  const status = panel.querySelector<HTMLElement>('[data-status]');
  if (!status) return;
  status.querySelector('.stxt')!.textContent = title;
  status.querySelector('.ssub')!.textContent = sub;
  status.querySelector('.status-count')!.textContent = `${Math.min(step + 1, total)}/${total}`;
  status.querySelectorAll<HTMLElement>('.status-track i').forEach((seg, n) => {
    seg.classList.toggle('done', n < step);
    seg.classList.toggle('cur', n === step);
  });
}

// ---------- Copy toast ----------
let toastTimer: ReturnType<typeof setTimeout> | undefined;
export function showToast(): void {
  const t = $('toast');
  t.classList.add('on');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('on'), 1500);
}

// ---------- Shared error/warning banner ----------
export function showBanner(text: string, kind: 'err' | 'warn'): void {
  const el = $('banner');
  el.textContent = text;
  el.className = `banner ${kind}`;
  el.hidden = false;
}

export function clearBanner(): void {
  $('banner').hidden = true;
}
