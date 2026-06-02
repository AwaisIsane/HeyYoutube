// Background service worker: gate the side panel so it is available only on
// YouTube watch pages, and enable it per-tab so each video tab gets its own
// panel instance. Transcript extraction and all AI work happen in the content
// script and side panel respectively (the worker has no DOM/WebGPU and gets
// killed mid-task).

import { isWatchUrl } from '@/lib/youtube';

const PANEL_PATH = 'src/sidepanel/index.html';

// Enable the panel on watch tabs, disable it everywhere else. Disabling per-tab
// means clicking the toolbar icon on a non-YouTube tab does nothing. Note the
// panel document itself is shared across tabs in a window (same path = one
// instance, not reloaded on tab switch); the side panel re-syncs to the active
// tab on its own (see sidepanel/main.ts syncToActiveTab).
async function syncPanelForTab(tabId: number, url: string | undefined): Promise<void> {
  try {
    if (isWatchUrl(url)) {
      await chrome.sidePanel.setOptions({ tabId, path: PANEL_PATH, enabled: true });
    } else {
      await chrome.sidePanel.setOptions({ tabId, enabled: false });
    }
  } catch (err) {
    // Tab may have closed between the event and this call; ignore.
    console.debug('setOptions failed', err);
  }
}

chrome.runtime.onInstalled.addListener(() => {
  // Open the panel on icon click (only fires for tabs where it's enabled).
  chrome.sidePanel
    .setPanelBehavior({ openPanelOnActionClick: true })
    .catch((err) => console.error('setPanelBehavior failed', err));
});

// Re-evaluate when a tab navigates (including SPA URL changes that flip status).
chrome.tabs.onUpdated.addListener((tabId, info, tab) => {
  if (info.status === 'loading' || info.url) void syncPanelForTab(tabId, tab.url);
});

// And when the user switches tabs, so the icon state matches the active tab.
chrome.tabs.onActivated.addListener(({ tabId }) => {
  chrome.tabs.get(tabId, (tab) => {
    if (!chrome.runtime.lastError) void syncPanelForTab(tabId, tab.url);
  });
});
