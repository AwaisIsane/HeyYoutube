import { defineManifest } from '@crxjs/vite-plugin';
import pkg from './package.json';

export default defineManifest({
  manifest_version: 3,
  name: 'Yt AI companion',
  version: pkg.version,
  description: pkg.description,
  minimum_chrome_version: '138',
  icons: {
    16: 'icons/icon-16.png',
    32: 'icons/icon-32.png',
    48: 'icons/icon-48.png',
    128: 'icons/icon-128.png',
  },
  action: {
    default_title: 'Yt AI companion',
    default_icon: {
      16: 'icons/icon-16.png',
      32: 'icons/icon-32.png',
      48: 'icons/icon-48.png',
      128: 'icons/icon-128.png',
    },
  },
  background: {
    service_worker: 'src/background/service-worker.ts',
    type: 'module',
  },
  side_panel: {
    default_path: 'src/sidepanel/index.html',
  },
  permissions: ['sidePanel', 'storage', 'activeTab', 'scripting'],
  host_permissions: ['*://www.youtube.com/*'],
  content_scripts: [
    {
      matches: ['*://www.youtube.com/watch*'],
      js: ['src/content/transcript.ts'],
      run_at: 'document_idle',
    },
  ],
  // 'wasm-unsafe-eval' is required by the ONNX runtime used by Transformers.js.
  content_security_policy: {
    extension_pages:
      "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'; connect-src 'self'",
  },
});
