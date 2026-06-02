import { defineConfig } from 'vite';
import { crx } from '@crxjs/vite-plugin';
import { fileURLToPath } from 'node:url';
import manifest from './manifest.config';

export default defineConfig({
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  plugins: [crx({ manifest })],
  // Transformers.js pulls in onnxruntime-web which ships large prebuilt wasm;
  // don't let Vite try to pre-bundle/optimize it.
  optimizeDeps: {
    exclude: ['@huggingface/transformers'],
  },
  worker: {
    format: 'es',
  },
  build: {
    target: 'esnext',
  },
  server: {
    // Bind IPv4 explicitly. Left to default, Vite binds `localhost` which on
    // some systems resolves to IPv6 `[::1]` only, while Chrome dials the
    // extension's `http://localhost:5173` over IPv4 `127.0.0.1` — so the dev
    // pages can't reach the server ("Cannot connect to http://localhost:5173").
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    // Vite 6 hardened the dev-server CORS default and no longer reflects
    // arbitrary origins, so the extension's `chrome-extension://` pages get no
    // `Access-Control-Allow-Origin` header and CRXJS's loader fetch is blocked
    // ("Cannot connect to http://localhost:5173"). Allow the extension origin.
    cors: { origin: [/^chrome-extension:\/\//] },
    hmr: {
      host: '127.0.0.1',
      port: 5173,
    },
  },
});
