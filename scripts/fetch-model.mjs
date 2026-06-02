// Populate public/models so the extension runs embeddings fully offline:
//   1. Download the all-MiniLM-L6-v2 (q8) model files from the Hugging Face Hub.
//   2. Copy the ONNX runtime wasm assets out of node_modules.
//
// Run once before building:  pnpm fetch-model

import { mkdir, writeFile, copyFile, access, rm } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const MODELS_DIR = join(root, 'public', 'models');
const MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
const HF_BASE = `https://huggingface.co/${MODEL_ID}/resolve/main`;

const MODEL_FILES = [
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'onnx/model_quantized.onnx',
];

async function download(rel) {
  const url = `${HF_BASE}/${rel}`;
  const dest = join(MODELS_DIR, MODEL_ID, rel);
  await mkdir(dirname(dest), { recursive: true });
  process.stdout.write(`  ${rel} … `);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  const buf = Buffer.from(await res.arrayBuffer());
  await writeFile(dest, buf);
  console.log(`${(buf.length / 1024 / 1024).toFixed(1)} MB`);
}

// onnxruntime-web ships ~25 build flavors (jsep, jspi, webgl, all the ort.*.mjs
// API bundles, …) totalling >100 MB. The Transformers.js browser entry
// (transformers.web.js) only ever fetches these four at runtime — the plain wasm
// CPU build and the asyncify build it uses for the WebGPU/proxy path — so we copy
// only these and skip the rest to keep the packaged extension small.
// Verify with: grep -oE 'ort-wasm-simd-threaded\.[a-z.]*(mjs|wasm)' \
//   node_modules/@huggingface/transformers/dist/transformers.web.js
const ORT_FILES = [
  'ort-wasm-simd-threaded.mjs',
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd-threaded.asyncify.mjs',
  'ort-wasm-simd-threaded.asyncify.wasm',
];

async function copyOrtWasm() {
  const ortDest = join(MODELS_DIR, 'ort');
  // Wipe the dir first so stale flavors from older runs don't linger.
  await rm(ortDest, { recursive: true, force: true });
  await mkdir(ortDest, { recursive: true });
  // Locate the onnxruntime-web dist via Node module resolution rather than a
  // hardcoded node_modules path. onnxruntime-web is a *transitive* dep of
  // @huggingface/transformers, so under pnpm's default (non-hoisted) layout it
  // lives in .pnpm/… and never at top-level node_modules — the old hardcoded
  // path copied 0 files there, leaving the extension to 404 on the ort wasm.
  const req = createRequire(import.meta.url);
  const candidates = [
    join(dirname(req.resolve('@huggingface/transformers')), '..', 'dist'),
  ];
  try {
    // Resolve onnxruntime-web from transformers' context (it's transformers' dep).
    const tfReq = createRequire(req.resolve('@huggingface/transformers'));
    candidates.push(dirname(tfReq.resolve('onnxruntime-web')));
  } catch {
    // Fall back to the top-level path (hoisted/npm layouts).
    candidates.push(join(root, 'node_modules', 'onnxruntime-web', 'dist'));
  }
  let copied = 0;
  for (const name of ORT_FILES) {
    for (const dir of candidates) {
      const src = join(dir, name);
      try {
        await access(src);
      } catch {
        continue;
      }
      await copyFile(src, join(ortDest, name));
      copied++;
      break; // first candidate that has it wins
    }
  }
  console.log(`  copied ${copied}/${ORT_FILES.length} ONNX runtime asset(s) -> public/models/ort`);
  if (copied < ORT_FILES.length) {
    console.warn('  ⚠ some ORT assets missing — run `pnpm install` first.');
  }
}

console.log(`Fetching ${MODEL_ID} into public/models …`);
for (const f of MODEL_FILES) await download(f);
await copyOrtWasm();
console.log('Done.');
