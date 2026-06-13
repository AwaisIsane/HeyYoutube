// Embedding pipeline backed by Transformers.js (all-MiniLM-L6-v2, 384-dim).
//
// Model weights and the ONNX runtime wasm are bundled into the extension under
// public/models so embeddings run fully offline and pass the extension CSP.
// Run `pnpm fetch-model` to populate public/models before building.
//
// IMPORTANT: only call this from a full page context (the side panel). MV3
// service workers lack WebGPU and can be killed mid-inference.

import {
  pipeline,
  env,
  type FeatureExtractionPipeline,
} from '@huggingface/transformers';
import { retrievalChunks } from '@/ai/chunking';

const MODEL_ID = 'Xenova/all-MiniLM-L6-v2';

// Point Transformers.js at the bundled assets instead of any CDN.
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = chrome.runtime.getURL('models/');
if (env.backends?.onnx?.wasm) {
  env.backends.onnx.wasm.wasmPaths = chrome.runtime.getURL('models/ort/');
}

// `pipeline` has a huge overload union (TS2590); narrow it to the one task we use.
const createPipeline = pipeline as (
  task: 'feature-extraction',
  model: string,
  options?: Record<string, unknown>,
) => Promise<FeatureExtractionPipeline>;

let pipePromise: Promise<FeatureExtractionPipeline> | null = null;

/**
 * Whether a WebGPU adapter can actually be acquired. `'gpu' in navigator` is not
 * enough: the API can be present while `requestAdapter()` returns null (no
 * hardware adapter, or Chrome started without --enable-unsafe-webgpu). In that
 * case ONNX fails late with "no available backend found", so we probe first.
 */
async function canUseWebGPU(): Promise<boolean> {
  const gpu = (navigator as { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) return false;
  try {
    return (await gpu.requestAdapter()) != null;
  } catch {
    return false;
  }
}

async function getPipeline(): Promise<FeatureExtractionPipeline> {
  if (pipePromise) return pipePromise;
  pipePromise = (async () => {
    if (await canUseWebGPU()) {
      try {
        return await createPipeline('feature-extraction', MODEL_ID, {
          dtype: 'q8',
          device: 'webgpu',
        });
      } catch (err) {
        console.warn('[embeddings] WebGPU init failed, falling back to wasm', err);
      }
    }
    return createPipeline('feature-extraction', MODEL_ID, {
      dtype: 'q8',
      device: 'wasm',
    });
  })();
  // Don't cache a rejected promise — a later call should be able to retry.
  pipePromise.catch(() => {
    pipePromise = null;
  });
  return pipePromise;
}

// The ONNX session behind the pipeline supports only ONE in-flight run: the
// asyncify wasm build suspends/resumes a single wasm stack, so a second run()
// started while one is suspended corrupts it and neither promise ever settles
// (every later embedding call then hangs too, until the page reloads). Chain
// all inference through this queue so concurrent callers (e.g. Promise.all in
// the quiz flow, or quiz and query running together) serialize instead.
let inferenceChain: Promise<unknown> = Promise.resolve();

function enqueueInference<T>(task: () => Promise<T>): Promise<T> {
  const run = inferenceChain.then(task, task);
  inferenceChain = run.catch(() => {});
  return run;
}

/** Embed a batch of texts into L2-normalized 384-dim vectors (mean-pooled). */
export async function embed(texts: string[]): Promise<Float32Array[]> {
  if (texts.length === 0) return [];
  const pipe = await getPipeline();
  const out = await enqueueInference(() =>
    pipe(texts, { pooling: 'mean', normalize: true }),
  );
  const [rows, dim] = out.dims as [number, number];
  const data = out.data as Float32Array;
  const vectors: Float32Array[] = [];
  for (let r = 0; r < rows; r++) {
    vectors.push(data.slice(r * dim, (r + 1) * dim));
  }
  return vectors;
}

export async function embedOne(text: string): Promise<Float32Array> {
  const [vec] = await embed([text]);
  return vec!;
}

/**
 * Embed text of any length as the L2-normalized mean of its retrieval-sized
 * subchunk vectors. MiniLM truncates input at 256 wordpiece tokens, so embedding
 * a long text directly silently drops everything past the cap; averaging
 * subchunk vectors keeps the whole text in the signal.
 */
export async function embedLong(text: string): Promise<Float32Array> {
  const chunks = await retrievalChunks(text, countTokens);
  if (chunks.length <= 1) return embedOne(chunks[0]?.text ?? text);
  const vectors = await embed(chunks.map((c) => c.text));
  const dim = vectors[0]!.length;
  const mean = new Float32Array(dim);
  for (const v of vectors) for (let i = 0; i < dim; i++) mean[i]! += v[i]!;
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += mean[i]! * mean[i]!;
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < dim; i++) mean[i]! /= norm;
  return mean;
}

/**
 * Count tokens with MiniLM's own tokenizer (BERT wordpiece, max seq 256),
 * including the [CLS]/[SEP] special tokens that count toward that limit. This is
 * the accurate way to size retrieval chunks for the embedding model — the LLM's
 * tokenizer would give the wrong counts. Loads the same pipeline used for
 * embedding, so there's no extra model download.
 */
export async function countTokens(text: string): Promise<number> {
  const pipe = await getPipeline();
  return pipe.tokenizer.encode(text).length;
}
