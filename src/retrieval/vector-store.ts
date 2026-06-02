// In-memory vector store with cosine top-k retrieval. Vectors are already
// L2-normalized by the embedding pipeline, so cosine similarity == dot product.
// One store instance per video, held in the side panel for its lifetime.

import { embed, embedOne, countTokens } from './embeddings';
import { retrievalChunks, retrievalChunksFromSegments, type Chunk } from '@/ai/chunking';
import type { TranscriptSegment } from '@/lib/messages';

interface Entry {
  chunk: Chunk;
  vector: Float32Array;
}

export interface RetrievedChunk {
  text: string;
  index: number;
  score: number;
  /** Start time (ms) of the chunk in the video, when known. */
  startMs?: number;
}

/** A merged, contiguous span of transcript handed to the model. */
export interface Passage {
  text: string;
  /** Start time (ms) of the span's first chunk, when known. */
  startMs?: number;
  /** Per-sentence start times within `text` (char offset -> ms), so the model
   *  can cite an exact moment inside the span. Absent for untimed text. */
  times?: { offset: number; tStartMs: number }[];
}

/** JSON-safe snapshot of a built store (Float32Array -> number[]). */
export interface SerializedStore {
  chunks: Chunk[];
  /** Flat row-major vectors; `dim` columns per chunk. */
  vectors: number[];
  dim: number;
}

function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i]! * b[i]!;
  return sum;
}

export class VectorStore {
  private entries: Entry[] = [];

  get size(): number {
    return this.entries.length;
  }

  /**
   * Chunk + embed a transcript. When `segments` are supplied, chunks are built
   * from the timed cues so each carries a start time (enabling seek/citation);
   * otherwise we fall back to plain-text chunks. Reports progress as a fraction.
   */
  async build(
    text: string,
    segments?: TranscriptSegment[],
    onProgress?: (done: number, total: number) => void,
  ): Promise<void> {
    const chunks =
      segments && segments.length
        ? await retrievalChunksFromSegments(segments, countTokens)
        : await retrievalChunks(text, countTokens);
    const batchSize = 16;
    this.entries = [];
    for (let i = 0; i < chunks.length; i += batchSize) {
      const batch = chunks.slice(i, i + batchSize);
      const vectors = await embed(batch.map((c) => c.text));
      batch.forEach((chunk, j) => this.entries.push({ chunk, vector: vectors[j]! }));
      onProgress?.(Math.min(i + batchSize, chunks.length), chunks.length);
    }
  }

  /** Snapshot the built index so it can be cached and rehydrated later. */
  serialize(): SerializedStore {
    const dim = this.entries[0]?.vector.length ?? 0;
    const vectors = new Array<number>(this.entries.length * dim);
    this.entries.forEach((e, i) => {
      for (let j = 0; j < dim; j++) vectors[i * dim + j] = e.vector[j]!;
    });
    return { chunks: this.entries.map((e) => e.chunk), vectors, dim };
  }

  /** Rebuild a store from a snapshot — skips the (expensive) embedding pass. */
  static deserialize(data: SerializedStore): VectorStore {
    const store = new VectorStore();
    const { chunks, vectors, dim } = data;
    if (
      !Array.isArray(chunks) ||
      !Array.isArray(vectors) ||
      typeof dim !== 'number' ||
      !Number.isInteger(dim) ||
      dim <= 0 ||
      vectors.length !== chunks.length * dim
    ) {
      return store;
    }
    store.entries = chunks.map((chunk, i) => ({
      chunk,
      vector: Float32Array.from(vectors.slice(i * dim, (i + 1) * dim)),
    }));
    return store;
  }

  async search(query: string, k: number): Promise<RetrievedChunk[]> {
    if (this.entries.length === 0) return [];
    const q = await embedOne(query);
    return this.entries
      .map((e) => ({
        text: e.chunk.text,
        index: e.chunk.index,
        score: dot(q, e.vector),
        startMs: e.chunk.startMs,
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k);
  }

  /**
   * Search, then widen each hit to its `radius` neighbours and merge the result
   * into contiguous passages. A 200-token retrieval chunk often clips the
   * sentence that actually holds the answer; pulling in the surrounding chunks
   * restores the continuity a sequential transcript needs, and merging adjacent
   * spans hands the model a few coherent passages instead of many fragments.
   * Passages are returned in transcript order (not score order), each tagged
   * with the start time of its first chunk so the answer can cite/seek to it.
   */
  async searchMerged(query: string, k: number, radius = 1): Promise<Passage[]> {
    const hits = await this.search(query, k);
    if (hits.length === 0) return [];

    const last = this.entries.length - 1;
    const keep = new Set<number>();
    for (const h of hits) {
      for (let i = h.index - radius; i <= h.index + radius; i++) {
        if (i >= 0 && i <= last) keep.add(i);
      }
    }

    const ordered = [...keep].sort((a, b) => a - b);
    const passages: Passage[] = [];
    let run: number[] = [];
    const flush = () => {
      if (!run.length) return;
      // Concatenate the run's chunk texts (join with a space) while re-basing each
      // chunk's per-sentence offsets onto the merged text, so timing survives.
      let text = '';
      const times: { offset: number; tStartMs: number }[] = [];
      for (const i of run) {
        const chunk = this.entries[i]!.chunk;
        if (text) text += ' ';
        const base = text.length;
        for (const t of chunk.times ?? []) {
          times.push({ offset: base + t.offset, tStartMs: t.tStartMs });
        }
        text += chunk.text;
      }
      passages.push({
        text,
        startMs: this.entries[run[0]!]!.chunk.startMs,
        times: times.length ? times : undefined,
      });
      run = [];
    };
    for (const i of ordered) {
      if (run.length && i !== run[run.length - 1]! + 1) flush();
      run.push(i);
    }
    flush();
    return passages;
  }
}
