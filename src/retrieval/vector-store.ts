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
  /** Best retrieval score among the hits merged into this span. */
  score?: number;
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

// Hybrid retrieval: dense cosine plus a BM25 lexical score. MiniLM blurs proper
// nouns and exact numbers in ASR transcripts, so pure-dense search can miss
// exact-term questions; IDF-weighted term matching rescues those. The BM25
// scores are max-normalized onto the cosine scale before blending.
const LEXICAL_WEIGHT = 0.3;
const BM25_K1 = 1.2;
const BM25_B = 0.75;

// Terms too common to discriminate between transcript chunks.
const STOPWORDS = new Set([
  'the', 'and', 'for', 'that', 'this', 'with', 'what', 'when', 'where', 'who',
  'how', 'why', 'does', 'did', 'are', 'was', 'were', 'have', 'has', 'had',
  'about', 'from', 'they', 'their', 'his', 'her', 'its', 'you', 'your', 'will',
  'can', 'could', 'would', 'should', 'than', 'then', 'them', 'there', 'these',
  'those', 'into', 'not', 'but', 'all', 'any', 'some', 'one', 'just', 'like',
  'video', 'say', 'says', 'said', 'talk', 'talks', 'mention', 'mentions',
]);

function lexTokens(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^\p{L}\p{N}]+/u)
    .filter((t) => t.length >= 3 && !STOPWORDS.has(t));
}

/** Per-chunk term stats for BM25, derived from the chunk texts. */
interface LexicalIndex {
  /** term -> term frequency, per chunk (entry order). */
  termFreqs: Map<string, number>[];
  /** Token count per chunk (entry order). */
  docLens: number[];
  avgDocLen: number;
  /** term -> number of chunks containing it. */
  docFreq: Map<string, number>;
}

export class VectorStore {
  private entries: Entry[] = [];
  /** Built lazily on first search; derived from chunk texts, so it needs no
   *  serialization and works for deserialized (cached) stores too. */
  private lexical?: LexicalIndex;

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
    this.lexical = undefined;
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

  private buildLexical(): LexicalIndex {
    const termFreqs: Map<string, number>[] = [];
    const docLens: number[] = [];
    const docFreq = new Map<string, number>();
    for (const e of this.entries) {
      const tf = new Map<string, number>();
      const tokens = lexTokens(e.chunk.text);
      for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
      for (const t of tf.keys()) docFreq.set(t, (docFreq.get(t) ?? 0) + 1);
      termFreqs.push(tf);
      docLens.push(tokens.length);
    }
    let total = 0;
    for (const len of docLens) total += len;
    const avgDocLen = docLens.length ? total / docLens.length || 1 : 1;
    return { termFreqs, docLens, avgDocLen, docFreq };
  }

  /** BM25 score of every chunk against `query` (entry order). */
  private bm25Scores(query: string): number[] {
    this.lexical ??= this.buildLexical();
    const { termFreqs, docLens, avgDocLen, docFreq } = this.lexical;
    const n = this.entries.length;
    const scores = new Array<number>(n).fill(0);
    for (const term of new Set(lexTokens(query))) {
      const df = docFreq.get(term);
      if (!df) continue;
      const idf = Math.log(1 + (n - df + 0.5) / (df + 0.5));
      for (let i = 0; i < n; i++) {
        const tf = termFreqs[i]!.get(term);
        if (!tf) continue;
        const norm = 1 - BM25_B + BM25_B * (docLens[i]! / avgDocLen);
        scores[i]! += (idf * tf * (BM25_K1 + 1)) / (tf + BM25_K1 * norm);
      }
    }
    return scores;
  }

  async search(query: string, k: number): Promise<RetrievedChunk[]> {
    if (this.entries.length === 0) return [];
    const q = await embedOne(query);
    const lexical = this.bm25Scores(query);
    let maxLex = 0;
    for (const s of lexical) if (s > maxLex) maxLex = s;
    return this.entries
      .map((e, i) => ({
        text: e.chunk.text,
        index: e.chunk.index,
        score:
          dot(q, e.vector) +
          (maxLex > 0 ? LEXICAL_WEIGHT * (lexical[i]! / maxLex) : 0),
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
    const scoreByIndex = new Map<number, number>();
    for (const h of hits) {
      scoreByIndex.set(h.index, h.score);
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
      let score: number | undefined;
      const times: { offset: number; tStartMs: number }[] = [];
      for (const i of run) {
        const chunk = this.entries[i]!.chunk;
        if (text) text += ' ';
        const base = text.length;
        for (const t of chunk.times ?? []) {
          times.push({ offset: base + t.offset, tStartMs: t.tStartMs });
        }
        text += chunk.text;
        const s = scoreByIndex.get(i);
        if (s != null && (score == null || s > score)) score = s;
      }
      passages.push({
        text,
        score,
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
