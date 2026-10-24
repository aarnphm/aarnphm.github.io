import type {
  PdfSearchRequest,
  PdfSearchResult,
  SemanticInitRequest,
  SemanticSearchRequest,
  SemanticSearchResult,
  SemanticSimilarRequest,
  SemanticWorkerConfig,
  SemanticWorkerRequest,
  SemanticWorkerResponse,
} from '../util/semantic-protocol'
import { imageManifestUrl, readImageManifest, validateImageVectorShard } from '../util/image-search'
import { readPdfIndex, pdfSearchResults, validatePdfVectorShard } from '../util/pdf-search'
import { isRecord } from '../util/type-guards'
import { loadQueryEncoder, type QueryEncoder } from './semantic/encoder'
import {
  buildPdfSearchable,
  loadPdfPassages,
  pdfLexicalMatches,
  type PdfSearchable,
} from './semantic/pdf-index'
import { reduceGroups, scanScores, topK } from './semantic/topk'

type ShardLayout = { path: string; rows: number; rowOffset: number }

/** One loaded vector set. Row-major unit vectors; every query scans all of them. */
type IndexEntry = {
  model: string
  rows: number
  dims: number
  vectors: Float32Array
  groups?: Int32Array
  groupCount?: number
  /** Row identities, kept for image indexes so the page never reads the manifest. */
  ids?: string[]
  pdf?: PdfSearchable
}

type ParsedManifest = { key: string; model: string; materialize(): Promise<IndexEntry> }

const MANIFEST_URL = '/embeddings/manifest.json'
const ROOT_POINTER = /^\/embeddings\/(?:pdf|images)\/manifest\.json$/
const MAX_INDEXES = 4

let state: 'idle' | 'loading' | 'ready' | 'error' = 'idle'
let cfg: SemanticWorkerConfig | null = null
let encoder: QueryEncoder | null = null
let encoderLoading: Promise<QueryEncoder> | null = null
let abortController: AbortController | null = null
// Insertion order is the LRU order; `current` is the index the last request used.
const indexes = new Map<string, IndexEntry>()
const resolved = new Map<string, string>()
let current = ''
const queryVectors = new Map<string, Float32Array>()
let scores = new Float32Array(0)
let similarScores = new Float64Array(0)

const post = (message: SemanticWorkerResponse) => self.postMessage(message)

function toAssetUrl(path: string): URL {
  const url = new URL(path, self.location.origin)
  if (url.origin !== self.location.origin) {
    throw new Error(`refusing cross-origin semantic asset ${url.origin}`)
  }
  return url
}

async function fetchAsset(path: string, cache: RequestCache = 'default'): Promise<Response> {
  const res = await fetch(toAssetUrl(path), { cache, signal: abortController?.signal })
  if (!res.ok) {
    throw new Error(`failed to fetch ${path}: ${res.status} ${res.statusText}`)
  }
  return res
}

const fetchBinary = async (path: string) => (await fetchAsset(path)).arrayBuffer()

async function loadVectors(
  shards: readonly ShardLayout[],
  rows: number,
  dims: number,
  read: (bytes: ArrayBuffer, index: number) => Promise<Float32Array>,
): Promise<Float32Array> {
  const all = new Float32Array(rows * dims)
  let loadedRows = 0
  await Promise.all(
    shards.map(async (shard, index) => {
      const view = await read(await fetchBinary(shard.path), index)
      if (view.length !== shard.rows * dims) {
        throw new Error(`shard ${shard.path} has mismatched length`)
      }
      if (shard.rowOffset + shard.rows > rows) {
        throw new Error(`shard ${shard.path} reaches past the ${rows} rows of the manifest`)
      }
      all.set(view, shard.rowOffset * dims)
      loadedRows += shard.rows
      post({ type: 'progress', loadedRows, totalRows: rows })
    }),
  )
  return all
}

function textGroups(ids: unknown, chunkMetadata: unknown, rows: number) {
  if (!Array.isArray(ids) || ids.length !== rows) return {}
  const groupOf = new Map<string, number>()
  const groups = new Int32Array(rows)
  for (let row = 0; row < rows; row++) {
    const id = String(ids[row])
    const chunk = isRecord(chunkMetadata) ? chunkMetadata[id] : undefined
    const parent = isRecord(chunk) && typeof chunk.parentSlug === 'string' ? chunk.parentSlug : id
    let group = groupOf.get(parent)
    if (group === undefined) {
      group = groupOf.size
      groupOf.set(parent, group)
    }
    groups[row] = group
  }
  return { groups, groupCount: groupOf.size }
}

async function parseManifest(path: string): Promise<ParsedManifest> {
  const pathname = toAssetUrl(path).pathname
  const isRoot = ROOT_POINTER.test(pathname)
  // Root pointers move between generations; generation paths never change.
  const raw: unknown = await (await fetchAsset(path, isRoot ? 'no-cache' : 'default')).json()
  if (!isRecord(raw)) throw new Error(`invalid semantic manifest ${path}`)
  if (raw.kind === 'quartz-pdf-search-v1') {
    const index = readPdfIndex(raw)
    const key = `/embeddings/pdf/${index.generation}/manifest.json`
    if (!isRoot && pathname !== key)
      throw new Error('PDF embedding manifest does not match its immutable generation URL.')
    return {
      key,
      model: 'google/embeddinggemma-2',
      async materialize() {
        const [vectors, passages] = await Promise.all([
          loadVectors(index.shards, index.rows, 768, async (bytes, i) => {
            await validatePdfVectorShard(bytes, index.shards[i])
            return new Float32Array(bytes)
          }),
          loadPdfPassages(index, fetchBinary),
        ])
        const pdf = buildPdfSearchable(index, passages)
        return {
          model: 'google/embeddinggemma-2',
          rows: index.rows,
          dims: 768,
          vectors,
          groups: pdf.groups,
          groupCount: pdf.groupCount,
          pdf,
        }
      },
    }
  }
  if (raw.modality === 'image') {
    const data = readImageManifest(raw)
    const key = imageManifestUrl(data)
    if (!isRoot && pathname !== key)
      throw new Error('Image embedding manifest does not match its immutable generation URL.')
    return {
      key,
      model: data.model,
      async materialize() {
        const vectors = await loadVectors(data.vectors.shards, data.rows, data.dims, (bytes, i) =>
          validateImageVectorShard(bytes, data.vectors.shards[i], data.dims),
        )
        return { model: data.model, rows: data.rows, dims: data.dims, vectors, ids: data.ids }
      },
    }
  }
  const { model, rows, dims, vectors } = raw
  if (
    typeof model !== 'string' ||
    typeof rows !== 'number' ||
    typeof dims !== 'number' ||
    !isRecord(vectors) ||
    !Array.isArray(vectors.shards)
  )
    throw new Error(`invalid semantic manifest ${path}`)
  if (vectors.dtype !== 'fp32') {
    throw new Error(`unsupported embedding dtype '${String(vectors.dtype)}', regenerate with fp32`)
  }
  const shards = vectors.shards as ShardLayout[]
  const grouping = textGroups(raw.ids, raw.chunkMetadata, rows)
  return {
    key: pathname,
    model,
    async materialize() {
      const loaded = await loadVectors(shards, rows, dims, async bytes => new Float32Array(bytes))
      return { model, rows, dims, vectors: loaded, ...grouping }
    },
  }
}

function ensureEncoder(model: string): Promise<QueryEncoder> {
  if (encoder) return Promise.resolve(encoder)
  encoderLoading ??= loadQueryEncoder(model, cfg?.queryDtype)
    .then(loaded => (encoder = loaded))
    .catch((err: unknown) => {
      encoderLoading = null
      throw err
    })
  return encoderLoading
}

async function selectIndex(path: string): Promise<IndexEntry> {
  const key = resolved.get(path) ?? path
  const known = indexes.get(key)
  if (known) {
    indexes.delete(key)
    indexes.set(key, known)
    current = key
    return known
  }
  try {
    const parsed = await parseManifest(path)
    if (encoder && parsed.model !== encoder.model)
      throw new Error(
        `index model ${parsed.model} differs from the loaded encoder ${encoder.model}`,
      )
    const entry = indexes.get(parsed.key) ?? (await parsed.materialize())
    // Publish only after every shard loads. Failed swaps leave the old index intact.
    resolved.set(path, parsed.key)
    indexes.delete(parsed.key)
    indexes.set(parsed.key, entry)
    while (indexes.size > MAX_INDEXES) {
      const oldest = indexes.keys().next().value
      if (oldest === undefined) break
      indexes.delete(oldest)
    }
    current = parsed.key
    return entry
  } catch (error) {
    resolved.delete(path)
    throw error
  }
}

async function handleInit(msg: SemanticInitRequest) {
  if (state === 'loading' || state === 'ready') {
    throw new Error('worker already initialized or loading')
  }
  state = 'loading'
  abortController?.abort()
  abortController = new AbortController()
  try {
    cfg = msg.cfg ?? {}
    const path = cfg.manifestUrl ?? MANIFEST_URL
    const parsed = await parseManifest(path)
    // The encoder loads beside the vectors, so "ready" means a query can run and a missing
    // WebGPU adapter fails now, not after the vectors are downloaded. An index-only request
    // may have loaded the vectors before init.
    const [entry] = await Promise.all([
      indexes.get(parsed.key) ?? parsed.materialize(),
      ensureEncoder(parsed.model),
    ])
    resolved.set(path, parsed.key)
    indexes.delete(parsed.key)
    indexes.set(parsed.key, entry)
    current = parsed.key
    state = 'ready'
    post({ type: 'ready' })
  } catch (err) {
    state = 'error'
    throw err
  }
}

async function queryVector(entry: IndexEntry, text: string): Promise<Float32Array> {
  const key = `${entry.dims}\u0000${text}`
  const cached = queryVectors.get(key)
  if (cached) return cached
  const loaded = await ensureEncoder(entry.model)
  const vec = await loaded.embed(text, entry.dims)
  queryVectors.set(key, vec)
  if (queryVectors.size > 32) {
    const oldest = queryVectors.keys().next().value
    if (oldest !== undefined) queryVectors.delete(oldest)
  }
  return vec
}

/** Best rows, one per document when the index groups rows. */
async function scan(entry: IndexEntry, text: string, k: number) {
  const query = await queryVector(entry, text)
  if (scores.length < entry.rows) scores = new Float32Array(entry.rows)
  const rowScores = scores.subarray(0, entry.rows)
  scanScores(query, entry.vectors, entry.rows, entry.dims, rowScores)
  if (!entry.groups || entry.groupCount === undefined) return topK(rowScores, k)
  const { best, row } = reduceGroups(rowScores, entry.groups, entry.groupCount)
  return topK(best, k, row)
}

async function handleSearch(msg: SemanticSearchRequest) {
  if (state !== 'ready') throw new Error('worker not ready for search')
  const entry = await selectIndex(msg.manifestUrl ?? current)
  const semantic = await scan(entry, msg.text, Math.max(1, msg.k))
  const result: SemanticSearchResult = { type: 'search-result', seq: msg.seq, semantic }
  const { ids } = entry
  if (ids) result.ids = semantic.map(hit => ids[hit.id])
  post(result)
}

/** Stored image rows nearest to a stored seed row, seed excluded. Runs before or without init. */
async function handleSimilar(msg: SemanticSimilarRequest) {
  const entry = await selectIndex(msg.manifestUrl)
  const { ids } = entry
  if (!ids) throw new Error(`${msg.manifestUrl} is not an image index`)
  const seed = ids.indexOf(msg.seedId)
  if (seed < 0) throw new Error('This image is not indexed for visual similarity.')
  // Float64 scores reproduce the ordering of the former page-side scan, near ties included.
  if (similarScores.length < entry.rows) similarScores = new Float64Array(entry.rows)
  const rowScores = similarScores.subarray(0, entry.rows)
  const query = entry.vectors.subarray(seed * entry.dims, (seed + 1) * entry.dims)
  scanScores(query, entry.vectors, entry.rows, entry.dims, rowScores)
  rowScores[seed] = -Infinity
  const semantic = topK(rowScores, Math.max(1, msg.k))
  post({ type: 'search-result', seq: msg.seq, semantic, ids: semantic.map(hit => ids[hit.id]) })
}

async function handlePdfSearch(msg: PdfSearchRequest) {
  if (state !== 'ready') throw new Error('worker not ready for search')
  const entry = await selectIndex(msg.manifestUrl)
  const { pdf } = entry
  if (!pdf) throw new Error(`${msg.manifestUrl} is not a PDF search index`)
  const [hits, lexical] = await Promise.all([
    msg.semantic ? scan(entry, msg.text, pdf.groupCount) : [],
    pdfLexicalMatches(pdf, msg.text),
  ])
  const semantic = pdfSearchResults(hits, pdf.passages)
  const documents: PdfSearchResult['documents'] = {}
  for (const { slug } of [...semantic, ...lexical]) documents[slug] = pdf.documents[slug]
  post({
    type: 'pdf-result',
    seq: msg.seq,
    generation: pdf.generation,
    coverage: pdf.coverage,
    documents,
    semantic,
    lexical,
  })
}

let requests: Promise<void> = Promise.resolve()

self.onmessage = (event: MessageEvent<SemanticWorkerRequest>) => {
  const origin = typeof event.origin === 'string' ? event.origin : ''
  if (origin && origin !== self.location.origin) return
  const data = event.data
  requests = requests
    .then(async () => {
      if (data.type === 'init') await handleInit(data)
      else if (data.type === 'load') {
        const entry = await selectIndex(data.manifestUrl)
        post({ type: 'loaded', seq: data.seq, rows: entry.rows, ids: entry.ids })
      } else if (data.type === 'similar') await handleSimilar(data)
      else if (data.type === 'pdf-search') await handlePdfSearch(data)
      else await handleSearch(data)
    })
    .catch((err: unknown) => {
      post({
        type: 'error',
        seq: data.type === 'init' ? undefined : data.seq,
        message: err instanceof Error ? err.message : String(err),
      })
    })
}
