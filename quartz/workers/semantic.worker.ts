import { PGlite } from '@electric-sql/pglite'
import 'onnxruntime-web/webgpu'
import 'onnxruntime-web/wasm'
//@ts-ignore
import { vector as vectorExtension } from '@electric-sql/pglite/vector'
import { env, AutoModel, AutoTokenizer } from '@huggingface/transformers'
import { init, defaultDevice, numpy as np } from '@jax-js/jax'
import { dependencies } from '../../package.json'

export {}

type VectorShardMeta = {
  path: string
  rows: number
  rowOffset: number
  byteLength: number
  sha256?: string
  byteStride: number
}

type ChunkMetadata = { parentSlug: string; chunkId: number }

type Manifest = {
  version: number
  model: string
  dims: number
  dtype: string
  normalized: boolean
  rows: number
  shardSizeRows: number
  vectors: { dtype: string; rows: number; dims: number; shards: VectorShardMeta[] }
  ids: string[]
  titles?: string[]
  chunkMetadata?: Record<string, ChunkMetadata>
  hnsw?: { M: number; efConstruction: number }
}

type InitMessage = {
  type: 'init'
  cfg: any
  manifestUrl: string
  baseUrl?: string
  disableCache?: boolean
}

type SearchMessage = { type: 'search'; text: string; k: number; seq: number }

type ResetMessage = { type: 'reset' }

type WorkerMessage = InitMessage | SearchMessage | ResetMessage

type ReadyMessage = { type: 'ready' }

type ProgressMessage = { type: 'progress'; loadedRows: number; totalRows: number }

type SearchHit = { id: number; score: number }

type SearchResultMessage = { type: 'search-result'; seq: number; semantic: SearchHit[] }

type ErrorMessage = { type: 'error'; seq?: number; message: string }

type WorkerState = 'idle' | 'loading' | 'ready' | 'error'

type CandidateRow = { id: number; vec: string; score: number }

type MetaRow = { value: string }

const DB_NAME = 'semantic-search-cache'
const META_TABLE = 'semantic_meta'
const EMBEDDINGS_TABLE = 'semantic_embeddings'
const INDEX_NAME = 'semantic_embeddings_vec_hnsw'
const CDN_BASE = `https://cdn.jsdelivr.net/npm/@electric-sql/pglite@${dependencies['@electric-sql/pglite'].slice(1)}/dist`
const VECTOR_BUNDLE_URL = new URL(`${CDN_BASE}/vector.tar.gz`)

let state: WorkerState = 'idle'
let manifest: Manifest | null = null
let cfg: any = null
let dims = 0
let tokenizer: any = null
let model: any = null
let envConfigured = false
let abortController: AbortController | null = null
let dbPromise: Promise<PGlite> | null = null
let jaxPromise: Promise<void> | null = null
let manifestId: string | null = null

function toAbsolute(path: string, baseUrl?: string): string {
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return path
  }
  const base = baseUrl ?? self.location.origin
  return new URL(path, base).toString()
}

async function fetchBinary(path: string): Promise<ArrayBuffer> {
  const res = await fetch(path, { signal: abortController?.signal ?? undefined })
  if (!res.ok) {
    throw new Error(`failed to fetch ${path}: ${res.status} ${res.statusText}`)
  }
  return await res.arrayBuffer()
}

function vectorToLiteral(vec: Float32Array): string {
  return `[${vec.join(',')}]`
}

function parseVectorLiteral(text: string): number[] {
  return JSON.parse(text)
}

function buildManifestId(data: Manifest): string {
  const shardHashes = data.vectors.shards.map(shard => shard.sha256 ?? '').join('|')
  return [data.version, data.model, data.dims, data.rows, shardHashes].join(':')
}

async function openDatabase(disableCache: boolean | undefined): Promise<PGlite> {
  if (!dbPromise) {
    dbPromise = (async () => {
      const [wasmModule, fsBundle] = await Promise.all([
        WebAssembly.compileStreaming(fetch(`${CDN_BASE}/pglite.wasm`)),
        fetch(`${CDN_BASE}/pglite.data`).then(r => r.blob()),
      ])

      const vector = {
        ...vectorExtension,
        async setup(pg: unknown, emscriptenOpts: unknown) {
          const setupResult = await vectorExtension.setup(pg as never, emscriptenOpts)
          return { ...setupResult, bundlePath: VECTOR_BUNDLE_URL }
        },
      }
      const dataDir = disableCache ? `memory://${DB_NAME}` : `idb://${DB_NAME}`
      const db = await PGlite.create({ dataDir, wasmModule, fsBundle, extensions: { vector } })
      await db.exec('create extension if not exists vector')
      return db
    })()
  }
  return dbPromise
}

async function closeDatabase(): Promise<void> {
  if (!dbPromise) return
  const db = await dbPromise
  await db.close()
  dbPromise = null
}

async function ensureJax(): Promise<void> {
  if (!jaxPromise) {
    jaxPromise = (async () => {
      const devices = await init('webgpu')
      if (!devices.includes('webgpu')) {
        throw new Error('webgpu unavailable for jax-js')
      }
      defaultDevice('webgpu')
    })()
  }
  return jaxPromise
}

async function ensureSchema(db: PGlite, manifest: Manifest, manifestKey: string, baseUrl?: string) {
  await db.exec(
    `create table if not exists ${META_TABLE} (key text primary key, value text not null)`,
  )
  let currentKey: string | null = null
  try {
    const meta = await db.query<MetaRow>(`select value from ${META_TABLE} where key = $1`, [
      'manifest_id',
    ])
    currentKey = meta.rows[0]?.value ?? null
  } catch {
    currentKey = null
  }

  if (currentKey === manifestKey) {
    try {
      const res = await db.query<{ count: number }>(
        `select count(*)::int as count from ${EMBEDDINGS_TABLE}`,
      )
      if (res.rows[0]?.count === manifest.rows) {
        return
      }
    } catch {}
  }

  await db.exec(`drop index if exists ${INDEX_NAME}`)
  await db.exec(`drop table if exists ${EMBEDDINGS_TABLE}`)
  await db.exec(
    `create table ${EMBEDDINGS_TABLE} (id integer primary key, vec vector(${manifest.dims}))`,
  )

  const batchSize = 512
  let loadedRows = 0

  await Promise.all(
    manifest.vectors.shards.map(async shard => {
      const absolute = toAbsolute(shard.path, baseUrl)
      const payload = await fetchBinary(absolute)
      const view = new Float32Array(payload)
      if (view.length !== shard.rows * manifest.dims) {
        throw new Error(
          `shard ${shard.path} has mismatched length (expected ${shard.rows * manifest.dims}, got ${view.length})`,
        )
      }

      let batchIds: number[] = []
      let batchVecs: string[] = []

      for (let i = 0; i < shard.rows; i++) {
        const id = shard.rowOffset + i
        const offset = i * manifest.dims
        const vec = view.subarray(offset, offset + manifest.dims)
        batchIds.push(id)
        batchVecs.push(vectorToLiteral(vec))

        if (batchIds.length >= batchSize) {
          await insertBatch(db, batchIds, batchVecs)
          batchIds = []
          batchVecs = []
        }
      }

      if (batchIds.length > 0) {
        await insertBatch(db, batchIds, batchVecs)
      }

      loadedRows = Math.min(manifest.rows, loadedRows + shard.rows)
      const progress: ProgressMessage = { type: 'progress', loadedRows, totalRows: manifest.rows }
      self.postMessage(progress)
    }),
  )

  const m = manifest.hnsw?.M ?? 16
  const efc = manifest.hnsw?.efConstruction ?? 200
  try {
    await db.exec(
      `create index ${INDEX_NAME} on ${EMBEDDINGS_TABLE} using hnsw (vec vector_cosine_ops) with (m=${m}, ef_construction=${efc})`,
    )
  } catch {
    try {
      await db.exec(
        `create index ${INDEX_NAME} on ${EMBEDDINGS_TABLE} using ivfflat (vec vector_cosine_ops) with (lists=100)`,
      )
    } catch {}
  }
  await db.query(
    `insert into ${META_TABLE} (key, value) values ($1, $2) on conflict (key) do update set value = excluded.value`,
    ['manifest_id', manifestKey],
  )
}

async function insertBatch(db: PGlite, ids: number[], vecs: string[]) {
  const values: string[] = []
  const params: Array<number | string> = []
  let idx = 1
  for (let i = 0; i < ids.length; i++) {
    values.push(`($${idx}, $${idx + 1}::vector)`)
    params.push(ids[i], vecs[i])
    idx += 2
  }
  await db.query(`insert into ${EMBEDDINGS_TABLE} (id, vec) values ${values.join(',')}`, params)
}

function configureRuntimeEnv() {
  if (envConfigured) return
  env.allowLocalModels = false
  env.allowRemoteModels = true
  const wasmBackend = env.backends?.onnx?.wasm
  if (!wasmBackend) {
    throw new Error('transformers.js ONNX runtime backend unavailable')
  }
  const cdnBase = `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${env.version}/dist/`
  wasmBackend.wasmPaths = cdnBase
  envConfigured = true
}

const MODEL_MAPPING: Record<string, string> = {
  'intfloat/multilingual-e5-large': 'Xenova/multilingual-e5-large',
  'google/embeddinggemma-300m': 'onnx-community/embeddinggemma-300m-ONNX',
  'Qwen/Qwen3-Embedding-0.6B': 'onnx-community/Qwen3-Embedding-0.6B-ONNX',
}

async function ensureEncoder() {
  if (tokenizer && model) return
  const modelId = manifest?.model ?? cfg?.model
  const mappedModel = MODEL_MAPPING[modelId] ?? modelId
  configureRuntimeEnv()
  const dtype = typeof cfg?.dtype === 'string' && cfg.dtype.length > 0 ? cfg.dtype : 'fp32'
  tokenizer = await AutoTokenizer.from_pretrained(mappedModel)
  model = await AutoModel.from_pretrained(mappedModel, { dtype })
  cfg.dtype = dtype
}

async function embed(text: string, isQuery: boolean = false): Promise<Float32Array> {
  await ensureEncoder()
  let prefixedText = text
  const modelId = manifest?.model ?? cfg?.model
  if (modelId) {
    const modelName = modelId.toLowerCase()
    switch (true) {
      case modelName.includes('e5'): {
        prefixedText = isQuery ? `query: ${text}` : `passage: ${text}`
        break
      }
      case modelName.includes('qwen') && modelName.includes('embedding'): {
        if (isQuery) {
          const task = 'Given a web search query, retrieve relevant passages that answer the query'
          prefixedText = `Instruct: ${task}\nQuery: ${text}`
        }
        break
      }
      case modelName.includes('embeddinggemma'): {
        prefixedText = isQuery
          ? `task: search result | query: ${text}`
          : `title: none | text: ${text}`
        break
      }
      default:
        break
    }
  }
  const inputs = await tokenizer([prefixedText], { padding: true })
  const outputs = await model(inputs)

  let embedding
  if (outputs.sentence_embedding) {
    embedding = outputs.sentence_embedding
  } else if (outputs.last_hidden_state) {
    const lastHidden = outputs.last_hidden_state
    const attentionMask = inputs.attention_mask
    const [_, seqLen, hiddenSize] = lastHidden.dims
    const pooled = new Float32Array(hiddenSize)

    for (let i = 0; i < hiddenSize; i++) {
      let sum = 0
      let count = 0
      for (let j = 0; j < seqLen; j++) {
        if (attentionMask.data[j] > 0) {
          sum += lastHidden.data[j * hiddenSize + i]
          count++
        }
      }
      pooled[i] = count > 0 ? sum / count : 0
    }

    embedding = { data: pooled }
  } else {
    throw new Error('unsupported model output format')
  }

  const data = embedding.data
  const vec = new Float32Array(dims)
  for (let i = 0; i < dims; i++) vec[i] = data[i] ?? 0
  let norm = 0
  for (let i = 0; i < dims; i++) norm += vec[i] * vec[i]
  norm = Math.sqrt(norm)
  if (norm > 0) {
    for (let i = 0; i < dims; i++) vec[i] /= norm
  }
  return vec
}

async function rerank(queryVec: Float32Array, candidates: CandidateRow[]): Promise<SearchHit[]> {
  if (candidates.length === 0) return []
  await ensureJax()
  const queryArr = Array.from(queryVec)
  const flat = new Float32Array(candidates.length * dims)
  for (let i = 0; i < candidates.length; i++) {
    const parsed = parseVectorLiteral(candidates[i].vec)
    for (let j = 0; j < dims; j++) {
      flat[i * dims + j] = parsed[j] ?? 0
    }
  }
  const q = np.array(queryArr)
  const m = np.array(flat).reshape([candidates.length, dims])
  const scoresArr = np.dot(m.ref, q.ref)
  m.dispose()
  q.dispose()
  const scoresRaw = await scoresArr.ref.data()
  scoresArr.dispose()
  const scores = toNumberArray(scoresRaw)

  const hits: SearchHit[] = []
  for (let i = 0; i < candidates.length; i++) {
    const fallback = Number(candidates[i].score)
    const score = scores[i]
    const resolved = Number.isFinite(score) ? score : fallback
    const validated =
      Number.isFinite(fallback) && Math.abs(resolved - fallback) > 1e-3 ? fallback : resolved
    hits.push({ id: candidates[i].id, score: validated })
  }
  return hits
}

function toNumberArray(value: unknown): number[] {
  if (Array.isArray(value)) {
    return value.map(entry => Number(entry))
  }
  if (value instanceof Float32Array) return Array.from(value)
  if (value instanceof Float64Array) return Array.from(value)
  if (value instanceof Int32Array) return Array.from(value)
  if (value instanceof Uint32Array) return Array.from(value)
  if (value instanceof Int16Array) return Array.from(value)
  if (value instanceof Uint16Array) return Array.from(value)
  if (value instanceof Int8Array) return Array.from(value)
  if (value instanceof Uint8Array) return Array.from(value)
  throw new Error('unexpected score payload')
}

async function handleInit(msg: InitMessage) {
  if (state === 'loading' || state === 'ready') {
    throw new Error('worker already initialized or loading')
  }

  state = 'loading'
  abortController?.abort()
  abortController = new AbortController()

  try {
    cfg = msg.cfg

    const manifestUrl = toAbsolute(msg.manifestUrl, msg.baseUrl)
    const response = await fetch(manifestUrl, { signal: abortController.signal })
    if (!response.ok) {
      throw new Error(
        `failed to fetch manifest ${manifestUrl}: ${response.status} ${response.statusText}`,
      )
    }
    manifest = (await response.json()) as Manifest

    if (manifest.vectors.dtype !== 'fp32') {
      throw new Error(
        `unsupported embedding dtype '${manifest.vectors.dtype}', regenerate with fp32`,
      )
    }

    dims = manifest.dims
    manifestId = buildManifestId(manifest)

    const [db] = await Promise.all([openDatabase(Boolean(msg.disableCache)), ensureJax()])
    await ensureSchema(db, manifest, manifestId, msg.baseUrl)

    state = 'ready'
    const ready: ReadyMessage = { type: 'ready' }
    self.postMessage(ready)
  } catch (err) {
    state = 'error'
    throw err
  }
}

async function handleSearch(msg: SearchMessage) {
  if (state !== 'ready') {
    throw new Error('worker not ready for search')
  }
  if (!manifest) {
    throw new Error('semantic worker not configured')
  }

  const queryVec = await embed(msg.text, true)
  const queryLiteral = vectorToLiteral(queryVec)
  const limit = Math.max(1, msg.k)
  const db = await openDatabase(Boolean(cfg?.disableCache))
  const efSearch = Math.min(1000, Math.max(64, limit * 10))
  try {
    await db.exec(`set hnsw.ef_search = ${efSearch}`)
  } catch {}
  const res = await db.query<CandidateRow>(
    `select id, vec, 1 - (vec <=> $1::vector) as score from ${EMBEDDINGS_TABLE} order by vec <=> $1::vector limit $2`,
    [queryLiteral, limit],
  )
  const reranked = await rerank(queryVec, res.rows)
  reranked.sort((a, b) => b.score - a.score)
  const semanticHits = reranked.slice(0, limit)

  const message: SearchResultMessage = {
    type: 'search-result',
    seq: msg.seq,
    semantic: semanticHits,
  }
  self.postMessage(message)
}

function handleReset() {
  abortController?.abort()
  abortController = null
  state = 'idle'
  manifest = null
  cfg = null
  dims = 0
  tokenizer = null
  model = null
  envConfigured = false
  manifestId = null
  jaxPromise = null
  void closeDatabase()
}

self.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const data = event.data

  if (data.type === 'reset') {
    handleReset()
    return
  }

  if (data.type === 'init') {
    void handleInit(data).catch((err: unknown) => {
      const message: ErrorMessage = {
        type: 'error',
        message: err instanceof Error ? err.message : String(err),
      }
      self.postMessage(message)
    })
    return
  }

  if (data.type === 'search') {
    void handleSearch(data).catch((err: unknown) => {
      const message: ErrorMessage = {
        type: 'error',
        seq: data.seq,
        message: err instanceof Error ? err.message : String(err),
      }
      self.postMessage(message)
    })
  }
}
