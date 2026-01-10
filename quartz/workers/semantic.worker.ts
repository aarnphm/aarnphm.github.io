import { env, pipeline } from "@huggingface/transformers"
import "onnxruntime-web/webgpu"
import "onnxruntime-web/wasm"

export {}

type VectorShardMeta = {
  path: string
  rows: number
  rowOffset: number
  byteLength: number
  sha256?: string
  byteStride: number
}

type LevelSection = {
  level: number
  indptr: { offset: number; elements: number; byteLength: number }
  indices: { offset: number; elements: number; byteLength: number }
}

type ChunkMetadata = {
  parentSlug: string
  chunkId: number
}

type Manifest = {
  version: number
  model: string
  dims: number
  dtype: string
  normalized: boolean
  rows: number
  shardSizeRows: number
  vectors: {
    dtype: string
    rows: number
    dims: number
    shards: VectorShardMeta[]
  }
  ids: string[]
  titles?: string[]
  chunkMetadata?: Record<string, ChunkMetadata>
  hnsw: {
    M: number
    efConstruction: number
    entryPoint: number
    maxLevel: number
    graph: {
      path: string
      sha256?: string
      levels: LevelSection[]
    }
  }
}

type InitMessage = {
  type: "init"
  cfg: any
  manifestUrl: string
  baseUrl?: string
  disableCache?: boolean
}

type SearchMessage = { type: "search"; text: string; k: number; seq: number }
type ResetMessage = { type: "reset" }

type WorkerMessage = InitMessage | SearchMessage | ResetMessage

type ReadyMessage = { type: "ready" }

type ProgressMessage = {
  type: "progress"
  loadedRows: number
  totalRows: number
}

type SearchHit = { id: number; score: number }

type SearchResultMessage = {
  type: "search-result"
  seq: number
  semantic: SearchHit[]
}

type ErrorMessage = { type: "error"; seq?: number; message: string }

type WorkerState = "idle" | "loading" | "ready" | "error"

// IndexedDB configuration
const DB_NAME = "semantic-search-cache"
const STORE_NAME = "assets"
const DB_VERSION = 1
const hasIndexedDB = typeof indexedDB !== "undefined"
const supportsSharedArrayBuffer = typeof SharedArrayBuffer !== "undefined"

// State
let state: WorkerState = "idle"
let manifest: Manifest | null = null
let cfg: any = null
let vectorsView: Float32Array | null = null
let dims = 0
let rows = 0
let classifier: any = null
let envConfigured = false
let entryPoint = -1
let maxLevel = 0
let efDefault = 128
let levelGraph: { indptr: Uint32Array; indices: Uint32Array }[] = []
let abortController: AbortController | null = null
let dbPromise: Promise<IDBDatabase> | null = null

// IndexedDB helpers
function openDatabase(): Promise<IDBDatabase> {
  if (!hasIndexedDB) {
    return Promise.reject(new Error("indexedDB unavailable"))
  }
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION)
      req.onupgradeneeded = () => {
        const db = req.result
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME)
        }
      }
      req.onsuccess = () => resolve(req.result)
      req.onerror = () => reject(req.error ?? new Error("failed to open cache store"))
    })
  }
  return dbPromise
}

async function readAsset(hash: string): Promise<ArrayBuffer | null> {
  if (!hasIndexedDB) {
    return null
  }
  const db = await openDatabase()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly")
    const store = tx.objectStore(STORE_NAME)
    const req = store.get(hash)
    req.onsuccess = () => {
      const value = req.result
      if (value instanceof ArrayBuffer) {
        resolve(value)
      } else if (value && value.buffer instanceof ArrayBuffer) {
        resolve(value.buffer as ArrayBuffer)
      } else {
        resolve(null)
      }
    }
    req.onerror = () => reject(req.error ?? new Error("failed to read cached asset"))
  })
}

async function writeAsset(hash: string, buffer: ArrayBuffer): Promise<void> {
  if (!hasIndexedDB) {
    return
  }
  const db = await openDatabase()
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite")
    const store = tx.objectStore(STORE_NAME)
    const req = store.put(buffer, hash)
    req.onsuccess = () => resolve()
    req.onerror = () => reject(req.error ?? new Error("failed to cache asset"))
  })
}

function toAbsolute(path: string, baseUrl?: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path
  }
  const base = baseUrl ?? self.location.origin
  return new URL(path, base).toString()
}

async function fetchBinary(
  path: string,
  disableCache: boolean,
  sha?: string,
): Promise<ArrayBuffer> {
  if (!disableCache && sha && hasIndexedDB) {
    try {
      const cached = await readAsset(sha)
      if (cached) {
        return cached
      }
    } catch {
      // fall through to network fetch on cache errors
    }
  }
  const res = await fetch(path, { signal: abortController?.signal ?? undefined })
  if (!res.ok) {
    throw new Error(`failed to fetch ${path}: ${res.status} ${res.statusText}`)
  }
  const payload = await res.arrayBuffer()
  if (!disableCache && sha && hasIndexedDB) {
    try {
      await writeAsset(sha, payload)
    } catch {
      // ignore cache write failures
    }
  }
  return payload
}

async function populateVectors(
  manifest: Manifest,
  baseUrl: string | undefined,
  disableCache: boolean | undefined,
): Promise<{ buffer: Float32Array; rowsLoaded: number }> {
  if (manifest.vectors.dtype !== "fp32") {
    throw new Error(`unsupported embedding dtype '${manifest.vectors.dtype}', regenerate with fp32`)
  }
  const rows = manifest.rows
  const dims = manifest.dims
  const totalBytes = rows * dims * Float32Array.BYTES_PER_ELEMENT
  const buffer = supportsSharedArrayBuffer
    ? new Float32Array(new SharedArrayBuffer(totalBytes))
    : new Float32Array(totalBytes)
  let loadedRows = 0
  for (const shard of manifest.vectors.shards) {
    const absolute = toAbsolute(shard.path, baseUrl)
    const payload = await fetchBinary(absolute, Boolean(disableCache), shard.sha256)
    const view = new Float32Array(payload)
    if (view.length !== shard.rows * dims) {
      throw new Error(
        `shard ${shard.path} has mismatched length (expected ${shard.rows * dims}, got ${view.length})`,
      )
    }
    buffer.set(view, shard.rowOffset * dims)
    loadedRows = Math.min(rows, shard.rowOffset + shard.rows)
    const progress: ProgressMessage = {
      type: "progress",
      loadedRows,
      totalRows: rows,
    }
    self.postMessage(progress)
  }
  return { buffer, rowsLoaded: loadedRows }
}

async function populateGraph(
  manifest: Manifest,
  baseUrl: string | undefined,
  disableCache: boolean | undefined,
): Promise<ArrayBuffer> {
  const graphMeta = manifest.hnsw.graph
  const absolute = toAbsolute(graphMeta.path, baseUrl)
  return await fetchBinary(absolute, Boolean(disableCache), graphMeta.sha256)
}

function configureRuntimeEnv() {
  if (envConfigured) return
  env.allowLocalModels = false
  env.allowRemoteModels = true
  const wasmBackend = env.backends?.onnx?.wasm
  if (!wasmBackend) {
    throw new Error("transformers.js ONNX runtime backend unavailable")
  }
  const cdnBase = `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${env.version}/dist/`
  wasmBackend.wasmPaths = cdnBase
  envConfigured = true
}

const MODEL_MAPPING: Record<string, string> = {
  "intfloat/multilingual-e5-large": "Xenova/multilingual-e5-large",
  "google/embeddinggemma-300m": "onnx-community/embeddinggemma-300m-ONNX",
}

async function ensureEncoder() {
  if (classifier) return
  const modelId = manifest?.model ?? cfg?.model
  if (!modelId) {
    throw new Error("semantic worker missing model identifier")
  }
  const model = MODEL_MAPPING[modelId] ?? modelId
  configureRuntimeEnv()
  const dtype = typeof cfg?.dtype === "string" && cfg.dtype.length > 0 ? cfg.dtype : "fp32"
  const pipelineOpts: Record<string, unknown> = {
    device: "wasm",
    dtype,
    local_files_only: false,
  }
  classifier = await pipeline("feature-extraction", model, pipelineOpts)
  cfg.dtype = dtype
}

function vectorSlice(id: number): Float32Array {
  if (!vectorsView) {
    throw new Error("vector buffer not configured")
  }
  const start = id * dims
  const end = start + dims
  return vectorsView.subarray(start, end)
}

function dot(a: Float32Array, b: Float32Array): number {
  let s = 0
  for (let i = 0; i < dims; i++) {
    s += a[i] * b[i]
  }
  return s
}

function neighborsFor(level: number, node: number): Uint32Array {
  const meta = levelGraph[level]
  if (!meta) return new Uint32Array()
  const { indptr, indices } = meta
  if (node < 0 || node + 1 >= indptr.length) return new Uint32Array()
  const start = indptr[node]
  const end = indptr[node + 1]
  return indices.subarray(start, end)
}

function insertSortedDescending(arr: SearchHit[], item: SearchHit) {
  let idx = arr.length
  while (idx > 0 && arr[idx - 1].score < item.score) {
    idx -= 1
  }
  arr.splice(idx, 0, item)
}

function hnswSearch(query: Float32Array, k: number): SearchHit[] {
  if (!manifest || !vectorsView || entryPoint < 0 || levelGraph.length === 0) {
    throw new Error("semantic graph not initialised; ensure embeddings include HNSW metadata")
  }
  const ef = Math.max(efDefault, k * 10)
  let ep = entryPoint
  let epScore = dot(query, vectorSlice(ep))
  for (let level = maxLevel; level > 0; level--) {
    let changed = true
    while (changed) {
      changed = false
      const neigh = neighborsFor(level, ep)
      for (let i = 0; i < neigh.length; i++) {
        const candidate = neigh[i]
        if (candidate >= rows) continue
        const score = dot(query, vectorSlice(candidate))
        if (score > epScore) {
          epScore = score
          ep = candidate
          changed = true
        }
      }
    }
  }

  const visited = new Set<number>()
  const candidateQueue: SearchHit[] = []
  const best: SearchHit[] = []
  insertSortedDescending(candidateQueue, { id: ep, score: epScore })
  insertSortedDescending(best, { id: ep, score: epScore })
  visited.add(ep)

  while (candidateQueue.length > 0) {
    const current = candidateQueue.shift()!
    const worstBest = best.length >= ef ? best[best.length - 1].score : -Infinity
    if (current.score < worstBest && best.length >= ef) {
      break
    }
    const neigh = neighborsFor(0, current.id)
    for (let i = 0; i < neigh.length; i++) {
      const candidate = neigh[i]
      if (candidate >= rows || visited.has(candidate)) continue
      visited.add(candidate)
      const score = dot(query, vectorSlice(candidate))
      const hit = { id: candidate, score }
      insertSortedDescending(candidateQueue, hit)
      if (best.length < ef || score > best[best.length - 1].score) {
        insertSortedDescending(best, hit)
        if (best.length > ef) {
          best.pop()
        }
      }
    }
  }

  best.sort((a, b) => b.score - a.score)
  return best.slice(0, k)
}

async function embed(text: string, isQuery: boolean = false): Promise<Float32Array> {
  await ensureEncoder()
  let prefixedText = text
  const model = manifest?.model ?? cfg?.model
  if (model) {
    const modelName = model.toLowerCase()
    switch (true) {
      case modelName.includes("e5"): {
        // E5 instruct models
        prefixedText = isQuery ? `query: ${text}` : `passage: ${text}`
        break
      }
      case modelName.includes("qwen") && modelName.includes("embedding"): {
        // Qwen3-Embedding requires task instruction for queries only
        if (isQuery) {
          const task = "Given a web search query, retrieve relevant passages that answer the query"
          prefixedText = `Instruct: ${task}\nQuery: ${text}`
        }
        // Documents use plain text (no prefix)
        break
      }
      case modelName.includes("embeddinggemma"): {
        // embeddinggemma requires specific prefixes
        prefixedText = isQuery
          ? `task: search result | query: ${text}`
          : `title: none | text: ${text}`
        break
      }
      default:
        break
    }
  }
  const out = await classifier(prefixedText, { pooling: "mean", normalize: true })
  const data = Array.from(out?.data ?? out) as number[]
  const vec = new Float32Array(dims)
  for (let i = 0; i < dims; i++) vec[i] = data[i] ?? 0
  return vec
}

async function handleInit(msg: InitMessage) {
  if (state === "loading" || state === "ready") {
    throw new Error("worker already initialized or loading")
  }

  state = "loading"
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

    if (manifest.vectors.dtype !== "fp32") {
      throw new Error(
        `unsupported embedding dtype '${manifest.vectors.dtype}', regenerate with fp32`,
      )
    }

    dims = manifest.dims
    rows = manifest.rows

    const { buffer: vectorBuffer } = await populateVectors(manifest, msg.baseUrl, msg.disableCache)
    vectorsView = vectorBuffer

    const graphBuffer = await populateGraph(manifest, msg.baseUrl, msg.disableCache)

    entryPoint = manifest.hnsw.entryPoint
    maxLevel = manifest.hnsw.maxLevel
    efDefault = Math.max(64, manifest.hnsw.M * 4)
    levelGraph = manifest.hnsw.graph.levels.map((level) => {
      const indptr = new Uint32Array(graphBuffer, level.indptr.offset, level.indptr.elements)
      const indices = new Uint32Array(graphBuffer, level.indices.offset, level.indices.elements)
      return { indptr, indices }
    })

    state = "ready"
    const ready: ReadyMessage = { type: "ready" }
    self.postMessage(ready)
  } catch (err) {
    state = "error"
    throw err
  }
}

async function handleSearch(msg: SearchMessage) {
  if (state !== "ready") {
    throw new Error("worker not ready for search")
  }
  if (!manifest || !vectorsView) {
    throw new Error("semantic worker not configured")
  }

  const queryVec = await embed(msg.text, true)
  const semanticHits = hnswSearch(queryVec, Math.max(1, msg.k))
  const message: SearchResultMessage = {
    type: "search-result",
    seq: msg.seq,
    semantic: semanticHits,
  }
  self.postMessage(message)
}

function handleReset() {
  abortController?.abort()
  abortController = null
  state = "idle"
  manifest = null
  cfg = null
  vectorsView = null
  dims = 0
  rows = 0
  classifier = null
  envConfigured = false
  levelGraph = []
  entryPoint = -1
  maxLevel = 0
}

self.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const data = event.data

  if (data.type === "reset") {
    handleReset()
    return
  }

  if (data.type === "init") {
    void handleInit(data).catch((err: unknown) => {
      const message: ErrorMessage = {
        type: "error",
        message: err instanceof Error ? err.message : String(err),
      }
      self.postMessage(message)
    })
    return
  }

  if (data.type === "search") {
    void handleSearch(data).catch((err: unknown) => {
      const message: ErrorMessage = {
        type: "error",
        seq: data.seq,
        message: err instanceof Error ? err.message : String(err),
      }
      self.postMessage(message)
    })
  }
}
