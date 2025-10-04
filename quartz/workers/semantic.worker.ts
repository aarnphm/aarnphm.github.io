// Semantic query worker: consumes shared embeddings and performs similarity search.

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

type Manifest = {
  version: number
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

type ConfigureMessage = {
  type: "configure"
  cfg: any
  manifest: Manifest
  vectorBuffer: ArrayBufferLike
  graphBuffer: ArrayBufferLike
}

type SearchMessage = { type: "search"; text: string; k: number; seq: number }
type ResetMessage = { type: "reset" }

type WorkerMessage = ConfigureMessage | SearchMessage | ResetMessage

type ReadyMessage = { type: "ready" }

type SearchHit = { id: number; score: number }

type SearchResultMessage = {
  type: "search-result"
  seq: number
  semantic: SearchHit[]
}

type ErrorMessage = { type: "error"; seq?: number; message: string }

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

function configureRuntimeEnv() {
  if (envConfigured) return
  const modelRoot =
    typeof cfg?.modelLocalPath === "string" && cfg.modelLocalPath.length > 0
      ? cfg.modelLocalPath
      : "/models"
  env.localModelPath = modelRoot
  const allowRemote = cfg?.allowRemoteModels ?? true
  const hasConfiguredLocalModel = Boolean(cfg?.modelLocalPath)
  env.allowLocalModels = hasConfiguredLocalModel
  env.allowRemoteModels = allowRemote
  const wasmBackend = env.backends?.onnx?.wasm
  if (!wasmBackend) {
    throw new Error("transformers.js ONNX runtime backend unavailable")
  }
  const cdnBase = `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${env.version}/dist/`
  wasmBackend.wasmPaths = cdnBase
  envConfigured = true
}

async function ensureEncoder() {
  if (classifier) return
  if (!cfg?.model) {
    throw new Error("semantic worker missing model identifier")
  }
  configureRuntimeEnv()
  const dtype = typeof cfg?.dtype === "string" && cfg.dtype.length > 0 ? cfg.dtype : "fp32"
  const allowRemote = cfg?.allowRemoteModels ?? true
  const hasConfiguredLocalModel = Boolean(cfg?.modelLocalPath)
  const pipelineOpts: Record<string, unknown> = {
    device: "wasm",
    dtype,
    local_files_only: hasConfiguredLocalModel && !allowRemote,
  }
  classifier = await pipeline("feature-extraction", cfg.model, pipelineOpts)
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

function normalize(vec: Float32Array) {
  let norm = 0
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i]
  norm = Math.sqrt(norm) || 1
  for (let i = 0; i < vec.length; i++) vec[i] /= norm
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

function bruteForceSearch(query: Float32Array, k: number): SearchHit[] {
  if (!vectorsView) return []
  const hits: SearchHit[] = []
  for (let id = 0; id < rows; id++) {
    const score = dot(query, vectorSlice(id))
    if (hits.length < k) {
      insertSortedDescending(hits, { id, score })
    } else if (score > hits[hits.length - 1].score) {
      insertSortedDescending(hits, { id, score })
      hits.length = k
    }
  }
  return hits
}

function hnswSearch(query: Float32Array, k: number): SearchHit[] {
  if (!manifest || !vectorsView || entryPoint < 0 || levelGraph.length === 0) {
    return bruteForceSearch(query, k)
  }
  const ef = Math.max(efDefault, k * 4)
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
  // Apply model-specific prefixes for asymmetric search
  let prefixedText = text
  if (cfg?.model) {
    const modelName = cfg.model.toLowerCase()
    if (modelName.includes("e5")) {
      // E5 models require query: or passage: prefix
      prefixedText = isQuery ? `query: ${text}` : `passage: ${text}`
    } else if (modelName.includes("qwen") && modelName.includes("embedding")) {
      // Qwen3-Embedding requires task instruction for queries only
      if (isQuery) {
        const task = "Given a web search query, retrieve relevant passages that answer the query"
        prefixedText = `Instruct: ${task}\nQuery: ${text}`
      }
      // Documents use plain text (no prefix)
    } else if (modelName.includes("embeddinggemma")) {
      // embeddinggemma requires specific prefixes
      prefixedText = isQuery
        ? `task: search result | query: ${text}`
        : `title: none | text: ${text}`
    }
  }
  const out = await classifier(prefixedText, { pooling: "mean", normalize: true })
  const data = Array.from(out?.data ?? out) as number[]
  const vec = new Float32Array(dims)
  for (let i = 0; i < dims; i++) vec[i] = data[i] ?? 0
  normalize(vec)
  return vec
}

function configureState(message: ConfigureMessage) {
  manifest = message.manifest
  cfg = message.cfg
  if (manifest.vectors.dtype !== "fp32") {
    throw new Error(`unsupported embedding dtype '${manifest.vectors.dtype}', regenerate with fp32`)
  }
  dims = manifest.dims
  rows = manifest.rows
  vectorsView = new Float32Array(message.vectorBuffer)
  entryPoint = manifest.hnsw.entryPoint
  maxLevel = manifest.hnsw.maxLevel
  efDefault = Math.max(64, manifest.hnsw.M * 4)
  levelGraph = manifest.hnsw.graph.levels.map((level) => {
    const indptr = new Uint32Array(message.graphBuffer, level.indptr.offset, level.indptr.elements)
    const indices = new Uint32Array(
      message.graphBuffer,
      level.indices.offset,
      level.indices.elements,
    )
    return { indptr, indices }
  })
  classifier = null
  envConfigured = false
}

self.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const data = event.data
  if (data.type === "reset") {
    manifest = null
    cfg = null
    vectorsView = null
    classifier = null
    envConfigured = false
    levelGraph = []
    entryPoint = -1
    maxLevel = 0
    return
  }
  if (data.type === "configure") {
    try {
      configureState(data)
      const ready: ReadyMessage = { type: "ready" }
      self.postMessage(ready)
    } catch (err) {
      const message: ErrorMessage = {
        type: "error",
        message: err instanceof Error ? err.message : String(err),
      }
      self.postMessage(message)
    }
    return
  }
  if (data.type === "search") {
    void (async () => {
      try {
        if (!manifest || !vectorsView) {
          throw new Error("semantic worker not configured")
        }
        const queryVec = await embed(data.text, true)
        const semanticHits = hnswSearch(queryVec, Math.max(1, data.k))
        const message: SearchResultMessage = {
          type: "search-result",
          seq: data.seq,
          semantic: semanticHits,
        }
        self.postMessage(message)
      } catch (err) {
        const message: ErrorMessage = {
          type: "error",
          seq: data.seq,
          message: err instanceof Error ? err.message : String(err),
        }
        self.postMessage(message)
      }
    })()
  }
}
