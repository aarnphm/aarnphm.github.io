import { env, AutoConfig, AutoModel, AutoTokenizer } from '@huggingface/transformers'
import { SEMANTIC_ONNX_MODELS, type SemanticQueryDType } from '../util/semantic-search'
import { isRecord } from '../util/type-guards'

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
}

type InitMessage = { type: 'init'; cfg?: SemanticWorkerConfig }

type SearchMessage = { type: 'search'; text: string; k: number; seq: number; manifestUrl?: string }
type LoadMessage = { type: 'load'; manifestUrl: string; seq: number }

type WorkerMessage = InitMessage | SearchMessage | LoadMessage

type ReadyMessage = { type: 'ready' }

type ProgressMessage = { type: 'progress'; loadedRows: number; totalRows: number }

type SearchHit = { id: number; score: number }

type SearchResultMessage = { type: 'search-result'; seq: number; semantic: SearchHit[] }

type ErrorMessage = { type: 'error'; seq?: number; message: string }

type WorkerState = 'idle' | 'loading' | 'ready' | 'error'

type SemanticWorkerConfig = {
  model?: string
  queryDtype?: SemanticQueryDType
  /** Same-origin manifest path; a base index lives under /embeddings/bases/<slug>/. */
  manifestUrl?: string
}

const MANIFEST_URL = '/embeddings/manifest.json'

let state: WorkerState = 'idle'
let manifest: Manifest | null = null
let cfg: SemanticWorkerConfig | null = null
let dims = 0
let tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>> | null = null
let model: Awaited<ReturnType<typeof AutoModel.from_pretrained>> | null = null
let encoderLoading: Promise<void> | null = null
let envConfigured = false
let abortController: AbortController | null = null
// Row-major unit vectors, `rows * dims` floats. Every query scans all of them.
let vectors: Float32Array | null = null
const indexes = new Map<string, { manifest: Manifest; vectors: Float32Array }>()
let manifestPath = ''
const queryVectors = new Map<string, Float32Array>()

function toAssetUrl(path: string): string {
  const url = new URL(path, self.location.origin)
  if (url.origin !== self.location.origin) {
    throw new Error(`refusing cross-origin semantic asset ${url.origin}`)
  }
  return url.toString()
}

async function fetchBinary(path: string): Promise<ArrayBuffer> {
  const res = await fetchAsset(path)
  return await res.arrayBuffer()
}

async function fetchAsset(path: string): Promise<Response> {
  const res = await fetch(path, { signal: abortController?.signal ?? undefined })
  if (!res.ok) {
    throw new Error(`failed to fetch ${path}: ${res.status} ${res.statusText}`)
  }
  return res
}

async function loadVectors(data: Manifest): Promise<Float32Array> {
  if (data.vectors.shards.length === 1 && data.vectors.shards[0].rowOffset === 0) {
    const shard = data.vectors.shards[0]
    const view = new Float32Array(await fetchBinary(toAssetUrl(shard.path)))
    if (view.length !== shard.rows * data.dims) {
      throw new Error(
        `shard ${shard.path} has mismatched length (expected ${shard.rows * data.dims}, got ${view.length})`,
      )
    }
    if (shard.rows > data.rows) {
      throw new Error(`shard ${shard.path} reaches past the ${data.rows} rows of the manifest`)
    }
    const progress: ProgressMessage = {
      type: 'progress',
      loadedRows: shard.rows,
      totalRows: data.rows,
    }
    self.postMessage(progress)
    return view
  }
  const all = new Float32Array(data.rows * data.dims)
  let loadedRows = 0
  await Promise.all(
    data.vectors.shards.map(async shard => {
      const view = new Float32Array(await fetchBinary(toAssetUrl(shard.path)))
      if (view.length !== shard.rows * data.dims) {
        throw new Error(
          `shard ${shard.path} has mismatched length (expected ${shard.rows * data.dims}, got ${view.length})`,
        )
      }
      if (shard.rowOffset + shard.rows > data.rows) {
        throw new Error(`shard ${shard.path} reaches past the ${data.rows} rows of the manifest`)
      }
      all.set(view, shard.rowOffset * data.dims)
      loadedRows += shard.rows
      const progress: ProgressMessage = { type: 'progress', loadedRows, totalRows: data.rows }
      self.postMessage(progress)
    }),
  )
  return all
}

// Exact cosine search. The rows and the query are unit vectors, so the dot product is the cosine.
// A scan costs rows * dims multiply-adds (about 25 million for 32k rows of 768), which is cheaper
// than building an approximate index in the page on every cold visit.
function topK(query: Float32Array, k: number): SearchHit[] {
  if (!manifest || !vectors) throw new Error('semantic index is not loaded')
  const hits: SearchHit[] = []
  let floor = -Infinity
  for (let row = 0, base = 0; row < manifest.rows; row++, base += dims) {
    let score = 0
    for (let d = 0; d < dims; d++) score += query[d] * vectors[base + d]
    if (!Number.isFinite(score) || score <= floor) continue
    let pos = hits.length
    while (pos > 0 && hits[pos - 1].score < score) pos--
    hits.splice(pos, 0, { id: row, score })
    if (hits.length > k) hits.pop()
    if (hits.length === k) floor = hits[k - 1].score
  }
  return hits
}

function configureRuntimeEnv() {
  if (envConfigured) return
  env.allowLocalModels = false
  env.allowRemoteModels = true
  // WebGPU runs the query encoder; keep the ONNX WASM orchestrator single-threaded.
  const onnxWasm = env.backends.onnx.wasm
  if (onnxWasm) onnxWasm.numThreads = 1
  envConfigured = true
}

function pruneTokenizerMemory(loadedTokenizer: NonNullable<typeof tokenizer>) {
  loadedTokenizer._tokenizerJSON = null
  const inner: unknown = loadedTokenizer._tokenizer
  if (isRecord(inner)) {
    inner.tokenizer = null
    const innerModel = inner.model
    if (isRecord(innerModel)) {
      innerModel.config = null
      innerModel.merges = null
      innerModel.vocab = null
    }
  }
}

const QUERY_DTYPES: readonly SemanticQueryDType[] = ['q4', 'q8', 'fp32']

function resolveQueryDType(value: unknown): SemanticQueryDType {
  return QUERY_DTYPES.find(dtype => dtype === value) ?? 'q4'
}

function ensureEncoder(): Promise<void> {
  encoderLoading ??= loadEncoder().catch((err: unknown) => {
    encoderLoading = null
    throw err
  })
  return encoderLoading
}

async function loadEncoder() {
  const modelId = manifest?.model ?? cfg?.model
  if (!modelId) {
    throw new Error('semantic model is not configured')
  }
  const mappedModel = SEMANTIC_ONNX_MODELS[modelId] ?? modelId
  configureRuntimeEnv()
  // Every quantized embeddinggemma graph stores its token table with GatherBlockQuantized.
  // The wasm backend has no kernel for it (session creation fails), and the fp32 graph is over 1 GB.
  // Without WebGPU the search stays lexical.
  if (!(await navigator.gpu?.requestAdapter().catch(() => null))) {
    throw new Error('semantic query encoder needs WebGPU')
  }
  const dtype = resolveQueryDType(cfg?.queryDtype)
  const config = await AutoConfig.from_pretrained(mappedModel)
  // embeddinggemma-2 ships vision and audio encoders (1.8 GB at fp32); queries only need the text graph.
  if (config.model_type === 'embedding_gemma2') {
    Object.assign(config, { vision_config: null, audio_config: null })
  }
  const [loadedTokenizer, loadedModel] = await Promise.all([
    AutoTokenizer.from_pretrained(mappedModel).then(loaded => {
      pruneTokenizerMemory(loaded)
      return loaded
    }),
    AutoModel.from_pretrained(mappedModel, { config, dtype, device: 'webgpu' }),
  ])
  tokenizer = loadedTokenizer
  model = loadedModel
  const onnxWasm = env.backends.onnx.wasm
  if (onnxWasm) onnxWasm.wasmBinary = undefined
}

async function embed(text: string, isQuery: boolean = false): Promise<Float32Array> {
  await ensureEncoder()
  if (!tokenizer || !model) throw new Error('semantic encoder unavailable')
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
  try {
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
    if (data.length < dims) {
      throw new Error(`model emits ${data.length} dims but the index stores ${dims}`)
    }
    // Matryoshka models (embeddinggemma-2: 768/512/256/128) keep the leading dims, then renormalize below.
    const vec = new Float32Array(dims)
    for (let i = 0; i < dims; i++) vec[i] = data[i]
    let norm = 0
    for (let i = 0; i < dims; i++) norm += vec[i] * vec[i]
    norm = Math.sqrt(norm)
    if (norm > 0) {
      for (let i = 0; i < dims; i++) vec[i] /= norm
    }
    return vec
  } finally {
    for (const map of [outputs, inputs]) {
      if (map && typeof map === 'object') {
        for (const key of Object.keys(map)) {
          const val = map[key]
          if (val && typeof val.dispose === 'function') {
            try {
              val.dispose()
            } catch {}
          }
        }
      }
    }
  }
}

async function fetchManifest(path: string): Promise<Manifest> {
  const manifestUrl = toAssetUrl(path)
  const response = await fetch(manifestUrl, { signal: abortController?.signal ?? undefined })
  if (!response.ok) {
    throw new Error(
      `failed to fetch manifest ${manifestUrl}: ${response.status} ${response.statusText}`,
    )
  }
  const data: Manifest = await response.json()
  if (data.vectors.dtype !== 'fp32') {
    throw new Error(`unsupported embedding dtype '${data.vectors.dtype}', regenerate with fp32`)
  }
  if (manifest && data.model !== manifest.model) {
    throw new Error(`index model ${data.model} differs from the loaded encoder ${manifest.model}`)
  }
  return { ...data, ids: [], titles: undefined, chunkMetadata: undefined }
}

async function handleInit(msg: InitMessage) {
  if (state === 'loading' || state === 'ready') {
    throw new Error('worker already initialized or loading')
  }

  state = 'loading'
  abortController?.abort()
  abortController = new AbortController()

  try {
    cfg = msg.cfg ?? {}

    manifest = await fetchManifest(cfg.manifestUrl ?? MANIFEST_URL)
    dims = manifest.dims

    // The encoder loads beside the vectors, so "ready" means a query can run and a missing
    // WebGPU adapter fails now, not after the vectors are downloaded.
    const [loaded] = await Promise.all([loadVectors(manifest), ensureEncoder()])
    vectors = loaded
    manifestPath = cfg.manifestUrl ?? MANIFEST_URL
    indexes.set(manifestPath, { manifest, vectors })

    state = 'ready'
    const ready: ReadyMessage = { type: 'ready' }
    self.postMessage(ready)
  } catch (err) {
    state = 'error'
    throw err
  }
}

async function selectIndex(path: string) {
  if (manifestPath === path) return
  let next = indexes.get(path)
  if (!next) {
    const data = await fetchManifest(path)
    next = { manifest: data, vectors: await loadVectors(data) }
  }
  // Publish the new inventory only after all shards load. Failed swaps leave the old index intact.
  indexes.delete(path)
  indexes.set(path, next)
  if (indexes.size > 2) {
    const oldest = indexes.keys().next().value
    if (oldest !== undefined) indexes.delete(oldest)
  }
  manifest = next.manifest
  dims = manifest.dims
  vectors = next.vectors
  manifestPath = path
}

async function handleSearch(msg: SearchMessage) {
  if (state !== 'ready') {
    throw new Error('worker not ready for search')
  }
  if (!manifest) {
    throw new Error('semantic worker not configured')
  }

  await selectIndex(msg.manifestUrl ?? manifestPath)
  const key = `${dims}\u0000${msg.text}`
  let queryVec = queryVectors.get(key)
  if (!queryVec) {
    queryVec = await embed(msg.text, true)
    queryVectors.set(key, queryVec)
    if (queryVectors.size > 32) {
      const oldest = queryVectors.keys().next().value
      if (oldest !== undefined) queryVectors.delete(oldest)
    }
  }
  const message: SearchResultMessage = {
    type: 'search-result',
    seq: msg.seq,
    semantic: topK(queryVec, Math.max(1, msg.k)),
  }
  self.postMessage(message)
}

let requests: Promise<void> = Promise.resolve()

self.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const origin = typeof event.origin === 'string' ? event.origin : ''
  if (origin && origin !== self.location.origin) return

  const data = event.data

  // The worker URL is stable, so an already open page may still send the previous protocol.
  requests = requests
    .then(async () => {
      if (data.type === 'init') await handleInit(data)
      else if (data.type === 'load') {
        await selectIndex(data.manifestUrl)
        self.postMessage({ type: 'loaded', seq: data.seq, rows: manifest?.rows ?? 0 })
      } else await handleSearch(data)
    })
    .catch((err: unknown) => {
      const message: ErrorMessage = {
        type: 'error',
        seq: data.type === 'init' ? undefined : data.seq,
        message: err instanceof Error ? err.message : String(err),
      }
      self.postMessage(message)
    })
}
