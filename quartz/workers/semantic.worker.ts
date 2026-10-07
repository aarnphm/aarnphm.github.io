import { env, AutoConfig, AutoModel, AutoTokenizer } from '@huggingface/transformers'
import { SEMANTIC_ONNX_MODELS, type SemanticQueryDType } from '../util/semantic-search'

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

type SearchMessage = { type: 'search'; text: string; k: number; seq: number }

type WorkerMessage = InitMessage | SearchMessage

type ReadyMessage = { type: 'ready' }

type ProgressMessage = { type: 'progress'; loadedRows: number; totalRows: number }

type SearchHit = { id: number; score: number }

type SearchResultMessage = { type: 'search-result'; seq: number; semantic: SearchHit[] }

type ErrorMessage = { type: 'error'; seq?: number; message: string }

type WorkerState = 'idle' | 'loading' | 'ready' | 'error'

type SemanticWorkerConfig = { model?: string; queryDtype?: SemanticQueryDType }

const MANIFEST_URL = '/embeddings/manifest.json'

let state: WorkerState = 'idle'
let manifest: Manifest | null = null
let cfg: SemanticWorkerConfig | null = null
let dims = 0
let tokenizer: any = null
let model: any = null
let encoderLoading: Promise<void> | null = null
let envConfigured = false
let abortController: AbortController | null = null
// Row-major unit vectors, `rows * dims` floats. Every query scans all of them.
let vectors: Float32Array | null = null

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
  // The ONNX wasm paths stay at the transformers.js default: it pins them to the onnxruntime-web build it bundles.
  envConfigured = true
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
  tokenizer = await AutoTokenizer.from_pretrained(mappedModel)
  model = await AutoModel.from_pretrained(mappedModel, { config, dtype, device: 'webgpu' })
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

    const response = await fetch(MANIFEST_URL, { signal: abortController.signal })
    if (!response.ok) {
      throw new Error(
        `failed to fetch manifest ${MANIFEST_URL}: ${response.status} ${response.statusText}`,
      )
    }
    manifest = (await response.json()) as Manifest

    if (manifest.vectors.dtype !== 'fp32') {
      throw new Error(
        `unsupported embedding dtype '${manifest.vectors.dtype}', regenerate with fp32`,
      )
    }

    dims = manifest.dims

    // The encoder loads beside the vectors, so "ready" means a query can run and a missing
    // WebGPU adapter fails now, not after the vectors are downloaded.
    const [loaded] = await Promise.all([loadVectors(manifest), ensureEncoder()])
    vectors = loaded

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
  const message: SearchResultMessage = {
    type: 'search-result',
    seq: msg.seq,
    semantic: topK(queryVec, Math.max(1, msg.k)),
  }
  self.postMessage(message)
}

self.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const origin = typeof event.origin === 'string' ? event.origin : ''
  if (origin && origin !== self.location.origin) return

  const data = event.data

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
