export {}

type VectorShard = {
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

type Bm25Data = {
  N: number
  avgdl: number
  docLen: number[]
  postings: Record<string, [number, number][]>
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
    shards: VectorShard[]
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
  bm25?: {
    path: string
  }
}

type InitMessage = {
  type: "init"
  manifestUrl: string
  baseUrl?: string
  disableCache?: boolean
}

type ResetMessage = { type: "reset" }

type WorkerMessage = InitMessage | ResetMessage

type TransferableBuffer = SharedArrayBuffer | ArrayBuffer

type ReadyMessage = {
  type: "ready"
  manifest: Manifest
  vectorBuffer: TransferableBuffer
  graphBuffer: TransferableBuffer
  bm25?: Bm25Data
}

type ProgressMessage = {
  type: "progress"
  loadedRows: number
  totalRows: number
}

type ErrorMessage = {
  type: "error"
  message: string
}

const DB_NAME = "semantic-search-cache"
const STORE_NAME = "assets"
const DB_VERSION = 1
const hasIndexedDB = typeof indexedDB !== "undefined"
const supportsSharedArrayBuffer = typeof SharedArrayBuffer !== "undefined"

let dbPromise: Promise<IDBDatabase> | null = null
let abortController: AbortController | null = null

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

function allocateBuffer(byteLength: number): TransferableBuffer {
  return supportsSharedArrayBuffer ? new SharedArrayBuffer(byteLength) : new ArrayBuffer(byteLength)
}

function cloneToTransferable(buffer: ArrayBuffer): TransferableBuffer {
  if (!supportsSharedArrayBuffer) {
    return buffer
  }
  const shared = new SharedArrayBuffer(buffer.byteLength)
  new Uint8Array(shared).set(new Uint8Array(buffer))
  return shared
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
): Promise<{ buffer: TransferableBuffer; rowsLoaded: number }> {
  if (manifest.vectors.dtype !== "fp32") {
    throw new Error(`unsupported embedding dtype '${manifest.vectors.dtype}', regenerate with fp32`)
  }
  const rows = manifest.rows
  const dims = manifest.dims
  const buffer = allocateBuffer(rows * dims * Float32Array.BYTES_PER_ELEMENT)
  const dest = new Float32Array(buffer)
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
    dest.set(view, shard.rowOffset * dims)
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
): Promise<TransferableBuffer> {
  const graphMeta = manifest.hnsw.graph
  const absolute = toAbsolute(graphMeta.path, baseUrl)
  const payload = await fetchBinary(absolute, Boolean(disableCache), graphMeta.sha256)
  return cloneToTransferable(payload)
}

async function loadBm25(
  manifest: Manifest,
  baseUrl: string | undefined,
): Promise<Bm25Data | undefined> {
  if (!manifest.bm25?.path) return undefined
  const url = toAbsolute(manifest.bm25.path, baseUrl)
  const res = await fetch(url, { signal: abortController?.signal ?? undefined })
  if (!res.ok) {
    throw new Error(`failed to fetch BM25 index ${url}: ${res.status} ${res.statusText}`)
  }
  return (await res.json()) as Bm25Data
}

async function handleInit(msg: InitMessage) {
  abortController?.abort()
  abortController = new AbortController()
  const manifestUrl = toAbsolute(msg.manifestUrl, msg.baseUrl)
  const response = await fetch(manifestUrl, { signal: abortController.signal })
  if (!response.ok) {
    throw new Error(
      `failed to fetch manifest ${manifestUrl}: ${response.status} ${response.statusText}`,
    )
  }
  const manifest = (await response.json()) as Manifest
  const { buffer: vectorBuffer } = await populateVectors(manifest, msg.baseUrl, msg.disableCache)
  const graphBuffer = await populateGraph(manifest, msg.baseUrl, msg.disableCache)
  const bm25 = await loadBm25(manifest, msg.baseUrl)
  const ready: ReadyMessage = {
    type: "ready",
    manifest,
    vectorBuffer,
    graphBuffer,
    bm25,
  }
  const transfers: ArrayBuffer[] = []
  if (vectorBuffer instanceof ArrayBuffer) transfers.push(vectorBuffer)
  if (graphBuffer instanceof ArrayBuffer) transfers.push(graphBuffer)
  if (transfers.length > 0) {
    // @ts-ignore
    self.postMessage(ready, transfers)
  } else {
    self.postMessage(ready)
  }
}

self.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const data = event.data
  if (data.type === "reset") {
    abortController?.abort()
    abortController = null
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
  }
}
