export type SemanticResult = { id: number; score: number }

type ManifestLevel = {
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
  dims: number
  dtype: string
  normalized: boolean
  rows: number
  shardSizeRows: number
  vectors: {
    dtype: string
    rows: number
    dims: number
    shards: {
      path: string
      rows: number
      rowOffset: number
      byteLength: number
      sha256?: string
      byteStride: number
    }[]
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
      levels: ManifestLevel[]
    }
  }
}

type IngestReadyMessage = {
  type: "ready"
  manifest: Manifest
  vectorBuffer: ArrayBufferLike
  graphBuffer: ArrayBufferLike
}

type IngestProgressMessage = {
  type: "progress"
  loadedRows: number
  totalRows: number
}

type IngestErrorMessage = { type: "error"; message: string }

type QueryReadyMessage = { type: "ready" }

type QueryResultMessage = {
  type: "search-result"
  seq: number
  semantic: SemanticResult[]
}

type QueryErrorMessage = { type: "error"; seq?: number; message: string }

type SearchPayload = {
  semantic: SemanticResult[]
}

type PendingResolver = {
  resolve: (payload: SearchPayload) => void
  reject: (err: Error) => void
}

export class SemanticClient {
  private ready: Promise<void>
  private resolveReady!: () => void
  private ingestWorker: Worker | null = null
  private queryWorker: Worker | null = null
  private pending = new Map<number, PendingResolver>()
  private seq = 0
  private disposed = false
  private readySettled = false
  private configured = false
  private lastError: Error | null = null

  constructor(private cfg?: any) {
    this.ready = new Promise((resolve) => {
      this.resolveReady = () => {
        if (this.readySettled) return
        this.readySettled = true
        resolve()
      }
    })

    if (this.cfg?.enable === false) {
      this.lastError = new Error("semantic search disabled by configuration")
      this.resolveReady()
      return
    }

    this.boot()
  }

  private boot() {
    try {
      this.ingestWorker = new Worker("/semantic-ingest.worker.js", { type: "module" })
    } catch (err) {
      this.handleFatal(err)
      return
    }
    try {
      this.queryWorker = new Worker("/semantic.worker.js", { type: "module" })
    } catch (err) {
      this.handleFatal(err)
      this.ingestWorker?.terminate()
      this.ingestWorker = null
      return
    }
    this.setupIngestion()
    this.setupQuery()
    this.startIngestion()
  }

  private setupIngestion() {
    if (!this.ingestWorker) return
    this.ingestWorker.onmessage = (
      event: MessageEvent<IngestReadyMessage | IngestProgressMessage | IngestErrorMessage>,
    ) => {
      const msg = event.data
      if (msg.type === "progress") {
        if (msg.totalRows > 0) {
          console.debug(
            `[SemanticClient] ingestion progress ${(msg.loadedRows / msg.totalRows) * 100}%`,
          )
        }
        return
      }
      if (msg.type === "error") {
        this.handleFatal(msg.message)
        return
      }
      if (!this.queryWorker) {
        this.handleFatal("query worker unavailable")
        return
      }
      this.queryWorker.postMessage(
        {
          type: "configure",
          cfg: this.cfg,
          manifest: msg.manifest,
          vectorBuffer: msg.vectorBuffer,
          graphBuffer: msg.graphBuffer,
        },
        [
          ...(msg.vectorBuffer instanceof ArrayBuffer ? [msg.vectorBuffer] : []),
          ...(msg.graphBuffer instanceof ArrayBuffer ? [msg.graphBuffer] : []),
        ],
      )
      this.ingestWorker?.terminate()
      this.ingestWorker = null
    }
  }

  private setupQuery() {
    if (!this.queryWorker) return
    this.queryWorker.onmessage = (
      event: MessageEvent<QueryReadyMessage | QueryResultMessage | QueryErrorMessage>,
    ) => {
      const msg = event.data
      if (msg.type === "ready") {
        this.configured = true
        this.lastError = null
        this.resolveReady()
        return
      }
      if (msg.type === "search-result") {
        const pending = this.pending.get(msg.seq)
        if (pending) {
          this.pending.delete(msg.seq)
          pending.resolve({ semantic: msg.semantic ?? [] })
        }
        return
      }
      if (msg.type === "error") {
        if (typeof msg.seq === "number") {
          const pending = this.pending.get(msg.seq)
          if (pending) {
            this.pending.delete(msg.seq)
            pending.reject(new Error(msg.message))
          }
        } else {
          this.handleFatal(msg.message)
        }
      }
    }
  }

  private startIngestion() {
    if (!this.ingestWorker) return
    const manifestUrl =
      typeof this.cfg?.manifestUrl === "string" && this.cfg.manifestUrl.length > 0
        ? this.cfg.manifestUrl
        : "/embeddings/manifest.json"
    const disableCache = Boolean(this.cfg?.disableCache)
    const baseUrl =
      typeof this.cfg?.manifestBaseUrl === "string" ? this.cfg.manifestBaseUrl : undefined
    this.ingestWorker.postMessage({ type: "init", manifestUrl, baseUrl, disableCache })
  }

  private rejectAll(err: Error, fatal = false) {
    for (const [id, pending] of this.pending.entries()) {
      pending.reject(err)
      this.pending.delete(id)
    }
    if (fatal) {
      this.lastError = err
      this.configured = false
      if (!this.readySettled) {
        this.resolveReady()
      }
    }
  }

  private handleFatal(err: unknown) {
    const error = err instanceof Error ? err : new Error(String(err))
    console.error("[SemanticClient] initialization failure:", error)
    this.rejectAll(error, true)
    this.ingestWorker?.terminate()
    this.ingestWorker = null
    if (this.queryWorker) {
      this.queryWorker.postMessage({ type: "reset" })
      this.queryWorker.terminate()
      this.queryWorker = null
    }
  }

  async ensureReady() {
    await this.ready
    if (!this.configured) {
      throw this.lastError ?? new Error("semantic search unavailable")
    }
  }

  async search(text: string, k: number): Promise<SearchPayload> {
    if (this.disposed) {
      throw new Error("semantic client has been disposed")
    }
    await this.ensureReady()
    if (!this.queryWorker || !this.configured) {
      throw this.lastError ?? new Error("query worker unavailable")
    }
    return new Promise<SearchPayload>((resolve, reject) => {
      const seq = ++this.seq
      this.pending.set(seq, { resolve, reject })
      this.queryWorker?.postMessage({ type: "search", text, k, seq })
    })
  }

  dispose() {
    if (this.disposed) return
    this.disposed = true
    this.rejectAll(new Error("semantic client disposed"))
    this.ingestWorker?.terminate()
    if (this.queryWorker) {
      this.queryWorker.postMessage({ type: "reset" })
      this.queryWorker.terminate()
    }
    this.ingestWorker = null
    this.queryWorker = null
    this.configured = false
  }
}
