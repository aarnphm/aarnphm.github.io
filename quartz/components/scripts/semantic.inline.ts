export type SemanticResult = { id: number; score: number }

type ProgressMessage = { type: "progress"; loadedRows: number; totalRows: number }

type ReadyMessage = { type: "ready" }

type ResultMessage = { type: "search-result"; seq: number; semantic: SemanticResult[] }

type ErrorMessage = { type: "error"; seq?: number; message: string }

type SearchPayload = { semantic: SemanticResult[] }

type PendingResolver = { resolve: (payload: SearchPayload) => void; reject: (err: Error) => void }

export class SemanticClient {
  private ready: Promise<void>
  private resolveReady!: () => void
  private worker: Worker | null = null
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
      this.worker = new Worker("/semantic.worker.js", { type: "module" })
    } catch (err) {
      this.handleFatal(err)
      return
    }
    this.setupWorker()
    this.startInit()
  }

  private setupWorker() {
    if (!this.worker) return
    this.worker.onmessage = (
      event: MessageEvent<ProgressMessage | ReadyMessage | ResultMessage | ErrorMessage>,
    ) => {
      const msg = event.data
      if (msg.type === "progress") {
        return
      }
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

  private startInit() {
    if (!this.worker) return
    const manifestUrl =
      typeof this.cfg?.manifestUrl === "string" && this.cfg.manifestUrl.length > 0
        ? this.cfg.manifestUrl
        : "/embeddings/manifest.json"
    const disableCache = Boolean(this.cfg?.disableCache)
    const baseUrl =
      typeof this.cfg?.manifestBaseUrl === "string" ? this.cfg.manifestBaseUrl : undefined
    this.worker.postMessage({ type: "init", cfg: this.cfg, manifestUrl, baseUrl, disableCache })
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
    if (this.worker) {
      this.worker.postMessage({ type: "reset" })
      this.worker.terminate()
      this.worker = null
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
    if (!this.worker || !this.configured) {
      throw this.lastError ?? new Error("worker unavailable")
    }
    return new Promise<SearchPayload>((resolve, reject) => {
      const seq = ++this.seq
      this.pending.set(seq, { resolve, reject })
      this.worker?.postMessage({ type: "search", text, k, seq })
    })
  }

  dispose() {
    if (this.disposed) return
    this.disposed = true
    this.rejectAll(new Error("semantic client disposed"))
    if (this.worker) {
      this.worker.postMessage({ type: "reset" })
      this.worker.terminate()
    }
    this.worker = null
    this.configured = false
  }
}
