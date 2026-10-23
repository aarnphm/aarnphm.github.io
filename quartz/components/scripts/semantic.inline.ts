import type { SemanticQueryDType } from '../../util/semantic-search'

export type SemanticResult = { id: number; score: number }
type SearchPayload = { semantic: SemanticResult[] }
type WorkerResponse =
  | { type: 'progress'; loadedRows: number; totalRows: number }
  | { type: 'ready' }
  | { type: 'search-result'; seq: number; semantic: SemanticResult[] }
  | { type: 'error'; seq?: number; message: string }
type SemanticConfig = {
  enable?: boolean
  model?: string
  queryDtype?: SemanticQueryDType
  manifestUrl?: string
}
type PendingResolver = { resolve: (payload: SearchPayload) => void; reject: (err: Error) => void }
const SEMANTIC_IDLE_MS = 30_000

export function isSemanticCancellation(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError'
}

function cancelled(): DOMException {
  return new DOMException('semantic query superseded', 'AbortError')
}

// Inline entry points are bundled separately. Keep the encoder on window so Base and site
// search share its GPU session even when their copies of this module differ.
export class SemanticRuntime {
  readonly ready: Promise<void>
  private worker: Worker | null = null
  private resolveReady!: () => void
  private error: Error | null = null
  private pending = new Map<number, PendingResolver>()
  private seq = 0
  private clients = 0
  private queue: Promise<unknown> = Promise.resolve()
  private timer: ReturnType<typeof setTimeout> | undefined
  private idleTimer: ReturnType<typeof setTimeout> | undefined
  private searches = 0

  get alive() {
    return this.worker !== null && this.error === null
  }

  private releaseWhenIdle() {
    clearTimeout(this.idleTimer)
    if (!this.alive || this.searches > 0) return
    // Clients retain result caches across navigation; the model has a shorter lifetime.
    this.idleTimer = setTimeout(() => this.fail(cancelled()), SEMANTIC_IDLE_MS)
  }

  constructor(readonly cfg: SemanticConfig) {
    this.ready = new Promise(resolve => (this.resolveReady = resolve))
    try {
      this.worker = new Worker(new URL('semantic.worker.js', import.meta.url), { type: 'module' })
      this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
        const msg = event.data
        if (msg.type === 'progress') return
        if (msg.type === 'ready') {
          clearTimeout(this.timer)
          this.resolveReady()
          this.releaseWhenIdle()
        } else if (msg.type === 'search-result') {
          const pending = this.pending.get(msg.seq)
          this.pending.delete(msg.seq)
          pending?.resolve({ semantic: msg.semantic })
        } else if (typeof msg.seq === 'number') {
          const pending = this.pending.get(msg.seq)
          this.pending.delete(msg.seq)
          pending?.reject(new Error(msg.message))
        } else {
          this.fail(new Error(msg.message))
        }
      }
      this.worker.onerror = event => this.fail(new Error(event.message || 'semantic worker error'))
      this.timer = setTimeout(() => this.fail(new Error('semantic model load timed out')), 120_000)
      this.worker.postMessage({ type: 'init', cfg })
    } catch (error) {
      this.fail(error instanceof Error ? error : new Error(String(error)))
    }
  }

  retain() {
    this.clients++
  }

  release() {
    if (--this.clients === 0) this.fail(new Error('semantic encoder released'))
  }

  private fail(error: Error) {
    this.error = error
    clearTimeout(this.timer)
    clearTimeout(this.idleTimer)
    this.worker?.terminate()
    this.worker = null
    this.resolveReady()
    for (const pending of this.pending.values()) pending.reject(error)
    this.pending.clear()
    if (window.quartzSemanticRuntime === this) window.quartzSemanticRuntime = undefined
  }

  async ensureReady() {
    await this.ready
    if (this.error) {
      if (window.quartzSemanticRuntime === this) window.quartzSemanticRuntime = undefined
      throw this.error
    }
    this.releaseWhenIdle()
  }

  search(text: string, k: number, manifestUrl: string, current: () => boolean) {
    this.searches++
    clearTimeout(this.idleTimer)
    const result = this.queue.then(async () => {
      if (!current()) throw cancelled()
      await this.ensureReady()
      if (!current()) throw cancelled()
      const payload = await new Promise<SearchPayload>((resolve, reject) => {
        const seq = ++this.seq
        this.pending.set(seq, { resolve, reject })
        this.timer = setTimeout(() => this.fail(new Error('semantic query timed out')), 15_000)
        this.worker?.postMessage({ type: 'search', text, k, seq, manifestUrl })
      }).finally(() => clearTimeout(this.timer))
      if (!current()) throw cancelled()
      return payload
    })
    this.queue = result
      .catch(() => {})
      .finally(() => {
        this.searches--
        this.releaseWhenIdle()
      })
    return result
  }
}

export class SemanticClient {
  private runtime: SemanticRuntime | null = null
  private generation = 0
  private disposed = false

  constructor(private cfg: SemanticConfig = {}) {
    if (cfg.enable === false) return
    this.acquireRuntime()
  }

  private acquireRuntime(): SemanticRuntime {
    if (this.runtime?.alive) return this.runtime
    this.runtime?.release()
    const previous = window.quartzSemanticRuntime
    if (
      previous &&
      (previous.cfg.model !== this.cfg.model || previous.cfg.queryDtype !== this.cfg.queryDtype)
    )
      throw new Error('semantic encoder configuration differs from the active encoder')
    this.runtime = previous ?? new SemanticRuntime(this.cfg)
    window.quartzSemanticRuntime = this.runtime
    this.runtime.retain()
    return this.runtime
  }

  async ensureReady() {
    if (this.disposed) throw new Error('semantic client has been disposed')
    if (this.cfg.enable === false) throw new Error('semantic search disabled by configuration')
    await this.acquireRuntime().ensureReady()
    if (this.disposed) throw new Error('semantic client has been disposed')
  }

  search(text: string, k: number): Promise<SearchPayload> {
    const generation = ++this.generation
    if (this.disposed || this.cfg.enable === false)
      return Promise.reject(new Error('semantic client unavailable'))
    return this.acquireRuntime().search(
      text,
      k,
      this.cfg.manifestUrl ?? '/embeddings/manifest.json',
      () => !this.disposed && generation === this.generation,
    )
  }

  cancel() {
    this.generation++
  }

  dispose() {
    if (this.disposed) return
    this.disposed = true
    this.cancel()
    this.runtime?.release()
    this.runtime = null
  }
}
