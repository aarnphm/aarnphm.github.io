import type {
  PdfSearchRequest,
  PdfSearchResult,
  SemanticLoadedResult,
  SemanticLoadRequest,
  SemanticSearchRequest,
  SemanticSearchResult,
  SemanticSimilarRequest,
  SemanticWorkerConfig,
  SemanticWorkerResponse,
} from '../../util/semantic-protocol'
import { reportSemanticProgress } from './search-progress'

export type SemanticResult = { id: number; score: number }
type SearchPayload = { semantic: SemanticResult[]; ids?: string[] }
type SemanticConfig = SemanticWorkerConfig
type Request =
  | Omit<SemanticSearchRequest, 'seq'>
  | Omit<PdfSearchRequest, 'seq'>
  | Omit<SemanticSimilarRequest, 'seq'>
  | Omit<SemanticLoadRequest, 'seq'>
type Reply = SemanticSearchResult | PdfSearchResult | SemanticLoadedResult
type PendingResolver = { resolve: (payload: Reply) => void; reject: (err: Error) => void }
// The encoder holds about 200 MB of GPU memory. A visible page keeps it for a while between
// searches; a hidden page gives it up quickly. Clients keep their result caches either way.
const IDLE_VISIBLE_MS = 5 * 60_000
const IDLE_HIDDEN_MS = 30_000

export function isSemanticCancellation(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError'
}

function cancelled(): DOMException {
  return new DOMException('semantic query superseded', 'AbortError')
}

// Inline entry points are bundled separately. Keep the encoder on window so Base and site
// search share its GPU session even when their copies of this module differ. The encoder
// loads on the first text query; stored-vector requests (`load`, `similar`) never start it.
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
  private initTimer: ReturnType<typeof setTimeout> | undefined
  private initialized = false
  private idleTimer: ReturnType<typeof setTimeout> | undefined
  private searches = 0
  private readonly onVisibility = () => this.releaseWhenIdle()

  get alive() {
    return this.worker !== null && this.error === null
  }

  private releaseWhenIdle() {
    clearTimeout(this.idleTimer)
    if (!this.alive || this.searches > 0) return
    const delay = document.visibilityState === 'hidden' ? IDLE_HIDDEN_MS : IDLE_VISIBLE_MS
    this.idleTimer = setTimeout(() => this.fail(cancelled()), delay)
  }

  constructor(readonly cfg: SemanticConfig) {
    this.ready = new Promise(resolve => (this.resolveReady = resolve))
    try {
      this.worker = new Worker(new URL('semantic.worker.js', import.meta.url), { type: 'module' })
      this.worker.onmessage = (event: MessageEvent<SemanticWorkerResponse>) => {
        const msg = event.data
        if (msg.type === 'model-progress') {
          const bytes = (value: number) =>
            (value / 1_000_000).toLocaleString(undefined, { maximumFractionDigits: 1 })
          const label =
            msg.phase === 'initializing'
              ? 'initializing model…'
              : msg.totalBytes > 0
                ? `loading model · ${bytes(msg.loadedBytes)}/${bytes(msg.totalBytes)} MB`
                : 'loading model files…'
          reportSemanticProgress(label, msg.loadedBytes, msg.totalBytes)
          return
        }
        if (msg.type === 'progress') {
          reportSemanticProgress(
            `loading index · ${msg.loadedRows.toLocaleString()}/${msg.totalRows.toLocaleString()}`,
            msg.loadedRows,
            msg.totalRows,
          )
          return
        }
        if (msg.type === 'ready') {
          clearTimeout(this.initTimer)
          this.resolveReady()
          this.releaseWhenIdle()
        } else if (
          msg.type === 'search-result' ||
          msg.type === 'pdf-result' ||
          msg.type === 'loaded'
        ) {
          const pending = this.pending.get(msg.seq)
          this.pending.delete(msg.seq)
          pending?.resolve(msg)
        } else if (typeof msg.seq === 'number') {
          const pending = this.pending.get(msg.seq)
          this.pending.delete(msg.seq)
          pending?.reject(new Error(msg.message))
        } else {
          this.fail(new Error(msg.message))
        }
      }
      this.worker.onerror = event => this.fail(new Error(event.message || 'semantic worker error'))
      document.addEventListener('visibilitychange', this.onVisibility)
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

  private init() {
    if (this.initialized || !this.worker) return
    this.initialized = true
    this.initTimer = setTimeout(
      () => this.fail(new Error('semantic model load timed out')),
      120_000,
    )
    this.worker.postMessage({ type: 'init', cfg: this.cfg })
  }

  private fail(error: Error) {
    this.error = error
    clearTimeout(this.timer)
    clearTimeout(this.initTimer)
    clearTimeout(this.idleTimer)
    document.removeEventListener('visibilitychange', this.onVisibility)
    this.worker?.terminate()
    this.worker = null
    this.resolveReady()
    for (const pending of this.pending.values()) pending.reject(error)
    this.pending.clear()
    if (window.quartzSemanticRuntime === this) window.quartzSemanticRuntime = undefined
  }

  async ensureReady() {
    reportSemanticProgress('preparing semantic model…')
    this.init()
    await this.ready
    if (this.error) {
      if (window.quartzSemanticRuntime === this) window.quartzSemanticRuntime = undefined
      throw this.error
    }
    this.releaseWhenIdle()
  }

  request(message: Request, current: () => boolean): Promise<Reply> {
    this.searches++
    clearTimeout(this.idleTimer)
    const result = this.queue.then(async () => {
      if (!current()) throw cancelled()
      if (message.type === 'search' || message.type === 'pdf-search') await this.ensureReady()
      // The worker answers in order, so a stored-vector request waits behind a started init.
      else if (this.initialized) await this.ready
      if (this.error) throw this.error
      if (!current()) throw cancelled()
      reportSemanticProgress('encoding query and ranking matches…')
      const payload = await new Promise<Reply>((resolve, reject) => {
        const seq = ++this.seq
        this.pending.set(seq, { resolve, reject })
        this.timer = setTimeout(() => this.fail(new Error('semantic query timed out')), 15_000)
        this.worker?.postMessage({ ...message, seq })
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

  /** `supersede: false` keeps earlier requests current; `cancel()` still drops it. */
  private request(message: Request, supersede = true): Promise<Reply> {
    const generation = supersede ? ++this.generation : this.generation
    if (this.disposed || this.cfg.enable === false)
      return Promise.reject(new Error('semantic client unavailable'))
    return this.acquireRuntime().request(
      message,
      () => !this.disposed && generation === this.generation,
    )
  }

  private get manifestUrl() {
    return this.cfg.manifestUrl ?? '/embeddings/manifest.json'
  }

  async search(text: string, k: number): Promise<SearchPayload> {
    const reply = await this.request({ type: 'search', text, k, manifestUrl: this.manifestUrl })
    if (reply.type !== 'search-result') throw new Error('unexpected semantic reply')
    return { semantic: reply.semantic, ids: reply.ids }
  }

  /** Stored rows nearest to `seedId`, seed excluded. Loads no encoder. */
  async similar(seedId: string, k: number): Promise<SearchPayload> {
    const reply = await this.request({ type: 'similar', seedId, k, manifestUrl: this.manifestUrl })
    if (reply.type !== 'search-result') throw new Error('unexpected semantic reply')
    return { semantic: reply.semantic, ids: reply.ids }
  }

  /** Row identities of an image index. Loads no encoder. */
  async ids(): Promise<string[]> {
    const reply = await this.request({ type: 'load', manifestUrl: this.manifestUrl }, false)
    if (reply.type !== 'loaded' || !reply.ids)
      throw new Error(`${this.manifestUrl} has no row identities`)
    return reply.ids
  }

  async pdfSearch(text: string, semantic: boolean): Promise<PdfSearchResult> {
    const reply = await this.request({
      type: 'pdf-search',
      text,
      semantic,
      manifestUrl: this.manifestUrl,
    })
    if (reply.type !== 'pdf-result') throw new Error('unexpected semantic reply')
    return reply
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
