import type { ContentDetails } from '../../plugins'
import type { LexicalDocumentIndex } from './lexical-search-index'
import type {
  LexicalFieldResult,
  LexicalIndexPriority,
  LexicalSearchSpec,
  LexicalWorkerRequest,
  LexicalWorkerResponse,
} from './lexical-search-types'
import { prepareLexicalDocument } from './lexical-search-document'

type PendingSearch = {
  resolve: (results: LexicalFieldResult[]) => void
  reject: (err: Error) => void
}

type PendingCommit = { seq: number; resolve: () => void; reject: (err: Error) => void }

type FallbackModule = typeof import('./lexical-search-index')

const STREAM_YIELD_BUDGET_MS = 4

function yieldMainThread(): Promise<void> {
  return new Promise(resolve => {
    window.setTimeout(resolve, 0)
  })
}

export class LexicalSearchClient {
  private worker: Worker | null = null
  private workerFailed = false
  private workerReady = false
  private indexedData: Record<string, ContentDetails> | null = null
  private indexingPromise: Promise<void> | null = null
  private pendingCommit: PendingCommit | null = null
  private pendingRestore: {
    seq: number
    resolve: (restored: boolean) => void
    reject: (err: Error) => void
  } | null = null
  private pendingSearches = new Map<number, PendingSearch>()
  private seq = 0
  private generation = 0
  private indexAttempt = 0

  private fallbackModulePromise: Promise<FallbackModule> | null = null
  private fallbackIndex: LexicalDocumentIndex | null = null
  private fallbackIndexedData: Record<string, ContentDetails> | null = null
  private fallbackIndexingPromise: Promise<void> | null = null
  private deferredFallbackData: Record<string, ContentDetails> | null = null

  async ensureIndexed(
    data: Record<string, ContentDetails>,
    priority: LexicalIndexPriority = 'user-visible',
  ): Promise<void> {
    if (!this.workerFailed) {
      if (this.worker && this.workerReady && this.indexedData === data) {
        return
      }
      if (this.indexingPromise && this.indexedData === data) {
        return this.indexingPromise
      }
      this.indexedData = data
      const generation = this.generation
      const attempt = ++this.indexAttempt
      const currentAttempt = this.populateWorker(data).catch(async err => {
        if (generation !== this.generation || attempt !== this.indexAttempt) throw err
        if (this.indexingPromise === currentAttempt) {
          this.indexingPromise = null
        }
        this.handleWorkerFatal(err)
        if (priority === 'background') {
          this.deferredFallbackData = data
          return
        }
        await this.ensureFallbackIndexed(data)
      })
      this.indexingPromise = currentAttempt
      return currentAttempt
    }

    if (priority === 'background') {
      this.deferredFallbackData = data
      return
    }

    await this.ensureFallbackIndexed(data)
  }

  async search(
    data: Record<string, ContentDetails>,
    spec: LexicalSearchSpec,
  ): Promise<LexicalFieldResult[]> {
    const generation = this.generation
    await this.ensureIndexed(data, 'user-visible')

    if (!this.workerFailed && this.worker && this.workerReady) {
      try {
        return await this.searchWorker(spec)
      } catch (err) {
        if (generation !== this.generation) throw err
        this.handleWorkerFatal(err)
      }
    }

    const fallbackData = this.deferredFallbackData ?? data
    await this.ensureFallbackIndexed(fallbackData)
    const mod = await this.loadFallbackModule()
    if (!this.fallbackIndex) {
      throw new Error('lexical fallback index is unavailable')
    }
    return mod.searchLexicalDocumentIndex(this.fallbackIndex, spec)
  }

  disposeForPagehide(): void {
    this.generation++
    this.rejectPending(new Error('lexical search terminated on pagehide'))
    if (this.worker) {
      this.worker.terminate()
      this.worker = null
    }
    this.workerFailed = false
    this.workerReady = false
    this.indexedData = null
    this.indexingPromise = null
  }

  recoverFromBfcache(): void {
    if (!this.worker && !this.workerFailed) {
      this.workerReady = false
      this.indexedData = null
      this.indexingPromise = null
    }
  }

  private ensureWorker(): Worker {
    if (this.worker) return this.worker
    const workerUrl = new URL('lexical.worker.js', import.meta.url)
    const worker = new Worker(workerUrl, { type: 'module' })
    worker.onmessage = (event: MessageEvent<LexicalWorkerResponse>) => {
      if (worker !== this.worker) return
      const msg = event.data
      if (msg.type === 'prebuilt') {
        if (this.pendingRestore?.seq === msg.seq) {
          const pending = this.pendingRestore
          this.pendingRestore = null
          pending.resolve(msg.restored)
        }
        return
      }
      if (msg.type === 'ready') {
        if (this.pendingCommit && this.pendingCommit.seq === msg.seq) {
          const commit = this.pendingCommit
          this.pendingCommit = null
          this.workerReady = true
          commit.resolve()
        }
        return
      }
      if (msg.type === 'search-result') {
        const pending = this.pendingSearches.get(msg.seq)
        if (pending) {
          this.pendingSearches.delete(msg.seq)
          pending.resolve(msg.results)
        }
        return
      }
      if (msg.type === 'error') {
        const error = new Error(msg.message)
        if (typeof msg.seq === 'number') {
          if (this.pendingCommit && this.pendingCommit.seq === msg.seq) {
            const commit = this.pendingCommit
            this.pendingCommit = null
            this.pendingRestore?.reject(error)
            this.pendingRestore = null
            commit.reject(error)
            return
          }
          const pending = this.pendingSearches.get(msg.seq)
          if (pending) {
            this.pendingSearches.delete(msg.seq)
            pending.reject(error)
            return
          }
        }
        this.handleWorkerFatal(error)
      }
    }
    worker.onerror = event => {
      event.preventDefault()
      if (worker !== this.worker) return
      this.handleWorkerFatal(new Error(event.message || 'lexical worker error'))
    }
    worker.onmessageerror = () => {
      if (worker !== this.worker) return
      this.handleWorkerFatal(new Error('lexical worker message deserialization error'))
    }
    this.worker = worker
    return worker
  }

  private postToWorker(worker: Worker, message: LexicalWorkerRequest): void {
    worker.postMessage(message)
  }

  private async populateWorker(data: Record<string, ContentDetails>): Promise<void> {
    const worker = this.ensureWorker()
    const seq = ++this.seq
    this.workerReady = false
    this.pendingRestore?.reject(new Error('superseded lexical restore'))
    this.pendingRestore = null

    const commitPromise = new Promise<void>((resolve, reject) => {
      if (this.pendingCommit) {
        this.pendingCommit.reject(new Error('superseded lexical index initialization'))
      }
      this.pendingCommit = { seq, resolve, reject }
    })

    // Worker failure can reject the commit while the document stream is still yielding.
    void commitPromise.catch(() => {})
    let sliceStart = performance.now()
    const documents = []
    let id = 0
    for (const [slug, fileData] of Object.entries<ContentDetails>(data)) {
      if (this.worker !== worker || this.pendingCommit?.seq !== seq) {
        throw new Error('lexical worker reset during indexing')
      }
      const item = prepareLexicalDocument(id, slug, fileData)
      documents.push(item)
      id += 1
      if (performance.now() - sliceStart >= STREAM_YIELD_BUDGET_MS) {
        await yieldMainThread()
        sliceStart = performance.now()
      }
    }

    if (this.worker !== worker || this.pendingCommit?.seq !== seq) {
      throw new Error('lexical worker reset before commit')
    }
    if (window.crypto?.subtle) {
      const digest = await crypto.subtle.digest(
        'SHA-256',
        new TextEncoder().encode(JSON.stringify(documents)),
      )
      if (this.worker !== worker || this.pendingCommit?.seq !== seq)
        throw new Error('lexical worker reset before restore')
      const fingerprint = Array.from(new Uint8Array(digest), byte =>
        byte.toString(16).padStart(2, '0'),
      ).join('')
      const restoring = new Promise<boolean>((resolve, reject) => {
        this.pendingRestore = { seq, resolve, reject }
      })
      this.postToWorker(worker, { type: 'init', seq, fingerprint, count: id })
      if (await restoring) {
        if (this.worker !== worker || this.pendingCommit?.seq !== seq)
          throw new Error('lexical worker reset after restore')
        this.workerReady = true
        const pending = this.pendingCommit
        this.pendingCommit = null
        pending.resolve()
        return
      }
    } else {
      this.postToWorker(worker, { type: 'init', seq })
    }
    sliceStart = performance.now()
    for (const item of documents) {
      if (this.worker !== worker || this.pendingCommit?.seq !== seq)
        throw new Error('lexical worker reset during fallback indexing')
      this.postToWorker(worker, { type: 'add', seq, item })
      if (performance.now() - sliceStart >= STREAM_YIELD_BUDGET_MS) {
        await yieldMainThread()
        sliceStart = performance.now()
      }
    }
    this.postToWorker(worker, { type: 'commit', seq, count: id })
    await commitPromise
  }

  private searchWorker(spec: LexicalSearchSpec): Promise<LexicalFieldResult[]> {
    const worker = this.worker
    if (!worker || !this.workerReady) {
      return Promise.reject(new Error('lexical worker is not ready'))
    }
    const seq = ++this.seq
    return new Promise<LexicalFieldResult[]>((resolve, reject) => {
      this.pendingSearches.set(seq, { resolve, reject })
      this.postToWorker(worker, { type: 'search', seq, spec })
    })
  }

  private rejectPending(error: Error): void {
    this.pendingRestore?.reject(error)
    this.pendingRestore = null
    if (this.pendingCommit) {
      const commit = this.pendingCommit
      this.pendingCommit = null
      commit.reject(error)
    }
    for (const [id, pending] of this.pendingSearches.entries()) {
      this.pendingSearches.delete(id)
      pending.reject(error)
    }
  }

  private handleWorkerFatal(err: unknown): void {
    if (this.workerFailed) return
    const error = err instanceof Error ? err : new Error(String(err))
    console.warn('[LexicalSearch] worker failed, switching to lazy fallback:', error)
    this.workerFailed = true
    this.workerReady = false
    this.indexingPromise = null
    this.rejectPending(error)
    if (this.worker) {
      this.worker.terminate()
      this.worker = null
    }
  }

  private loadFallbackModule(): Promise<FallbackModule> {
    this.fallbackModulePromise ??= import('./lexical-search-index').catch(err => {
      this.fallbackModulePromise = null
      throw err
    })
    return this.fallbackModulePromise
  }

  private ensureFallbackIndexed(data: Record<string, ContentDetails>): Promise<void> {
    if (this.fallbackIndex && this.fallbackIndexedData === data) {
      this.deferredFallbackData = null
      return Promise.resolve()
    }
    if (this.fallbackIndexingPromise && this.fallbackIndexedData === data) {
      return this.fallbackIndexingPromise
    }

    this.fallbackIndexedData = data
    const task = (async () => {
      const mod = await this.loadFallbackModule()
      const nextIndex = mod.createLexicalDocumentIndex()
      let id = 0
      // FlexSearch addAsync yields via timer per document; a single library call per document is not mid-call preemptible.
      for (const [slug, fileData] of Object.entries<ContentDetails>(data)) {
        const item = prepareLexicalDocument(id, slug, fileData)
        await nextIndex.addAsync(item)
        id += 1
      }
      this.fallbackIndex = nextIndex
      this.deferredFallbackData = null
    })().catch(err => {
      this.fallbackIndexingPromise = null
      this.fallbackIndexedData = null
      this.fallbackIndex = null
      throw err
    })

    this.fallbackIndexingPromise = task
    return task
  }
}

let sharedLexicalClient: LexicalSearchClient | null = null
export function getLexicalSearchClient(): LexicalSearchClient {
  sharedLexicalClient ??= new LexicalSearchClient()
  return sharedLexicalClient
}
