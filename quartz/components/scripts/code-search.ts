import type { CodeWorkerRequest, CodeWorkerResponse } from '../../util/code-search-protocol'
import {
  CODE_SEARCH_MANIFEST,
  type CodeQuery,
  type CodeSearchProgress,
  type CodeSearchResult,
} from '../../util/code-search-types'

export interface CodeSearchClientOptions {
  manifestUrl?: string
  workerUrl?: string
  idleMs?: number
  timeoutMs?: number
}

type Pending = {
  id: number
  resolve: (result: CodeSearchResult) => void
  reject: (error: Error) => void
  progress?: (value: CodeSearchProgress) => void
}

export class CodeSearchClient {
  private worker: Worker | undefined
  private pending: Pending | undefined
  private sequence = 0
  private disposed = false
  private timer: ReturnType<typeof setTimeout> | undefined
  constructor(private readonly options: CodeSearchClientOptions = {}) {}

  search(
    query: CodeQuery,
    progress?: (value: CodeSearchProgress) => void,
  ): Promise<CodeSearchResult> {
    if (this.disposed) return Promise.reject(new Error('Code search has been disposed'))
    if (this.pending) this.cancel()
    this.clearTimer()
    return new Promise((resolve, reject) => {
      const id = ++this.sequence
      this.pending = { id, resolve, reject, progress }
      try {
        const worker = this.ensureWorker()
        const request: CodeWorkerRequest = {
          type: 'search',
          id,
          query,
          manifestUrl: new URL(this.options.manifestUrl ?? CODE_SEARCH_MANIFEST, location.href)
            .href,
        }
        worker.postMessage(request)
        this.timer = setTimeout(
          () => this.fail(new Error('Code search exceeded its execution limit')),
          this.options.timeoutMs ?? 60_000,
        )
      } catch (error) {
        this.fail(error instanceof Error ? error : new Error(String(error)))
      }
    })
  }

  private ensureWorker(): Worker {
    if (this.worker) return this.worker
    const worker = new Worker(
      this.options.workerUrl ?? new URL('code-search.worker.js', import.meta.url),
      { type: 'module' },
    )
    worker.addEventListener('message', (event: MessageEvent<CodeWorkerResponse>) => {
      if (worker !== this.worker || event.data.id !== this.pending?.id) return
      const message = event.data
      if (message.type === 'progress') {
        this.pending.progress?.(message.progress)
        return
      }
      const pending = this.pending
      this.pending = undefined
      this.clearTimer()
      if (message.type === 'result') pending.resolve(message.result)
      else {
        const error = new Error(message.message)
        error.name = message.name
        pending.reject(error)
      }
      this.timer = setTimeout(() => this.stopWorker(), this.options.idleMs ?? 30_000)
    })
    worker.addEventListener('error', event => {
      if (worker === this.worker) this.fail(new Error(event.message || 'Code search worker failed'))
    })
    worker.addEventListener('messageerror', () => {
      if (worker === this.worker) this.fail(new Error('Invalid code search worker response'))
    })
    this.worker = worker
    return worker
  }

  private clearTimer(): void {
    if (this.timer !== undefined) clearTimeout(this.timer)
    this.timer = undefined
  }
  private stopWorker(): void {
    this.clearTimer()
    this.worker?.terminate()
    this.worker = undefined
  }
  private fail(error: Error): void {
    const pending = this.pending
    this.pending = undefined
    this.stopWorker()
    pending?.reject(error)
  }

  cancelPending(): void {
    if (this.pending) this.cancel()
  }

  cancel(): void {
    this.fail(new DOMException('Code search superseded', 'AbortError'))
  }
  dispose(): void {
    if (this.disposed) return
    this.disposed = true
    this.cancel()
  }
}
