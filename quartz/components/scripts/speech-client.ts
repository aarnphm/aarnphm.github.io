import type {
  SpeechAudio,
  SpeechOptions,
  SpeechRequest,
  SpeechResponse,
} from '../../util/speech-protocol'
import { SPEECH_MODEL_INFO } from '../../util/speech-protocol'

type Pending = { resolve: (audio: SpeechAudio) => void; reject: (error: Error) => void }

export class SpeechClient {
  private worker: Worker | undefined
  private ready: Promise<void> | undefined
  private resolveReady: (() => void) | undefined
  private rejectReady: ((error: Error) => void) | undefined
  private pending = new Map<number, Pending>()
  private seq = 0
  private disposed = false
  private loaded = false
  private warmup: Promise<void> | undefined

  constructor(
    private onProgress: (message: string) => void = () => {},
    private options: SpeechOptions = {},
  ) {}

  get isReady(): boolean {
    return this.loaded
  }

  async warmCached(): Promise<void> {
    if (this.disposed || this.ready || !('caches' in window)) return
    try {
      const names = await caches.keys()
      if (!names.some(name => name.startsWith(`garden-speech-${SPEECH_MODEL_INFO.revision}-ort-`)))
        return
      if (this.disposed || this.ready) return
      this.warmup = this.initialize(true).catch(() => {})
      await this.warmup
    } catch {
      // Missing or unavailable storage must not trigger an unsolicited model download.
    } finally {
      this.warmup = undefined
    }
  }

  private initialize(cacheOnly = false): Promise<void> {
    if (this.ready) return this.ready
    const ready = new Promise<void>((resolve, reject) => {
      this.resolveReady = resolve
      this.rejectReady = reject
    })
    this.ready = ready
    try {
      this.worker = new Worker(new URL('/speech.worker.js', window.location.origin), {
        type: 'module',
      })
      this.worker.onmessage = (event: MessageEvent<SpeechResponse>) => this.receive(event.data)
      this.worker.onerror = event =>
        this.fail(new Error(event.message || 'Le moteur vocal a échoué.'))
      this.worker.onmessageerror = () =>
        this.fail(new Error('La réponse du moteur vocal est invalide.'))
      this.send({ type: 'init', seq: ++this.seq, options: { ...this.options, cacheOnly } })
    } catch (error) {
      this.fail(error instanceof Error ? error : new Error(String(error)))
    }
    return ready
  }

  private send(message: SpeechRequest): void {
    if (!this.worker) throw new Error('Le moteur vocal est indisponible.')
    this.worker.postMessage(message)
  }

  private receive(message: SpeechResponse): void {
    if (message.type === 'progress') {
      this.onProgress(message.message)
    } else if (message.type === 'ready') {
      this.loaded = true
      this.resolveReady?.()
      this.resolveReady = undefined
      this.rejectReady = undefined
    } else if (message.type === 'result') {
      const pending = this.pending.get(message.seq)
      this.pending.delete(message.seq)
      pending?.resolve(message)
    } else {
      const error = new Error(message.message)
      const pending = this.pending.get(message.seq)
      if (pending) {
        this.pending.delete(message.seq)
        pending.reject(error)
      } else {
        this.fail(error)
      }
    }
  }

  private fail(error: Error): void {
    this.loaded = false
    this.rejectReady?.(error)
    this.resolveReady = undefined
    this.rejectReady = undefined
    for (const pending of this.pending.values()) pending.reject(error)
    this.pending.clear()
    this.worker?.terminate()
    this.worker = undefined
    this.ready = undefined
  }

  async synthesize(text: string): Promise<SpeechAudio> {
    if (this.disposed) throw new Error('Le moteur vocal a été fermé.')
    await this.warmup
    if (this.disposed) throw new Error('Le moteur vocal a été fermé.')
    await this.initialize()
    if (this.disposed) throw new Error('Le moteur vocal a été fermé.')
    return new Promise((resolve, reject) => {
      const seq = ++this.seq
      this.pending.set(seq, { resolve, reject })
      try {
        this.send({ type: 'synthesize', seq, text })
      } catch (error) {
        this.pending.delete(seq)
        reject(error instanceof Error ? error : new Error(String(error)))
      }
    })
  }

  dispose(): void {
    if (this.disposed) return
    this.disposed = true
    this.fail(new Error('Le moteur vocal a été fermé.'))
  }
}
