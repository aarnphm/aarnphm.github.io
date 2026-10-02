import { SPEECH_MODEL_INFO } from '../../util/speech-protocol'
import { SpeechClient } from './speech-client'

const MAX_AUDIO_CACHE = 32
const MAX_STORED_AUDIO_BYTES = 2_000_000
const AUDIO_CACHE_NAME = `garden-speech-audio-${SPEECH_MODEL_INFO.revision}-${SPEECH_MODEL_INFO.voice}-v1`

function wavBlob(audio: Float32Array, samplingRate: number): Blob {
  const bytes = new ArrayBuffer(44 + audio.length * 2)
  const view = new DataView(bytes)
  const text = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index++)
      view.setUint8(offset + index, value.charCodeAt(index))
  }
  text(0, 'RIFF')
  view.setUint32(4, bytes.byteLength - 8, true)
  text(8, 'WAVE')
  text(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, samplingRate, true)
  view.setUint32(28, samplingRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  text(36, 'data')
  view.setUint32(40, audio.length * 2, true)
  for (let index = 0; index < audio.length; index++) {
    const sample = Math.max(-1, Math.min(1, audio[index]))
    view.setInt16(44 + index * 2, sample * (sample < 0 ? 32768 : 32767), true)
  }
  return new Blob([bytes], { type: 'audio/wav' })
}

class SpeechRuntime {
  onProgress: ((message: string) => void) | undefined
  readonly client: SpeechClient
  private audio = new Map<string, string>()
  private preparing = new Map<string, Promise<string>>()
  private storage: Promise<Cache | undefined> | undefined
  private disposed = false
  private prefetching = false

  constructor(readonly modelBaseUrl: string | undefined) {
    this.client = new SpeechClient(message => this.onProgress?.(message), { modelBaseUrl })
  }

  private openStorage(): Promise<Cache | undefined> {
    this.storage ??=
      'caches' in window
        ? caches.open(AUDIO_CACHE_NAME).catch(() => undefined)
        : Promise.resolve(undefined)
    return this.storage
  }

  private async audioKey(text: string): Promise<string> {
    const bytes = new TextEncoder().encode(JSON.stringify([this.modelBaseUrl ?? '', text]))
    const digest = await crypto.subtle.digest('SHA-256', bytes)
    const hash = Array.from(new Uint8Array(digest), byte =>
      byte.toString(16).padStart(2, '0'),
    ).join('')
    return new URL(`/static/speech/audio/${hash}.wav`, window.location.origin).href
  }

  private async storedAudio(key: string): Promise<Blob | undefined> {
    try {
      const cached = await (await this.openStorage())?.match(key)
      return cached?.blob()
    } catch {
      return undefined
    }
  }

  private async storeAudio(key: string, blob: Blob): Promise<void> {
    if (blob.size > MAX_STORED_AUDIO_BYTES) return
    try {
      const cache = await this.openStorage()
      if (!cache) return
      await cache.put(key, new Response(blob))
      const keys = await cache.keys()
      await Promise.all(
        keys.slice(0, Math.max(0, keys.length - MAX_AUDIO_CACHE)).map(key => cache.delete(key)),
      )
    } catch {
      // The memory cache remains usable when browser storage is unavailable or full.
    }
  }

  prepare(text: string): Promise<string> {
    const cached = this.audio.get(text)
    if (cached) {
      this.audio.delete(text)
      this.audio.set(text, cached)
      return Promise.resolve(cached)
    }
    const pending = this.preparing.get(text)
    if (pending) return pending
    const job = this.generate(text).finally(() => this.preparing.delete(text))
    this.preparing.set(text, job)
    return job
  }

  prefetch(text: string): void {
    if (!this.client.isReady || this.prefetching || this.preparing.size > 0 || this.audio.has(text))
      return
    this.prefetching = true
    void this.prepare(text)
      .catch(() => {})
      .finally(() => {
        this.prefetching = false
      })
  }

  private async generate(text: string): Promise<string> {
    const key = await this.audioKey(text)
    let blob = await this.storedAudio(key)
    if (!blob) {
      const result = await this.client.synthesize(text)
      blob = wavBlob(result.audio, result.samplingRate)
      void this.storeAudio(key, blob)
    }
    if (this.disposed) throw new Error('Le moteur vocal a été fermé.')
    const url = URL.createObjectURL(blob)
    this.audio.set(text, url)
    if (this.audio.size > MAX_AUDIO_CACHE) {
      const oldest = this.audio.entries().next().value
      if (oldest) {
        URL.revokeObjectURL(oldest[1])
        this.audio.delete(oldest[0])
      }
    }
    return url
  }

  dispose(): void {
    this.disposed = true
    this.client.dispose()
    for (const url of this.audio.values()) URL.revokeObjectURL(url)
    this.audio.clear()
  }
}

let sharedRuntime: SpeechRuntime | undefined

export function speechRuntime(modelBaseUrl?: string): SpeechRuntime {
  if (!sharedRuntime || sharedRuntime.modelBaseUrl !== modelBaseUrl) {
    sharedRuntime?.dispose()
    sharedRuntime = new SpeechRuntime(modelBaseUrl)
  }
  return sharedRuntime
}
