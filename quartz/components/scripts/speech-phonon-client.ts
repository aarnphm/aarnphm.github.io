import type { LoadProgress, ModelSpec, PhononTTS } from 'phonon-tts'
import { PHONON_MODELS, type PhononModelId, type SpeechAudio } from '../../util/speech-protocol'

// phonon-tts keys its Cache API entries by URL in this cache; every model URL carries the revision.
const PHONON_CACHE_NAME = 'phonon-tts-v1'
const RUNTIME_BASE = '/static/speech/phonon/'

function modelSpec(id: PhononModelId): ModelSpec {
  const base = new URL(PHONON_MODELS[id].baseUrl, window.location.href)
  const file = (name: string) => new URL(name, base).href
  return {
    weights: { q8: file('model.q8.gguf') },
    tokenizer: file('tokenizer.json'),
    config: file('config.json'),
    voices: { default: file('default-voice.safetensors') },
    defaultVoice: 'default',
  }
}

async function isCached(spec: ModelSpec): Promise<boolean> {
  if (!('caches' in window) || !(await caches.has(PHONON_CACHE_NAME))) return false
  const cache = await caches.open(PHONON_CACHE_NAME)
  const urls = [spec.weights.q8, spec.tokenizer, spec.config, ...Object.values(spec.voices ?? {})]
  const hits = await Promise.all(urls.map(url => (url ? cache.match(String(url)) : undefined)))
  return hits.every(Boolean)
}

/**
 * Same surface as SpeechClient. phonon-tts runs inference in its own module worker; a nested
 * worker inside speech.worker.js cannot load it in the app's Chromium, so this owns it from the
 * page. The package is imported from the emitted runtime only on the first playback request.
 */
export class PhononClient {
  private tts: Promise<PhononTTS> | undefined
  private loaded: PhononTTS | undefined
  private disposed = false

  constructor(
    private id: PhononModelId,
    private onProgress: (message: string) => void = () => {},
  ) {}

  get isReady(): boolean {
    return this.loaded !== undefined
  }

  async warmCached(): Promise<void> {
    if (this.disposed || this.tts) return
    try {
      if (await isCached(modelSpec(this.id))) await this.initialize().catch(() => {})
    } catch {
      // Missing or unavailable storage must not trigger an unsolicited model download.
    }
  }

  private initialize(): Promise<PhononTTS> {
    this.tts ??= this.load().then(
      tts => {
        if (this.disposed) {
          tts.dispose()
          throw new Error('Le moteur vocal a été fermé.')
        }
        this.loaded = tts
        return tts
      },
      error => {
        this.tts = undefined
        throw error
      },
    )
    return this.tts
  }

  private async load(): Promise<PhononTTS> {
    const runtime = new URL(RUNTIME_BASE, window.location.origin)
    const module: typeof import('phonon-tts') = await import(new URL('index.js', runtime).href)
    let lastUpdate = 0
    const onProgress = ({ loaded, total, cached }: LoadProgress) => {
      if (!cached && total && loaded < total && performance.now() - lastUpdate < 150) return
      lastUpdate = performance.now()
      const percent = total && !cached ? ` (${Math.round((loaded / total) * 100)} %)` : ''
      this.onProgress(
        cached ? 'Chargement de la voix depuis le cache…' : `Téléchargement de la voix${percent}…`,
      )
    }
    return module.PhononTTS.load({
      lang: 'fr',
      model: modelSpec(this.id),
      quant: 'q8',
      voices: ['default'],
      device: 'auto',
      onProgress,
      workerUrl: new URL('worker.js', runtime),
      wasmUrl: new URL('wasm/phonon_tts_bg.wasm', runtime),
      simdWasmUrl: new URL('wasm/phonon_tts_simd_bg.wasm', runtime),
      threadsWasmUrl: new URL('wasm-threads/phonon_tts_bg.wasm', runtime),
      threadsSimdWasmUrl: new URL('wasm-threads/phonon_tts_simd_bg.wasm', runtime),
    })
  }

  async synthesize(text: string): Promise<SpeechAudio> {
    if (this.disposed) throw new Error('Le moteur vocal a été fermé.')
    const trimmed = text.trim()
    if (!trimmed) throw new Error('Le texte à prononcer est vide.')
    if (trimmed.length > 5_000)
      throw new Error('Sélectionnez un texte de moins de 5 000 caractères.')
    const tts = await this.initialize()
    this.onProgress('Prononciation en préparation…')
    const started = performance.now()
    let firstAudioMs: number | undefined
    const chunks: Float32Array[] = []
    const speech = tts.stream(trimmed)
    for await (const chunk of speech) {
      firstAudioMs ??= performance.now() - started
      chunks.push(chunk)
    }
    await speech.done
    const audio = new Float32Array(chunks.reduce((size, chunk) => size + chunk.length, 0))
    let offset = 0
    for (const chunk of chunks) {
      audio.set(chunk, offset)
      offset += chunk.length
    }
    if (!audio.length) throw new Error('Le modèle a produit un audio invalide.')
    const totalMs = performance.now() - started
    const info = PHONON_MODELS[this.id]
    return {
      audio,
      samplingRate: tts.sampleRate,
      model: info.model,
      voice: info.voice,
      timing: {
        firstAudioMs: firstAudioMs ?? totalMs,
        totalMs,
        device: tts.device === 'webgpu' ? 'webgpu' : `cpu×${tts.threads}`,
      },
    }
  }

  dispose(): void {
    if (this.disposed) return
    this.disposed = true
    this.loaded?.dispose()
    this.loaded = undefined
  }
}
