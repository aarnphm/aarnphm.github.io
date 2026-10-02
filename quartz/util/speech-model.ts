import { env, InferenceSession, Tensor } from 'onnxruntime-web/wasm'
import { SPEECH_MODEL_INFO, type SpeechOptions, type SpeechProgress } from './speech-protocol'
import { isRecord } from './type-guards'

// Adapted from Supertone's MIT browser inference example at commit 1e9799e964ea4c0dad7cde993b65c3c813a7b373.
// https://github.com/supertone-oss-archive/supertonic/blob/1e9799e964ea4c0dad7cde993b65c3c813a7b373/web/helper.js
// The model weights have a separate OpenRAIL-M license; see speech-model/LICENSE for code attribution.

type ModelConfig = {
  sampleRate: number
  baseChunkSize: number
  chunkCompressFactor: number
  latentDim: number
}

type VoiceStyle = { ttl: Tensor; dp: Tensor }
type Sessions = {
  duration: InferenceSession
  text: InferenceSession
  vector: InferenceSession
  vocoder: InferenceSession
}
type ProgressCallback = (progress: Omit<SpeechProgress, 'type' | 'seq'>) => void

const CACHE_NAME = `garden-speech-${SPEECH_MODEL_INFO.revision}-ort-${env.versions.web}`
const MODEL_BASE = `https://huggingface.co/${SPEECH_MODEL_INFO.model}/resolve/${SPEECH_MODEL_INFO.revision}/`
const INFERENCE_STEPS = 5
const MAX_CHUNK_LENGTH = 300

function positiveInteger(value: unknown): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1) {
    throw new Error('Le modèle vocal contient une configuration invalide.')
  }
  return value
}

function parseConfig(value: unknown): ModelConfig {
  if (!isRecord(value) || !isRecord(value.ae) || !isRecord(value.ttl)) {
    throw new Error('La configuration du modèle vocal est invalide.')
  }
  return {
    sampleRate: positiveInteger(value.ae.sample_rate),
    baseChunkSize: positiveInteger(value.ae.base_chunk_size),
    chunkCompressFactor: positiveInteger(value.ttl.chunk_compress_factor),
    latentDim: positiveInteger(value.ttl.latent_dim),
  }
}

function parseIndexer(value: unknown): number[] {
  if (
    !Array.isArray(value) ||
    !value.every((item: unknown) => typeof item === 'number' && Number.isInteger(item))
  ) {
    throw new Error("L'index de caractères du modèle vocal est invalide.")
  }
  return value
}

function flattenNumbers(value: unknown, output: number[]): void {
  if (Array.isArray(value)) {
    for (const item of value) flattenNumbers(item, output)
  } else if (typeof value === 'number' && Number.isFinite(value)) {
    output.push(value)
  } else {
    throw new Error('Le style vocal contient une valeur invalide.')
  }
}

function parseStyleTensor(value: unknown): Tensor {
  if (!isRecord(value) || !Array.isArray(value.dims) || value.dims.length !== 3) {
    throw new Error('Le style vocal est invalide.')
  }
  const dims = value.dims.map(positiveInteger)
  const values: number[] = []
  flattenNumbers(value.data, values)
  if (dims[0] !== 1 || dims.reduce((size, dim) => size * dim, 1) !== values.length) {
    throw new Error('Les dimensions du style vocal sont invalides.')
  }
  return new Tensor('float32', Float32Array.from(values), dims)
}

function parseStyle(value: unknown): VoiceStyle {
  if (!isRecord(value)) throw new Error('Le style vocal est invalide.')
  return { ttl: parseStyleTensor(value.style_ttl), dp: parseStyleTensor(value.style_dp) }
}

function floatData(tensor: Tensor | undefined, name: string): Float32Array {
  if (!tensor || !(tensor.data instanceof Float32Array)) {
    throw new Error(`La sortie vocale ${name} est invalide.`)
  }
  return tensor.data
}

function normalizeFrench(text: string): string {
  let normalized = text
    .normalize('NFKD')
    .replace(/[\p{Extended_Pictographic}\p{Regional_Indicator}]/gu, '')
    .replace(/[–‑—]/g, '-')
    .replace(/[“”]/g, '"')
    .replace(/[‘’´`]/g, "'")
    .replace(/[_[\]|/#→←]/g, ' ')
    .replace(/[♥☆♡©\\]/g, '')
    .replace(/\s+/g, ' ')
    .replace(/\s+([,.!?;:'])/g, '$1')
    .trim()
  if (!normalized) throw new Error('Le texte à prononcer est vide.')
  if (!/[.!?;:,'"')\]}…。」』】〉》›»]$/.test(normalized)) normalized += '.'
  return `<fr>${normalized}</fr>`
}

function chunkFrench(text: string): string[] {
  if (text.length > 5_000) throw new Error('Sélectionnez un texte de moins de 5 000 caractères.')
  const sentences = new Intl.Segmenter('fr', { granularity: 'sentence' }).segment(text.trim())
  const chunks: string[] = []
  let current = ''
  for (const { segment } of sentences) {
    for (const word of segment.trim().split(/\s+/)) {
      if (!word) continue
      if (word.length > MAX_CHUNK_LENGTH) {
        throw new Error('Un mot est trop long pour le modèle vocal.')
      }
      if (current.length + word.length + 1 > MAX_CHUNK_LENGTH) {
        chunks.push(current)
        current = ''
      }
      current += `${current ? ' ' : ''}${word}`
    }
  }
  if (current) chunks.push(current)
  if (!chunks.length) throw new Error('Le texte à prononcer est vide.')
  return chunks
}

class ModelAssets {
  private cache: Cache | undefined
  private base: URL

  constructor(
    private options: SpeechOptions,
    private progress: ProgressCallback,
  ) {
    this.base = new URL(options.modelBaseUrl ?? MODEL_BASE, self.location.href)
    if (!this.base.pathname.endsWith('/')) this.base.pathname += '/'
  }

  async open(): Promise<void> {
    if (!('caches' in globalThis)) return
    try {
      this.cache = await caches.open(CACHE_NAME)
    } catch {
      this.progress({ message: 'Cache indisponible. Le modèle reste utilisable pour cette page.' })
    }
  }

  async json(path: string): Promise<unknown> {
    const bytes = await this.fetch(new URL(path, this.base), path)
    return JSON.parse(new TextDecoder().decode(bytes))
  }

  async model(path: string): Promise<InferenceSession> {
    const bytes = await this.fetch(new URL(`onnx/${path}`, this.base), path)
    this.progress({ message: 'Préparation de la voix…', asset: path })
    return InferenceSession.create(bytes, { executionProviders: ['wasm'] })
  }

  async wasm(): Promise<Uint8Array<ArrayBuffer>> {
    return this.fetch(
      new URL('static/speech/ort-wasm-simd-threaded.wasm', self.location.href),
      'moteur WASM',
    )
  }

  private async fetch(url: URL, asset: string): Promise<Uint8Array<ArrayBuffer>> {
    const cached = await this.cache?.match(url.href)
    if (cached) {
      this.progress({ message: 'Chargement de la voix depuis le cache…', asset, cached: true })
      return new Uint8Array(await cached.arrayBuffer())
    }
    if (this.options.cacheOnly)
      throw new Error('La voix doit être téléchargée à la première lecture.')
    this.progress({ message: 'Téléchargement de la voix…', asset, cached: false })
    const response = await fetch(url, { credentials: 'omit' })
    if (!response.ok) {
      throw new Error(`Téléchargement du modèle impossible (${asset}, HTTP ${response.status}).`)
    }
    const length = Number(response.headers.get('content-length'))
    const totalBytes = Number.isFinite(length) && length > 0 ? length : undefined
    const reader = response.body?.getReader()
    const chunks: Uint8Array[] = []
    let loadedBytes = 0
    if (reader) {
      let lastUpdate = 0
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        chunks.push(value)
        loadedBytes += value.byteLength
        if (performance.now() - lastUpdate > 150) {
          const percent = totalBytes ? ` (${Math.round((loadedBytes / totalBytes) * 100)} %)` : ''
          this.progress({
            message: `Téléchargement de la voix${percent}…`,
            asset,
            loadedBytes,
            totalBytes,
            cached: false,
          })
          lastUpdate = performance.now()
        }
      }
    } else {
      const bytes = new Uint8Array(await response.arrayBuffer())
      chunks.push(bytes)
      loadedBytes = bytes.byteLength
    }
    const bytes = new Uint8Array(loadedBytes)
    let offset = 0
    for (const chunk of chunks) {
      bytes.set(chunk, offset)
      offset += chunk.byteLength
    }
    if (this.cache) {
      try {
        await this.cache.put(url.href, new Response(bytes, { headers: response.headers }))
      } catch {
        this.progress({
          message: 'Stockage du cache insuffisant. Le téléchargement reste utilisable.',
        })
      }
    }
    return bytes
  }
}

export class FrenchSpeechModel {
  private constructor(
    private config: ModelConfig,
    private indexer: number[],
    private style: VoiceStyle,
    private sessions: Sessions,
  ) {}

  static async load(
    options: SpeechOptions,
    progress: ProgressCallback,
  ): Promise<FrenchSpeechModel> {
    const assets = new ModelAssets(options, progress)
    await assets.open()
    env.wasm.proxy = false
    env.wasm.numThreads = 1
    env.wasm.wasmBinary = await assets.wasm()
    const config = parseConfig(await assets.json('onnx/tts.json'))
    const indexer = parseIndexer(await assets.json('onnx/unicode_indexer.json'))
    const style = parseStyle(await assets.json(`voice_styles/${SPEECH_MODEL_INFO.voice}.json`))
    const opened: InferenceSession[] = []
    try {
      const duration = await assets.model('duration_predictor.onnx')
      opened.push(duration)
      const text = await assets.model('text_encoder.onnx')
      opened.push(text)
      const vector = await assets.model('vector_estimator.onnx')
      opened.push(vector)
      const vocoder = await assets.model('vocoder.onnx')
      return new FrenchSpeechModel(config, indexer, style, { duration, text, vector, vocoder })
    } catch (error) {
      await Promise.allSettled(opened.map(session => session.release()))
      throw error
    }
  }

  async synthesize(text: string, progress: ProgressCallback): Promise<Float32Array<ArrayBuffer>> {
    const chunks = chunkFrench(text)
    const outputs: Float32Array[] = []
    const silenceSamples = Math.floor(this.config.sampleRate * 0.3)
    for (let index = 0; index < chunks.length; index++) {
      progress({ message: `Prononciation en préparation (${index + 1}/${chunks.length})…` })
      outputs.push(await this.infer(chunks[index]))
    }
    const total = outputs.reduce((size, output) => size + output.length, 0)
    const audio = new Float32Array(total + silenceSamples * (outputs.length - 1))
    let offset = 0
    for (const output of outputs) {
      audio.set(output, offset)
      offset += output.length + silenceSamples
    }
    return audio
  }

  get samplingRate(): number {
    return this.config.sampleRate
  }

  private async infer(text: string): Promise<Float32Array<ArrayBuffer>> {
    const characters = Array.from(normalizeFrench(text))
    const ids = BigInt64Array.from(characters, character => {
      const point = character.codePointAt(0)
      const value = point === undefined ? undefined : this.indexer[point]
      if (value === undefined || value < 0) {
        throw new Error(`Le modèle vocal ne reconnaît pas le caractère « ${character} ».`)
      }
      return BigInt(value)
    })
    const textIds = new Tensor('int64', ids, [1, ids.length])
    const textMask = new Tensor('float32', new Float32Array(ids.length).fill(1), [1, 1, ids.length])
    const predicted = await this.sessions.duration.run({
      text_ids: textIds,
      text_mask: textMask,
      style_dp: this.style.dp,
    })
    const duration = floatData(predicted.duration, 'duration')[0]
    if (!Number.isFinite(duration) || duration <= 0 || duration > 60) {
      throw new Error('Le modèle a produit une durée vocale invalide.')
    }
    const encoded = await this.sessions.text.run({
      text_ids: textIds,
      text_mask: textMask,
      style_ttl: this.style.ttl,
    })
    const textEmb = encoded.text_emb
    if (!textEmb) throw new Error("L'encodage du texte a échoué.")
    const sampleCount = Math.floor(duration * this.config.sampleRate)
    const chunkSize = this.config.baseChunkSize * this.config.chunkCompressFactor
    const latentLength = Math.ceil(sampleCount / chunkSize)
    const latentChannels = this.config.latentDim * this.config.chunkCompressFactor
    const shape = [1, latentChannels, latentLength]
    const noise = new Float32Array(latentLength * latentChannels)
    for (let index = 0; index < noise.length; index++) {
      const u1 = Math.max(0.0001, Math.random())
      noise[index] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * Math.random())
    }
    let latent = new Tensor('float32', noise, shape)
    const latentMask = new Tensor('float32', new Float32Array(latentLength).fill(1), [
      1,
      1,
      latentLength,
    ])
    const totalStep = new Tensor('float32', Float32Array.of(INFERENCE_STEPS), [1])
    for (let step = 0; step < INFERENCE_STEPS; step++) {
      const denoised = await this.sessions.vector.run({
        noisy_latent: latent,
        text_emb: textEmb,
        style_ttl: this.style.ttl,
        latent_mask: latentMask,
        text_mask: textMask,
        current_step: new Tensor('float32', Float32Array.of(step), [1]),
        total_step: totalStep,
      })
      latent = new Tensor('float32', floatData(denoised.denoised_latent, 'denoised_latent'), shape)
    }
    const result = await this.sessions.vocoder.run({ latent })
    const audio = floatData(result.wav_tts, 'wav_tts').slice(0, sampleCount)
    if (!audio.length || !audio.every(Number.isFinite)) {
      throw new Error('Le modèle a produit un audio invalide.')
    }
    return new Float32Array(audio)
  }
}
