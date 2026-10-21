export const SPEECH_MODEL_INFO = {
  label: 'Supertonic 2',
  model: 'supertone-oss-archive/supertonic-2',
  revision: '7edbcd0bba8cd579100a353a9e0b2b49cad7c875',
  voice: 'F1',
  locale: 'fr',
  modelDownloadBytes: 263_520_219,
  runtimeDownloadBytes: 14_239_897,
  downloadBytes: 277_760_116,
  sourceUrl: 'https://huggingface.co/supertone-oss-archive/supertonic-2',
  licenseUrl:
    'https://huggingface.co/supertone-oss-archive/supertonic-2/blob/7edbcd0bba8cd579100a353a9e0b2b49cad7c875/LICENSE',
}

export type PhononModelInfo = {
  label: string
  model: string
  revision: string
  voice: string
  /** Serves config.json, tokenizer.json, model.q8.gguf and default-voice.safetensors. */
  baseUrl: string
  modelDownloadBytes: number
  runtimeDownloadBytes: number
  downloadBytes: number
  sourceUrl: string
  license: string
}

// Chromium's path: the single-threaded relaxed-SIMD wasm plus the package's JS. Pages without
// cross-origin isolation never load the threaded build.
const PHONON_RUNTIME_BYTES = 4_387_046 + 96_812
const phononFileBytes = (weights: number, tokenizer: number, config: number) =>
  weights + tokenizer + config + 256_136

function phonon(
  label: string,
  repo: string,
  revision: string,
  modelDownloadBytes: number,
  license: string,
): PhononModelInfo {
  return {
    label,
    model: `gradium/${repo}`,
    revision,
    voice: 'default',
    baseUrl: `/models/speech/${repo}/${revision}/`,
    modelDownloadBytes,
    runtimeDownloadBytes: PHONON_RUNTIME_BYTES,
    downloadBytes: modelDownloadBytes + PHONON_RUNTIME_BYTES,
    sourceUrl: `https://huggingface.co/gradium/${repo}`,
    license,
  }
}

// The 40M repo is gated, so a browser cannot read it from the Hub. Its pinned files live in the
// LFS_BUCKET R2 bucket under models/speech/, served by the worker's /models/ route.
export const PHONON_MODELS = {
  'phonon-40m-fr': phonon(
    'Phonon 40M français',
    'phonon-40m-fr',
    '65b5b451d235a8c24245f51d2774ec9806a9443f',
    phononFileBytes(74_553_568, 277_749, 2_851),
    'CC BY-NC 4.0',
  ),
  'phonon-40m-multilingual': phonon(
    'Phonon 40M multilingue',
    'phonon-40m-multilingual',
    '1267486cbcb55d2eaa61ba521c2ed8e4d22923c2',
    phononFileBytes(74_553_568, 277_749, 2_891),
    'CC BY-NC 4.0',
  ),
} satisfies Record<string, PhononModelInfo>

export type PhononModelId = keyof typeof PHONON_MODELS
export type SpeechModelId = 'supertonic-2' | PhononModelId

// The multilingual checkpoint reads French with an English accent; the French one is the default.
export const DEFAULT_SPEECH_MODEL: SpeechModelId = 'phonon-40m-fr'
export const SPEECH_MODEL_IDS: SpeechModelId[] = [
  'phonon-40m-fr',
  'phonon-40m-multilingual',
  'supertonic-2',
]

export function isSpeechModelId(value: unknown): value is SpeechModelId {
  return SPEECH_MODEL_IDS.some(id => id === value)
}

export function speechModelLabel(id: SpeechModelId): string {
  return id === 'supertonic-2' ? SPEECH_MODEL_INFO.label : PHONON_MODELS[id].label
}

export type SpeechOptions = {
  /** A mirror containing onnx/ and voice_styles/ from the pinned Supertonic revision. */
  modelBaseUrl?: string
  /** Warm an already-cached model without downloading missing assets. */
  cacheOnly?: boolean
}

export type SpeechTiming = {
  /** From the request reaching the worker to the first PCM samples. */
  firstAudioMs: number
  totalMs: number
  /** Where generation ran, such as `webgpu` or `cpu×1`. */
  device: string
}

export type SpeechAudio = {
  audio: Float32Array<ArrayBuffer>
  samplingRate: number
  model: string
  voice: string
  timing: SpeechTiming
}

export type SpeechProgress = {
  type: 'progress'
  seq: number
  message: string
  asset?: string
  loadedBytes?: number
  totalBytes?: number
  cached?: boolean
}

export type SpeechRequest =
  | { type: 'init'; seq: number; options: SpeechOptions }
  | { type: 'synthesize'; seq: number; text: string }

export type SpeechResponse =
  | SpeechProgress
  | { type: 'ready'; seq: number }
  | ({ type: 'result'; seq: number } & SpeechAudio)
  | { type: 'error'; seq: number; message: string }
