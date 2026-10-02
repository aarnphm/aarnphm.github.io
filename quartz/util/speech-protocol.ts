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

export type SpeechOptions = {
  /** A mirror containing onnx/ and voice_styles/ from the pinned model revision. */
  modelBaseUrl?: string
  /** Warm an already-cached model without downloading missing assets. */
  cacheOnly?: boolean
}

export type SpeechAudio = {
  audio: Float32Array<ArrayBuffer>
  samplingRate: number
  model: string
  voice: string
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
