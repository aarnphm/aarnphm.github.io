import { FrenchSpeechModel } from '../util/speech-model'
import {
  SPEECH_MODEL_INFO,
  type SpeechRequest,
  type SpeechResponse,
  type SpeechProgress,
} from '../util/speech-protocol'

let model: FrenchSpeechModel | undefined
let queue = Promise.resolve()

function send(message: SpeechResponse): void {
  self.postMessage(message)
}

function progress(seq: number): (update: Omit<SpeechProgress, 'type' | 'seq'>) => void {
  return update => send({ type: 'progress', seq, ...update })
}

async function handle(message: SpeechRequest): Promise<void> {
  try {
    if (message.type === 'init') {
      model ??= await FrenchSpeechModel.load(message.options, progress(message.seq))
      send({ type: 'ready', seq: message.seq })
      return
    }
    if (!model) throw new Error("Le modèle vocal n'est pas encore chargé.")
    const audio = await model.synthesize(message.text, progress(message.seq))
    const result: SpeechResponse = {
      type: 'result',
      seq: message.seq,
      audio,
      samplingRate: model.samplingRate,
      model: SPEECH_MODEL_INFO.model,
      voice: SPEECH_MODEL_INFO.voice,
    }
    self.postMessage(result, { transfer: [audio.buffer] })
  } catch (error) {
    send({
      type: 'error',
      seq: message.seq,
      message: error instanceof Error ? error.message : String(error),
    })
  }
}

self.onmessage = (event: MessageEvent<SpeechRequest>): void => {
  queue = queue.then(() => handle(event.data))
}
