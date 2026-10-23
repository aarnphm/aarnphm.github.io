import { AutoConfig, AutoModel, AutoProcessor, RawImage, Tensor } from '@huggingface/transformers'
import { isRecord } from '../../util/type-guards'
import { prepareImage } from './prepare-image'

export interface LocalImageEncoder {
  image(source: Uint8Array): Promise<Float32Array>
  query(text: string): Promise<Float32Array>
  dispose(): Promise<void>
}
import { IMAGE_ENCODER_CONTRACT, normalizeImageVector } from './store'

function disposeTensors(value: unknown): void {
  if (!isRecord(value)) return
  for (const tensor of Object.values(value)) if (tensor instanceof Tensor) tensor.dispose()
}

function embeddingFrom(outputs: unknown): Float32Array {
  if (!isRecord(outputs) || !(outputs.sentence_embedding instanceof Tensor))
    throw new Error('Image encoder did not return a sentence embedding.')
  const data = outputs.sentence_embedding.data
  if (!(data instanceof Float32Array))
    throw new Error('Image encoder returned a non-fp32 embedding.')
  return normalizeImageVector(data)
}

export async function createImageEncoder(threads: number) {
  const { repo, onnxRevision: revision } = IMAGE_ENCODER_CONTRACT
  const config = await AutoConfig.from_pretrained(repo, { revision })
  Object.assign(config, { audio_config: null })
  const processor = await AutoProcessor.from_pretrained(repo, { revision })
  const model = await AutoModel.from_pretrained(repo, {
    config,
    revision,
    dtype: 'q4',
    device: 'cpu',
    session_options: { intraOpNumThreads: threads },
  })
  async function embed(inputs: unknown): Promise<Float32Array> {
    let outputs: unknown
    try {
      outputs = await model(inputs)
      return embeddingFrom(outputs)
    } finally {
      disposeTensors(inputs)
      disposeTensors(outputs)
    }
  }
  return {
    async image(source: Uint8Array): Promise<Float32Array> {
      const { data, info } = await prepareImage(source)
      return embed(
        await processor(null, new RawImage(new Uint8Array(data), info.width, info.height, 3)),
      )
    },
    async query(text: string): Promise<Float32Array> {
      return embed(await processor(`task: search result | query: ${text}`))
    },
    async dispose(): Promise<void> {
      await model.dispose()
    },
  }
}
