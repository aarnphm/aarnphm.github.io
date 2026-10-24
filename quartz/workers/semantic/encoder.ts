import { env, AutoConfig, AutoModel, AutoTokenizer } from '@huggingface/transformers'
import { IMAGE_EMBEDDING_REVISION } from '../../util/image-search'
import { SEMANTIC_ONNX_MODELS, type SemanticQueryDType } from '../../util/semantic-search'
import { isRecord } from '../../util/type-guards'

export type QueryEncoder = {
  readonly model: string
  /** Unit query vector cut to `dims` (Matryoshka models keep the leading dims). */
  embed(text: string, dims: number): Promise<Float32Array>
}

type Tokenizer = Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>

const QUERY_DTYPES: readonly SemanticQueryDType[] = ['q4', 'q8', 'fp32']

function resolveQueryDType(value: unknown): SemanticQueryDType {
  return QUERY_DTYPES.find(dtype => dtype === value) ?? 'q4'
}

let envConfigured = false

function configureRuntimeEnv() {
  if (envConfigured) return
  env.allowLocalModels = false
  env.allowRemoteModels = true
  // WebGPU runs the query encoder; keep the ONNX WASM orchestrator single-threaded.
  const onnxWasm = env.backends.onnx.wasm
  if (onnxWasm) onnxWasm.numThreads = 1
  envConfigured = true
}

function pruneTokenizerMemory(loadedTokenizer: Tokenizer) {
  loadedTokenizer._tokenizerJSON = null
  const inner: unknown = loadedTokenizer._tokenizer
  if (isRecord(inner)) {
    inner.tokenizer = null
    const innerModel = inner.model
    if (isRecord(innerModel)) {
      innerModel.config = null
      innerModel.merges = null
      innerModel.vocab = null
    }
  }
}

function queryPrompt(modelId: string, text: string): string {
  const modelName = modelId.toLowerCase()
  if (modelName.includes('e5')) return `query: ${text}`
  if (modelName.includes('qwen') && modelName.includes('embedding')) {
    const task = 'Given a web search query, retrieve relevant passages that answer the query'
    return `Instruct: ${task}\nQuery: ${text}`
  }
  if (modelName.includes('embeddinggemma')) return `task: search result | query: ${text}`
  return text
}

function disposeTensors(...maps: unknown[]) {
  for (const map of maps) {
    if (!isRecord(map)) continue
    for (const value of Object.values(map)) {
      if (isRecord(value) && typeof value.dispose === 'function') {
        try {
          value.dispose()
        } catch {}
      }
    }
  }
}

export async function loadQueryEncoder(
  modelId: string,
  queryDtype: unknown,
): Promise<QueryEncoder> {
  const mappedModel = SEMANTIC_ONNX_MODELS[modelId] ?? modelId
  const revision = modelId === 'google/embeddinggemma-2' ? IMAGE_EMBEDDING_REVISION : 'main'
  configureRuntimeEnv()
  // Every quantized embeddinggemma graph stores its token table with GatherBlockQuantized.
  // The wasm backend has no kernel for it (session creation fails), and the fp32 graph is over 1 GB.
  // Without WebGPU the search stays lexical.
  if (!(await navigator.gpu?.requestAdapter().catch(() => null))) {
    throw new Error('semantic query encoder needs WebGPU')
  }
  const dtype = resolveQueryDType(queryDtype)
  const config = await AutoConfig.from_pretrained(mappedModel, { revision })
  // embeddinggemma-2 ships vision and audio encoders (1.8 GB at fp32); queries only need the text graph.
  if (config.model_type === 'embedding_gemma2') {
    Object.assign(config, { vision_config: null, audio_config: null })
  }
  const [tokenizer, model] = await Promise.all([
    AutoTokenizer.from_pretrained(mappedModel, { revision }).then(loaded => {
      pruneTokenizerMemory(loaded)
      return loaded
    }),
    AutoModel.from_pretrained(mappedModel, { config, dtype, device: 'webgpu', revision }),
  ])
  const onnxWasm = env.backends.onnx.wasm
  if (onnxWasm) onnxWasm.wasmBinary = undefined

  return {
    model: modelId,
    async embed(text, dims) {
      const inputs = tokenizer([queryPrompt(modelId, text)], { padding: true })
      const outputs = await model(inputs)
      try {
        let data: ArrayLike<number>
        if (outputs.sentence_embedding) {
          data = outputs.sentence_embedding.data
        } else if (outputs.last_hidden_state) {
          const lastHidden = outputs.last_hidden_state
          const attentionMask = inputs.attention_mask
          const [, seqLen, hiddenSize] = lastHidden.dims
          const pooled = new Float32Array(hiddenSize)
          for (let i = 0; i < hiddenSize; i++) {
            let sum = 0
            let count = 0
            for (let j = 0; j < seqLen; j++) {
              if (attentionMask.data[j] > 0) {
                sum += lastHidden.data[j * hiddenSize + i]
                count++
              }
            }
            pooled[i] = count > 0 ? sum / count : 0
          }
          data = pooled
        } else {
          throw new Error('unsupported model output format')
        }
        if (data.length < dims) {
          throw new Error(`model emits ${data.length} dims but the index stores ${dims}`)
        }
        const vec = new Float32Array(dims)
        let norm = 0
        for (let i = 0; i < dims; i++) {
          vec[i] = data[i]
          norm += vec[i] * vec[i]
        }
        norm = Math.sqrt(norm)
        if (norm > 0) for (let i = 0; i < dims; i++) vec[i] /= norm
        return vec
      } finally {
        disposeTensors(outputs, inputs)
      }
    },
  }
}
