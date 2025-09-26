import { write } from "./helpers"
import { QuartzEmitterPlugin } from "../types"
import { FilePath, FullSlug } from "../../util/path"
import { ReadTimeResults } from "reading-time"
import { GlobalConfiguration } from "../../cfg"
import { computeModelLocalPath, ensureLocalModel } from "../../util/semantic"
import { encode } from "../../components/scripts/util"

const DEFAULT_MODEL_ID = "onnx-community/Qwen3-Embedding-0.6B-ONNX"

const defaults: GlobalConfiguration["semanticSearch"] = {
  enable: true,
  model: DEFAULT_MODEL_ID,
  dims: 1024,
  dtype: "fp32",
  shardSizeRows: 1024,
  hnsw: { M: 16, efConstruction: 200 },
  modelLocalPath: computeModelLocalPath(DEFAULT_MODEL_ID),
}

type ContentDetails = {
  slug: string
  title: string
  filePath: FilePath
  content: string
  readingTime?: Partial<ReadTimeResults>
}

export const SemanticIndex: QuartzEmitterPlugin<Partial<GlobalConfiguration["semanticSearch"]>> = (
  opts,
) => {
  const o = { ...defaults, ...opts }
  if (!o.model) {
    throw new Error("Semantic search requires a model identifier")
  }
  o.modelLocalPath ??= computeModelLocalPath(o.model)

  return {
    name: "SemanticIndex",
    getQuartzComponents() {
      return []
    },
    async *partialEmit() {},
    async *emit(ctx, content, _resources) {
      if (!o.enable) return

      const token =
        process.env.HF_TOKEN ??
        process.env.HUGGINGFACEHUB_API_TOKEN ??
        process.env.HUGGING_FACE_TOKEN
      try {
        const result = await ensureLocalModel({
          outputDir: ctx.argv.output,
          modelId: o.model!,
          token: token ?? undefined,
        })
        if (result.downloaded > 0) {
          console.info(
            `[SemanticIndex] staged ${result.downloaded} files (${result.skipped} cached) for ${o.model} @ ${result.revision}`,
          )
        }
      } catch (err) {
        const reason = err instanceof Error ? err.message : String(err)
        console.warn(`[SemanticIndex] failed to stage model assets for ${o.model}: ${reason}`)
      }

      const docs: ContentDetails[] = []
      for (const [_, file] of content) {
        const slug = file.data.slug!
        const title = file.data.frontmatter?.title ?? slug
        const text = file.data.text
        if (text) {
          docs.push({
            slug,
            title,
            filePath: file.data.filePath!,
            content: text,
            readingTime: file.data.readingTime,
          })
        }
      }
      // emit a JSONL with the exact text used for embeddings (one per line)
      // format: { slug, title, text }
      const jsonl = docs
        .map((d) => ({ slug: d.slug, title: d.title, text: d.content }))
        .map((o) => JSON.stringify(o))
        .join("\n")
      yield write({
        ctx,
        slug: "embeddings-text" as FullSlug,
        ext: ".jsonl",
        content: jsonl,
      })

      // Build BM25 postings
      const N = docs.length
      const docLen: number[] = new Array(N).fill(0)
      const postings = new Map<string, Map<number, number>>()
      const minDf = 2
      for (let i = 0; i < N; i++) {
        const tokens = encode(docs[i].content)
        docLen[i] = tokens.length
        const tf = new Map<string, number>()
        for (const token of tokens) {
          tf.set(token, (tf.get(token) ?? 0) + 1)
        }
        for (const [term, freq] of tf) {
          let bucket = postings.get(term)
          if (!bucket) {
            bucket = new Map<number, number>()
            postings.set(term, bucket)
          }
          bucket.set(i, freq)
        }
      }
      const avgdl = docLen.reduce((acc, len) => acc + len, 0) / Math.max(1, N)
      const postingsObj: Record<string, [number, number][]> = {}
      for (const [term, bucket] of postings) {
        if (bucket.size < minDf) continue
        postingsObj[term] = Array.from(bucket.entries()).map(([docId, freq]) => [docId, freq])
      }
      const bm25 = {
        N,
        avgdl,
        docLen,
        postings: postingsObj,
      }
      yield write({
        ctx,
        slug: "embeddings/bm25" as FullSlug,
        ext: ".json",
        content: JSON.stringify(bm25),
      })
    },
    externalResources(_ctx) {
      return {}
    },
  }
}
