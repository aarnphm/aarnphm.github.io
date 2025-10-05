import { write } from "./helpers"
import { QuartzEmitterPlugin } from "../types"
import { FilePath, FullSlug } from "../../util/path"
import { ReadTimeResults } from "reading-time"
import { GlobalConfiguration } from "../../cfg"

const DEFAULT_MODEL_ID = "onnx-community/Qwen3-Embedding-0.6B-ONNX"

const defaults: GlobalConfiguration["semanticSearch"] = {
  enable: true,
  model: DEFAULT_MODEL_ID,
  dims: 1024,
  dtype: "fp32",
  shardSizeRows: 1024,
  hnsw: { M: 16, efConstruction: 200 },
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

  return {
    name: "SemanticIndex",
    getQuartzComponents() {
      return []
    },
    async *partialEmit() {},
    async *emit(ctx, content, _resources) {
      if (!o.enable) return

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
    },
    externalResources(_ctx) {
      return {}
    },
  }
}
