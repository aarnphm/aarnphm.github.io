import { write } from "./helpers"
import { QuartzEmitterPlugin } from "../types"
import { FilePath, FullSlug, joinSegments, QUARTZ } from "../../util/path"
import { ReadTimeResults } from "reading-time"
import { GlobalConfiguration } from "../../cfg"
import { spawn } from "child_process"

const DEFAULT_MODEL_ID = "onnx-community/Qwen3-Embedding-0.6B-ONNX"

const defaults: GlobalConfiguration["semanticSearch"] = {
  enable: true,
  model: DEFAULT_MODEL_ID,
  aot: false,
  dims: 1024,
  dtype: "fp32",
  shardSizeRows: 1024,
  hnsw: { M: 16, efConstruction: 200 },
  chunking: {
    chunkSize: 512,
    chunkOverlap: 128,
    noChunking: false,
  },
  vllm: {
    useVllm: false,
    vllmUrl:
      process.env.VLLM_URL || process.env.VLLM_EMBED_URL || "http://127.0.0.1:8000/v1/embeddings",
    concurrency: parseInt(process.env.VLLM_CONCURRENCY || "8", 10),
    batchSize: parseInt(process.env.VLLM_BATCH_SIZE || "64", 10),
  },
}

type ContentDetails = {
  slug: string
  title: string
  filePath: FilePath
  content: string
  readingTime?: Partial<ReadTimeResults>
}

/**
 * Check if uv is installed
 */
function checkUvInstalled(): Promise<boolean> {
  return new Promise((resolve) => {
    const proc = spawn("uv", ["--version"], { shell: true })
    proc.on("error", () => resolve(false))
    proc.on("close", (code) => resolve(code === 0))
  })
}

/**
 * Run the Python embedding build script using uv
 * Script uses PEP 723 inline metadata for dependency management
 */
function runEmbedBuild(
  jsonlPath: string,
  outDir: string,
  opts: {
    model: string
    dtype: string
    dims: number
    shardSizeRows: number
    chunking: { chunkSize: number; chunkOverlap: number; noChunking: boolean }
    vllm: { useVllm: boolean; vllmUrl?: string; concurrency: number; batchSize: number }
  },
): Promise<void> {
  return new Promise((resolve, reject) => {
    const scriptPath = joinSegments(QUARTZ, "embed_build.py")
    const args = [
      "run",
      scriptPath,
      "--jsonl",
      jsonlPath,
      "--model",
      opts.model,
      "--out",
      outDir,
      "--dtype",
      opts.dtype,
      "--dims",
      String(opts.dims),
      "--shard-size",
      String(opts.shardSizeRows),
      "--chunk-size",
      String(opts.chunking.chunkSize),
      "--chunk-overlap",
      String(opts.chunking.chunkOverlap),
    ]

    if (opts.chunking.noChunking) {
      args.push("--no-chunking")
    }

    if (opts.vllm.useVllm) {
      args.push("--use-vllm")
      if (opts.vllm.vllmUrl) {
        args.push("--vllm-url", opts.vllm.vllmUrl)
      }
      args.push("--concurrency", String(opts.vllm.concurrency))
      args.push("--batch-size", String(opts.vllm.batchSize))
    }

    console.log("\nRunning embedding generation:")
    console.log(`  uv ${args.join(" ")}`)

    const env = { ...process.env }
    if (opts.vllm.useVllm && !env.USE_VLLM) {
      env.USE_VLLM = "1"
    }

    const proc = spawn("uv", args, {
      stdio: "inherit",
      shell: true,
      env,
    })

    proc.on("error", (err) => {
      reject(new Error(`Failed to spawn uv: ${err.message}`))
    })

    proc.on("close", (code) => {
      if (code === 0) {
        console.log("Embedding generation completed successfully")
        resolve()
      } else {
        reject(new Error(`embed_build.py exited with code ${code}`))
      }
    })
  })
}

export const SemanticIndex: QuartzEmitterPlugin<Partial<GlobalConfiguration["semanticSearch"]>> = (
  opts,
) => {
  const merged = { ...defaults, ...opts }
  const o = {
    enable: merged.enable!,
    model: merged.model!,
    aot: merged.aot!,
    dims: merged.dims!,
    dtype: merged.dtype!,
    shardSizeRows: merged.shardSizeRows!,
    hnsw: {
      M: merged.hnsw?.M ?? defaults.hnsw!.M!,
      efConstruction: merged.hnsw?.efConstruction ?? defaults.hnsw!.efConstruction!,
      efSearch: merged.hnsw?.efSearch,
    },
    chunking: {
      chunkSize: merged.chunking?.chunkSize ?? defaults.chunking!.chunkSize!,
      chunkOverlap: merged.chunking?.chunkOverlap ?? defaults.chunking!.chunkOverlap!,
      noChunking: merged.chunking?.noChunking ?? defaults.chunking!.noChunking!,
    },
    vllm: {
      useVllm: merged.vllm?.useVllm ?? defaults.vllm!.useVllm!,
      vllmUrl: merged.vllm?.vllmUrl ?? defaults.vllm!.vllmUrl,
      concurrency: merged.vllm?.concurrency ?? defaults.vllm!.concurrency!,
      batchSize: merged.vllm?.batchSize ?? defaults.vllm!.batchSize!,
    },
  }

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

      // Emit JSONL with the exact text used for embeddings
      const jsonl = docs
        .map((d) => ({ slug: d.slug, title: d.title, text: d.content }))
        .map((o) => JSON.stringify(o))
        .join("\n")

      const jsonlSlug = "embeddings-text" as FullSlug
      yield write({
        ctx,
        slug: jsonlSlug,
        ext: ".jsonl",
        content: jsonl,
      })

      // If aot is false, run the embedding generation script
      if (!o.aot) {
        console.log("\nGenerating embeddings (aot=false)...")

        // Check for uv
        const hasUv = await checkUvInstalled()
        if (!hasUv) {
          throw new Error(
            "uv is required for embedding generation. Install it from https://docs.astral.sh/uv/",
          )
        }

        const jsonlPath = joinSegments(ctx.argv.output, "embeddings-text.jsonl")
        const outDir = joinSegments(ctx.argv.output, "embeddings")

        try {
          await runEmbedBuild(jsonlPath, outDir, o)
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err)
          throw new Error(`Embedding generation failed: ${message}`)
        }
      } else {
        console.log(
          "\nSkipping embedding generation (aot=true). Expecting pre-generated embeddings in public/embeddings/",
        )
      }
    },
    externalResources(_ctx) {
      return {}
    },
  }
}
