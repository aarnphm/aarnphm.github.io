import { spawn } from "child_process"
import { styleText } from "node:util"
import { GlobalConfiguration } from "../../cfg"
import { QuartzEmitterPlugin } from "../../types/plugin"
import { FullSlug, joinSegments, QUARTZ } from "../../util/path"
import { write } from "./helpers"

const DEFAULT_MODEL_ID = "onnx-community/Qwen3-Embedding-0.6B-ONNX"

const defaults: GlobalConfiguration["semanticSearch"] = {
  enable: true,
  model: DEFAULT_MODEL_ID,
  aot: false,
  dims: 1024,
  dtype: "fp32",
  shardSizeRows: 1024,
  hnsw: { M: 16, efConstruction: 200 },
  chunking: { chunkSize: 512, chunkOverlap: 128, noChunking: false },
  vllm: {
    useVllm: false,
    vllmUrl:
      process.env.VLLM_URL || process.env.VLLM_EMBED_URL || "http://127.0.0.1:8000/v1/embeddings",
    concurrency: parseInt(process.env.VLLM_CONCURRENCY || "8", 10),
    batchSize: parseInt(process.env.VLLM_BATCH_SIZE || "64", 10),
  },
}

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
    const args = [
      "run",
      joinSegments(QUARTZ, "embed_build.py"),
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

    const proc = spawn("uv", args, { stdio: "inherit", shell: true })

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

  return {
    name: "SemanticIndex",
    getQuartzComponents() {
      return []
    },
    async *partialEmit() {},
    async *emit(ctx, content, _resources) {
      if (!o.enable) return

      const lines: string[] = []
      for (const [_, file] of content) {
        const slug = file.data.slug!
        const title = file.data.frontmatter?.title ?? slug
        const text = file.data.text
        if (!text) continue
        lines.push(JSON.stringify({ slug, title, text }))
      }

      yield write({
        ctx,
        slug: "embeddings-text" as FullSlug,
        ext: ".jsonl",
        content: lines.join("\n"),
      })

      if (!o.aot) {
        console.log(styleText("blue", `\n[emit:Semantic] Generating embeddings (aot=${o.aot})...`))

        await runEmbedBuild(
          joinSegments(ctx.argv.output, "embeddings-text.jsonl"),
          joinSegments(ctx.argv.output, "embeddings"),
          o,
        )
      } else {
        console.log(
          styleText("yellow", `[emit:Semantic] Skipping embedding generation (aot=${o.aot})`),
        )
      }
    },
    externalResources(_ctx) {
      return {}
    },
  }
}
