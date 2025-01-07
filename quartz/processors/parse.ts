import esbuild from "esbuild"
import remarkParse from "remark-parse"
import remarkRehype from "remark-rehype"
import { Processor, unified } from "unified"
import { Root as MDRoot } from "mdast"
import { Root as HTMLRoot } from "hast"
import { HtmlContent, MarkdownContent } from "../plugins/vfile"
import { PerfTimer } from "../util/perf"
import { read } from "to-vfile"
import { FilePath, FullSlug, QUARTZ, slugifyFilePath } from "../util/path"
import path from "path"
import workerpool, { Promise as WorkerPromise } from "workerpool"
import { QuartzLogger } from "../util/log"
import { trace } from "../util/trace"
import { BuildCtx } from "../util/ctx"

export type QuartzMarkdownProcessor = Processor<MDRoot, MDRoot, MDRoot>
export type QuartzHtmlProcessor = Processor<MDRoot, MDRoot, HTMLRoot>

export function createMarkdownProcessor(ctx: BuildCtx): QuartzMarkdownProcessor {
  const transformers = ctx.cfg.plugins.transformers

  return (
    (unified() as unknown as QuartzMarkdownProcessor)
      // base Markdown -> MD AST
      .use(remarkParse)
      // MD AST -> MD AST transforms
      .use(
        transformers
          .filter((p) => p.markdownPlugins)
          .flatMap((plugin) => plugin.markdownPlugins!(ctx)),
      )
  )
}

export function createHtmlProcessor(ctx: BuildCtx): QuartzHtmlProcessor {
  const transformers = ctx.cfg.plugins.transformers

  return (
    (unified() as unknown as QuartzHtmlProcessor)
      // MD AST -> HTML AST
      .use(remarkRehype, { allowDangerousHtml: true })
      // HTML AST -> HTML AST transforms
      .use(transformers.filter((p) => p.htmlPlugins).flatMap((plugin) => plugin.htmlPlugins!(ctx)))
  )
}

function* chunks<T>(arr: T[], n: number) {
  for (let i = 0; i < arr.length; i += n) {
    yield arr.slice(i, i + n)
  }
}

async function transpileWorkerScript() {
  // transpile worker script
  const cacheFile = "./.quartz-cache/transpiled-worker.mjs"
  const fp = "./quartz/worker.ts"
  return esbuild.build({
    entryPoints: [fp],
    outfile: path.join(QUARTZ, cacheFile),
    bundle: true,
    keepNames: true,
    platform: "node",
    format: "esm",
    packages: "external",
    sourcemap: true,
    sourcesContent: false,
    plugins: [
      {
        name: "css-and-scripts-as-text",
        setup(build) {
          build.onLoad({ filter: /\.scss$/ }, (_) => ({
            contents: "",
            loader: "text",
          }))
          build.onLoad({ filter: /\.inline\.(ts|js)$/ }, (_) => ({
            contents: "",
            loader: "text",
          }))
        },
      },
    ],
  })
}

export function createFileParser(ctx: BuildCtx, fps: FilePath[]) {
  const { argv, cfg } = ctx
  return async (processor: QuartzMarkdownProcessor) => {
    const res: MarkdownContent[] = []
    for (const fp of fps) {
      try {
        const perf = new PerfTimer()
        const file = await read(fp)

        // strip leading and trailing whitespace
        file.value = file.value.toString().trim()

        // Text -> Text transforms
        for (const plugin of cfg.plugins.transformers.filter((p) => p.textTransform)) {
          file.value = plugin.textTransform!(ctx, file.value.toString())
        }

        // base data properties that plugins may use
        file.data.filePath = file.path as FilePath
        file.data.relativePath = path.posix.relative(argv.directory, file.path) as FilePath
        file.data.slug = slugifyFilePath(file.data.relativePath)

        const ast = processor.parse(file)
        const newAst = await processor.run(ast, file)
        res.push([newAst, file])

        if (argv.verbose) {
          console.log(`[process:markdown] ${fp}`)
          console.log(`[process:markdown] ├─ slug: ${file.data.slug}`)
          console.log(`[process:markdown] └─ time: ${perf.timeSince()}`)
        }
      } catch (err) {
        trace(`\n[process:markdown] Failed to process \`${fp}\``, err as Error)
      }
    }

    return res
  }
}

export function createMarkdownParser(ctx: BuildCtx, mdContent: MarkdownContent[]) {
  return async (processor: QuartzHtmlProcessor) => {
    const res: HtmlContent[] = []
    for (const [ast, file] of mdContent) {
      const fp = file.data.filePath

      try {
        const perf = new PerfTimer()
        const newAst = await processor.run(ast, file)
        res.push([newAst, file])

        if (ctx.argv.verbose) {
          console.log(`[process:html] ${fp}`)
          console.log(`[process:html] ├─ slug: ${file.data.slug}`)
          console.log(`[process:html] └─ time: ${perf.timeSince()}`)
        }
      } catch (err) {
        trace(`\n[process:html] Failed to process html \`${fp}\``, err as Error)
      }
    }

    return res
  }
}

const clamp = (num: number, min: number, max: number) =>
  Math.min(Math.max(Math.round(num), min), max)
export async function parseMarkdown(ctx: BuildCtx, fps: FilePath[]): Promise<HtmlContent[]> {
  const { argv } = ctx
  const perf = new PerfTimer()
  const log = new QuartzLogger(argv.verbose)

  // rough heuristics: 128 gives enough time for v8 to JIT and optimize parsing code paths
  const CHUNK_SIZE = 128
  const concurrency = ctx.argv.concurrency ?? clamp(fps.length / CHUNK_SIZE, 1, 4)

  let res: HtmlContent[] = []
  log.start(`[process] Parsing input files using ${concurrency} threads`)
  if (concurrency === 1) {
    try {
      const processor = createMarkdownProcessor(ctx)
      const fileParser = createFileParser(ctx, fps)
      const markdown = await fileParser(processor)

      const html = createHtmlProcessor(ctx)
      const htmlParser = createMarkdownParser(ctx, markdown)
      res = await htmlParser(html)
    } catch (error) {
      log.end()
      throw error
    }
  } else {
    await transpileWorkerScript()
    const pool = workerpool.pool("./quartz/bootstrap-worker.mjs", {
      minWorkers: "max",
      maxWorkers: concurrency,
      workerType: "thread",
    })
    const errorHandler = (err: any) => {
      console.error(`${err}`.replace(/^error:\s*/i, ""))
      process.exit(1)
    }

    const mdPromises: WorkerPromise<[MarkdownContent[], FullSlug[]]>[] = []
    for (const chunk of chunks(fps, CHUNK_SIZE)) {
      mdPromises.push(pool.exec("parseMarkdown", [ctx.buildId, argv, chunk]))
    }
    const mdResults: [MarkdownContent[], FullSlug[]][] =
      await WorkerPromise.all(mdPromises).catch(errorHandler)

    const childPromises: WorkerPromise<HtmlContent[]>[] = []
    for (const [_, extraSlugs] of mdResults) {
      ctx.allSlugs.push(...extraSlugs)
    }
    for (const [mdChunk, _] of mdResults) {
      childPromises.push(pool.exec("parseHtml", [ctx.buildId, argv, mdChunk, ctx.allSlugs]))
    }
    const results: HtmlContent[][] = await WorkerPromise.all(childPromises).catch(errorHandler)
    res = results.flat()
    await pool.terminate()
  }

  log.end(`[process] Parsed ${res.length} Markdown files in ${perf.timeSince()}`)
  return res
}
