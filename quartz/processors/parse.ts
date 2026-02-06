import esbuild from 'esbuild'
import { Root as HtmlRoot } from 'hast'
import { Root as MdRoot } from 'mdast'
import path from 'path'
import remarkParse from 'remark-parse'
import remarkRehype from 'remark-rehype'
import { read } from 'to-vfile'
import { Processor, unified } from 'unified'
import { styleText } from 'util'
import workerpool, { Promise as WorkerPromise } from 'workerpool'
import { MarkdownContent, ProcessedContent } from '../plugins/vfile'
import { BuildCtx, WorkerSerializableBuildCtx } from '../util/ctx'
import { QuartzLogger } from '../util/log'
import { FilePath, QUARTZ, slugifyFilePath } from '../util/path'
import { PerfTimer } from '../util/perf'
import { trace } from '../util/trace'

export type QuartzMdProcessor = Processor<MdRoot, MdRoot, MdRoot>
export type QuartzHtmlProcessor = Processor<undefined, MdRoot, HtmlRoot>

export function createMdProcessor(ctx: BuildCtx): QuartzMdProcessor {
  const transformers = ctx.cfg.plugins.transformers

  return (
    unified()
      // base Markdown -> MD AST
      .use(remarkParse)
      // MD AST -> MD AST transforms
      .use(
        transformers.flatMap(plugin => plugin.markdownPlugins?.(ctx) ?? []),
      ) as unknown as QuartzMdProcessor
    //  ^ sadly the typing of `use` is not smart enough to infer the correct type from our plugin list
  )
}

export function createHtmlProcessor(ctx: BuildCtx): QuartzHtmlProcessor {
  const transformers = ctx.cfg.plugins.transformers
  return (
    unified()
      // MD AST -> HTML AST
      .use(remarkRehype as any, {
        allowDangerousHtml: true,
        handlers: {
          // Pass through sidenote nodes with their children converted to HAST
          sidenote: (state: any, node: any) => {
            const content = state.all(node)
            const labelNodes = node.data?.sidenoteParsed?.labelNodes ?? []
            const label = labelNodes.map((n: any) => state.one(n, node))

            return {
              type: 'element',
              tagName: 'span',
              properties: {
                className: ['sidenote-placeholder'],
                dataType: 'sidenote',
                sidenoteId: node.data?.sidenoteId,
                baseId: node.data?.baseId,
                forceInline: node.data?.forceInline,
                allowLeft: node.data?.allowLeft,
                allowRight: node.data?.allowRight,
                label: node.data?.label,
                internal: node.data?.internal,
              },
              children: [
                {
                  type: 'element',
                  tagName: 'span',
                  properties: { className: ['sidenote-label-hast'] },
                  children: label,
                },
                {
                  type: 'element',
                  tagName: 'span',
                  properties: { className: ['sidenote-content-hast'] },
                  children: content,
                },
              ],
            }
          },
          sidenoteReference: (state: any, node: any) => {
            const labelNodes = node.labelNodes ?? []
            const label = labelNodes.map((n: any) => state.one(n, node))

            return {
              type: 'element',
              tagName: 'span',
              properties: {
                className: ['sidenote-ref-placeholder'],
                dataType: 'sidenote-ref',
                label: node.label,
                sidenoteId: node.data?.sidenoteId,
                baseId: node.data?.baseId,
              },
              children: [
                {
                  type: 'element',
                  tagName: 'span',
                  properties: { className: ['sidenote-label-hast'] },
                  children: label,
                },
              ],
            }
          },
          sidenoteDefinition: (state: any, node: any) => {
            const content = state.all(node)
            const labelNodes = node.labelNodes ?? []
            const label = labelNodes.map((n: any) => state.one(n, node))

            return {
              type: 'element',
              tagName: 'div',
              properties: {
                className: ['sidenote-def-placeholder'],
                dataType: 'sidenote-def',
                label: node.label,
              },
              children: [
                {
                  type: 'element',
                  tagName: 'span',
                  properties: { className: ['sidenote-label-hast'] },
                  children: label,
                },
                {
                  type: 'element',
                  tagName: 'div',
                  properties: { className: ['sidenote-content-hast'] },
                  children: content,
                },
              ],
            }
          },
        },
      })
      // HTML AST -> HTML AST transforms
      .use(
        transformers.flatMap(plugin => plugin.htmlPlugins?.(ctx) ?? []),
      ) as unknown as QuartzHtmlProcessor
  )
}

function* chunks<T>(arr: T[], n: number) {
  for (let i = 0; i < arr.length; i += n) {
    yield arr.slice(i, i + n)
  }
}

async function transpileWorkerScript() {
  // transpile worker script
  const cacheFile = './.quartz-cache/transpiled-worker.mjs'
  const fp = './quartz/worker.ts'
  return esbuild.build({
    entryPoints: [fp],
    outfile: path.join(QUARTZ, cacheFile),
    bundle: true,
    keepNames: true,
    platform: 'node',
    format: 'esm',
    packages: 'external',
    sourcemap: true,
    sourcesContent: false,
    plugins: [
      {
        name: 'css-and-scripts-as-text',
        setup(build) {
          build.onLoad({ filter: /\.scss$/ }, _ => ({ contents: '', loader: 'text' }))
          build.onLoad({ filter: /\.inline\.(ts|js)$/ }, _ => ({ contents: '', loader: 'text' }))
        },
      },
    ],
  })
}

export function createFileParser(ctx: BuildCtx, fps: FilePath[]) {
  const { argv, cfg } = ctx
  return async (processor: QuartzMdProcessor) => {
    const res: MarkdownContent[] = []
    for (const fp of fps) {
      try {
        const perf = new PerfTimer()
        const file = await read(fp)

        // strip leading and trailing whitespace
        file.value = file.value.toString().trim()

        // Text -> Text transforms
        for (const plugin of cfg.plugins.transformers.filter(p => p.textTransform)) {
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
          console.log(`[markdown] ${fp} -> ${file.data.slug} (${perf.timeSince()})`)
        }
      } catch (err) {
        trace(`\nFailed to process markdown \`${fp}\``, err as Error)
      }
    }

    return res
  }
}

export function createMarkdownParser(ctx: BuildCtx, mdContent: MarkdownContent[]) {
  return async (processor: QuartzHtmlProcessor) => {
    const res: ProcessedContent[] = []
    for (const [ast, file] of mdContent) {
      try {
        const perf = new PerfTimer()

        const newAst = await processor.run(ast as MdRoot, file)
        res.push([newAst, file])

        if (ctx.argv.verbose) {
          console.log(`[html] ${file.data.slug} (${perf.timeSince()})`)
        }
      } catch (err) {
        trace(`\nFailed to process html \`${file.data.filePath}\``, err as Error)
      }
    }

    return res
  }
}

const clamp = (num: number, min: number, max: number) =>
  Math.min(Math.max(Math.round(num), min), max)

export async function parseMarkdown(ctx: BuildCtx, fps: FilePath[]): Promise<ProcessedContent[]> {
  const { argv } = ctx
  const perf = new PerfTimer()
  const log = new QuartzLogger(argv.verbose)

  // rough heuristics: 128 gives enough time for v8 to JIT and optimize parsing code paths
  const CHUNK_SIZE = 128
  const concurrency = ctx.argv.concurrency ?? clamp(fps.length / CHUNK_SIZE, 1, 4)

  let res: ProcessedContent[] = []
  log.start(`Parsing input files using ${concurrency} threads`)
  if (concurrency === 1) {
    try {
      const mdRes = await createFileParser(ctx, fps)(createMdProcessor(ctx))
      res = await createMarkdownParser(ctx, mdRes)(createHtmlProcessor(ctx))
    } catch (error) {
      log.end()
      throw error
    }
  } else {
    await transpileWorkerScript()
    const pool = workerpool.pool('./quartz/bootstrap-worker.mjs', {
      minWorkers: 'max',
      maxWorkers: concurrency,
      workerType: 'thread',
    })
    const errorHandler = (err: any) => {
      console.error(err)
      process.exit(1)
    }

    const serializableCtx: WorkerSerializableBuildCtx = {
      buildId: ctx.buildId,
      argv: ctx.argv,
      allSlugs: ctx.allSlugs,
      allFiles: ctx.allFiles,
      incremental: ctx.incremental,
    }

    const textToMarkdownPromises: WorkerPromise<MarkdownContent[]>[] = []
    let processedFiles = 0
    for (const chunk of chunks(fps, CHUNK_SIZE)) {
      textToMarkdownPromises.push(pool.exec('parseMarkdown', [serializableCtx, chunk]))
    }

    const mdResults: Array<MarkdownContent[]> = await Promise.all(
      textToMarkdownPromises.map(async promise => {
        const result = await promise
        processedFiles += result.length
        log.updateText(`text->markdown ${styleText('gray', `${processedFiles}/${fps.length}`)}`)
        return result
      }),
    ).catch(errorHandler)

    const markdownToHtmlPromises: WorkerPromise<ProcessedContent[]>[] = []
    processedFiles = 0
    for (const mdChunk of mdResults) {
      markdownToHtmlPromises.push(pool.exec('processHtml', [serializableCtx, mdChunk]))
    }
    const results: ProcessedContent[][] = await Promise.all(
      markdownToHtmlPromises.map(async promise => {
        const result = await promise
        processedFiles += result.length
        log.updateText(`markdown->html ${styleText('gray', `${processedFiles}/${fps.length}`)}`)
        return result
      }),
    ).catch(errorHandler)

    res = results.flat()
    await pool.terminate()
  }

  log.end(`Parsed ${res.length} Markdown files in ${perf.timeSince()}`)
  return res
}
