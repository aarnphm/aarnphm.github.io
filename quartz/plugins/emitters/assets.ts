import { readFile } from 'node:fs/promises'
import path from 'path'
import { QuartzEmitterPlugin } from '../../types/plugin'
import { defaultIoConcurrency, mapConcurrent } from '../../util/async-pool'
import { Argv, BuildCtx } from '../../util/ctx'
import { isFlashcardPath } from '../../util/flashcards-path'
import { glob } from '../../util/glob'
import { emitOutputAsset, removeOutputAsset, type OutputAssetClaim } from '../../util/output-assets'
import { FilePath, joinSegments, slugifyFilePath, stripSlashes } from '../../util/path'
import { logBuildSpan, PerfTimer } from '../../util/perf'

const heavyWatchAssetExts = new Set(['.ddl', '.mat'])
const pdfEmbedPattern = /!\[\[([^\]\r\n]+)\]\]/g

function isMarkdownReferenceSource(fp: FilePath): boolean {
  const ext = path.extname(fp).toLowerCase()
  return ext === '.md' || ext === '.base' || isFlashcardPath(fp)
}

function pdfTargetKey(target: string): string | undefined {
  const normalized = stripSlashes(target.trim())
  const pathOnly = normalized.split('#', 1)[0]?.trim()
  if (!pathOnly || path.extname(pathOnly).toLowerCase() !== '.pdf') return undefined
  return slugifyFilePath(pathOnly as FilePath)
}

async function referencedPdfFiles(ctx: BuildCtx, files: FilePath[]): Promise<Set<FilePath>> {
  if (!ctx.argv.watch || process.env.CF_PAGES === '1') return new Set()

  const pdfBySlug = new Map<string, FilePath>()
  for (const fp of files) {
    if (path.extname(fp).toLowerCase() === '.pdf') {
      pdfBySlug.set(slugifyFilePath(fp), fp)
    }
  }

  const referenced = new Set<FilePath>()
  const sources = files.filter(isMarkdownReferenceSource)
  await mapConcurrent(sources, defaultIoConcurrency, async fp => {
    const source = joinSegments(ctx.argv.directory, fp)
    const body = await readFile(source, 'utf8')
    for (const match of body.matchAll(pdfEmbedPattern)) {
      const rawTarget = match[1]?.split('|', 1)[0]
      const key = rawTarget ? pdfTargetKey(rawTarget) : undefined
      const pdf = key ? pdfBySlug.get(key) : undefined
      if (pdf) referenced.add(pdf)
    }
  })
  return referenced
}

function shouldIgnoreAssetFile(
  argv: Argv,
  fp: FilePath,
  referencedPdfs: ReadonlySet<FilePath>,
): boolean {
  const ext = path.extname(fp).toLowerCase()
  if (ext === '.md' || ext === '.base' || isFlashcardPath(fp)) return true
  if (process.env.CF_PAGES === '1') return ext === '.pdf' || heavyWatchAssetExts.has(ext)
  if (argv.watch && ext === '.pdf') return !referencedPdfs.has(fp)
  return argv.watch && heavyWatchAssetExts.has(ext)
}

async function contentAssetFilesFrom(ctx: BuildCtx, files: FilePath[]): Promise<FilePath[]> {
  const referencedPdfs = await referencedPdfFiles(ctx, files)
  return files.filter(fp => !shouldIgnoreAssetFile(ctx.argv, fp, referencedPdfs))
}

function contentAssetClaim(argv: Argv, fp: FilePath): OutputAssetClaim {
  const src = joinSegments(argv.directory, fp) as FilePath
  const name = slugifyFilePath(fp)
  const output = joinSegments(argv.output, name) as FilePath
  return { owner: 'content-asset', source: src, output }
}

export async function contentAssetClaims(ctx: BuildCtx): Promise<OutputAssetClaim[]> {
  return (await contentAssetFilesFrom(ctx, ctx.allFiles)).map(fp => contentAssetClaim(ctx.argv, fp))
}

const filesToCopy = async (ctx: BuildCtx): Promise<FilePath[]> => {
  const { argv, cfg } = ctx
  const perf = new PerfTimer()
  if (ctx.allFiles.length > 0) {
    const fps = await contentAssetFilesFrom(ctx, ctx.allFiles)
    logBuildSpan(argv, 'assets:scan', `${fps.length} files`, perf.elapsedMs())
    return fps
  }

  const patterns = [
    '**/*.md',
    '**/*.base',
    '**/*.fc',
    '**/*.flashcards',
    ...cfg.configuration.ignorePatterns,
  ]

  if (process.env.CF_PAGES === '1' || argv.watch) {
    patterns.push('**.ddl', '**.mat')
  }

  const allFiles = await glob('**', argv.directory, patterns)
  const fps = await contentAssetFilesFrom(ctx, allFiles)
  logBuildSpan(argv, 'assets:glob', `${fps.length} files`, perf.elapsedMs())
  return fps
}

export const Assets: QuartzEmitterPlugin = () => {
  return {
    name: 'Assets',
    async *emit(ctx) {
      const { argv } = ctx
      const fps = await filesToCopy(ctx)
      const perf = new PerfTimer()
      const files = await mapConcurrent(fps, defaultIoConcurrency, fp =>
        emitOutputAsset(ctx, contentAssetClaim(argv, fp)),
      )
      logBuildSpan(argv, 'assets:copy', `${fps.length} files`, perf.elapsedMs())
      for (const file of files) {
        yield file
      }
    },
    async *partialEmit(ctx, _content, _resources, changeEvents) {
      const referencedPdfs = await referencedPdfFiles(ctx, ctx.allFiles)
      const refreshReferencedPdfs = changeEvents.some(
        changeEvent =>
          isMarkdownReferenceSource(changeEvent.path) ||
          path.extname(changeEvent.path).toLowerCase() === '.pdf',
      )
      const emitted = new Set<string>()
      for (const changeEvent of changeEvents) {
        if (shouldIgnoreAssetFile(ctx.argv, changeEvent.path, referencedPdfs)) continue

        const claim = contentAssetClaim(ctx.argv, changeEvent.path)

        if (changeEvent.type === 'add' || changeEvent.type === 'change') {
          emitted.add(claim.output)
          yield emitOutputAsset(ctx, claim)
        } else if (changeEvent.type === 'delete') {
          await removeOutputAsset(ctx, claim.output)
        }
      }
      if (refreshReferencedPdfs) {
        for (const fp of referencedPdfs) {
          const claim = contentAssetClaim(ctx.argv, fp)
          if (emitted.has(claim.output)) continue
          yield emitOutputAsset(ctx, claim)
        }
      }
    },
  }
}
