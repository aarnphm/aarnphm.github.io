import { readFile, stat } from 'node:fs/promises'
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

const pdfReferenceCache = new Map<FilePath, { signature: string; keys: string[] }>()
let copiedReferencedPdfs: ReadonlySet<FilePath> = new Set()

async function pdfReferenceKeys(ctx: BuildCtx, fp: FilePath): Promise<string[]> {
  const source = joinSegments(ctx.argv.directory, fp) as FilePath
  const signature = await stat(source).then(
    info => `${info.mtimeMs}:${info.size}`,
    () => 'missing',
  )
  const cached = pdfReferenceCache.get(fp)
  if (cached?.signature === signature) return cached.keys

  const keys: string[] = []
  if (signature !== 'missing') {
    const body = await readFile(source, 'utf8')
    for (const match of body.matchAll(pdfEmbedPattern)) {
      const rawTarget = match[1]?.split('|', 1)[0]
      const key = rawTarget ? pdfTargetKey(rawTarget) : undefined
      if (key) keys.push(key)
    }
  }
  pdfReferenceCache.set(fp, { signature, keys })
  return keys
}

async function referencedPdfFiles(
  ctx: BuildCtx,
  files: FilePath[],
  rescan?: readonly FilePath[],
): Promise<Set<FilePath>> {
  if (!ctx.argv.watch || process.env.CF_PAGES === '1') return new Set()

  const pdfBySlug = new Map<string, FilePath>()
  for (const fp of files) {
    if (path.extname(fp).toLowerCase() === '.pdf') {
      pdfBySlug.set(slugifyFilePath(fp), fp)
    }
  }

  const perf = new PerfTimer()
  const sources = files.filter(isMarkdownReferenceSource)
  const stale = rescan ?? sources
  await mapConcurrent(stale, defaultIoConcurrency, fp => pdfReferenceKeys(ctx, fp))

  const referenced = new Set<FilePath>()
  for (const fp of sources) {
    for (const key of pdfReferenceCache.get(fp)?.keys ?? []) {
      const pdf = pdfBySlug.get(key)
      if (pdf) referenced.add(pdf)
    }
  }
  logBuildSpan(ctx.argv, 'assets:pdfrefs', `${stale.length} rescanned`, perf.elapsedMs())
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
      copiedReferencedPdfs = await referencedPdfFiles(ctx, ctx.allFiles)
      for (const file of files) {
        yield file
      }
    },
    async *partialEmit(ctx, _content, _resources, changeEvents) {
      const changedSources = changeEvents
        .filter(changeEvent => isMarkdownReferenceSource(changeEvent.path))
        .map(changeEvent => changeEvent.path)
      const refreshReferencedPdfs =
        changedSources.length > 0 ||
        changeEvents.some(changeEvent => path.extname(changeEvent.path).toLowerCase() === '.pdf')
      const referencedPdfs = refreshReferencedPdfs
        ? await referencedPdfFiles(ctx, ctx.allFiles, changedSources)
        : copiedReferencedPdfs
      const newlyReferencedPdfs = [...referencedPdfs].filter(fp => !copiedReferencedPdfs.has(fp))
      copiedReferencedPdfs = referencedPdfs
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
      for (const fp of newlyReferencedPdfs) {
        const claim = contentAssetClaim(ctx.argv, fp)
        if (emitted.has(claim.output)) continue
        yield emitOutputAsset(ctx, claim)
      }
    },
  }
}
