import fs from 'fs'
import path from 'path'
import { Readable } from 'stream'
import type { BuildCtx } from '../../util/ctx'
import type { FilePath, FullSlug } from '../../util/path'
import { joinSegments } from '../../util/path'
import { logBuildSpan, PerfTimer } from '../../util/perf'

type WriteOptions = {
  ctx: BuildCtx
  slug: FullSlug | string
  ext: `.${string}` | ''
  content: string | Buffer | Readable
}

type KnownChangedWriteOptions = Omit<WriteOptions, 'content'> & { content: string | Buffer }

type ContentCacheEntry = { size: number; content: string }
type WriteCacheState = {
  writtenContent: Map<FilePath, ContentCacheEntry>
  ensuredDirs: Set<string>
}

declare global {
  var __quartzWriteCache: WriteCacheState | undefined
}

const writeCache = (globalThis.__quartzWriteCache ??= {
  writtenContent: new Map<FilePath, ContentCacheEntry>(),
  ensuredDirs: new Set<string>(),
})
const { writtenContent, ensuredDirs } = writeCache

export function resetWriteCache(): void {
  writtenContent.clear()
  ensuredDirs.clear()
}

function contentSize(content: WriteOptions['content']): number | undefined {
  if (typeof content === 'string') return Buffer.byteLength(content)
  if (Buffer.isBuffer(content)) return content.byteLength
  return undefined
}

function contentCacheEntry(content: WriteOptions['content']): ContentCacheEntry | undefined {
  if (typeof content !== 'string') return undefined
  return { size: Buffer.byteLength(content), content }
}

function contentEquals(existing: Buffer, content: WriteOptions['content']): boolean {
  if (typeof content === 'string') return existing.equals(Buffer.from(content))
  if (Buffer.isBuffer(content)) return existing.equals(content)
  return false
}

async function shouldWrite(
  pathToPage: FilePath,
  content: WriteOptions['content'],
  cacheEntry: ContentCacheEntry | undefined,
): Promise<boolean> {
  if (cacheEntry) {
    const previous = writtenContent.get(pathToPage)
    if (previous) {
      return previous.size !== cacheEntry.size || previous.content !== cacheEntry.content
    }
  }

  const size = contentSize(content)
  if (size === undefined) return true

  try {
    const stat = await fs.promises.stat(pathToPage)
    if (stat.size !== size) return true
    const existing = await fs.promises.readFile(pathToPage)
    const changed = !contentEquals(existing, content)
    if (!changed && cacheEntry) {
      writtenContent.set(pathToPage, cacheEntry)
    }
    return changed
  } catch {
    return true
  }
}

function cachedContentStatus(
  pathToPage: FilePath,
  cacheEntry: ContentCacheEntry,
): 'changed' | 'same' | undefined {
  const previous = writtenContent.get(pathToPage)
  if (!previous) return undefined
  if (previous.size !== cacheEntry.size || previous.content !== cacheEntry.content) return 'changed'
  return 'same'
}

function writeCachedContent(pathToPage: FilePath, content: string | Buffer): void {
  fs.writeFileSync(pathToPage, content)
}

function ensureOutputDir(dir: string): Promise<void> | undefined {
  if (ensuredDirs.has(dir)) return undefined
  return fs.promises.mkdir(dir, { recursive: true }).then(() => {
    ensuredDirs.add(dir)
  })
}

function canWriteCachedContent(content: WriteOptions['content']): content is string | Buffer {
  return typeof content === 'string' || Buffer.isBuffer(content)
}

function cacheWrittenContent(pathToPage: FilePath, cacheEntry: ContentCacheEntry | undefined) {
  if (cacheEntry) {
    writtenContent.set(pathToPage, cacheEntry)
  }
}

function isCachedSame(status: ReturnType<typeof cachedContentStatus>): boolean {
  return status !== undefined && status === 'same'
}

export const write = async ({ ctx, slug, ext, content }: WriteOptions): Promise<FilePath> => {
  const perf = new PerfTimer()
  const pathToPage = joinSegments(ctx.argv.output, slug + ext) as FilePath
  const dir = path.dirname(pathToPage)
  const dirWrite = ensureOutputDir(dir)
  if (dirWrite) await dirWrite
  const cacheEntry = contentCacheEntry(content)
  if (!ctx.incremental) {
    await fs.promises.writeFile(pathToPage, content)
    cacheWrittenContent(pathToPage, cacheEntry)
    logBuildSpan(ctx.argv, 'write', pathToPage, perf.elapsedMs())
    return pathToPage
  }

  const cachedStatus = cacheEntry ? cachedContentStatus(pathToPage, cacheEntry) : undefined
  if (isCachedSame(cachedStatus)) {
    logBuildSpan(ctx.argv, 'write:skip', pathToPage, perf.elapsedMs())
    return pathToPage
  }
  if (cachedStatus === 'changed' && canWriteCachedContent(content)) {
    writeCachedContent(pathToPage, content)
    cacheWrittenContent(pathToPage, cacheEntry)
    logBuildSpan(ctx.argv, 'write', pathToPage, perf.elapsedMs())
    return pathToPage
  }
  if (!(await shouldWrite(pathToPage, content, cacheEntry))) {
    logBuildSpan(ctx.argv, 'write:skip', pathToPage, perf.elapsedMs())
    return pathToPage
  }
  await fs.promises.writeFile(pathToPage, content)
  cacheWrittenContent(pathToPage, cacheEntry)
  logBuildSpan(ctx.argv, 'write', pathToPage, perf.elapsedMs())
  return pathToPage
}

export async function writeKnownChanged({
  ctx,
  slug,
  ext,
  content,
}: KnownChangedWriteOptions): Promise<FilePath> {
  const perf = new PerfTimer()
  const pathToPage = joinSegments(ctx.argv.output, slug + ext) as FilePath
  const dirWrite = ensureOutputDir(path.dirname(pathToPage))
  if (dirWrite) await dirWrite
  await fs.promises.writeFile(pathToPage, content)
  cacheWrittenContent(pathToPage, contentCacheEntry(content))
  logBuildSpan(ctx.argv, 'write', pathToPage, perf.elapsedMs())
  return pathToPage
}
