import { createHash } from 'crypto'
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

type ContentCacheEntry = { size: number; kind: 'text' | 'bytes'; fingerprint: string }
type WriteCacheState = {
  writtenContent: Map<FilePath, ContentCacheEntry>
  ensuredDirs: Map<string, Promise<void>>
}

declare global {
  var __quartzWriteCache: WriteCacheState | undefined
}

const existingWriteCache = globalThis.__quartzWriteCache
const writeCache =
  existingWriteCache && existingWriteCache.ensuredDirs instanceof Map
    ? existingWriteCache
    : {
        writtenContent:
          existingWriteCache?.writtenContent ?? new Map<FilePath, ContentCacheEntry>(),
        ensuredDirs: new Map<string, Promise<void>>(),
      }
globalThis.__quartzWriteCache = writeCache
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
  if (typeof content === 'string') {
    return {
      size: Buffer.byteLength(content),
      kind: 'text',
      fingerprint: createHash('sha256').update(content).digest('base64url'),
    }
  }

  if (Buffer.isBuffer(content)) {
    return {
      size: content.byteLength,
      kind: 'bytes',
      fingerprint: createHash('sha256').update(content).digest('base64url'),
    }
  }

  return undefined
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
      return (
        previous.size !== cacheEntry.size ||
        previous.kind !== cacheEntry.kind ||
        previous.fingerprint !== cacheEntry.fingerprint
      )
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
  if (
    previous.size !== cacheEntry.size ||
    previous.kind !== cacheEntry.kind ||
    previous.fingerprint !== cacheEntry.fingerprint
  ) {
    return 'changed'
  }
  return 'same'
}

function ensureOutputDir(dir: string): Promise<void> {
  const existing = ensuredDirs.get(dir)
  if (existing) return existing
  const pending = fs.promises
    .mkdir(dir, { recursive: true })
    .then(() => undefined)
    .catch(error => {
      ensuredDirs.delete(dir)
      throw error
    })
  ensuredDirs.set(dir, pending)
  return pending
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
  await ensureOutputDir(dir)
  if (ctx.cleanOutput) {
    await fs.promises.writeFile(pathToPage, content)
    logBuildSpan(ctx.argv, 'write', pathToPage, perf.elapsedMs())
    return pathToPage
  }
  const cacheEntry = ctx.incremental || ctx.argv.watch ? contentCacheEntry(content) : undefined
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
    await fs.promises.writeFile(pathToPage, content)
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
  await ensureOutputDir(path.dirname(pathToPage))
  if (ctx.cleanOutput) {
    await fs.promises.writeFile(pathToPage, content)
    logBuildSpan(ctx.argv, 'write', pathToPage, perf.elapsedMs())
    return pathToPage
  }
  await fs.promises.writeFile(pathToPage, content)
  cacheWrittenContent(pathToPage, contentCacheEntry(content))
  logBuildSpan(ctx.argv, 'write', pathToPage, perf.elapsedMs())
  return pathToPage
}
