import { createHash, randomUUID } from 'crypto'
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

export async function removeWritten(
  ctx: BuildCtx,
  slug: FullSlug | string,
  ext: `.${string}` | '',
): Promise<void> {
  const pathToPage = joinSegments(ctx.argv.output, slug + ext) as FilePath
  writtenContent.delete(pathToPage)
  await fs.promises.rm(pathToPage, { force: true })
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
  if (!cacheEntry) return true
  const previous = writtenContent.get(pathToPage)
  if (previous) {
    return (
      previous.size !== cacheEntry.size ||
      previous.kind !== cacheEntry.kind ||
      previous.fingerprint !== cacheEntry.fingerprint
    )
  }

  try {
    const stat = await fs.promises.stat(pathToPage)
    if (stat.size !== cacheEntry.size) return true
    const existing = await fs.promises.readFile(pathToPage)
    const changed = !contentEquals(existing, content)
    if (!changed) {
      writtenContent.set(pathToPage, cacheEntry)
    }
    return changed
  } catch {
    return true
  }
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

function cacheWrittenContent(pathToPage: FilePath, cacheEntry: ContentCacheEntry | undefined) {
  if (cacheEntry) {
    writtenContent.set(pathToPage, cacheEntry)
  } else {
    writtenContent.delete(pathToPage)
  }
}

async function writeOutputFile(
  ctx: BuildCtx,
  pathToPage: FilePath,
  content: WriteOptions['content'],
): Promise<void> {
  await ensureOutputDir(path.dirname(pathToPage))
  if (!ctx.incremental) {
    await fs.promises.writeFile(pathToPage, content)
    return
  }

  // Stage beside the output tree so Wrangler cannot index a temporary file before rename.
  const temporary = path.join(
    path.dirname(path.resolve(ctx.argv.output)),
    `.quartz-write-${randomUUID()}`,
  )
  try {
    await fs.promises.writeFile(temporary, content)
    await fs.promises.rename(temporary, pathToPage)
  } finally {
    await fs.promises.rm(temporary, { force: true })
  }
}

export const write = async ({ ctx, slug, ext, content }: WriteOptions): Promise<FilePath> => {
  const perf = new PerfTimer()
  const pathToPage = joinSegments(ctx.argv.output, slug + ext) as FilePath
  const cacheEntry = ctx.incremental || ctx.argv.watch ? contentCacheEntry(content) : undefined
  if (
    ctx.incremental &&
    !ctx.cleanOutput &&
    !(await shouldWrite(pathToPage, content, cacheEntry))
  ) {
    logBuildSpan(ctx.argv, 'write:skip', pathToPage, perf.elapsedMs())
    return pathToPage
  }
  await writeOutputFile(ctx, pathToPage, content)
  if (!ctx.cleanOutput) cacheWrittenContent(pathToPage, cacheEntry)
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
  await writeOutputFile(ctx, pathToPage, content)
  if (!ctx.cleanOutput) cacheWrittenContent(pathToPage, contentCacheEntry(content))
  logBuildSpan(ctx.argv, 'write', pathToPage, perf.elapsedMs())
  return pathToPage
}
