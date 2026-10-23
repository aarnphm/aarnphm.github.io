import { createHash } from 'node:crypto'
import fs from 'node:fs/promises'
import { styleText } from 'node:util'
import v8 from 'node:v8'
import { VFile } from 'vfile'
import { BuildCtx } from '../../util/ctx'
import {
  ensureExtractedNotebookAssets,
  type ExtractedAsset,
} from '../../util/notebook/extract-assets'
import { FilePath, FullSlug, joinSegments, QUARTZ } from '../../util/path'
import { isRecord } from '../../util/type-guards'
import { ProcessedContent, QuartzPluginData } from '../vfile'
import { notebookCacheReceipt } from './notebook-cache-receipts'

export type NotebookPage = { fp: FilePath; slug: FullSlug; content: ProcessedContent }

export type NotebookPageCacheEntry = {
  fp: FilePath
  slug: FullSlug
  key: string
  cacheKey: string
  blob: FilePath
  htmlBlobPath: FilePath
  extractedAssets: ExtractedAsset[]
  data?: QuartzPluginData
}
type NotebookPageBlob = {
  tree: ProcessedContent[0]
  data: QuartzPluginData
  value: string
  extractedAssets?: ExtractedAsset[]
}

declare global {
  var __quartzNotebookPageBlobs: Map<FilePath, NotebookPageCacheEntry> | undefined
}

// Parsed pages live on disk across rebuilds and CLI restarts: 220 pages hold ~0.4 GiB of hast in
// memory and load from blobs in ~3 s, against ~55 s to parse.
const notebookPageBlobDir = joinSegments(QUARTZ, '.quartz-cache', 'notebook-pages') as FilePath
const notebookPageIndexFile = joinSegments(notebookPageBlobDir, 'index.json') as FilePath

function isExtractedAsset(value: unknown): value is ExtractedAsset {
  return (
    isRecord(value) &&
    typeof value.hash === 'string' &&
    typeof value.url === 'string' &&
    typeof value.mime === 'string' &&
    typeof value.byteLength === 'number'
  )
}

export function compactNotebookPluginData(data: QuartzPluginData): QuartzPluginData {
  return {
    filePath: data.filePath,
    relativePath: data.relativePath,
    slug: data.slug,
    frontmatter: data.frontmatter,
    description: data.description,
    links: data.links,
    dates: data.dates,
  }
}

function parseNotebookCachedData(value: unknown): QuartzPluginData | undefined {
  if (!isRecord(value) || typeof value.slug !== 'string') return undefined
  const dates = isRecord(value.dates)
    ? {
        created: new Date(String(value.dates.created)),
        modified: new Date(String(value.dates.modified)),
        published: new Date(String(value.dates.published)),
      }
    : undefined
  return { ...(value as QuartzPluginData), ...(dates ? { dates } : {}) }
}

function parseNotebookPageCacheEntry(value: unknown): NotebookPageCacheEntry | undefined {
  if (!isRecord(value)) return undefined
  const fp = typeof value.fp === 'string' ? (value.fp as FilePath) : undefined
  const slug = typeof value.slug === 'string' ? (value.slug as FullSlug) : undefined
  const key =
    typeof value.key === 'string'
      ? value.key
      : typeof value.cacheKey === 'string'
        ? value.cacheKey
        : undefined
  const blob =
    typeof value.blob === 'string'
      ? (value.blob as FilePath)
      : typeof value.htmlBlobPath === 'string'
        ? (value.htmlBlobPath as FilePath)
        : undefined
  if (!fp || !slug || !key || !blob) return undefined
  const extractedAssets = Array.isArray(value.extractedAssets)
    ? value.extractedAssets.filter(isExtractedAsset)
    : []
  const data = parseNotebookCachedData(value.data)
  return {
    fp,
    slug,
    key,
    cacheKey: key,
    blob,
    htmlBlobPath: blob,
    extractedAssets,
    ...(data ? { data } : {}),
  }
}

async function loadNotebookPageCacheFromDisk(): Promise<Map<FilePath, NotebookPageCacheEntry>> {
  const entries = new Map<FilePath, NotebookPageCacheEntry>()
  await fs.mkdir(notebookPageBlobDir, { recursive: true })
  try {
    const rawIndex: unknown = JSON.parse(await fs.readFile(notebookPageIndexFile, 'utf8'))
    if (isRecord(rawIndex) && isRecord(rawIndex.entries)) {
      for (const [fp, rawEntry] of Object.entries(rawIndex.entries)) {
        const parsed = parseNotebookPageCacheEntry(
          isRecord(rawEntry) ? { fp, ...rawEntry } : rawEntry,
        )
        if (parsed) entries.set(parsed.fp, parsed)
      }
    }
  } catch {}

  try {
    const dirEntries = await fs.readdir(notebookPageBlobDir)
    await Promise.all(
      dirEntries
        .filter(name => name.endsWith('.meta.json'))
        .map(async name => {
          try {
            const raw: unknown = JSON.parse(
              await fs.readFile(joinSegments(notebookPageBlobDir, name), 'utf8'),
            )
            const parsed = parseNotebookPageCacheEntry(raw)
            if (parsed) entries.set(parsed.fp, parsed)
          } catch {}
        }),
    )
  } catch {}

  return entries
}

export async function writeNotebookPageCacheIndex(
  entries: ReadonlyMap<FilePath, NotebookPageCacheEntry>,
): Promise<void> {
  try {
    await fs.mkdir(notebookPageBlobDir, { recursive: true })
    const serialized: Record<string, NotebookPageCacheEntry> = {}
    for (const [fp, entry] of [...entries.entries()].sort(([left], [right]) =>
      left.localeCompare(right),
    )) {
      serialized[fp] = entry
    }
    const temp = joinSegments(notebookPageBlobDir, `index.${process.pid}.${Date.now()}.tmp`)
    await fs.writeFile(temp, JSON.stringify({ version: 1, entries: serialized }))
    await fs.rename(temp, notebookPageIndexFile)
  } catch {}
}

export async function notebookPageCacheEntries(): Promise<Map<FilePath, NotebookPageCacheEntry>> {
  if (!globalThis.__quartzNotebookPageBlobs) {
    globalThis.__quartzNotebookPageBlobs = await loadNotebookPageCacheFromDisk()
  }
  return globalThis.__quartzNotebookPageBlobs
}

export function resetNotebookCaches(): void {
  globalThis.__quartzNotebookPageBlobs = undefined
  globalThis.__quartzNotebookDependencyCache?.clear()
}

function isNotebookPageBlob(value: unknown): value is NotebookPageBlob {
  return (
    isRecord(value) &&
    isRecord(value.data) &&
    typeof value.data.slug === 'string' &&
    isRecord(value.tree) &&
    value.tree.type === 'root' &&
    Array.isArray(value.tree.children) &&
    (value.extractedAssets === undefined ||
      (Array.isArray(value.extractedAssets) && value.extractedAssets.every(isExtractedAsset)))
  )
}

export async function readNotebookPageBlob(
  ctx: BuildCtx,
  entry: NotebookPageCacheEntry,
  fp: FilePath,
  src: FilePath,
  slug: FullSlug,
  origin: 'cache' | 'spill' = 'cache',
): Promise<NotebookPage | undefined> {
  let bytes: Buffer
  try {
    bytes = await fs.readFile(entry.blob)
  } catch {
    notebookCacheReceipt(ctx, fp, 'restore', 'miss', 'blob-unavailable')
    return undefined
  }
  let decoded: unknown
  try {
    decoded = v8.deserialize(bytes)
  } catch {
    notebookCacheReceipt(ctx, fp, 'restore', 'miss', 'blob-invalid')
    return undefined
  }
  if (!isNotebookPageBlob(decoded)) {
    notebookCacheReceipt(ctx, fp, 'restore', 'miss', 'blob-invalid')
    return undefined
  }
  const extractedAssets = entry.extractedAssets ?? decoded.extractedAssets ?? []
  const assetsReady = await ensureExtractedNotebookAssets(ctx, extractedAssets).catch(() => false)
  if (!assetsReady) {
    notebookCacheReceipt(ctx, fp, 'restore', 'miss', 'assets-unavailable')
    return undefined
  }
  delete decoded.data.htmlAst
  entry.data ??= compactNotebookPluginData(decoded.data)
  const file = new VFile({ path: src, value: '' })
  file.data = decoded.data
  notebookCacheReceipt(
    ctx,
    fp,
    'restore',
    origin === 'cache' ? 'hit' : 'reload',
    origin === 'cache' ? 'blob-restored' : 'spill-restored',
  )
  return { fp, slug, content: [decoded.tree, file] }
}

export async function writeNotebookPageBlob(
  entries: Map<FilePath, NotebookPageCacheEntry>,
  key: string,
  page: NotebookPage,
  extractedAssets: ExtractedAsset[],
): Promise<void> {
  const [tree, file] = page.content
  delete file.data.htmlAst
  const name = createHash('sha256').update(page.fp).digest('hex').slice(0, 24)
  const blob = joinSegments(notebookPageBlobDir, `${name}.bin`) as FilePath
  const meta = joinSegments(notebookPageBlobDir, `${name}.meta.json`) as FilePath
  try {
    await fs.mkdir(notebookPageBlobDir, { recursive: true })
    const value: NotebookPageBlob = { tree, data: file.data, value: '', extractedAssets }
    const entry: NotebookPageCacheEntry = {
      fp: page.fp,
      slug: page.slug,
      key,
      cacheKey: key,
      blob,
      htmlBlobPath: blob,
      extractedAssets,
      data: compactNotebookPluginData(file.data),
    }
    await Promise.all([
      fs.writeFile(blob, v8.serialize(value)),
      fs.writeFile(meta, JSON.stringify(entry)),
    ])
    entries.set(page.fp, entry)
  } catch (error) {
    entries.delete(page.fp)
    console.warn(styleText('yellow', `[emit:NotebookViewer] page cache skipped ${page.fp}`), error)
  }
}
