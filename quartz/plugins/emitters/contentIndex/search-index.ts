import crypto from 'node:crypto'
import fs from 'node:fs/promises'
import { gzipSync } from 'node:zlib'
import type { BuildCtx } from '../../../util/ctx'
import type { ContentDetails, ContentIndexMap } from './link-index'
import { prepareLexicalDocument } from '../../../components/scripts/lexical-search-document'
import { createLexicalDocumentIndex } from '../../../components/scripts/lexical-search-index'
import {
  encodeCompactLexicalIndex,
  lexicalPrebuiltKind,
} from '../../../components/scripts/lexical-search-prebuilt'
import { defaultIoConcurrency, mapConcurrent } from '../../../util/async-pool'
import { FilePath, FullSlug, joinSegments } from '../../../util/path'
import { write, writeKnownChanged } from '../helpers'

const SEARCH_INDEX_CHUNK_COUNT = 64
const SEARCH_INDEX_KIND = 'quartz-search-index-v1'

export type SerializedIndexCache = { entries: Map<FullSlug, string> }

export function serializeContentDetails(content: ContentDetails, includeContent: boolean): string {
  const serializable = { ...content }
  if (!includeContent) {
    delete serializable.content
  }
  delete serializable.fileData
  delete serializable.richContent
  return JSON.stringify(serializable)
}

function serializeIndexJson(
  idx: ContentIndexMap,
  includeContent: boolean,
  previous?: SerializedIndexCache,
  changedSlugs?: ReadonlySet<FullSlug>,
): { content: string; cache: SerializedIndexCache } {
  if (!previous || !changedSlugs) {
    const entries = new Map<FullSlug, string>()
    for (const [slug, content] of idx) {
      entries.set(slug, serializeContentDetails(content, includeContent))
    }
    return { content: serializeEntries(entries), cache: { entries } }
  }

  const entries = new Map(previous.entries)
  for (const slug of entries.keys()) {
    if (!idx.has(slug)) {
      entries.delete(slug)
    }
  }
  for (const slug of changedSlugs) {
    const details = idx.get(slug)
    if (details) {
      entries.set(slug, serializeContentDetails(details, includeContent))
    }
  }
  return { content: serializeEntries(entries), cache: { entries } }
}

function serializeEntries(entries: ReadonlyMap<FullSlug, string>): string {
  const parts: string[] = []
  const sortedEntries = Array.from(entries).sort(([slugA], [slugB]) => slugA.localeCompare(slugB))
  for (const [slug, content] of sortedEntries) {
    parts.push(serializedEntry(slug, content))
  }
  return `{${parts.join(',')}}`
}

function serializedEntry(slug: FullSlug, content: string): string {
  return `${JSON.stringify(slug)}:${content}`
}

function searchChunkId(slug: FullSlug): string {
  let hash = 0x811c9dc5
  for (let i = 0; i < slug.length; i += 1) {
    hash ^= slug.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193) >>> 0
  }
  return (hash % SEARCH_INDEX_CHUNK_COUNT).toString(16).padStart(2, '0')
}

function searchChunkIds(): string[] {
  return Array.from({ length: SEARCH_INDEX_CHUNK_COUNT }, (_, index) =>
    index.toString(16).padStart(2, '0'),
  )
}

function searchChunkSlug(id: string): FullSlug {
  return joinSegments('static', 'search-index', id) as FullSlug
}

function searchChunkManifest(lexical: {
  kind: string
  path: string
  fingerprint: string
  sha256: string
}): string {
  return JSON.stringify({
    kind: SEARCH_INDEX_KIND,
    lexical,
    chunks: searchChunkIds().map(id => `search-index/${id}.json`),
  })
}

const searchChunksByEntries = new WeakMap<
  ReadonlyMap<FullSlug, string>,
  Map<string, Map<FullSlug, string>>
>()

function searchChunkEntries(
  entries: ReadonlyMap<FullSlug, string>,
  id: string,
): Map<FullSlug, string> {
  let buckets = searchChunksByEntries.get(entries)
  if (!buckets) {
    buckets = new Map()
    for (const chunkId of searchChunkIds()) {
      buckets.set(chunkId, new Map())
    }
    for (const [slug, content] of entries) {
      const chunkId = searchChunkId(slug)
      let bucket = buckets.get(chunkId)
      if (!bucket) {
        bucket = new Map()
        buckets.set(chunkId, bucket)
      }
      bucket.set(slug, content)
    }
    searchChunksByEntries.set(entries, buckets)
  }
  return buckets.get(id) ?? new Map()
}

async function writeSearchChunk(
  ctx: BuildCtx,
  id: string,
  entries: ReadonlyMap<FullSlug, string>,
  knownChanged = false,
): Promise<FilePath> {
  const content = serializeEntries(searchChunkEntries(entries, id))
  const slug = searchChunkSlug(id)
  return knownChanged
    ? writeKnownChanged({ ctx, content, slug, ext: '.json' })
    : write({ ctx, content, slug, ext: '.json' })
}

async function writeSearchManifest(
  ctx: BuildCtx,
  entries: ReadonlyMap<FullSlug, string>,
): Promise<FilePath[]> {
  // Match Object.assign({}, ...chunks) in loadSearchIndex, including its document ID order.
  const orderedEntries = searchChunkIds().flatMap(id =>
    Array.from(searchChunkEntries(entries, id)).sort(([a], [b]) => a.localeCompare(b)),
  )
  const ordered: Record<string, string> = Object.fromEntries(orderedEntries)
  const documents = Object.entries(ordered).map(([slug, serialized], id) => {
    const details: ContentDetails = JSON.parse(serialized)
    return prepareLexicalDocument(id, slug, details)
  })
  const fingerprint = crypto.createHash('sha256').update(JSON.stringify(documents)).digest('hex')
  const index = createLexicalDocumentIndex()
  for (const document of documents) index.add(document)
  const { bytes } = encodeCompactLexicalIndex(index)
  const sha256 = crypto.createHash('sha256').update(bytes).digest('hex')
  const asset = `search-lexical/${sha256}/index-${sha256.slice(0, 8)}`
  const binary = await write({ ctx, content: gzipSync(bytes), slug: `static/${asset}`, ext: '.gz' })
  const manifest = await write({
    ctx,
    content: searchChunkManifest({
      kind: lexicalPrebuiltKind,
      path: `${asset}.gz`,
      fingerprint,
      sha256,
    }),
    slug: joinSegments('static', 'searchIndex'),
    ext: '.json',
  })
  return [binary, manifest]
}

export async function writeContentIndexJson(
  ctx: BuildCtx,
  linkIndex: ContentIndexMap,
  previous?: SerializedIndexCache,
  changedSlugs?: ReadonlySet<FullSlug>,
  knownChanged = false,
): Promise<{ file: FilePath; cache: SerializedIndexCache }> {
  const serialized = serializeIndexJson(linkIndex, false, previous, changedSlugs)
  const slug = joinSegments('static', 'contentIndex') as FullSlug
  const file = knownChanged
    ? await writeKnownChanged({ ctx, content: serialized.content, slug, ext: '.json' })
    : await write({ ctx, content: serialized.content, slug, ext: '.json' })
  return { file, cache: serialized.cache }
}

function patchSerializedEntry(
  content: string,
  slug: FullSlug,
  previousEntry: string,
  currentEntry: string,
): string | undefined {
  const previous = serializedEntry(slug, previousEntry)
  const current = serializedEntry(slug, currentEntry)
  const next = content.replace(previous, current)
  return next === content ? undefined : next
}

export async function writePatchedSerializedIndexJson(
  ctx: BuildCtx,
  outputSlug: FullSlug,
  previous: SerializedIndexCache,
  updates: ReadonlyMap<FullSlug, string>,
): Promise<{ file: FilePath; cache: SerializedIndexCache } | undefined> {
  const entries = new Map(previous.entries)
  const pathToIndex = joinSegments(ctx.argv.output, `${outputSlug}.json`) as FilePath
  let content = await fs.readFile(pathToIndex, 'utf8')

  for (const [slug, currentEntry] of updates) {
    const previousEntry = entries.get(slug)
    if (!previousEntry) return undefined
    const patched = patchSerializedEntry(content, slug, previousEntry, currentEntry)
    if (!patched) return undefined
    content = patched
    entries.set(slug, currentEntry)
  }

  return {
    file: await writeKnownChanged({ ctx, content, slug: outputSlug, ext: '.json' }),
    cache: { entries },
  }
}

export async function writeSearchIndexJson(
  ctx: BuildCtx,
  searchIndex: ContentIndexMap,
  previous?: SerializedIndexCache,
  changedSlugs?: ReadonlySet<FullSlug>,
): Promise<{ files: FilePath[]; cache: SerializedIndexCache }> {
  const serialized = serializeIndexJson(searchIndex, true, previous, changedSlugs)
  const chunkIds = changedSlugs
    ? Array.from(new Set(Array.from(changedSlugs, searchChunkId))).sort()
    : searchChunkIds()
  const [manifestFile, chunkFiles] = await Promise.all([
    writeSearchManifest(ctx, serialized.cache.entries),
    mapConcurrent(chunkIds, defaultIoConcurrency, id =>
      writeSearchChunk(ctx, id, serialized.cache.entries),
    ),
  ])
  const files = [...manifestFile, ...chunkFiles]
  return { files, cache: serialized.cache }
}

export async function writePatchedSearchIndexJson(
  ctx: BuildCtx,
  previous: SerializedIndexCache,
  updates: ReadonlyMap<FullSlug, string>,
  deletes: ReadonlySet<FullSlug>,
  knownChanged = false,
): Promise<{ files: FilePath[]; cache: SerializedIndexCache }> {
  const entries = new Map(previous.entries)
  for (const slug of deletes) {
    entries.delete(slug)
  }
  for (const [slug, content] of updates) {
    entries.set(slug, content)
  }
  const changedChunks = new Set<string>()
  for (const slug of deletes) {
    changedChunks.add(searchChunkId(slug))
  }
  for (const slug of updates.keys()) {
    changedChunks.add(searchChunkId(slug))
  }
  if (changedChunks.size === 0) {
    return { files: [], cache: { entries } }
  }
  const files = await mapConcurrent(Array.from(changedChunks), defaultIoConcurrency, id =>
    writeSearchChunk(ctx, id, entries, knownChanged),
  )
  return { files: [...files, ...(await writeSearchManifest(ctx, entries))], cache: { entries } }
}

export async function writePatchedSearchIndexChunks(
  ctx: BuildCtx,
  previous: SerializedIndexCache,
  updates: ReadonlyMap<FullSlug, string>,
): Promise<{ files: FilePath[]; cache: SerializedIndexCache } | undefined> {
  const entries = new Map(previous.entries)
  const updatesByChunk = new Map<string, Map<FullSlug, string>>()

  for (const [slug, currentEntry] of updates) {
    if (!entries.has(slug)) return undefined
    const id = searchChunkId(slug)
    const chunkUpdates = updatesByChunk.get(id)
    if (chunkUpdates) {
      chunkUpdates.set(slug, currentEntry)
    } else {
      updatesByChunk.set(id, new Map([[slug, currentEntry]]))
    }
  }

  const files: FilePath[] = []
  for (const [id, chunkUpdates] of updatesByChunk) {
    const slug = searchChunkSlug(id)
    const pathToChunk = joinSegments(ctx.argv.output, `${slug}.json`) as FilePath
    let content = await fs.readFile(pathToChunk, 'utf8')

    for (const [entrySlug, currentEntry] of chunkUpdates) {
      const previousEntry = entries.get(entrySlug)
      if (!previousEntry) return undefined
      const patched = patchSerializedEntry(content, entrySlug, previousEntry, currentEntry)
      if (!patched) return undefined
      content = patched
      entries.set(entrySlug, currentEntry)
    }

    files.push(await writeKnownChanged({ ctx, content, slug, ext: '.json' }))
  }

  return { files: [...files, ...(await writeSearchManifest(ctx, entries))], cache: { entries } }
}
