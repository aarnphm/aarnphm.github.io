import { isSearchExcluded } from './search-exclude'
import { fuseSearchRanks, type SemanticDocument, type SemanticHit } from './semantic-search'
import { isRecord } from './type-guards'

export const PDF_MANIFEST_URL = '/embeddings/pdf/manifest.json'

export interface PdfPassage {
  slug: string
  page: number
  pageLabel: string
  text: string
  title: string
  chunk: number
}

export interface PdfMetadata {
  title: string
  aliases?: readonly string[]
  acronyms?: readonly string[]
}

export function pdfMetadata(
  data: Readonly<
    Record<
      string,
      { title: string; fileName: string; aliases?: readonly string[]; protected?: boolean }
    >
  >,
): Record<string, PdfMetadata> {
  return Object.fromEntries(
    Object.entries(data)
      .filter(
        ([slug, entry]) =>
          slug.endsWith('.pdf') &&
          !entry.protected &&
          !isSearchExcluded(slug) &&
          !isSearchExcluded(entry.fileName) &&
          !slug.includes('.encrypted.'),
      )
      .map(([slug, entry]) => [slug, { title: entry.title, aliases: entry.aliases }]),
  )
}

export type PdfSearchDocument = SemanticDocument & { passage?: PdfPassage }
export interface PdfVectorShard {
  path: string
  rows: number
  rowOffset: number
  byteLength: number
  byteStride: number
  sha256: string
}
/** JSON array of passages, aligned row for row with the vector shard of the same index. */
export interface PdfPassageShard {
  path: string
  rows: number
  rowOffset: number
  byteLength: number
  sha256: string
}
export interface PdfIndex {
  generation: string
  rows: number
  documents: Record<string, PdfMetadata>
  coverage: { indexedDocuments: number; eligibleDocuments: number; indexedPages: number }
  shards: PdfVectorShard[]
  /** Version 3 stores passage text beside the vectors; version 2 inlines it as `passages`. */
  passageShards: PdfPassageShard[]
  passages?: PdfPassage[]
}

const integer = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
const strings = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every(item => typeof item === 'string')
const digest = /^[a-f\d]{64}$/
const unsafePath = /(?:\.{2}|[\\?#%])/

function readShardLayout(
  value: unknown,
  generation: string,
  fileName: RegExp,
  offset: number,
): { path: string; rows: number; byteLength: number; sha256: string } {
  if (
    !isRecord(value) ||
    typeof value.path !== 'string' ||
    !value.path.startsWith(`/embeddings/pdf/${generation}/`) ||
    !fileName.test(value.path.split('/').at(-1) ?? '') ||
    unsafePath.test(value.path) ||
    !integer(value.rows) ||
    value.rows === 0 ||
    value.rows > 1024 ||
    value.rowOffset !== offset ||
    !integer(value.byteLength) ||
    typeof value.sha256 !== 'string' ||
    !digest.test(value.sha256)
  )
    throw new Error('Invalid PDF shard')
  return { path: value.path, rows: value.rows, byteLength: value.byteLength, sha256: value.sha256 }
}

function readPassage(value: unknown, documents: Record<string, PdfMetadata>): PdfPassage {
  if (
    !isRecord(value) ||
    typeof value.slug !== 'string' ||
    !documents[value.slug] ||
    !integer(value.page) ||
    value.page === 0 ||
    typeof value.pageLabel !== 'string' ||
    typeof value.text !== 'string' ||
    !value.text.trim() ||
    typeof value.title !== 'string' ||
    !integer(value.chunk)
  )
    throw new Error('Invalid PDF passage reference')
  return {
    slug: value.slug,
    page: value.page,
    pageLabel: value.pageLabel,
    text: value.text,
    title: value.title,
    chunk: value.chunk,
  }
}

export function readPdfIndex(data: unknown): PdfIndex {
  if (
    !isRecord(data) ||
    data.kind !== 'quartz-pdf-search-v1' ||
    (data.version !== 2 && data.version !== 3) ||
    data.model !== 'google/embeddinggemma-2' ||
    data.dims !== 768 ||
    data.dtype !== 'fp32' ||
    data.normalized !== true ||
    typeof data.generation !== 'string' ||
    !digest.test(data.generation) ||
    !integer(data.rows) ||
    !isRecord(data.documents) ||
    !isRecord(data.coverage) ||
    !integer(data.coverage.indexedDocuments) ||
    !integer(data.coverage.eligibleDocuments) ||
    !integer(data.coverage.indexedPages)
  )
    throw new Error('Invalid PDF search index')
  if (
    !isRecord(data.vectors) ||
    data.vectors.dtype !== 'fp32' ||
    data.vectors.dims !== 768 ||
    data.vectors.rows !== data.rows ||
    !Array.isArray(data.vectors.shards)
  )
    throw new Error('Invalid PDF vector inventory')
  const { generation } = data
  const shards: PdfVectorShard[] = []
  let offset = 0
  for (const shard of data.vectors.shards) {
    const layout = readShardLayout(shard, generation, /^vectors-\d+\.bin$/, offset)
    if (
      !isRecord(shard) ||
      shard.byteStride !== 768 * 4 ||
      layout.byteLength !== layout.rows * 768 * 4
    )
      throw new Error('Invalid PDF vector shard')
    shards.push({ ...layout, rowOffset: offset, byteStride: 768 * 4 })
    offset += layout.rows
  }
  if (offset !== data.rows) throw new Error('PDF shards do not cover the passage inventory')
  const documents: Record<string, PdfMetadata> = {}
  for (const [slug, value] of Object.entries(data.documents)) {
    if (
      !slug.endsWith('.pdf') ||
      isSearchExcluded(slug) ||
      !isRecord(value) ||
      typeof value.title !== 'string' ||
      !strings(value.aliases) ||
      !strings(value.acronyms)
    )
      throw new Error('Invalid PDF document metadata')
    documents[slug] = { title: value.title, aliases: value.aliases, acronyms: value.acronyms }
  }
  const index: PdfIndex = {
    generation,
    rows: data.rows,
    documents,
    shards,
    passageShards: [],
    coverage: {
      indexedDocuments: data.coverage.indexedDocuments,
      eligibleDocuments: data.coverage.eligibleDocuments,
      indexedPages: data.coverage.indexedPages,
    },
  }
  if (data.version === 2) {
    if (
      !strings(data.ids) ||
      !Array.isArray(data.passages) ||
      data.rows !== data.passages.length ||
      data.ids.length !== data.passages.length ||
      new Set(data.ids).size !== data.ids.length
    )
      throw new Error('Invalid PDF passage inventory')
    index.passages = data.passages.map((value, row) => {
      const passage = readPassage(value, documents)
      if ((data.ids as string[])[row] !== passageId(passage))
        throw new Error('Invalid PDF passage reference')
      return passage
    })
    return index
  }
  if (!isRecord(data.passages) || !Array.isArray(data.passages.shards))
    throw new Error('Invalid PDF passage inventory')
  offset = 0
  for (const [position, shard] of data.passages.shards.entries()) {
    const layout = readShardLayout(shard, generation, /^passages-\d+\.json$/, offset)
    if (layout.rows !== shards[position]?.rows)
      throw new Error('PDF passage shards do not align with the vector shards')
    index.passageShards.push({ ...layout, rowOffset: offset })
    offset += layout.rows
  }
  if (offset !== data.rows) throw new Error('PDF passage shards do not cover the inventory')
  return index
}

export function passageId(passage: Pick<PdfPassage, 'slug' | 'page' | 'chunk'>): string {
  return `${passage.slug}#page=${passage.page}&chunk=${passage.chunk}`
}

async function sha256Hex(bytes: ArrayBuffer): Promise<string> {
  const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))
  return Array.from(hash, value => value.toString(16).padStart(2, '0')).join('')
}

export async function validatePdfVectorShard(
  bytes: ArrayBuffer,
  shard: PdfVectorShard,
): Promise<void> {
  if (bytes.byteLength !== shard.byteLength) throw new Error('PDF shard length mismatch')
  if ((await sha256Hex(bytes)) !== shard.sha256) throw new Error('PDF shard checksum mismatch')
  const values = new Float32Array(bytes)
  for (let row = 0; row < shard.rows; row++) {
    let norm = 0
    for (let column = 0; column < 768; column++) norm += values[row * 768 + column] ** 2
    if (!Number.isFinite(norm) || Math.abs(norm - 1) > 0.01)
      throw new Error('PDF shard contains an invalid unit vector')
  }
}

/** Parse one passage shard after checking its bytes against the manifest. */
export async function readPdfPassageShard(
  bytes: ArrayBuffer,
  shard: PdfPassageShard,
  documents: Record<string, PdfMetadata>,
): Promise<PdfPassage[]> {
  if (bytes.byteLength !== shard.byteLength) throw new Error('PDF passage shard length mismatch')
  if ((await sha256Hex(bytes)) !== shard.sha256)
    throw new Error('PDF passage shard checksum mismatch')
  const parsed: unknown = JSON.parse(new TextDecoder().decode(bytes))
  if (!Array.isArray(parsed) || parsed.length !== shard.rows)
    throw new Error('PDF passage shard row count mismatch')
  return parsed.map(value => readPassage(value, documents))
}

export function pdfSearchResults(
  hits: readonly SemanticHit[],
  passages: readonly PdfPassage[],
): (SemanticDocument & { passage: PdfPassage })[] {
  const best = new Map<string, SemanticDocument & { passage: PdfPassage }>()
  for (const { id, score } of hits) {
    const passage = passages[id]
    if (!passage || !Number.isFinite(score) || isSearchExcluded(passage.slug)) continue
    const previous = best.get(passage.slug)
    if (!previous || score > previous.score)
      best.set(passage.slug, { slug: passage.slug, score, passage })
  }
  return [...best.values()].sort((a, b) => b.score - a.score || a.slug.localeCompare(b.slug))
}

function words(value: string): string {
  return value
    .normalize('NFKC')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .trim()
}

export function rankPdfResults(
  query: string,
  lexical: readonly string[],
  semantic: readonly PdfSearchDocument[],
  metadata: Readonly<Record<string, PdfMetadata>>,
  allowed: ReadonlySet<string>,
): PdfSearchDocument[] {
  const text = words(query)
  const tokens = text.split(' ').filter(Boolean)
  const priority = (slug: string) => {
    const document = metadata[slug]
    if (!document || !text) return 0
    const names = [
      document.title,
      ...(document.aliases ?? []),
      slug
        .split('/')
        .at(-1)
        ?.replace(/\.pdf$/i, '') ?? '',
    ].map(words)
    if (names.includes(text)) return 3
    if (document.acronyms?.some(acronym => words(acronym) === text)) return 2
    return names.some(name => tokens.every(token => name.split(' ').includes(token))) ? 1 : 0
  }
  const titleMatches = Object.keys(metadata)
    .filter(slug => allowed.has(slug) && priority(slug) > 0)
    .sort((a, b) => priority(b) - priority(a) || a.localeCompare(b))
  const bySlug = new Map(semantic.map(document => [document.slug, document]))
  return fuseSearchRanks([
    { ids: [...titleMatches, ...lexical], weight: 1 },
    { ids: semantic.map(document => document.slug), weight: 1 },
  ])
    .filter(({ id }) => allowed.has(id) && !isSearchExcluded(id))
    .sort((a, b) => priority(b.id) - priority(a.id) || b.score - a.score)
    .map(({ id }) => bySlug.get(id) ?? { slug: id, score: Number.NaN })
}
