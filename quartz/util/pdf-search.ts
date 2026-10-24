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
export interface PdfIndex {
  generation: string
  passages: PdfPassage[]
  documents: Record<string, PdfMetadata>
  coverage: { indexedDocuments: number; eligibleDocuments: number; indexedPages: number }
  shards: PdfVectorShard[]
}

const integer = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
const strings = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every(item => typeof item === 'string')

export function readPdfIndex(data: unknown): PdfIndex {
  if (
    !isRecord(data) ||
    data.kind !== 'quartz-pdf-search-v1' ||
    data.version !== 2 ||
    data.model !== 'google/embeddinggemma-2' ||
    data.dims !== 768 ||
    data.dtype !== 'fp32' ||
    data.normalized !== true ||
    typeof data.generation !== 'string' ||
    !/^[a-f\d]{64}$/.test(data.generation) ||
    !strings(data.ids) ||
    !Array.isArray(data.passages) ||
    data.rows !== data.passages.length ||
    data.ids.length !== data.passages.length ||
    new Set(data.ids).size !== data.ids.length ||
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
  const shards: PdfVectorShard[] = []
  let offset = 0
  for (const shard of data.vectors.shards) {
    if (
      !isRecord(shard) ||
      typeof shard.path !== 'string' ||
      !shard.path.startsWith(`/embeddings/pdf/${data.generation}/`) ||
      !/^vectors-\d+\.bin$/.test(shard.path.split('/').at(-1) ?? '') ||
      /(?:\.{2}|[\\?#%])/.test(shard.path) ||
      !integer(shard.rows) ||
      shard.rows === 0 ||
      shard.rows > 1024 ||
      shard.rowOffset !== offset ||
      shard.byteStride !== 768 * 4 ||
      shard.byteLength !== shard.rows * 768 * 4 ||
      typeof shard.sha256 !== 'string' ||
      !/^[a-f\d]{64}$/.test(shard.sha256)
    )
      throw new Error('Invalid PDF vector shard')
    shards.push({
      path: shard.path,
      rows: shard.rows,
      rowOffset: offset,
      byteLength: shard.byteLength,
      byteStride: shard.byteStride,
      sha256: shard.sha256,
    })
    offset += shard.rows
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
  const passages: PdfPassage[] = []
  for (const [row, value] of data.passages.entries()) {
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
      !integer(value.chunk) ||
      data.ids[row] !== `${value.slug}#page=${value.page}&chunk=${value.chunk}`
    )
      throw new Error('Invalid PDF passage reference')
    passages.push({
      slug: value.slug,
      page: value.page,
      pageLabel: value.pageLabel,
      text: value.text,
      title: value.title,
      chunk: value.chunk,
    })
  }
  return {
    generation: data.generation,
    passages,
    documents,
    shards,
    coverage: {
      indexedDocuments: data.coverage.indexedDocuments,
      eligibleDocuments: data.coverage.eligibleDocuments,
      indexedPages: data.coverage.indexedPages,
    },
  }
}

export async function validatePdfVectorShard(
  bytes: ArrayBuffer,
  shard: PdfVectorShard,
): Promise<void> {
  if (bytes.byteLength !== shard.byteLength) throw new Error('PDF shard length mismatch')
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))
  if (Array.from(digest, value => value.toString(16).padStart(2, '0')).join('') !== shard.sha256)
    throw new Error('PDF shard checksum mismatch')
  const values = new Float32Array(bytes)
  for (let row = 0; row < shard.rows; row++) {
    let norm = 0
    for (let column = 0; column < 768; column++) norm += values[row * 768 + column] ** 2
    if (!Number.isFinite(norm) || Math.abs(norm - 1) > 0.01)
      throw new Error('PDF shard contains an invalid unit vector')
  }
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
