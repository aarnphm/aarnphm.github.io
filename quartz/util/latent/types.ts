/**
 * `layout: latent` sidecar contract. Written by quartz/scripts/latent/build.py under
 * `<directory>/latent/<hash>/`, read at build time (quartz/util/latent/sidecar.ts) and in the
 * browser (quartz/components/scripts/latent.client.ts). Row ids are global indices over every
 * included store in discovery order; document ids index docs.json, which is time-ordered.
 */

export type LatentKind = 'note' | 'pdf' | 'image' | 'base'

export const LATENT = 'latent' as const
export const KIND_ORDER: LatentKind[] = ['note', 'pdf', 'image', 'base']

export interface LatentStore {
  id: string
  kind: LatentKind | null
  /** Pointer manifest, site-absolute (`/embeddings/pdf/manifest.json`). */
  path: string
  generation: string | null
  rows: number | null
  rowOffset: number | null
  model: string | null
  dims: number | null
  included: boolean
  reason: string | null
  coverage?: { eligibleDocuments: number; indexedDocuments: number; indexedPages: number } | null
}

export interface LatentIndex {
  version: 1
  hash: string
  built: string
  model: string
  dims: number
  k: number
  block: number
  /** chunks per text part (`text/<d>.json` holds part 0, `text/<d>-NNN.json` the rest) */
  part: number
  axes: number
  rows: number
  docs: number
  datedDocs: number
  kinds: LatentKind[]
  counts: Partial<Record<LatentKind, number>>
  stores: LatentStore[]
}

export interface LatentDoc {
  d: number
  id: string
  s: LatentKind
  t: string
  date: string | null
  prec: 'day' | 'month' | 'year' | null
  r0: number
  n: number
  sha?: string
  pages?: number | null
}

/** One row of a `prev/NNNNN.json` block. */
export interface LatentPreview {
  r: number
  d: number
  s: LatentKind
  t: string
  c: number
  p: number | null
  l: string
  x: string
}

export interface LatentChunk {
  r: number
  c: number
  text: string
  p?: number
  l?: string
  /** Page box in permille of the page, resolved from pdf-regions at build. */
  box?: [number, number, number, number]
  /** 1 when `text` is the note's markdown slice rather than its rendered text */
  m?: 1
}

export interface LatentText {
  d: number
  id: string
  store: LatentKind
  title: string
  date: string | null
  prec: string | null
  /** chunks of the document */
  total: number
  /** chunks per part */
  part: number
  /** part 0 */
  chunks: LatentChunk[]
  sha?: string
  pages?: number | null
}

export interface LatentTextPart {
  d: number
  part: number
  chunks: LatentChunk[]
}

export function partName(doc: number, part: number): string {
  return part === 0 ? `${doc}.json` : `${doc}-${String(part).padStart(3, '0')}.json`
}

export interface LatentNeighbour {
  r: number
  /** cosine × 255 */
  score: number
}

export interface LatentAxisPole {
  r: number
  d: number
  t: string
  s: LatentKind
}

export interface LatentAxis {
  id: string
  name: string
  a: string
  b: string
  /** Sarle's bimodality coefficient of the projections; > 0.555 reads as two modes. */
  bc: number
  scale: number
  hist: number[]
  terms: { a: string[]; b: string[] }
  poles: { a: LatentAxisPole[]; b: LatentAxisPole[] }
  /** Set on an axis the page's `axes:` frontmatter names (build.py `axes_spec`). */
  source?: 'frontmatter'
  spec?: { a: LatentAxisSpecPole; b: LatentAxisSpecPole }
  /** documents in each pole (one in both counts in neither) */
  docs?: { a: number; b: number }
  /** held-out AUC: direction from half of each pole, scored on the other half */
  sep?: number
}

export interface LatentAxisSpecPole {
  tags: string[]
  paths: string[]
}

export interface LatentProbe {
  axes: LatentAxis[]
}

/** Per-row record decoded from a `knn/NNNNN.bin` block. */
export interface LatentRowLinks {
  r: number
  neighbours: LatentNeighbour[]
  /** Projections on the probe axes, int8 of each axis's scale. */
  axes: number[]
}

export const TRACK_STRIDE = 10

export function knnStride(index: Pick<LatentIndex, 'k' | 'axes'>): number {
  return index.k * 5 + index.axes
}

export function blockOf(row: number, block: number): number {
  return Math.floor(row / block)
}

export function blockName(block: number): string {
  return String(block).padStart(5, '0')
}

/** Decode every row of one kNN block. */
export function decodeKnnBlock(
  bytes: ArrayBuffer,
  block: number,
  index: Pick<LatentIndex, 'k' | 'axes' | 'block'>,
): LatentRowLinks[] {
  const stride = knnStride(index)
  const view = new DataView(bytes)
  const u8 = new Uint8Array(bytes)
  const rows = Math.floor(bytes.byteLength / stride)
  const out: LatentRowLinks[] = []
  for (let i = 0; i < rows; i++) {
    const base = i * stride
    const neighbours: LatentNeighbour[] = []
    for (let j = 0; j < index.k; j++) {
      neighbours.push({ r: view.getUint32(base + j * 4, true), score: u8[base + index.k * 4 + j] })
    }
    const axes: number[] = []
    for (let a = 0; a < index.axes; a++) {
      axes.push(view.getInt8(base + index.k * 5 + a))
    }
    out.push({ r: block * index.block + i, neighbours, axes })
  }
  return out
}

export interface LatentTrackDoc {
  day: number
  kind: LatentKind
  prec: number
  r0: number
  n: number
}

export function decodeTrack(bytes: ArrayBuffer, kinds: LatentKind[]): LatentTrackDoc[] {
  const view = new DataView(bytes)
  const docs: LatentTrackDoc[] = []
  for (let off = 0; off + TRACK_STRIDE <= bytes.byteLength; off += TRACK_STRIDE) {
    docs.push({
      day: view.getUint16(off, true),
      kind: kinds[view.getUint8(off + 2)] ?? 'note',
      prec: view.getUint8(off + 3),
      r0: view.getUint32(off + 4, true),
      n: view.getUint16(off + 8, true),
    })
  }
  return docs
}

/** Days since 1900-01-01 (+1; 0 = undated) to an ISO year. */
export function yearOfDay(day: number): number {
  if (!day) return 0
  const ms = Date.UTC(1900, 0, 1) + (day - 1) * 86_400_000
  return new Date(ms).getUTCFullYear()
}

export function scopePrefixes(embeddings: unknown): string[] {
  if (!Array.isArray(embeddings)) return []
  const out: string[] = []
  for (const entry of embeddings) {
    if (typeof entry !== 'string') continue
    const inner = entry.trim().replace(/^\[\[/, '').replace(/\]\]$/, '')
    const target = inner
      .split('|')[0]
      .split('#')[0]
      .trim()
      .replace(/^\/+|\/+$/g, '')
    if (target) out.push(target)
  }
  return out
}

export function inScope(docId: string, prefixes: string[]): boolean {
  if (prefixes.length === 0) return true
  return prefixes.some(prefix => docId === prefix || docId.startsWith(`${prefix}/`))
}

/** Site path of a row's source: a note, a page of the pdf reader, an image, a base entry. */
export function rowHref(doc: Pick<LatentDoc, 'id' | 's'>, page?: number | null): string {
  switch (doc.s) {
    case 'pdf':
      return `/read/${doc.id}${page ? `#page=${page}` : ''}`
    case 'image':
      return `/${doc.id}`
    default:
      return `/${doc.id}`
  }
}
