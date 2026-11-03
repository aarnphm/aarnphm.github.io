import { createHash } from 'node:crypto'
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import path from 'node:path'
import type {
  LatentChunk,
  LatentDoc,
  LatentIndex,
  LatentPreview,
  LatentProbe,
  LatentRowLinks,
  LatentText,
  LatentTextPart,
  LatentTrackDoc,
} from './types'
import { blockName, blockOf, decodeKnnBlock, decodeTrack, inScope, partName } from './types'

/**
 * Build-time access to the latent sidecar. The sidecar is pinned to a hash over every included
 * manifest under the directory (quartz/scripts/latent/build.py); the same walk runs here so a
 * regenerated store makes the page fall back to prose plus ledger instead of serving stale rows.
 */

const SHA = /^[a-f0-9]{64}$/

function classify(manifest: Record<string, unknown>): string | null {
  if (manifest.kind === 'quartz-pdf-search-v1') return 'pdf'
  if (manifest.modality === 'image') return 'image'
  if (manifest.version === 2 && 'chunkMetadata' in manifest) return 'note'
  if (manifest.version === 2 && 'encoder' in manifest) return 'base'
  return null
}

function* walk(dir: string): Generator<string> {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      if (entry.name === 'latent') continue
      yield* walk(full)
    } else if (entry.name === 'manifest.json') yield full
  }
}

/** The hash the builder would compute for the stores under `directory` right now. */
export function expectedHash(content: string, directory: string): string | null {
  const root = path.join(content, directory)
  if (!existsSync(root) || !statSync(root).isDirectory()) return null
  const lines: string[] = []
  const found: {
    rel: string
    digest: string
    manifest: Record<string, unknown>
    kind: string | null
  }[] = []
  for (const file of [...walk(root)].sort()) {
    const raw = readFileSync(file)
    const parent = path.dirname(file)
    if (SHA.test(path.basename(parent))) {
      const pointer = path.join(path.dirname(parent), 'manifest.json')
      if (existsSync(pointer)) continue
    }
    let manifest: Record<string, unknown>
    try {
      manifest = JSON.parse(raw.toString()) as Record<string, unknown>
    } catch {
      continue
    }
    const rel = path.relative(content, file).split(path.sep).join('/')
    found.push({
      rel,
      digest: createHash('sha256').update(raw).digest('hex'),
      manifest,
      kind: classify(manifest),
    })
  }
  const reference = found.find(f => f.kind === 'note') ?? found[0] ?? null
  if (!reference) return null
  for (const f of found) {
    if (
      f.kind &&
      f.manifest.model === reference.manifest.model &&
      f.manifest.dims === reference.manifest.dims
    )
      lines.push(`${f.rel}\t${f.digest}`)
  }
  lines.sort()
  return createHash('sha256').update(lines.join('\n')).digest('hex')
}

export interface Sidecar {
  dir: string
  /** site-absolute base, `/embeddings/latent/<hash>` */
  base: string
  index: LatentIndex
  docs: LatentDoc[]
  track: LatentTrackDoc[]
  degree: Uint16Array
  probe: LatentProbe
  knn(row: number): LatentRowLinks
  preview(row: number): LatentPreview
  text(doc: number): LatentText
  /** chunks of one part; part 0 is the head file's own chunks */
  textPart(doc: number, part: number): LatentChunk[]
}

const cache = new Map<string, Sidecar | null>()

export function openSidecar(content: string, directory: string): Sidecar | null {
  const key = `${content}::${directory}`
  if (cache.has(key)) return cache.get(key)!
  const sidecar = load(content, directory)
  cache.set(key, sidecar)
  return sidecar
}

function load(content: string, directory: string): Sidecar | null {
  const root = path.join(content, directory, 'latent')
  const pointer = path.join(root, 'index.json')
  if (!existsSync(pointer)) return null
  const { hash } = JSON.parse(readFileSync(pointer, 'utf8')) as { hash: string }
  const dir = path.join(root, hash)
  if (!existsSync(path.join(dir, 'index.json'))) return null
  const expected = expectedHash(content, directory)
  if (expected !== hash) {
    console.warn(
      `[latent] sidecar ${hash.slice(0, 8)} is stale for ${directory} (stores hash ${expected?.slice(0, 8)}); rendering without it`,
    )
    return null
  }
  const index = JSON.parse(readFileSync(path.join(dir, 'index.json'), 'utf8')) as LatentIndex
  const docs = JSON.parse(readFileSync(path.join(dir, 'docs.json'), 'utf8')) as LatentDoc[]
  const trackBytes = readFileSync(path.join(dir, 'track.bin'))
  const track = decodeTrack(
    trackBytes.buffer.slice(trackBytes.byteOffset, trackBytes.byteOffset + trackBytes.byteLength),
    index.kinds,
  )
  const degreeBytes = readFileSync(path.join(dir, 'degree.u16'))
  const degree = new Uint16Array(
    degreeBytes.buffer.slice(
      degreeBytes.byteOffset,
      degreeBytes.byteOffset + degreeBytes.byteLength,
    ),
  )
  const probe = JSON.parse(readFileSync(path.join(dir, 'probe.json'), 'utf8')) as LatentProbe
  const knnBlocks = new Map<number, LatentRowLinks[]>()
  const prevBlocks = new Map<number, LatentPreview[]>()
  const base = `/${directory.replace(/^\/+|\/+$/g, '')}/latent/${hash}`
  return {
    dir,
    base,
    index,
    docs,
    track,
    degree,
    probe,
    knn(row) {
      const b = blockOf(row, index.block)
      let block = knnBlocks.get(b)
      if (!block) {
        const bytes = readFileSync(path.join(dir, 'knn', `${blockName(b)}.bin`))
        block = decodeKnnBlock(
          bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
          b,
          index,
        )
        knnBlocks.set(b, block)
      }
      return block[row - b * index.block]
    },
    preview(row) {
      const b = blockOf(row, index.block)
      let block = prevBlocks.get(b)
      if (!block) {
        block = JSON.parse(
          readFileSync(path.join(dir, 'prev', `${blockName(b)}.json`), 'utf8'),
        ) as LatentPreview[]
        prevBlocks.set(b, block)
      }
      return block[row - b * index.block]
    },
    text(doc) {
      return JSON.parse(
        readFileSync(path.join(dir, 'text', partName(doc, 0)), 'utf8'),
      ) as LatentText
    },
    textPart(doc, part) {
      if (part === 0) return this.text(doc).chunks
      const file = path.join(dir, 'text', partName(doc, part))
      if (!existsSync(file)) return []
      return (JSON.parse(readFileSync(file, 'utf8')) as LatentTextPart).chunks
    },
  }
}

/**
 * The row the page opens on. Widest reach alone picks boilerplate (an interlibrary-loan cover
 * sheet links to every other cover sheet), so the document comes first: among the scoped notes
 * and pdfs, the ones nearest the scope's centroid in the 256-d space are the most typical of the
 * collection, and within the nearest few the row with the widest in-scope reach, then the widest
 * reach, then the highest degree, opens the page. Front and back matter are skipped.
 */
export function defaultFocus(sidecar: Sidecar, prefixes: string[]): number {
  const docOf = new Uint32Array(sidecar.index.rows)
  for (const doc of sidecar.docs) docOf.fill(doc.d, doc.r0, doc.r0 + doc.n)
  const scopedDoc = new Uint8Array(sidecar.docs.length)
  for (const doc of sidecar.docs) scopedDoc[doc.d] = inScope(doc.id, prefixes) ? 1 : 0
  const scoped = sidecar.docs.filter(
    d => scopedDoc[d.d] && (d.s === 'note' || d.s === 'pdf') && d.n >= 3,
  )
  if (scoped.length === 0) return 0
  const candidates = centralDocs(sidecar, scoped, 12)
  let best = -1
  let bestKey: [number, number, number] = [-1, -1, -1]
  for (const doc of candidates) {
    for (let r = doc.r0; r < doc.r0 + doc.n; r++) {
      const preview = sidecar.preview(r)
      if (preview.x.length < 160) continue
      // front and back matter (copyright pages, series lists) link to every other book's
      if (doc.s === 'pdf' && doc.pages && preview.p) {
        if (preview.p < doc.pages * 0.1 || preview.p > doc.pages * 0.95) continue
      } else if (preview.c === 0 && doc.n > 2) continue
      const others = new Set<number>()
      const inside = new Set<number>()
      for (const n of sidecar.knn(r).neighbours) {
        const d = docOf[n.r]
        if (d === doc.d) continue
        others.add(d)
        if (scopedDoc[d]) inside.add(d)
      }
      const key: [number, number, number] = [inside.size, others.size, sidecar.degree[r]]
      if (
        key[0] > bestKey[0] ||
        (key[0] === bestKey[0] &&
          (key[1] > bestKey[1] || (key[1] === bestKey[1] && key[2] > bestKey[2])))
      ) {
        best = r
        bestKey = key
      }
    }
  }
  return best >= 0 ? best : scoped[0].r0
}

/** The `count` documents nearest the centroid of `docs`, by the int8 document vectors. */
function centralDocs(sidecar: Sidecar, docs: LatentDoc[], count: number): LatentDoc[] {
  const file = path.join(sidecar.dir, 'vec', 'docs-256-int8.bin')
  if (!existsSync(file)) return docs.slice(0, count)
  const bytes = readFileSync(file)
  const dims = 256
  const vec = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const centroid = new Float64Array(dims)
  for (const doc of docs) for (let i = 0; i < dims; i++) centroid[i] += vec[doc.d * dims + i]
  let norm = 0
  for (let i = 0; i < dims; i++) norm += centroid[i] * centroid[i]
  norm = Math.sqrt(norm) || 1
  const scored = docs.map(doc => {
    let dot = 0
    let len = 0
    for (let i = 0; i < dims; i++) {
      const v = vec[doc.d * dims + i]
      dot += v * centroid[i]
      len += v * v
    }
    return { doc, cos: dot / (norm * (Math.sqrt(len) || 1)) }
  })
  scored.sort((a, b) => b.cos - a.cos)
  return scored.slice(0, count).map(s => s.doc)
}

export function docOfRow(sidecar: Sidecar, row: number): LatentDoc {
  // docs are not sorted by r0 (time order), so binary search a cached r0 index
  const sorted = byRow(sidecar)
  let lo = 0
  let hi = sorted.length - 1
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (sorted[mid].r0 <= row) lo = mid
    else hi = mid - 1
  }
  return sorted[lo]
}

const rowIndexCache = new WeakMap<Sidecar, LatentDoc[]>()
function byRow(sidecar: Sidecar): LatentDoc[] {
  let sorted = rowIndexCache.get(sidecar)
  if (!sorted) {
    sorted = [...sidecar.docs].sort((a, b) => a.r0 - b.r0)
    rowIndexCache.set(sidecar, sorted)
  }
  return sorted
}
