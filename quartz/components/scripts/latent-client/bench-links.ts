import type { Bench, Runtime } from './runtime'
import { simplifySlug, type FullSlug } from '../../../util/path'
import { knnOf } from './runtime'

/**
 * The wikilinks between the bench's documents as a compressed adjacency: the neighbours of doc
 * `d` are `to[start[d]]..to[start[d + 1]]`, each pair once per side, and `dir` at the same slot
 * says which way the link runs from `d` (OUT, IN, or both). Nothing is drawn from it until a
 * point is hovered or picked, and then only that point's row of it.
 */
export type BenchLinks = { start: Int32Array; to: Int32Array; dir: Uint8Array }

const OUT = 1
const IN = 2

// one build per bench; a new runtime (another visit) gets a new bench and builds again
const building = new WeakMap<Bench, Promise<BenchLinks>>()

async function buildBenchLinks(rt: Runtime): Promise<BenchLinks> {
  const index = await fetchData
  const byId = new Map<string, number>()
  for (const doc of rt.docs) byId.set(simplifySlug(doc.id as FullSlug), doc.d)
  const n = rt.docs.length
  // pair key (low · n + high) → bit OUT when low links high, bit IN when high links low
  const flags = new Map<number, number>()
  for (const [slug, details] of Object.entries(index)) {
    const a = byId.get(simplifySlug(slug as FullSlug))
    if (a === undefined) continue
    for (const target of details.links ?? []) {
      const b = byId.get(target)
      if (b === undefined || b === a) continue
      const key = a < b ? a * n + b : b * n + a
      flags.set(key, (flags.get(key) ?? 0) | (a < b ? OUT : IN))
    }
  }
  const start = new Int32Array(n + 1)
  for (const key of flags.keys()) {
    start[Math.floor(key / n) + 1]++
    start[(key % n) + 1]++
  }
  for (let d = 0; d < n; d++) start[d + 1] += start[d]
  const to = new Int32Array(start[n])
  const dir = new Uint8Array(start[n])
  const fill = start.slice(0, n)
  for (const [key, bits] of flags) {
    const lo = Math.floor(key / n)
    const hi = key % n
    to[fill[lo]] = hi
    dir[fill[lo]++] = bits
    to[fill[hi]] = lo
    // seen from the high side the directions swap
    dir[fill[hi]++] = ((bits & OUT) << 1) | ((bits & IN) >> 1)
  }
  return { start, to, dir }
}

/**
 * The adjacency if it is built; otherwise start building it (once) and redraw when it lands.
 * The first hover or pick pays one pass over the content index, about 2k pairs on this garden.
 */
export function benchLinks(rt: Runtime, bench: Bench, redraw: () => void): BenchLinks | null {
  if (bench.links) return bench.links
  if (!building.has(bench)) {
    const pending = buildBenchLinks(rt)
    building.set(bench, pending)
    pending.then(
      links => {
        if (rt.signal.aborted) return
        bench.links = links
        redraw()
      },
      error => {
        building.delete(bench)
        if (!rt.signal.aborted) console.warn('[latent] bench links', error)
      },
    )
  }
  return null
}

// nearest chunks per bench point, read from the knn blocks the reader already caches; null
// while the block is in flight
const nearCache = new WeakMap<Bench, Map<string, Int32Array | null>>()
let dwell = 0
let dwellKey = ''

/**
 * The points nearest a bench point in the embedding: at the chunk level its k neighbour chunks,
 * at the document level the documents those neighbours of its first chunk belong to.
 */
function nearOf(rt: Runtime, bench: Bench, point: number, redraw: () => void): Int32Array | null {
  let cache = nearCache.get(bench)
  if (!cache) nearCache.set(bench, (cache = new Map()))
  const key = `${bench.level}:${point}`
  if (cache.has(key)) return cache.get(key) ?? null
  const docs = bench.level === 'docs'
  const row = docs ? rt.docs[point]?.r0 : point
  if (row === undefined) return null
  // the pointer crosses many points on its way; fetch the knn block of the one it rests on
  // (a sweep over the cloud would otherwise pull most of the blocks). A redraw of the same
  // point leaves its timer alone
  if (dwellKey === key) return null
  dwellKey = key
  window.clearTimeout(dwell)
  dwell = window.setTimeout(() => {
    dwellKey = ''
    if (bench.hover !== point || cache.has(key)) return
    cache.set(key, null)
    knnOf(rt, row).then(
      links => {
        if (rt.signal.aborted || !links) return
        const out: number[] = []
        for (const { r } of links.neighbours) {
          const p = docs ? rt.docByRow[r]?.d : r
          if (p === undefined || p === point || out.includes(p)) continue
          out.push(p)
        }
        cache.set(key, Int32Array.from(out))
        if (bench.hover === point) redraw()
      },
      () => cache.delete(key),
    )
  }, 90)
  return null
}

type Screen = { xs: Float32Array; ys: Float32Array; depth: Float32Array }

/** What a point's edges came to, for the status line; null where the data has not landed. */
export type EdgeCounts = { near: number | null; out: number | null; back: number | null }

/**
 * A point's edges, drawn over the points (they are a handful; under 53k chunks they would not
 * be seen). Wikilinks are ink: a link out is a solid hairline, a backlink dashed, a link both
 * ways a heavier solid. With `near`, the point's nearest chunks are joined in grey under them.
 * The far half of each kind is drawn fainter, as the far points are.
 */
export function drawBenchLinks(
  ctx: CanvasRenderingContext2D,
  rt: Runtime,
  bench: Bench,
  at: Screen,
  shown: (i: number) => boolean,
  point: number,
  colours: { ink: string; near: string },
  redraw: () => void,
  near = false,
): EdgeCounts | null {
  if (point < 0 || !shown(point)) return null
  const counts: EdgeCounts = { near: null, out: null, back: null }
  const docs = bench.level === 'docs'
  const from = (path: Path2D, b: number) => {
    path.moveTo(at.xs[point], at.ys[point])
    path.lineTo(at.xs[b], at.ys[b])
  }
  const bands = () => [new Path2D(), new Path2D()]
  const deep = (b: number) => (at.depth[point] + at.depth[b] > 0 ? 1 : 0)
  ctx.save()
  if (near) {
    const nearest = nearOf(rt, bench, point, redraw)
    if (nearest) {
      const paths = bands()
      counts.near = 0
      for (const b of nearest)
        if (shown(b)) {
          from(paths[deep(b)], b)
          counts.near++
        }
      ctx.strokeStyle = colours.near
      ctx.lineWidth = 0.75
      ctx.globalAlpha = 0.35
      ctx.stroke(paths[0])
      ctx.globalAlpha = 0.7
      ctx.stroke(paths[1])
    }
  }
  const links = benchLinks(rt, bench, redraw)
  const doc = docs ? point : rt.docByRow[point]?.d
  if (links && doc !== undefined) {
    counts.out = 0
    counts.back = 0
    // out, in, both
    const kinds = [bands(), bands(), bands()]
    for (let k = links.start[doc]; k < links.start[doc + 1]; k++) {
      const d = links.to[k]
      const b = docs ? d : rt.docs[d].r0
      if (!shown(b)) continue
      const bits = links.dir[k]
      if (bits & OUT) counts.out++
      if (bits & IN) counts.back++
      from(kinds[bits === (OUT | IN) ? 2 : bits === IN ? 1 : 0][deep(b)], b)
    }
    ctx.strokeStyle = colours.ink
    const style: [number, number[]][] = [
      [1, []],
      [1, [3, 2]],
      [1.75, []],
    ]
    kinds.forEach((paths, i) => {
      ctx.lineWidth = style[i][0]
      ctx.setLineDash(style[i][1])
      ctx.globalAlpha = 0.4
      ctx.stroke(paths[0])
      ctx.globalAlpha = 0.85
      ctx.stroke(paths[1])
    })
  }
  ctx.restore()
  return counts
}
