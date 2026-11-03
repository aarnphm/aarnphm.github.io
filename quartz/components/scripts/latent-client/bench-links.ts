import type { Bench, Runtime } from './runtime'
import { simplifySlug, type FullSlug } from '../../../util/path'

/**
 * The wikilinks between the bench's documents, the edges the graph view draws, as flat pairs of
 * doc indices. Each pair appears once whichever side links the other.
 */
export async function loadBenchLinks(rt: Runtime): Promise<Int32Array> {
  const index = await fetchData
  const byId = new Map<string, number>()
  for (const doc of rt.docs) byId.set(simplifySlug(doc.id as FullSlug), doc.d)
  const n = rt.docs.length
  const seen = new Set<number>()
  const pairs: number[] = []
  for (const [slug, details] of Object.entries(index)) {
    const a = byId.get(simplifySlug(slug as FullSlug))
    if (a === undefined) continue
    for (const target of details.links ?? []) {
      const b = byId.get(target)
      if (b === undefined || b === a) continue
      const key = a < b ? a * n + b : b * n + a
      if (seen.has(key)) continue
      seen.add(key)
      pairs.push(a, b)
    }
  }
  return Int32Array.from(pairs)
}

/**
 * The links under the points, in four depth bands (one stroke each) so the far ones fade like
 * the far points. At the chunk level a link joins the two documents' first chunks. The hovered
 * point's links are drawn in ink, as the graph lights a hovered node's edges.
 */
export function drawBenchLinks(
  ctx: CanvasRenderingContext2D,
  rt: Runtime,
  bench: Bench,
  at: { xs: Float32Array; ys: Float32Array; depth: Float32Array },
  shown: (i: number) => boolean,
  colour: string,
  ink: string,
): void {
  const links = bench.links
  if (!links || !bench.showLinks) return
  const point = (d: number) => (bench.level === 'docs' ? d : rt.docs[d].r0)
  const bands = [new Path2D(), new Path2D(), new Path2D(), new Path2D()]
  const lit = new Path2D()
  let anyLit = false
  for (let k = 0; k < links.length; k += 2) {
    const a = point(links[k])
    const b = point(links[k + 1])
    if (!shown(a) || !shown(b)) continue
    const t = (at.depth[a] + at.depth[b] + 2) / 4
    const path = a === bench.hover || b === bench.hover ? lit : bands[Math.min(3, (t * 4) | 0)]
    if (path === lit) anyLit = true
    path.moveTo(at.xs[a], at.ys[a])
    path.lineTo(at.xs[b], at.ys[b])
  }
  ctx.save()
  ctx.lineWidth = 0.6
  ctx.strokeStyle = colour
  bands.forEach((path, i) => {
    ctx.globalAlpha = 0.1 + i * 0.08
    ctx.stroke(path)
  })
  if (anyLit) {
    ctx.globalAlpha = 0.8
    ctx.lineWidth = 1
    ctx.strokeStyle = ink
    ctx.stroke(lit)
  }
  ctx.restore()
}
