import type { Element } from 'hast'
import { s } from 'hastscript'
import type { Items, Pt, Shape } from './vendor/initialis/ornament.js'
import { stemPoly } from './vendor/initialis/ornament.js'

// The grown items as SVG. Stems are filled outlines (initialis stemPoly); leaves, grapes and
// dots are filled polygons with their detail lines stroked. Colours are the theme tokens set in
// dropcap.scss, so one markup prints in both modes.

const f = (n: number) => String(Math.round(n))
// One unit is 1/1000 em (a tile prints at 70 to 135 px), so two units is well under a pixel.
const TOL = 2

// Ramer-Douglas-Peucker, iterative: drops points that stay within `tol` of the chord.
function simplify(pts: Pt[], tol: number): Pt[] {
  if (pts.length < 3) return pts
  const keep = new Uint8Array(pts.length)
  keep[0] = keep[pts.length - 1] = 1
  const stack: [number, number][] = [[0, pts.length - 1]]
  while (stack.length) {
    const [a, b] = stack.pop()!
    const [ax, ay] = pts[a]
    const [bx, by] = pts[b]
    const len = Math.hypot(bx - ax, by - ay) || 1
    let far = -1
    let dmax = tol
    for (let i = a + 1; i < b; i++) {
      const [x, y] = pts[i]
      const d = Math.abs((bx - ax) * (ay - y) - (ax - x) * (by - ay)) / len
      if (d > dmax) {
        dmax = d
        far = i
      }
    }
    if (far < 0) continue
    keep[far] = 1
    stack.push([a, far], [far, b])
  }
  return pts.filter((_, i) => keep[i])
}

function path(polys: Pt[][], closed = true, tol = TOL): string {
  let d = ''
  for (const poly of polys) {
    if (poly.length < 2) continue
    // a closed ring is split at its far point so neither half is a degenerate chord
    const pts = closed
      ? [
          ...simplify(poly.slice(0, poly.length >> 1), tol),
          ...simplify(poly.slice(poly.length >> 1), tol),
        ]
      : simplify(poly, tol)
    let x0 = NaN
    let y0 = NaN
    for (const [x, y] of pts) {
      const rx = Math.round(x)
      const ry = Math.round(y)
      if (rx === x0 && ry === y0) continue
      d += `${Number.isNaN(x0) ? 'M' : 'L'}${rx} ${ry}`
      x0 = rx
      y0 = ry
    }
    if (closed) d += 'Z'
  }
  return d
}

const circles = (dots: number[][]) =>
  dots.map(([x, y, r]) => s('circle', { cx: f(x), cy: f(y), r: f(Math.max(0.6, r)) }))

/** `tol` is the outline simplification in field units; the rules print smaller than the tile.
 * `contrast` is initialis's broad-nib thick/thin (0 leaves the stem width alone). */
export function drawItems(
  items: Items,
  detailW: number,
  tol = TOL,
  paint: { contrast?: number; cells?: Pt[][] } = {},
): Element[] {
  const grapes: Shape[] = []
  const flowers: Shape[] = []
  for (const fl of items.flowers)
    ((fl as Shape & { kind?: string }).kind === 'grapes' ? grapes : flowers).push(fl)
  const lines = [...items.leaves, ...flowers].flatMap(sh => sh.lines).filter(l => l.length > 1)
  const contrast = paint.contrast ?? 0
  const cells = paint.cells?.length
    ? [s('g', { class: 'gv-cells' }, [s('path', { d: path(paint.cells, true, tol) })])]
    : []
  return [
    ...cells,
    s('g', { class: 'gv-stem' }, [
      s('path', {
        d: path(
          items.stems.map(st => stemPoly(st, contrast, (35 * Math.PI) / 180)),
          true,
          tol,
        ),
      }),
      ...circles([...items.dots, ...items.beads]),
    ]),
    s('g', { class: 'gv-leaf' }, [
      s('path', {
        d: path(
          [...items.leaves, ...flowers].flatMap(sh => sh.polys),
          true,
          tol,
        ),
      }),
      ...(lines.length
        ? [s('path', { class: 'gv-vein', d: path(lines, false, tol), strokeWidth: f(detailW) })]
        : []),
    ]),
    s(
      'g',
      { class: 'gv-grapes' },
      grapes.flatMap(g => g.polys.map(p => s('path', { d: path([p], true, tol) }))),
    ),
  ]
}
