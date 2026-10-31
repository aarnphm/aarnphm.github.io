import type { Element } from 'hast'
import { s } from 'hastscript'
import type { Pt } from './glyph'
import type { Grapes, Leaf, Tendril } from './grow'

// The drawn parts of the vine as hast. Stems and tendrils are filled ribbons (an SVG stroke
// cannot taper); leaves and clusters are symbols placed with a transform, one definition per
// drawing. Colours are theme tokens set in dropcap.scss, so the same markup prints in both modes.

const f = (n: number) => String(Math.round(n))
const DEG = 180 / Math.PI

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

/** A tapered ribbon along `pts`, from width w0 to w1, as one closed path. */
export function ribbon(pts: Pt[], w0: number, w1: number): Element {
  const n = pts.length
  const left: Pt[] = []
  const right: Pt[] = []
  for (let i = 0; i < n; i++) {
    const a = pts[Math.max(0, i - 1)]
    const b = pts[Math.min(n - 1, i + 1)]
    const dx = b[0] - a[0]
    const dy = b[1] - a[1]
    const len = Math.hypot(dx, dy) || 1
    const w = (w0 + (w1 - w0) * (i / (n - 1))) / 2
    const nx = (-dy / len) * w
    const ny = (dx / len) * w
    left.push([pts[i][0] + nx, pts[i][1] + ny])
    right.push([pts[i][0] - nx, pts[i][1] - ny])
  }
  const outline = [...simplify(left, 1.2), ...simplify(right, 1.2).reverse()]
  const d = outline.map(([x, y], i) => `${i ? 'L' : 'M'}${f(x)} ${f(y)}`).join('') + 'Z'
  return s('path', { d })
}

export function tendril(t: Tendril): Element {
  return ribbon(t.pts, t.w, 1.5)
}

// A grapevine leaf, unit size, blade along +x from the petiole at the origin: five pointed lobes
// with rounded sinuses, drawn as a Catmull-Rom loop whose handles collapse at the tips.
const LOBES: [number, number, boolean][] = [
  [180, 0.2, false],
  [-130, 0.4, true],
  [-96, 0.3, false],
  [-58, 0.54, true],
  [-27, 0.38, false],
  [0, 0.6, true],
  [27, 0.38, false],
  [58, 0.54, true],
  [96, 0.3, false],
  [130, 0.4, true],
  [154, 0.22, false],
]

function leafOutline(): string {
  const c: Pt = [0.44, 0]
  const pts: Pt[] = LOBES.map(([deg, r]) => [
    c[0] + Math.cos(deg / DEG) * r,
    c[1] + Math.sin(deg / DEG) * r,
  ])
  const n = pts.length
  let d = `M${pts[0][0].toFixed(3)} ${pts[0][1].toFixed(3)}`
  for (let i = 0; i < n; i++) {
    const p0 = pts[(i - 1 + n) % n]
    const p1 = pts[i]
    const p2 = pts[(i + 1) % n]
    const p3 = pts[(i + 2) % n]
    const tip1 = LOBES[i][2]
    const tip2 = LOBES[(i + 1) % n][2]
    const c1: Pt = tip1 ? p1 : [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6]
    const c2: Pt = tip2 ? p2 : [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6]
    d += `C${c1[0].toFixed(3)} ${c1[1].toFixed(3)} ${c2[0].toFixed(3)} ${c2[1].toFixed(3)} ${p2[0].toFixed(3)} ${p2[1].toFixed(3)}`
  }
  return d + 'Z'
}

function leafVeins(): string {
  const c: Pt = [0.44, 0]
  return LOBES.filter(l => l[2])
    .map(([deg, r]) => {
      const x = c[0] + Math.cos(deg / DEG) * r * 0.86
      const y = c[1] + Math.sin(deg / DEG) * r * 0.86
      return `M0.06 0Q${(0.1 + x * 0.45).toFixed(3)} ${(y * 0.35).toFixed(3)} ${x.toFixed(3)} ${y.toFixed(3)}`
    })
    .join('')
}

/**
 * One definition of the leaf and the cluster per drawing; `id` keeps the page's ids unique. A
 * <use> clones these into a shadow tree where the page's selectors do not reach, so the inks go
 * on as attributes: the custom properties still inherit through the <use>.
 */
export function defs(id: string): Element {
  const berries: Element[] = []
  for (let row = 0; row < 4; row++)
    for (let k = 0; k < 4 - row; k++)
      berries.push(
        s('circle', {
          cx: f((k - (3 - row) / 2) * 25),
          cy: f(row * 22),
          r: 13.5,
          fill: 'var(--mag-grape)',
          stroke: 'var(--light)',
          strokeWidth: 3,
        }),
      )
  return s('defs', [
    s('g', { id: `${id}l` }, [
      s('path', {
        d: leafOutline(),
        fill: 'var(--mag-leaf)',
        stroke: 'var(--mag-vine)',
        strokeWidth: 0.045,
      }),
      s('path', { d: leafVeins(), fill: 'none', stroke: 'var(--mag-vine)', strokeWidth: 0.03 }),
    ]),
    s('g', { id: `${id}g` }, berries),
  ])
}

export function leaf(id: string, l: Leaf): Element[] {
  const out: Element[] = []
  if (l.stalk)
    out.push(
      s('path', { class: 'gv-stalk', d: `M${f(l.stalk[0])} ${f(l.stalk[1])}L${f(l.x)} ${f(l.y)}` }),
    )
  out.push(
    s('use', {
      href: `#${id}l`,
      transform: `translate(${f(l.x)} ${f(l.y)}) rotate(${(l.angle * DEG).toFixed(0)}) scale(${l.size.toFixed(0)})`,
    }),
  )
  return out
}

export function grapes(id: string, g: Grapes): Element[] {
  const out: Element[] = []
  // The cluster hangs: its top row sits a third up from the centre, and the stalk bends to it.
  const top: Pt = [g.x, g.y - g.size * 0.34]
  if (g.stalk)
    out.push(
      s('path', {
        class: 'gv-stalk',
        d: `M${f(g.stalk[0])} ${f(g.stalk[1])}Q${f((g.stalk[0] + top[0]) / 2)} ${f(g.stalk[1])} ${f(top[0])} ${f(top[1])}`,
      }),
    )
  out.push(
    s('use', {
      href: `#${id}g`,
      transform: `translate(${f(top[0])} ${f(top[1])}) scale(${(g.size / 100).toFixed(2)})`,
    }),
  )
  return out
}
