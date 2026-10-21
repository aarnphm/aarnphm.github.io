import { GATE, hourAngle, stoneOuter } from '../404-gate'
import { mix, mul, put, step, type Grid, type RGB } from '../404-pixel'
import { run, trace } from './svg-trace'

// Stone tone at a point in the ring's upright frame: 0.5 is the flat plate, each 0.25 is one dither
// stop. The sun sits up and to the right. Every voussoir is chamfered, lit on its sunward edges and
// shaded on the rest, and the soffit under the ring is in deep shade.
function stoneTone(x: number, y: number, side: number) {
  const dx = x - GATE.x
  const dy = y - GATE.y
  const r = Math.hypot(dx, dy)
  if (r < GATE.r) return 0.08 + 0.3 * (dx / GATE.r) ** 2
  const a = Math.atan2(dy, dx)
  const h = (Math.round((a * 6) / Math.PI) + 15) % 12
  const out = stoneOuter(h)
  if (r <= out + 2) {
    const u = (r - GATE.r) / (out - GATE.r)
    const v = (Math.atan2(Math.sin(a - hourAngle(h)), Math.cos(a - hourAngle(h))) * 12) / Math.PI
    const eu = 8 / (out - GATE.r)
    const ev = (8 * 12) / (Math.PI * r)
    const cu = u > 1 - eu ? 1 : u < eu ? -1 : 0
    const cv = v > 1 - ev ? 1 : v < ev - 1 ? -1 : 0
    const nx = Math.cos(a) * cu - Math.sin(a) * cv
    const ny = Math.sin(a) * cu + Math.cos(a) * cv
    return 0.5 + 0.45 * (nx * 0.6 * side - ny * 0.8)
  }
  return 0.5
}

const probe = (() => {
  let ctx: CanvasRenderingContext2D | null = null
  return (css: string): RGB => {
    if (!ctx) {
      const c = document.createElement('canvas')
      c.width = 1
      c.height = 1
      ctx = c.getContext('2d', { willReadFrequently: true })!
    }
    ctx.clearRect(0, 0, 1, 1)
    ctx.fillStyle = css
    ctx.fillRect(0, 0, 1, 1)
    const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data
    return [r, g, b]
  }
})()

const MATERIALS = ['nf-fill-stone', 'nf-fill-rose', 'nf-fill-sage']
// Stops 0..4 of the tone dither map onto deep, shade, plate, plate, light.
const PLATE = [0, 1, 2, 2, 3]

export type Print = { ink: RGB; stone: RGB[]; rose: RGB[]; sage: RGB[] }
export type Masonry = { print: Print; spill: number[]; bands: number[] }

let ruinOwner = new Uint8Array(0)
let ruinVoided = new Uint8Array(0)
let ruinFrame = new Uint8Array(0)

// Prints the ruin from the hidden SVG geometry: plates dithered through the stone's lighting,
// keylines one pixel thick, dial marks wherever they cover half a pixel, and chips bitten out
// wherever an `nf-void` says, inked where they break into stone. Every stone is lit in its own
// upright place in the ring, so its chamfers travel with it when it slips and when the ring leans.
// Below `ground` the ring is buried in the island and prints nothing, so the turf's own edge is the
// line it sinks behind. The light that spills onto the lower stones and the tablet animates on top
// of this, so its pixels are returned as a list.
export function rasterRuin(
  d: Uint8ClampedArray,
  g: Grid,
  svg: SVGSVGElement,
  ground: (x: number) => number,
  side: number,
  dim: number,
) {
  const art = svg.querySelector('#nf-temple-art')
  const root = svg.getScreenCTM()
  if (!art || !root) return null
  const rootInv = root.inverse()
  const { bw, bh } = g
  const sheet = document.createElement('canvas')
  sheet.width = bw
  sheet.height = bh
  const scratch = sheet.getContext('2d', { willReadFrequently: true })!
  const stops: RGB[][] = MATERIALS.map(() => [])
  const paper: RGB = [255, 252, 240]
  let ink: RGB | null = null
  const toBuffer = (el: SVGGraphicsElement) => {
    const ctm = el.getScreenCTM()
    return ctm && g.toBuffer.multiply(rootInv.multiply(ctm))
  }
  const tone = (el: Element, id: number) => {
    if (stops[id].length === 0) {
      const base = mul(probe(getComputedStyle(el).fill), dim)
      stops[id] = [mul(base, 0.7), mul(base, 0.86), base, mix(base, paper, 0.55)]
    }
    return stops[id]
  }
  const inside = (i: number, j: number) => i >= 0 && j >= 0 && i < bw && j < bh
  // Which paint owns each pixel, stacked in document order as the SVG would stack them: 1..3 a
  // material plate, 4 ink, 0 bare. Later elements cover earlier ones, keylines included.
  const INK = 4
  const total = bw * bh
  if (ruinOwner.length < total) {
    ruinOwner = new Uint8Array(total)
    ruinVoided = new Uint8Array(total)
    ruinFrame = new Uint8Array(total)
  } else {
    ruinOwner.fill(0, 0, total)
    ruinVoided.fill(0, 0, total)
    ruinFrame.fill(0, 0, total)
  }
  const owner = ruinOwner
  const voided = ruinVoided
  // Which frame lights each plate pixel: 0 the scene, n a stone's upright place in the ring.
  const frame = ruinFrame
  const frames: DOMMatrix[] = [new DOMMatrix()]
  const frameOf = new Map<Element, number>()
  const frameFor = (el: Element) => {
    const stone = el.closest<SVGGElement>('.nf-stone')
    const m = stone?.getScreenCTM()
    if (!stone || !m) return 0
    let f = frameOf.get(stone)
    if (f === undefined) {
      f = frames.push(rootInv.multiply(m).inverse()) - 1
      frameOf.set(stone, f)
    }
    return f
  }
  const cover = (el: SVGPathElement, m: DOMMatrix, set: number, threshold: number, f = 0) => {
    const b = el.getBBox()
    const xs: number[] = []
    const ys: number[] = []
    for (const [x, y] of [
      [b.x, b.y],
      [b.x + b.width, b.y],
      [b.x, b.y + b.height],
      [b.x + b.width, b.y + b.height],
    ]) {
      const q = m.transformPoint(new DOMPoint(x, y))
      xs.push(q.x)
      ys.push(q.y)
    }
    const x0 = Math.max(0, Math.floor(Math.min(...xs)) - 1)
    const y0 = Math.max(0, Math.floor(Math.min(...ys)) - 1)
    const w = Math.min(bw, Math.ceil(Math.max(...xs)) + 1) - x0
    const h = Math.min(bh, Math.ceil(Math.max(...ys)) + 1) - y0
    if (w <= 0 || h <= 0) return
    scratch.resetTransform()
    scratch.clearRect(x0, y0, w, h)
    scratch.setTransform(m)
    scratch.fill(new Path2D(el.getAttribute('d') ?? ''))
    const a = scratch.getImageData(x0, y0, w, h).data
    for (let j = 0; j < h; j++)
      for (let i = 0; i < w; i++) {
        if (a[(j * w + i) * 4 + 3] < threshold) continue
        const p = (y0 + j) * bw + x0 + i
        owner[p] = set
        frame[p] = f
        if (set === 0) voided[p] = 1
      }
  }

  for (const el of art.querySelectorAll<SVGPathElement>('path')) {
    const m = toBuffer(el)
    if (!m) continue
    const cls = el.getAttribute('class') ?? ''
    const id = MATERIALS.findIndex(c => cls.includes(c))
    if (cls.includes('nf-key')) {
      ink ??= probe(getComputedStyle(el).stroke)
      const px = run(trace(el.getAttribute('d') ?? ''), m)
      for (let s = 0; s < px.length; s += 2)
        if (inside(px[s], px[s + 1])) owner[px[s + 1] * bw + px[s]] = INK
    } else if (id >= 0) {
      tone(el, id)
      cover(el, m, id + 1, 128, frameFor(el))
    } else if (cls.includes('nf-ink')) cover(el, m, INK, 110)
    else if (cls.includes('nf-void')) cover(el, m, 0, 128)
  }
  sheet.width = 0
  sheet.height = 0
  ink ??= [16, 15, 15]
  for (let j = 1; j < bh - 1; j++)
    for (let i = 1; i < bw - 1; i++) {
      const p = j * bw + i
      if (owner[p] === 0 || owner[p] === INK) continue
      if (voided[p - 1] || voided[p + 1] || voided[p - bw] || voided[p + bw]) owner[p] = INK
    }

  const spill: number[] = []
  const bands: number[] = []
  const sill = GATE.y + GATE.r * 0.7
  for (let j = 0; j < bh; j++)
    for (let i = 0; i < bw; i++) {
      const p = j * bw + i
      const o = owner[p]
      const x = g.sx[i]
      const y = g.sy[j]
      if (o === 0 || (y >= ground(x) && Math.hypot(x - GATE.x, y - GATE.y) < 270)) continue
      if (o === INK) {
        put(d, p * 4, ink)
        continue
      }
      const f = frames[frame[p]]
      const lx = f.a * x + f.c * y + f.e
      const ly = f.b * x + f.d * y + f.f
      put(d, p * 4, stops[o - 1][PLATE[step(5, stoneTone(lx, ly, side) * 0.75 + 0.125, i, j)]])
      if (o === 1 && y > sill && Math.abs(x - GATE.x) < 150) {
        spill.push(p)
        bands.push(Math.min(3, Math.floor((y - sill) / 40)))
      }
    }

  const print = { ink, stone: stops[0], rose: stops[1], sage: stops[2] }
  return { print, spill, bands } satisfies Masonry
}
