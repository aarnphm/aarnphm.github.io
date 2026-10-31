import type { Rng } from './rng.js'
import { circle, ribbon, type Poly, type Vec } from './geometry.js'
import { cos, sin } from './math.js'

/**
 * A drawing surface that also remembers where ink already is. Growth checks
 * the occupancy grid before committing, which is what keeps procedural
 * scrollwork from knotting into itself.
 */

export type Layer = 'field' | 'axis'

export interface Shape {
  poly: Poly
  hole: boolean
  /**
   * `field` shapes get reflected/rotated by the layout's symmetry; `axis`
   * shapes sit on the mirror line and are drawn once.
   */
  layer: Layer
}

interface Mark {
  x: number
  y: number
  r: number
  owner: number
}

class Occupancy {
  private cells = new Map<number, Mark[]>()

  constructor(private cell: number) {}

  private key(ix: number, iy: number): number {
    return (ix + 4096) * 8192 + (iy + 4096)
  }

  add(mark: Mark): void {
    const k = this.key(Math.floor(mark.x / this.cell), Math.floor(mark.y / this.cell))
    const bucket = this.cells.get(k)
    if (bucket) bucket.push(mark)
    else this.cells.set(k, [mark])
  }

  hits(p: Vec, r: number, ignoreOwner = -1): boolean {
    // Marks never exceed a cell in radius, so a 1-cell halo past `r` is enough.
    const span = Math.ceil(r / this.cell) + 1
    const cx = Math.floor(p.x / this.cell)
    const cy = Math.floor(p.y / this.cell)
    for (let ix = cx - span; ix <= cx + span; ix++) {
      for (let iy = cy - span; iy <= cy + span; iy++) {
        const bucket = this.cells.get(this.key(ix, iy))
        if (!bucket) continue
        for (const m of bucket) {
          if (m.owner === ignoreOwner) continue
          const rr = r + m.r
          const dx = m.x - p.x
          const dy = m.y - p.y
          if (dx * dx + dy * dy < rr * rr) return true
        }
      }
    }
    return false
  }
}

export interface SketchOptions {
  rng: Rng
  /** Is a disc of radius `r` at `p` inside the drawable region? */
  inBounds: (p: Vec, r: number) => boolean
  /** Minimum air between separate strokes. */
  gap: number
  /** Stroke weight multiplier from the params. */
  weight: number
}

export class Sketch {
  readonly shapes: Shape[] = []
  readonly rng: Rng
  readonly gap: number
  readonly weight: number
  layer: Layer = 'field'
  private occ: Occupancy
  private owners = 0
  private ignored = -1

  constructor(private opts: SketchOptions) {
    this.rng = opts.rng
    this.gap = opts.gap
    this.weight = opts.weight
    this.occ = new Occupancy(Math.max(8, 6 + opts.gap))
  }

  newOwner(): number {
    return ++this.owners
  }

  inBounds(p: Vec, r: number): boolean {
    return this.opts.inBounds(p, r)
  }

  /** True when a disc (plus half the gap) touches neither ink nor the edge. */
  free(p: Vec, r: number, ignoreOwner = this.ignored): boolean {
    return this.opts.inBounds(p, r) && !this.occ.hits(p, r + this.gap / 2, ignoreOwner)
  }

  /** Is there ink within `r` of `p`? Ignores the region bounds. */
  nearInk(p: Vec, r: number): boolean {
    return this.occ.hits(p, r)
  }

  reserve(p: Vec, r: number, owner = 0): void {
    // Big discs are stored as a ring of smaller ones so grid lookups stay local.
    if (r > 6) {
      const rings = Math.ceil(r / 5)
      for (let k = 0; k <= rings; k++) {
        const rr = (r * k) / rings
        const count = k === 0 ? 1 : Math.ceil((2 * Math.PI * rr) / 5)
        for (let i = 0; i < count; i++) {
          const a = (i / count) * Math.PI * 2
          this.occ.add({ x: p.x + cos(a) * rr, y: p.y + sin(a) * rr, r: 3 + this.gap / 2, owner })
        }
      }
      return
    }
    this.occ.add({ x: p.x, y: p.y, r: r + this.gap / 2, owner })
  }

  reservePath(points: Vec[], halfWidths: number[], owner = 0): void {
    for (let i = 0; i < points.length; i++)
      this.reserve(points[i], Math.max(0.5, halfWidths[i]), owner)
  }

  fill(poly: Poly): void {
    if (poly.length >= 3) this.shapes.push({ poly, hole: false, layer: this.layer })
  }

  hole(poly: Poly): void {
    if (poly.length >= 3) this.shapes.push({ poly, hole: true, layer: this.layer })
  }

  line(points: Vec[], halfWidths: number[], owner = 0, reserve = true): void {
    this.fill(ribbon(points, halfWidths))
    if (reserve) this.reservePath(points, halfWidths, owner)
  }

  dot(p: Vec, r: number, owner = 0): void {
    this.fill(circle(p, r))
    this.reserve(p, r, owner)
  }

  /** Runs `draw` treating one owner's ink as absent — a motif may touch the stem it grows from. */
  ignoring<T>(owner: number, draw: () => T): T {
    const prev = this.ignored
    this.ignored = owner
    try {
      return draw()
    } finally {
      this.ignored = prev
    }
  }

  /** Runs `draw` with shapes routed to the unmirrored axis layer. */
  onAxis(draw: () => void): void {
    const prev = this.layer
    this.layer = 'axis'
    try {
      draw()
    } finally {
      this.layer = prev
    }
  }
}
