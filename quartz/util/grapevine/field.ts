import type { Pt } from './glyph'

// The clearance field the vine grows through, after azlen/initialis (MIT): a raster of what is
// blocked for good (the letter's ink, the frame) turned into a distance transform once, plus the
// discs the growing ornament stamps as it goes. clearance(p) is the radius of the largest empty
// circle at p; every stem step, leaf and cluster asks for the room it needs before taking it.

export interface Disc {
  x: number
  y: number
  r: number
  owner: number
  idx: number
  dead?: boolean
}

export type Ignore = (disc: Disc) => boolean

const BUCKET = 50

export class Field {
  private readonly nx: number
  private readonly ny: number
  private readonly dist: Float32Array
  private readonly order: Uint32Array
  private readonly buckets = new Map<number, Disc[]>()
  readonly discs: Disc[] = []
  private maxR = 0

  /**
   * @param width canvas width in units
   * @param height canvas height in units
   * @param cell raster cell in units
   * @param inset distance from the canvas edge that counts as blocked
   * @param rings blocked polygons (nonzero winding)
   * @param gap keep-away added around everything blocked
   */
  constructor(
    readonly width: number,
    readonly height: number,
    private readonly cell: number,
    inset: number,
    rings: Pt[][],
    private readonly gap: number,
  ) {
    this.nx = Math.ceil(width / cell)
    this.ny = Math.ceil(height / cell)
    const blocked = new Uint8Array(this.nx * this.ny)
    for (let j = 0; j < this.ny; j++)
      for (let i = 0; i < this.nx; i++) {
        const x = (i + 0.5) * cell
        const y = (j + 0.5) * cell
        if (x < inset || y < inset || x > width - inset || y > height - inset)
          blocked[j * this.nx + i] = 1
      }
    this.rasterise(rings, blocked)
    this.dist = this.transform(blocked)
    const open: number[] = []
    for (let k = 0; k < this.dist.length; k++) if (this.dist[k] > gap) open.push(k)
    this.order = Uint32Array.from(open.sort((a, b) => this.dist[b] - this.dist[a]))
  }

  private rasterise(rings: Pt[][], blocked: Uint8Array) {
    const edges: [number, number, number, number][] = []
    for (const ring of rings)
      for (let k = 0; k < ring.length; k++) {
        const [x0, y0] = ring[k]
        const [x1, y1] = ring[(k + 1) % ring.length]
        if (y0 !== y1) edges.push([x0, y0, x1, y1])
      }
    for (let j = 0; j < this.ny; j++) {
      const y = (j + 0.5) * this.cell
      const hits: [number, number][] = []
      for (const [x0, y0, x1, y1] of edges) {
        if (y < Math.min(y0, y1) || y >= Math.max(y0, y1)) continue
        hits.push([x0 + ((y - y0) * (x1 - x0)) / (y1 - y0), y1 > y0 ? 1 : -1])
      }
      hits.sort((a, b) => a[0] - b[0])
      let winding = 0
      for (let h = 0; h + 1 < hits.length; h++) {
        winding += hits[h][1]
        if (winding === 0) continue
        const from = Math.max(0, Math.ceil(hits[h][0] / this.cell - 0.5))
        const to = Math.min(this.nx - 1, Math.floor(hits[h + 1][0] / this.cell - 0.5))
        for (let i = from; i <= to; i++) blocked[j * this.nx + i] = 1
      }
    }
  }

  // Felzenszwalb and Huttenlocher's separable squared distance transform, in cells.
  private transform(blocked: Uint8Array): Float32Array {
    const { nx, ny } = this
    const INF = 1e9
    const f = new Float32Array(nx * ny)
    for (let k = 0; k < f.length; k++) f[k] = blocked[k] ? 0 : INF
    const line = (get: (k: number) => number, set: (k: number, v: number) => void, n: number) => {
      const v = new Int32Array(n)
      const z = new Float64Array(n + 1)
      let k = 0
      v[0] = 0
      z[0] = -INF
      z[1] = INF
      for (let q = 1; q < n; q++) {
        let s = 0
        for (;;) {
          const p = v[k]
          s = (get(q) + q * q - (get(p) + p * p)) / (2 * q - 2 * p)
          if (s > z[k]) break
          k--
        }
        k++
        v[k] = q
        z[k] = s
        z[k + 1] = INF
      }
      k = 0
      const out = new Float64Array(n)
      for (let q = 0; q < n; q++) {
        while (z[k + 1] < q) k++
        out[q] = (q - v[k]) ** 2 + get(v[k])
      }
      for (let q = 0; q < n; q++) set(q, out[q])
    }
    for (let i = 0; i < nx; i++)
      line(
        j => f[j * nx + i],
        (j, val) => (f[j * nx + i] = val),
        ny,
      )
    for (let j = 0; j < ny; j++)
      line(
        i => f[j * nx + i],
        (i, val) => (f[j * nx + i] = val),
        nx,
      )
    for (let k = 0; k < f.length; k++) f[k] = Math.sqrt(f[k]) * this.cell
    return f
  }

  /** Room to the nearest fixed obstacle, less the keep-away. */
  base(x: number, y: number): number {
    const i = Math.min(this.nx - 1, Math.max(0, Math.floor(x / this.cell)))
    const j = Math.min(this.ny - 1, Math.max(0, Math.floor(y / this.cell)))
    return this.dist[j * this.nx + i] - this.gap
  }

  private key(x: number, y: number) {
    return Math.floor(x / BUCKET) * 4096 + Math.floor(y / BUCKET)
  }

  stamp(x: number, y: number, r: number, owner: number, idx = 0): Disc {
    const disc: Disc = { x, y, r, owner, idx }
    this.discs.push(disc)
    if (r > this.maxR) this.maxR = r
    const key = this.key(x, y)
    const list = this.buckets.get(key)
    if (list) list.push(disc)
    else this.buckets.set(key, [disc])
    return disc
  }

  /** Forgets everything a failed stem stamped. */
  rollback(owner: number) {
    for (const disc of this.discs) if (disc.owner === owner) disc.dead = true
  }

  /**
   * Room at (x, y): the largest radius that touches nothing, fixed or stamped. Callers that only
   * ask whether there is `need` room get an answer once the search can no longer change it.
   */
  clearance(x: number, y: number, ignore?: Ignore, need = Infinity): number {
    let room = this.base(x, y)
    if (room <= 0 || room < need) return room
    const bx = Math.floor(x / BUCKET)
    const by = Math.floor(y / BUCKET)
    // Rings of buckets outward: ring k holds no disc edge nearer than (k - 1) buckets less the
    // largest radius, so the walk ends once the room found so far is closer than the next ring.
    for (let k = 0; (k - 1) * BUCKET - this.maxR < Math.min(room, need); k++) {
      for (let i = bx - k; i <= bx + k; i++)
        for (let j = by - k; j <= by + k; j++) {
          if (k > 0 && Math.abs(i - bx) !== k && Math.abs(j - by) !== k) continue
          const list = this.buckets.get(i * 4096 + j)
          if (!list) continue
          for (const disc of list) {
            if (disc.dead || (ignore && ignore(disc))) continue
            const d = Math.hypot(disc.x - x, disc.y - y) - disc.r
            if (d < room) room = d
          }
          if (room < need) return room
        }
    }
    return room
  }

  /** The emptiest spot left: cells in order of fixed room, until none can beat the best. */
  biggest(): { x: number; y: number; r: number } {
    let best = { x: 0, y: 0, r: -Infinity }
    for (const k of this.order) {
      if (this.dist[k] - this.gap <= best.r) break
      const x = ((k % this.nx) + 0.5) * this.cell
      const y = (Math.floor(k / this.nx) + 0.5) * this.cell
      const r = this.clearance(x, y, undefined, best.r)
      if (r > best.r) best = { x, y, r }
    }
    return best
  }
}
