// Clearance grid: distance (in units) from each cell to the nearest obstacle, capped.
// Ornament reads it to steer and to size things; every placed piece is stamped back in.
// begin/rollback lets a stem try growing and be thrown away if it comes out too short.

import { edt } from './util.js'

export class Field {
  constructor(W, H, cell, cap) {
    this.W = W
    this.H = H
    this.cell = cell
    this.cap = cap
    this.nx = Math.ceil(W / cell) + 1
    this.ny = Math.ceil(H / cell) + 1
    this.d = new Float32Array(this.nx * this.ny)
    this.logI = null
    this.logV = null
    this.logN = 0
    // per-block maxima so the biggest gap is cheap to find
    this.B = 16
    this.bx = Math.ceil(this.nx / this.B)
    this.by = Math.ceil(this.ny / this.B)
    this.bMax = new Float32Array(this.bx * this.by)
    this.bArg = new Int32Array(this.bx * this.by)
    this.bDirty = new Uint8Array(this.bx * this.by).fill(1)
  }

  dirtyRect(i0, i1, j0, j1) {
    const B = this.B
    for (let bj = (j0 / B) | 0; bj <= ((j1 / B) | 0); bj++)
      for (let bi = (i0 / B) | 0; bi <= ((i1 / B) | 0); bi++) this.bDirty[bj * this.bx + bi] = 1
  }

  // blocked: Uint8Array nx*ny, 1 where nothing may go
  setBlocked(blocked) {
    const { nx, ny, cell, cap } = this
    const f = new Float64Array(nx * ny)
    for (let i = 0; i < f.length; i++) f[i] = blocked[i] ? 0 : 1e20
    edt(f, nx, ny)
    for (let i = 0; i < f.length; i++) this.d[i] = Math.min(Math.sqrt(f[i]) * cell, cap)
    this.base = this.d.slice()
    this.bDirty.fill(1)
  }

  clone() {
    const f = new Field(this.W, this.H, this.cell, this.cap)
    f.d.set(this.d)
    f.base = this.base.slice()
    return f
  }

  sample(arr, x, y) {
    const { nx, ny, cell } = this
    const gx = x / cell,
      gy = y / cell
    if (gx < 0 || gy < 0 || gx >= nx - 1 || gy >= ny - 1) return 0
    const i = gx | 0,
      j = gy | 0,
      fx = gx - i,
      fy = gy - j
    const k = j * nx + i
    const a = arr[k],
      b = arr[k + 1],
      c = arr[k + nx],
      d = arr[k + nx + 1]
    return (a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy
  }
  at(x, y) {
    return this.sample(this.d, x, y)
  }
  atBase(x, y) {
    return this.sample(this.base, x, y)
  }

  grad(x, y) {
    const e = this.cell
    const gx = this.at(x + e, y) - this.at(x - e, y)
    const gy = this.at(x, y + e) - this.at(x, y - e)
    const l = Math.hypot(gx, gy) || 1
    return [gx / l, gy / l]
  }

  begin() {
    this.logI = new Int32Array(4096)
    this.logV = new Float32Array(4096)
    this.logN = 0
  }
  // stop logging but hand the log back, so the change can be lifted out later
  detach() {
    const log = { I: this.logI, V: this.logV, N: this.logN }
    this.commit()
    return log
  }
  lift(log) {
    const { nx, B, bx } = this
    for (let n = log.N - 1; n >= 0; n--) {
      const k = log.I[n]
      this.d[k] = log.V[n]
      this.bDirty[((((k / nx) | 0) / B) | 0) * bx + (((k % nx) / B) | 0)] = 1
    }
  }
  commit() {
    this.logI = null
    this.logV = null
    this.logN = 0
  }
  rollback() {
    const { nx, B, bx } = this
    for (let n = this.logN - 1; n >= 0; n--) {
      const k = this.logI[n]
      this.d[k] = this.logV[n]
      this.bDirty[((((k / nx) | 0) / B) | 0) * bx + (((k % nx) / B) | 0)] = 1
    }
    this.commit()
  }

  // a disc of radius r becomes an obstacle
  stamp(x, y, r) {
    const { nx, ny, cell, cap, d } = this
    const R = r + cap
    const i0 = Math.max(0, Math.floor((x - R) / cell)),
      i1 = Math.min(nx - 1, Math.ceil((x + R) / cell))
    const j0 = Math.max(0, Math.floor((y - R) / cell)),
      j1 = Math.min(ny - 1, Math.ceil((y + R) / cell))
    const logging = this.logI !== null
    this.dirtyRect(i0, i1, j0, j1)
    for (let j = j0; j <= j1; j++) {
      const dy = j * cell - y,
        dy2 = dy * dy
      let k = j * nx + i0
      for (let i = i0; i <= i1; i++, k++) {
        const dx = i * cell - x
        const v = Math.sqrt(dx * dx + dy2) - r
        if (v < d[k]) {
          if (logging) {
            if (this.logN >= this.logI.length) this.grow()
            this.logI[this.logN] = k
            this.logV[this.logN] = d[k]
            this.logN++
          }
          d[k] = v < 0 ? 0 : v
        }
      }
    }
  }
  grow() {
    const I = new Int32Array(this.logI.length * 2),
      V = new Float32Array(this.logV.length * 2)
    I.set(this.logI)
    V.set(this.logV)
    this.logI = I
    this.logV = V
  }

  // stamp a polyline of discs (x, y, r) spaced so they overlap
  stampPath(pts, radii) {
    for (let i = 0; i < pts.length; i++) {
      this.stamp(pts[i][0], pts[i][1], Array.isArray(radii) ? radii[i] : radii)
      if (i > 0) {
        const [ax, ay] = pts[i - 1],
          [bx, by] = pts[i]
        const L = Math.hypot(bx - ax, by - ay)
        const r = Array.isArray(radii) ? Math.min(radii[i], radii[i - 1]) : radii
        const n = Math.floor(L / Math.max(this.cell * 0.7, r * 0.8))
        for (let s = 1; s < n; s++)
          this.stamp(ax + ((bx - ax) * s) / n, ay + ((by - ay) * s) / n, r)
      }
    }
  }

  // biggest open spot
  biggest() {
    const { nx, ny, cell, d, B, bx, by, bMax, bArg, bDirty } = this
    for (let bj = 0; bj < by; bj++)
      for (let bi = 0; bi < bx; bi++) {
        const b = bj * bx + bi
        if (!bDirty[b]) continue
        let best = -1,
          arg = 0
        const i1 = Math.min(nx, (bi + 1) * B),
          j1 = Math.min(ny, (bj + 1) * B)
        for (let j = bj * B; j < j1; j++)
          for (let i = bi * B, k = j * nx + i; i < i1; i++, k++)
            if (d[k] > best) {
              best = d[k]
              arg = k
            }
        bMax[b] = best
        bArg[b] = arg
        bDirty[b] = 0
      }
    let best = -1,
      bi = 0
    for (let b = 0; b < bMax.length; b++)
      if (bMax[b] > best) {
        best = bMax[b]
        bi = bArg[b]
      }
    return { x: (bi % nx) * cell, y: Math.floor(bi / nx) * cell, d: best }
  }
}
