// Seeded randomness, gradient noise and small geometry helpers. Units are a 1000×1000 box.

export function rng(seed) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function hash(...xs) {
  let h = 2166136261
  for (const x of xs) {
    const s = String(x)
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i)
      h = Math.imul(h, 16777619)
    }
    h ^= 0x9e37
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

// 2D gradient noise in about [-1, 1]
export function makeNoise(seed) {
  const r = rng(seed)
  const p = new Uint8Array(512)
  const base = [...Array(256).keys()]
  for (let i = 255; i > 0; i--) {
    const j = (r() * (i + 1)) | 0
    ;[base[i], base[j]] = [base[j], base[i]]
  }
  for (let i = 0; i < 512; i++) p[i] = base[i & 255]
  const gx = new Float32Array(256),
    gy = new Float32Array(256)
  for (let i = 0; i < 256; i++) {
    const a = r() * Math.PI * 2
    gx[i] = Math.cos(a)
    gy[i] = Math.sin(a)
  }
  const fade = t => t * t * t * (t * (t * 6 - 15) + 10)
  return (x, y) => {
    const xi = Math.floor(x),
      yi = Math.floor(y)
    const xf = x - xi,
      yf = y - yi
    const X = xi & 255,
      Y = yi & 255
    const g = (h, dx, dy) => gx[h] * dx + gy[h] * dy
    const aa = p[p[X] + Y],
      ab = p[p[X] + Y + 1],
      ba = p[p[X + 1] + Y],
      bb = p[p[X + 1] + Y + 1]
    const u = fade(xf),
      v = fade(yf)
    const x1 = g(aa, xf, yf) + u * (g(ba, xf - 1, yf) - g(aa, xf, yf))
    const x2 = g(ab, xf, yf - 1) + u * (g(bb, xf - 1, yf - 1) - g(ab, xf, yf - 1))
    return (x1 + v * (x2 - x1)) * 1.42
  }
}

export const clamp = (x, a, b) => (x < a ? a : x > b ? b : x)
export const lerp = (a, b, t) => a + (b - a) * t
export const TAU = Math.PI * 2

export function pick(r, arr) {
  return arr[(r() * arr.length) | 0]
}
export function gauss(r) {
  return (r() + r() + r() - 1.5) * 1.15
}

// polyline helpers: points are [x, y]
export function polyLength(pts) {
  let L = 0
  for (let i = 1; i < pts.length; i++)
    L += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
  return L
}

export function resample(pts, step, closed = false) {
  const src = closed ? [...pts, pts[0]] : pts
  if (src.length < 2) return pts.slice()
  const out = [src[0]]
  let carry = 0
  for (let i = 1; i < src.length; i++) {
    const [ax, ay] = src[i - 1],
      [bx, by] = src[i]
    const seg = Math.hypot(bx - ax, by - ay)
    let t = step - carry
    while (t <= seg) {
      const f = t / seg
      out.push([ax + (bx - ax) * f, ay + (by - ay) * f])
      t += step
    }
    carry = seg - (t - step)
  }
  if (!closed) out.push(src[src.length - 1])
  else if (out.length > 2) out.pop()
  return out
}

export function signedArea(pts) {
  let a = 0
  for (let i = 0, n = pts.length; i < n; i++) {
    const [x1, y1] = pts[i],
      [x2, y2] = pts[(i + 1) % n]
    a += x1 * y2 - x2 * y1
  }
  return a / 2
}

// offset a closed polygon inward by d (negative = outward), mitred with a limit
export function offsetPolygon(pts, d) {
  const n = pts.length
  const s = signedArea(pts) > 0 ? 1 : -1 // y-down: positive area = clockwise on screen
  const out = []
  for (let i = 0; i < n; i++) {
    const p0 = pts[(i - 1 + n) % n],
      p1 = pts[i],
      p2 = pts[(i + 1) % n]
    let ax = p1[0] - p0[0],
      ay = p1[1] - p0[1]
    let bx = p2[0] - p1[0],
      by = p2[1] - p1[1]
    const la = Math.hypot(ax, ay) || 1,
      lb = Math.hypot(bx, by) || 1
    ax /= la
    ay /= la
    bx /= lb
    by /= lb
    // inward normal for each edge
    const nax = -ay * s,
      nay = ax * s,
      nbx = -by * s,
      nby = bx * s
    let mx = nax + nbx,
      my = nay + nby
    const ml = Math.hypot(mx, my) || 1
    mx /= ml
    my /= ml
    const cos = mx * nax + my * nay
    const k = d / Math.max(cos, 0.35)
    out.push([p1[0] + mx * k, p1[1] + my * k])
  }
  return out
}

export function bbox(contours) {
  let x0 = Infinity,
    y0 = Infinity,
    x1 = -Infinity,
    y1 = -Infinity
  for (const c of contours)
    for (const [x, y] of c) {
      if (x < x0) x0 = x
      if (y < y0) y0 = y
      if (x > x1) x1 = x
      if (y > y1) y1 = y
    }
  return { x0, y0, x1, y1, w: x1 - x0, h: y1 - y0 }
}

// exact 2D squared Euclidean distance transform (Felzenszwalb & Huttenlocher)
export function edt(f, nx, ny) {
  const INF = 1e20
  const n = Math.max(nx, ny)
  const d = new Float64Array(n),
    v = new Int32Array(n),
    z = new Float64Array(n + 1),
    g = new Float64Array(n)
  const one = len => {
    let k = 0
    v[0] = 0
    z[0] = -INF
    z[1] = INF
    for (let q = 1; q < len; q++) {
      let s
      while (true) {
        const r = v[k]
        s = (g[q] + q * q - (g[r] + r * r)) / (2 * q - 2 * r)
        if (s <= z[k]) {
          k--
          if (k < 0) {
            k = 0
            break
          }
        } else break
      }
      k++
      v[k] = q
      z[k] = s
      z[k + 1] = INF
    }
    k = 0
    for (let q = 0; q < len; q++) {
      while (z[k + 1] < q) k++
      const r = v[k]
      d[q] = (q - r) * (q - r) + g[r]
    }
  }
  for (let x = 0; x < nx; x++) {
    for (let y = 0; y < ny; y++) g[y] = f[y * nx + x]
    one(ny)
    for (let y = 0; y < ny; y++) f[y * nx + x] = d[y]
  }
  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) g[x] = f[y * nx + x]
    one(nx)
    for (let x = 0; x < nx; x++) f[y * nx + x] = d[x]
  }
  return f
}
