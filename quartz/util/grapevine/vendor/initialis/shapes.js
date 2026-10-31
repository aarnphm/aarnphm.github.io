// Leaf and flower geometry. Everything returns
//   { polys: [closed point lists], lines: [detail polylines], dots: [[x, y, r]], axis, half }
// so the renderer can fill, outline or knock out any of it the same way. Each kind of leaf and
// flower is a module of its own (kinds/leaf/, kinds/flower/); this has the blades they're built from.

import { kind as kindOf } from './kinds.js'
import { TAU, clamp } from './util.js'

const frac = x => x - Math.floor(x)

// half-width profile along the leaf, 0..1, for one side
// (type: a name below, or a function (t, side, o) of your own)
function profile(type, t, side, o) {
  if (typeof type === 'function') return type(t, side, o)
  const s = Math.sin
  switch (type) {
    case 'lance':
      return Math.pow(s(Math.PI * t), 0.9)
    case 'willow':
      return Math.pow(s(Math.PI * t), 1.2)
    case 'ovate':
      return Math.pow(s(Math.PI * Math.pow(t, 0.68)), 0.7)
    case 'teardrop':
      return t < 0.68
        ? Math.pow(s(((Math.PI / 2) * t) / 0.68), 0.75)
        : Math.sqrt(Math.max(0, 1 - ((t - 0.68) / 0.32) ** 2))
    case 'boteh':
      // paisley: a round bulb, then a long taper into the tail
      return t < 0.36
        ? Math.sqrt(Math.max(0, 1 - ((t - 0.36) / 0.36) ** 2))
        : Math.pow(1 - (t - 0.36) / 0.64, 1.25)
    case 'petal':
      return t < 0.55
        ? Math.pow(t / 0.55, 0.65)
        : Math.sqrt(Math.max(0, 1 - ((t - 0.55) / 0.45) ** 2))
    case 'acanthus': {
      const env = Math.pow(s(Math.PI * Math.pow(t, 0.8)), 0.7)
      const n = Math.max(1, o.lobes)
      const f = frac(t * n + (side > 0 ? 0 : 0.5) + 0.15)
      const lobe = f < 0.78 ? Math.pow(f / 0.78, 0.6) : 1 - ((f - 0.78) / 0.22) * 0.92
      return env * (1 - o.depth + o.depth * lobe)
    }
    case 'oak': {
      const env = Math.pow(s(Math.PI * Math.pow(t, 0.85)), 0.6)
      const n = Math.max(1, o.lobes)
      const f = frac(t * n + (side > 0 ? 0 : 0.5))
      return env * (1 - o.depth * 0.6 * (1 - Math.sin(Math.PI * f)))
    }
    default:
      return s(Math.PI * t)
  }
}

// a single bladed leaf along a bending axis
export function blade(type, o) {
  const N = o.N || 26
  const axis = [],
    dirs = []
  let x = o.x,
    y = o.y
  const step = o.len / N
  for (let i = 0; i <= N; i++) {
    const t = i / N
    const a = o.dir + o.bend * t + o.curl * t * t * t
    axis.push([x, y])
    dirs.push(a)
    x += Math.cos(a) * step
    y += Math.sin(a) * step
  }
  const W = o.len * o.width * 0.5
  const left = [],
    right = [],
    half = []
  for (let i = 0; i <= N; i++) {
    const t = i / N
    let hl = profile(type, t, 1, o),
      hr = profile(type, t, -1, o)
    if (o.serr) {
      const z = Math.abs(Math.sin(t * Math.PI * 14)) * o.serr * 0.12
      hl += z
      hr += z
    }
    const nx = -Math.sin(dirs[i]),
      ny = Math.cos(dirs[i])
    const [ax, ay] = axis[i]
    left.push([ax + nx * hl * W, ay + ny * hl * W])
    right.push([ax - nx * hr * W, ay - ny * hr * W])
    half.push(Math.max(hl, hr) * W)
  }
  const poly = [...left, ...right.reverse().slice(1, -1)]
  return { polys: [poly], axis, dirs, half, W, left, right: right.reverse(), lines: [], dots: [] }
}

// veins, hatching or an inline for a blade
export function bladeDetail(b, kind, o = {}) {
  const lines = []
  const { axis, dirs, half } = b
  const N = axis.length - 1
  if (kind === 'vein' || kind === 'veins') lines.push(axis.slice(1, Math.round(N * 0.82)))
  if (kind === 'veins') {
    const m = Math.max(2, Math.round(N / 5))
    for (let k = 1; k <= m; k++) {
      const i = Math.round((k / (m + 1)) * N * 0.8)
      const j = Math.min(N, i + 3)
      for (const sd of [1, -1]) {
        const [ax, ay] = axis[i]
        const nx = -Math.sin(dirs[j]) * sd,
          ny = Math.cos(dirs[j]) * sd
        const [bx, by] = axis[j]
        const r = half[j] * 0.7
        lines.push([
          [ax, ay],
          [bx + nx * r, by + ny * r],
        ])
      }
    }
  }
  if (kind === 'hatch') {
    const m = Math.max(3, Math.round(N * 0.7))
    for (let k = 1; k < m; k++) {
      const i = Math.round((k / m) * N * 0.92)
      const [ax, ay] = axis[i]
      const a = dirs[i]
      const nx = -Math.sin(a),
        ny = Math.cos(a)
      const r = half[i] * 0.85
      if (r < 1) continue
      lines.push([
        [ax, ay],
        [ax + nx * r + Math.cos(a) * r * 0.35, ay + ny * r + Math.sin(a) * r * 0.35],
      ])
    }
  }
  if (kind === 'inline') {
    const pts = []
    for (let i = 2; i <= N - 2; i++) {
      const [ax, ay] = axis[i],
        a = dirs[i],
        nx = -Math.sin(a),
        ny = Math.cos(a),
        r = half[i] * 0.55
      pts.push([ax + nx * r, ay + ny * r])
    }
    for (let i = N - 2; i >= 2; i--) {
      const [ax, ay] = axis[i],
        a = dirs[i],
        nx = -Math.sin(a),
        ny = Math.cos(a),
        r = half[i] * 0.55
      pts.push([ax - nx * r, ay - ny * r])
    }
    if (pts.length > 3) {
      pts.push(pts[0])
      lines.push(pts)
    }
  }
  return lines
}

// a leaf of any type: compound types are built from blades
// discs (x, y, r) that cover a blade, for fitting and stamping
function bladeCover(b, step = 2) {
  const out = []
  for (let i = 0; i < b.axis.length; i += step)
    out.push([b.axis[i][0], b.axis[i][1], b.half[i], i / (b.axis.length - 1)])
  return out
}

// a leaf of any type: compound types are built from blades.
// cover: [x, y, r, t] discs covering the whole leaf (t = how far from the base, 0..1)
export function leaf(type, o) {
  const m = kindOf('leaf', type)
  if (m)
    return m.draw({
      type,
      o,
      settings: o.settings || {},
      blade,
      bladeDetail,
      bladeCover,
      circlePoly,
      circleLine,
      resampleClosed,
      TAU,
      clamp,
    })
  // one nobody has: a plain blade of that profile
  const b = blade(type, o)
  b.lines = bladeDetail(b, o.detail)
  b.cover = bladeCover(b, 2)
  return b
}

function circleLine(x, y, r) {
  const p = circlePoly(x, y, r)
  p.push(p[0])
  return p
}
function resampleClosed(poly, step) {
  const out = []
  let carry = 0
  for (let i = 0; i < poly.length; i++) {
    const [ax, ay] = poly[i],
      [bx, by] = poly[(i + 1) % poly.length]
    const L = Math.hypot(bx - ax, by - ay)
    let t = step - carry
    while (t <= L) {
      out.push([ax + ((bx - ax) * t) / L, ay + ((by - ay) * t) / L])
      t += step
    }
    carry = L - (t - step)
  }
  return out
}

function circlePoly(x, y, r, n = 0) {
  n = n || clamp(Math.round(r * 1.2), 10, 64)
  const p = []
  for (let i = 0; i < n; i++) {
    const a = (i / n) * TAU
    p.push([x + Math.cos(a) * r, y + Math.sin(a) * r])
  }
  return p
}
export { circlePoly }

// a flower centred at (x, y) with radius r, or oriented along dir for the stem-end kinds (head:
// true): P(u, v) is a point u along dir and v across it
export function flower(type, o) {
  const { x, y, r } = o
  const rot = o.rot || 0
  const out = { polys: [], lines: [], dots: [], radius: r }
  const ring = (n, r0, r1, width, shape, off = 0) => {
    for (let i = 0; i < n; i++) {
      const a = rot + off + (i / n) * TAU
      const b = blade(shape, {
        x: x + Math.cos(a) * r0,
        y: y + Math.sin(a) * r0,
        dir: a,
        len: r1 - r0,
        width,
        bend: 0,
        curl: 0,
        N: 14,
      })
      out.polys.push(b.polys[0])
      if (o.detail && o.detail !== 'none')
        out.lines.push(...bladeDetail(b, o.detail === 'veins' ? 'vein' : o.detail))
    }
  }
  const m = kindOf('flower', type)
  // one nobody has: a disc
  if (!m) {
    out.polys.push(circlePoly(x, y, r))
    return out
  }
  const dir = o.dir,
    fx = Math.cos(dir),
    fy = Math.sin(dir),
    nx = -fy,
    ny = fx
  const P = (u, v) => [x + fx * u + nx * v, y + fy * u + ny * v]
  const det = o.detail && o.detail !== 'none'
  m.draw({
    type,
    o,
    settings: o.settings || {},
    x,
    y,
    r,
    rot,
    out,
    ring,
    blade,
    bladeDetail,
    circlePoly,
    circleLine,
    TAU,
    clamp,
    dir,
    fx,
    fy,
    nx,
    ny,
    P,
    det,
  })
  return out
}
