// Shared by the server render and the inline script, so the first paint and every drag agree.
// Every drawn mark and readout value comes from timeView or locusView.

export type Lock = 'none' | 'sigma' | 'omega' | 'zeta'
export type Comp = 'none' | 'integrator' | 'pi'

export type TimeState =
  | { kind: 'first'; p: number }
  | { kind: 'pair'; re: number; im: number }
  | { kind: 'real'; p1: number; p2: number }

type Range = readonly [number, number]
export type Scale = { re: Range; im: Range; t: Range; c: Range; tStep: number }

export const TIME_SCALE: Scale = { re: [-9, 3], im: [-7, 7], t: [0, 8], c: [0, 2], tStep: 1 }
export const LOCUS_SCALE: Scale = { re: [-13, 2], im: [-7, 7], t: [0, 12], c: [0, 2], tStep: 2 }

// One s-plane unit is the same length on both axes, so angles and the zeta ray read true.
export const UNIT = 18
export const PLANE_X = 24
export const PLANE_Y = 22
export const PLOT_X = 30
export const PLOT_Y = 22
export const PLOT_W = 300
const BOTTOM = 34

export const planeW = (s: Scale) => (s.re[1] - s.re[0]) * UNIT
export const planeH = (s: Scale) => (s.im[1] - s.im[0]) * UNIT
export const viewH = (s: Scale) => PLANE_Y + planeH(s) + BOTTOM
export const planeViewW = (s: Scale) => PLANE_X + planeW(s) + 10
export const plotViewW = PLOT_X + PLOT_W + 12

const r1 = (n: number) => Math.round(n * 10) / 10
export const sx = (s: Scale, re: number) => r1(PLANE_X + (re - s.re[0]) * UNIT)
export const sy = (s: Scale, im: number) => r1(PLANE_Y + (s.im[1] - im) * UNIT)
export const tx = (s: Scale, t: number) => r1(PLOT_X + ((t - s.t[0]) / (s.t[1] - s.t[0])) * PLOT_W)
export const cy = (s: Scale, c: number) =>
  r1(PLOT_Y + ((s.c[1] - c) / (s.c[1] - s.c[0])) * planeH(s))

// Pointer snaps: within a few pixels of an axis or of the other real pole, the pole lands on it.
const SNAP = 4.5 / UNIT
const MIN_IM = 5 / UNIT
const MIN_R = 0.1
const BAND = 0.02

export type Attrs = Record<string, string | number>

// ---------- formatting ----------

const MINUS = '−'
export const fmt = (x: number, d = 2) => {
  const v = Math.abs(x) < 0.5 * 10 ** -d ? 0 : x
  return (v < 0 ? MINUS : '') + Math.abs(v).toFixed(d)
}
const sec = (x: number | null) => (x === null ? '–' : `${fmt(x)} s`)
const rad = (x: number | null) => (x === null ? '–' : `${fmt(x)} rad/s`)

// ---------- first and second order: closed-form step responses ----------

export const stepAt = (st: TimeState, t: number): number => {
  if (st.kind === 'first') return 1 - Math.exp(st.p * t)
  if (st.kind === 'pair') {
    const sig = -st.re
    const w = st.im
    return 1 - Math.exp(-sig * t) * (Math.cos(w * t) + (sig / w) * Math.sin(w * t))
  }
  const s1 = -st.p1
  const s2 = -st.p2
  if (Math.abs(s1 - s2) < 1e-9) return 1 - (1 + s1 * t) * Math.exp(-s1 * t)
  return 1 - (s2 * Math.exp(-s1 * t) - s1 * Math.exp(-s2 * t)) / (s2 - s1)
}

export type Case = 'stable' | 'unstable' | 'undamped' | 'underdamped' | 'critical' | 'overdamped'

export const caseOf = (st: TimeState): Case => {
  if (st.kind === 'first') return st.p < 0 ? 'stable' : 'unstable'
  if (st.kind === 'pair') return st.re > 0 ? 'unstable' : st.re === 0 ? 'undamped' : 'underdamped'
  if (st.p1 > 0 || st.p2 > 0) return 'unstable'
  return st.p1 === st.p2 ? 'critical' : 'overdamped'
}

const bisect = (f: (t: number) => number, lo: number, hi: number) => {
  let a = lo
  let b = hi
  const fa = f(a) > 0
  for (let i = 0; i < 60; i++) {
    const m = (a + b) / 2
    if (f(m) > 0 === fa) a = m
    else b = m
  }
  return (a + b) / 2
}

const firstCross = (st: TimeState, level: number): number | null => {
  const h = 0.0015
  let prev = 0
  for (let t = h; t <= 60; t += h) {
    if (stepAt(st, t) >= level) return bisect(u => stepAt(st, u) - level, prev, t)
    prev = t
  }
  return null
}

// Exact 2% settling time: the last time |c - 1| leaves the band.
const settleBand = (st: TimeState): number | null => {
  const c = caseOf(st)
  if (c === 'unstable' || c === 'undamped') return null
  if (st.kind === 'first') return -Math.log(BAND) / -st.p
  if (st.kind === 'real') return bisect(t => 1 - stepAt(st, t) - BAND, 0, 400)
  const sig = -st.re
  const wn = Math.hypot(st.re, st.im)
  // |c - 1| <= (wn / wd) e^{-sigma t}, so nothing leaves the band after tEnv.
  const tEnv = Math.log(wn / st.im / BAND) / sig
  const h = Math.min(Math.PI / st.im / 100, tEnv / 20000)
  const out = (t: number) => Math.abs(stepAt(st, t) - 1) - BAND
  for (let t = tEnv; t > 0; t -= h) if (out(t) > 0) return bisect(out, t, t + h)
  return 0
}

export type TimeMetrics = {
  cls: Case
  zeta: number | null
  wn: number | null
  sigmaD: number | null
  wd: number | null
  t10: number | null
  t90: number | null
  tp: number | null
  os: number | null
  tsRule: number | null
  tsBand: number | null
}

export const timeMetrics = (st: TimeState): TimeMetrics => {
  const cls = caseOf(st)
  const settles = cls !== 'unstable' && cls !== 'undamped'
  const rises = cls !== 'unstable'
  let zeta: number | null = null
  let wn: number | null = null
  let sigmaD: number | null = null
  let wd: number | null = null
  let tp: number | null = null
  let os: number | null = null
  let tsRule: number | null = null
  if (st.kind === 'pair') {
    wn = Math.hypot(st.re, st.im)
    zeta = -st.re / wn
    sigmaD = -st.re
    wd = st.im
    if (cls !== 'unstable') {
      tp = Math.PI / st.im
      os = 100 * Math.exp((st.re * Math.PI) / st.im)
    }
    if (settles) tsRule = 4 / sigmaD
  } else if (st.kind === 'real') {
    if (st.p1 * st.p2 > 0) {
      wn = Math.sqrt(st.p1 * st.p2)
      zeta = -(st.p1 + st.p2) / (2 * wn)
    }
    if (settles) os = 0
  } else if (settles) tsRule = 4 / -st.p
  const t10 = rises ? firstCross(st, 0.1) : null
  const t90 = rises ? firstCross(st, 0.9) : null
  return {
    cls,
    zeta,
    wn,
    sigmaD,
    wd,
    t10,
    t90,
    tp,
    os,
    tsRule,
    tsBand: settles ? settleBand(st) : null,
  }
}

const CASE_WORDS: Record<Case, string> = {
  stable: 'stable',
  unstable: 'unstable',
  undamped: 'undamped',
  underdamped: 'underdamped',
  critical: 'critically damped',
  overdamped: 'overdamped',
}

export const polesText = (st: TimeState) =>
  st.kind === 'first'
    ? fmt(st.p)
    : st.kind === 'pair'
      ? `${fmt(st.re)} ± j${fmt(st.im)}`
      : st.p1 === st.p2
        ? `${fmt(st.p1)} (double)`
        : `${fmt(Math.max(st.p1, st.p2))}, ${fmt(Math.min(st.p1, st.p2))}`

const denomText = (st: TimeState) => {
  const [a, b] =
    st.kind === 'pair'
      ? [-2 * st.re, st.re * st.re + st.im * st.im]
      : st.kind === 'real'
        ? [-(st.p1 + st.p2), st.p1 * st.p2]
        : [0, 0]
  const term = (v: number, unit: string) => ` ${v < 0 ? MINUS : '+'} ${fmt(Math.abs(v))}${unit}`
  return `s²${term(a, 's')}${term(b, '')}`
}

export const TIME_ROWS = {
  first: ['case', 'pole', 'tau', 'tr', 'tsRule', 'tsBand'],
  second: [
    'case',
    'poles',
    'denom',
    'zeta',
    'wn',
    'sigmaD',
    'wd',
    'tr',
    'tp',
    'os',
    'tsRule',
    'tsBand',
  ],
} as const

export const timeReadout = (st: TimeState, m: TimeMetrics): Record<string, string> => {
  const tr = m.t10 !== null && m.t90 !== null ? m.t90 - m.t10 : null
  if (st.kind === 'first') {
    const a = -st.p
    return {
      case: CASE_WORDS[m.cls],
      pole: fmt(st.p),
      tau: sec(1 / a),
      tr: sec(m.cls === 'stable' ? Math.log(9) / a : null),
      tsRule: sec(m.tsRule),
      tsBand: sec(m.tsBand),
    }
  }
  return {
    case: CASE_WORDS[m.cls],
    poles: polesText(st),
    denom: denomText(st),
    zeta: m.zeta === null ? '–' : fmt(m.zeta, 3),
    wn: rad(m.wn),
    sigmaD: m.sigmaD === null ? '–' : `${fmt(m.sigmaD)} 1/s`,
    wd: rad(m.wd),
    tr: sec(tr),
    tp: sec(m.tp),
    os: m.os === null ? '–' : `${fmt(m.os, 1)}%`,
    tsRule: sec(m.tsRule),
    tsBand: sec(m.tsBand),
  }
}

// ---------- drag rules ----------

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
const awayFromOrigin = (p: number) => (Math.abs(p) < MIN_R ? (p < 0 ? -MIN_R : MIN_R) : p)
const round = (v: number) => Math.round(v * 1e6) / 1e6

const reBounds = (s: Scale) => [s.re[0] + 0.15, s.re[1] - 0.15] as const
const imMax = (s: Scale) => s.im[1] - 0.15

export const onRay = (s: Scale, ref: { re: number; im: number }, r: number) => {
  const len = Math.hypot(ref.re, ref.im)
  const ux = ref.re / len
  const uy = ref.im / len
  // Keep the pole inside the plane along the ray.
  const lim = Math.min(
    ux < 0 ? (s.re[0] + 0.15) / ux : ux > 0 ? (s.re[1] - 0.15) / ux : Infinity,
    imMax(s) / uy,
  )
  const rr = clamp(r, 0.3, lim)
  return { kind: 'pair', re: round(rr * ux), im: round(rr * uy) } as const
}

/** Moves handle `which` toward the s-plane point (re, im) under the lock. */
export const dragTo = (
  s: Scale,
  st: TimeState,
  lock: Lock,
  ref: TimeState,
  which: number,
  re: number,
  im: number,
  snap = SNAP,
): TimeState => {
  const [lo, hi] = reBounds(s)
  const x = clamp(re, lo, hi)
  if (st.kind === 'first') return { kind: 'first', p: round(awayFromOrigin(x)) }
  if (lock !== 'none' && ref.kind === 'pair') {
    if (lock === 'sigma')
      return { kind: 'pair', re: ref.re, im: round(clamp(im, MIN_IM, imMax(s))) }
    if (lock === 'omega') {
      const v = Math.abs(x) < snap ? 0 : x
      return { kind: 'pair', re: round(v), im: ref.im }
    }
    const len = Math.hypot(ref.re, ref.im)
    return onRay(s, ref, (re * ref.re + Math.abs(im) * ref.im) / len)
  }
  const y = Math.abs(im)
  if (st.kind === 'pair') {
    if (y < Math.max(snap, MIN_IM) || im < 0) {
      const p = round(awayFromOrigin(x))
      return { kind: 'real', p1: p, p2: p }
    }
    return { kind: 'pair', re: round(Math.abs(x) < snap ? 0 : x), im: round(Math.min(y, imMax(s))) }
  }
  if (y > Math.max(snap, MIN_IM) * 2)
    return { kind: 'pair', re: round(Math.abs(x) < snap ? 0 : x), im: round(Math.min(y, imMax(s))) }
  const other = which === 0 ? st.p2 : st.p1
  const p = Math.abs(x - other) < snap ? other : round(awayFromOrigin(x))
  return which === 0 ? { kind: 'real', p1: p, p2: other } : { kind: 'real', p1: other, p2: p }
}

const KEY_STEP = 0.1

/** Arrow-key nudge of handle `which`; returns null for keys the handle ignores. */
export const nudge = (
  s: Scale,
  st: TimeState,
  lock: Lock,
  ref: TimeState,
  which: number,
  key: string,
  big: boolean,
): TimeState | null => {
  const d = big ? 0.5 : KEY_STEP
  const dx = key === 'ArrowRight' ? d : key === 'ArrowLeft' ? -d : 0
  const dy = key === 'ArrowUp' ? d : key === 'ArrowDown' ? -d : 0
  if (!dx && !dy) return null
  const tiny = 0.5 * KEY_STEP
  if (st.kind === 'first') return dx ? dragTo(s, st, lock, ref, 0, st.p + dx, 0, tiny) : null
  if (lock === 'zeta' && st.kind === 'pair')
    return onRay(s, ref as { re: number; im: number }, Math.hypot(st.re, st.im) + (dx || dy))
  if (st.kind === 'pair') {
    if (lock === 'sigma' && !dy) return null
    if (lock === 'omega' && !dx) return null
    if (dy < 0 && st.im + dy < tiny && lock === 'none') {
      const p = round(awayFromOrigin(st.re))
      return { kind: 'real', p1: p, p2: p }
    }
    return dragTo(s, st, lock, ref, which, st.re + dx, Math.max(st.im + dy, MIN_IM), tiny)
  }
  const mine = which === 0 ? st.p1 : st.p2
  if (dy) return st.p1 === st.p2 && dy > 0 ? { kind: 'pair', re: st.p1, im: d } : null
  return dragTo(s, st, lock, ref, which, mine + dx, 0, tiny)
}

// ---------- view: attributes for every moving mark ----------

const SHOW = 'visible'
const HIDE = 'hidden'
const vis = (b: boolean) => (b ? SHOW : HIDE)
const at = (x: number, y: number) => `translate(${x} ${y})`

const curvePath = (s: Scale, f: (t: number) => number) => {
  const n = 320
  const lo = s.c[0] - 2
  const hi = s.c[1] + 2
  let d = ''
  for (let i = 0; i <= n; i++) {
    const t = s.t[0] + ((s.t[1] - s.t[0]) * i) / n
    const v = clamp(f(t), lo, hi)
    d += `${i ? 'L' : 'M'}${tx(s, t)} ${cy(s, v)}`
  }
  return d
}

// Bottom-row labels keep a minimum gap and stay inside the plot.
const spread = (items: { key: string; x: number }[], gap: number, lo: number, hi: number) => {
  const sorted = [...items].sort((a, b) => a.x - b.x)
  for (let i = 1; i < sorted.length; i++) sorted[i].x = Math.max(sorted[i].x, sorted[i - 1].x + gap)
  const over = sorted.length ? sorted[sorted.length - 1].x - hi : 0
  if (over > 0) {
    sorted[sorted.length - 1].x -= over
    for (let i = sorted.length - 2; i >= 0; i--)
      sorted[i].x = Math.min(sorted[i].x, sorted[i + 1].x - gap)
  }
  for (const it of sorted) it.x = Math.max(it.x, lo)
  return Object.fromEntries(sorted.map(it => [it.key, it.x]))
}

export const FO_W = 34
export const FO_H = 16
const fo = (x: number, y: number, show = true): Attrs => ({
  x: r1(x - FO_W / 2),
  y: r1(y - FO_H / 2),
  visibility: vis(show),
})

const line = (x1: number, y1: number, x2: number, y2: number, show = true): Attrs => ({
  x1,
  y1,
  x2,
  y2,
  visibility: vis(show),
})

// Where a ray from the origin along (ux, uy) leaves the plane.
const rayEnd = (s: Scale, ux: number, uy: number) => {
  const tre = ux < 0 ? s.re[0] / ux : ux > 0 ? s.re[1] / ux : Infinity
  const tim = uy < 0 ? s.im[0] / uy : uy > 0 ? s.im[1] / uy : Infinity
  const k = Math.min(tre, tim)
  return [ux * k, uy * k] as const
}

const polesOf = (st: TimeState): [number, number][] =>
  st.kind === 'first'
    ? [[st.p, 0]]
    : st.kind === 'pair'
      ? [
          [st.re, st.im],
          [st.re, -st.im],
        ]
      : [
          [st.p1, 0],
          [st.p2, 0],
        ]

export const TIME_ELEMENTS = [
  'h0',
  'h1',
  'conj',
  'ref0',
  'ref1',
  'guide0',
  'guide1',
  'radius',
  'projRe',
  'projIm',
  'arc',
  'lSig',
  'lWd',
  'lWn',
  'lTh',
  'rise',
  'band',
  'env0',
  'env1',
  'refCurve',
  'curve',
  'peakH',
  'peakV',
  'peak',
  'tsV',
  'ts',
  'tauH',
  'tauV',
  'tau',
  'slope',
  'lTr',
  'lTp',
  'lTs',
  'lTau',
] as const

export type TimeView = {
  attrs: Record<(typeof TIME_ELEMENTS)[number], Attrs>
  readout: Record<string, string>
  cls: Case
  aria: string
}

export const timeView = (s: Scale, st: TimeState, lock: Lock, ref: TimeState): TimeView => {
  const m = timeMetrics(st)
  const ox = sx(s, 0)
  const oy = sy(s, 0)
  const ps = polesOf(st)
  const isPair = st.kind === 'pair'
  const showRef = lock !== 'none' && ref.kind === 'pair'
  const refPs = polesOf(ref)

  let guide0 = line(0, 0, 0, 0, false)
  let guide1 = line(0, 0, 0, 0, false)
  if (showRef && ref.kind === 'pair') {
    const top = PLANE_Y
    const bot = PLANE_Y + planeH(s)
    const left = PLANE_X
    const right = PLANE_X + planeW(s)
    if (lock === 'sigma') guide0 = line(sx(s, ref.re), top, sx(s, ref.re), bot)
    else if (lock === 'omega') {
      guide0 = line(left, sy(s, ref.im), right, sy(s, ref.im))
      guide1 = line(left, sy(s, -ref.im), right, sy(s, -ref.im))
    } else {
      const len = Math.hypot(ref.re, ref.im)
      const [ex, ey] = rayEnd(s, ref.re / len, ref.im / len)
      guide0 = line(ox, oy, sx(s, ex), sy(s, ey))
      guide1 = line(ox, oy, sx(s, ex), sy(s, -ey))
    }
  }

  // Pole geometry: radius w_n, angle theta from the negative real axis, and the two projections.
  const [pre, pim] = ps[0]
  const px = sx(s, pre)
  const py = sy(s, pim)
  const geo = isPair
  const lhp = isPair && pre < 0
  const phi = Math.atan2(pim, pre)
  const arcR = 16
  const arc = lhp
    ? `M${r1(ox - arcR)} ${oy}A${arcR} ${arcR} 0 0 0 ${r1(ox + arcR * Math.cos(phi))} ${r1(oy - arcR * Math.sin(phi))}`
    : ''
  const half = (Math.PI + phi) / 2
  const wnLen = Math.hypot(px - ox, py - oy)
  const nx = (py - oy) / (wnLen || 1)
  const ny = -(px - ox) / (wnLen || 1)
  const wnFlip = ny > 0 ? -1 : 1
  const wnX = (ox + px) / 2 + wnFlip * nx * 14
  const wnY = (oy + py) / 2 + wnFlip * ny * 14
  // The omega_n label yields to a reference pole or the omega_d label it would cover.
  const wnHitsRef =
    (showRef && refPs.some(([re, im]) => Math.hypot(sx(s, re) - wnX, sy(s, im) - wnY) < 16)) ||
    Math.hypot((pre <= 0 ? ox + 20 : ox - 20) - wnX, py - wnY) < 22

  // Response marks.
  const yAxis = PLOT_X
  const tAxis = cy(s, s.c[0])
  const tIn = (t: number | null) => t !== null && t <= s.t[1]
  const tr = m.t10 !== null && m.t90 !== null
  const hasPeak = tIn(m.tp) && m.cls !== 'unstable'
  const cPeak = m.tp !== null ? stepAt(st, m.tp) : 0
  const hasTs = tIn(m.tsBand)
  const cTs = m.tsBand !== null ? stepAt(st, m.tsBand) : 0
  const first = st.kind === 'first'
  const tau = first ? 1 / -st.p : 0
  const hasTau = first && m.cls === 'stable' && tau <= s.t[1]

  const envOn = lock === 'sigma' && isPair && m.cls !== 'unstable'
  const envAmp = isPair ? Math.hypot(st.re, st.im) / st.im : 0
  const env = (sign: number) =>
    envOn ? curvePath(s, t => 1 + sign * envAmp * Math.exp((st as { re: number }).re * t)) : ''

  const row = PLOT_Y + planeH(s) + 24
  const lab: { key: string; x: number }[] = []
  if (tr && m.t90! <= s.t[1]) lab.push({ key: 'lTr', x: (tx(s, m.t10!) + tx(s, m.t90!)) / 2 })
  if (hasPeak) lab.push({ key: 'lTp', x: tx(s, m.tp!) })
  if (hasTs) lab.push({ key: 'lTs', x: tx(s, m.tsBand!) })
  if (hasTau) lab.push({ key: 'lTau', x: tx(s, tau) })
  // The right end of the row holds the t axis label.
  const lx = spread(lab, 24, PLOT_X + 10, PLOT_X + PLOT_W - 48)

  const attrs: TimeView['attrs'] = {
    h0: { transform: at(px, py) },
    h1: {
      transform: st.kind === 'real' ? at(sx(s, st.p2), oy) : at(px, py),
      visibility: vis(st.kind === 'real'),
    },
    conj: { transform: at(px, sy(s, -pim)), visibility: vis(isPair) },
    ref0: { transform: at(sx(s, refPs[0][0]), sy(s, refPs[0][1])), visibility: vis(showRef) },
    ref1: {
      transform: at(sx(s, refPs[1]?.[0] ?? 0), sy(s, refPs[1]?.[1] ?? 0)),
      visibility: vis(showRef),
    },
    guide0,
    guide1,
    radius: line(ox, oy, px, py, geo),
    projRe: line(px, py, px, oy, geo),
    projIm: line(px, py, ox, py, geo),
    arc: { d: arc, visibility: vis(lhp) },
    lSig: fo(px, oy + 11, geo && pim * UNIT > 24 && Math.abs(pre) * UNIT > 12),
    lWd: fo(pre <= 0 ? ox + 20 : ox - 20, py, geo && Math.abs(pre) * UNIT > 12),
    lWn: fo(wnX, wnY, geo && wnLen > 40 && !wnHitsRef),
    lTh: fo(ox + 28 * Math.cos(half), oy - 28 * Math.sin(half), lhp && wnLen > 40),
    rise: {
      x: tr ? tx(s, m.t10!) : 0,
      y: PLOT_Y,
      width: tr ? r1(Math.min(tx(s, m.t90!), PLOT_X + PLOT_W) - tx(s, m.t10!)) : 0,
      height: planeH(s),
      visibility: vis(tr),
    },
    band: { visibility: vis(m.tsBand !== null) },
    env0: { d: env(1), visibility: vis(envOn) },
    env1: { d: env(-1), visibility: vis(envOn) },
    refCurve: { d: showRef ? curvePath(s, t => stepAt(ref, t)) : '', visibility: vis(showRef) },
    curve: { d: curvePath(s, t => stepAt(st, t)) },
    peakH: line(yAxis, cy(s, cPeak), tx(s, m.tp ?? 0), cy(s, cPeak), hasPeak),
    peakV: line(tx(s, m.tp ?? 0), cy(s, cPeak), tx(s, m.tp ?? 0), tAxis, hasPeak),
    peak: { cx: tx(s, m.tp ?? 0), cy: cy(s, cPeak), visibility: vis(hasPeak) },
    tsV: line(tx(s, m.tsBand ?? 0), cy(s, cTs), tx(s, m.tsBand ?? 0), tAxis, hasTs),
    ts: { cx: tx(s, m.tsBand ?? 0), cy: cy(s, cTs), visibility: vis(hasTs) },
    tauH: line(yAxis, cy(s, 1 - Math.exp(-1)), tx(s, tau), cy(s, 1 - Math.exp(-1)), hasTau),
    tauV: line(tx(s, tau), cy(s, 1 - Math.exp(-1)), tx(s, tau), tAxis, hasTau),
    tau: { cx: tx(s, tau), cy: cy(s, 1 - Math.exp(-1)), visibility: vis(hasTau) },
    // The initial slope a meets the final value at t = tau.
    slope: line(tx(s, 0), tAxis, tx(s, tau), cy(s, 1), hasTau),
    lTr: fo(lx.lTr ?? 0, row, 'lTr' in lx),
    lTp: fo(lx.lTp ?? 0, row, 'lTp' in lx),
    lTs: fo(lx.lTs ?? 0, row, 'lTs' in lx),
    lTau: fo(lx.lTau ?? 0, row, 'lTau' in lx),
  }

  const readout = timeReadout(st, m)
  const aria = `${readout.case}; ${first ? 'pole' : 'poles'} at ${polesText(st)}.`
  return { attrs, readout, cls: m.cls, aria }
}

// Course examples with w_n = 3 (slides on time response), plus a right-half-plane pair.
export const CASES: { id: Case; label: string; state: TimeState }[] = [
  {
    id: 'overdamped',
    label: 'overdamped',
    state: { kind: 'real', p1: -(9 - Math.sqrt(45)) / 2, p2: -(9 + Math.sqrt(45)) / 2 },
  },
  { id: 'critical', label: 'critically damped', state: { kind: 'real', p1: -3, p2: -3 } },
  { id: 'underdamped', label: 'underdamped', state: { kind: 'pair', re: -1, im: Math.sqrt(8) } },
  { id: 'undamped', label: 'undamped', state: { kind: 'pair', re: 0, im: 3 } },
  { id: 'unstable', label: 'unstable', state: { kind: 'pair', re: 0.5, im: Math.sqrt(8.75) } },
]

// ---------- root locus: complex arithmetic and polynomial roots ----------

type Cx = [number, number]
const cadd = (a: Cx, b: Cx): Cx => [a[0] + b[0], a[1] + b[1]]
const csub = (a: Cx, b: Cx): Cx => [a[0] - b[0], a[1] - b[1]]
const cmul = (a: Cx, b: Cx): Cx => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
const cdiv = (a: Cx, b: Cx): Cx => {
  const d = b[0] * b[0] + b[1] * b[1]
  return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]
}
const cexp = (a: Cx): Cx => {
  const e = Math.exp(a[0])
  return [e * Math.cos(a[1]), e * Math.sin(a[1])]
}

// Coefficients run from the highest power down.
const polyFromRoots = (roots: number[]) =>
  roots.reduce<number[]>((p, r) => [...p, 0].map((c, i) => c - r * (i ? p[i - 1] : 0)), [1])
const polyEval = (p: number[], z: Cx): Cx =>
  p.reduce<Cx>((acc, c) => cadd(cmul(acc, z), [c, 0]), [0, 0])
const polyDeriv = (p: number[]) => p.slice(0, -1).map((c, i) => c * (p.length - 1 - i))
const polyAdd = (a: number[], b: number[], k: number) => {
  const n = Math.max(a.length, b.length)
  const pa = Array(n - a.length)
    .fill(0)
    .concat(a)
  const pb = Array(n - b.length)
    .fill(0)
    .concat(b)
  return pa.map((c, i) => c + k * pb[i])
}

// Durand-Kerner on a monic polynomial. A warm start from nearby roots keeps branch identity.
const polyRoots = (p: number[], guess?: Cx[]): Cx[] => {
  const n = p.length - 1
  let z: Cx[] =
    guess && guess.length === n
      ? guess.map(g => [g[0] + 1e-7, g[1] + 1e-7])
      : Array.from({ length: n }, (_, k) =>
          cmul([1 + 0.5 * k, 0], cexp([0, 0.4 + (2 * Math.PI * k) / n])),
        )
  for (let it = 0; it < 500; it++) {
    let moved = 0
    z = z.map((zi, i) => {
      let den: Cx = [1, 0]
      for (let j = 0; j < n; j++) if (j !== i) den = cmul(den, csub(zi, z[j]))
      const step = cdiv(polyEval(p, zi), den)
      moved = Math.max(moved, Math.hypot(step[0], step[1]))
      return csub(zi, step)
    })
    if (moved < 1e-13) break
  }
  // Real coefficients: snap tiny imaginary parts so real roots read as real.
  return z.map(([a, b]) => [a, Math.abs(b) < 1e-7 ? 0 : b])
}

// Nise, Example 9.1: G(s) = 1/((s+1)(s+2)(s+10)), design at zeta = 0.174, PI zero at -0.1.
export const PLANT_POLES = [-1, -2, -10]
export const DESIGN_ZETA = 0.174
export const PI_ZERO = -0.1
export const K_MAX = 400

export const compPoles = (c: Comp) => (c === 'none' ? PLANT_POLES : [...PLANT_POLES, 0])
export const compZeros = (c: Comp) => (c === 'pi' ? [PI_ZERO] : [])

const charPoly = (c: Comp, k: number) =>
  polyAdd(polyFromRoots(compPoles(c)), polyFromRoots(compZeros(c)), k)

export const closedLoopPoles = (c: Comp, k: number, guess?: Cx[]) =>
  polyRoots(charPoly(c, k), guess)

const dominantPair = (roots: Cx[]) =>
  roots.filter(r => r[1] > 0).sort((a, b) => b[0] - a[0])[0] as Cx | undefined
const zetaOf = (r: Cx) => -r[0] / Math.hypot(r[0], r[1])

const LOCUS_SAMPLES = 600
const LOCUS_K_TOP = 4000

export type Branch = { k: number; re: number; im: number }[]

const branchCache = new Map<Comp, Branch[]>()
export const locusBranches = (c: Comp): Branch[] => {
  const hit = branchCache.get(c)
  if (hit) return hit
  let prev = closedLoopPoles(c, 0)
  const n = prev.length
  const branches: Branch[] = prev.map(r => [{ k: 0, re: r[0], im: r[1] }])
  for (let i = 1; i <= LOCUS_SAMPLES; i++) {
    const k = LOCUS_K_TOP * (i / LOCUS_SAMPLES) ** 3
    const next = closedLoopPoles(c, k, prev)
    // Match each branch to its nearest new root so the polylines never jump branches.
    const used = new Set<number>()
    const ordered: Cx[] = prev.map(p => {
      let best = -1
      let bd = Infinity
      next.forEach((q, j) => {
        const d = Math.hypot(q[0] - p[0], q[1] - p[1])
        if (!used.has(j) && d < bd) {
          bd = d
          best = j
        }
      })
      used.add(best)
      return next[best]
    })
    for (let b = 0; b < n; b++) branches[b].push({ k, re: ordered[b][0], im: ordered[b][1] })
    prev = ordered
  }
  branchCache.set(c, branches)
  return branches
}

/** Gain where the dominant pair crosses the design zeta ray. */
export const designGain = (c: Comp): number => {
  const pts = locusBranches(c)[0].map((_, i) => i)
  const zAt = (k: number) => {
    const d = dominantPair(closedLoopPoles(c, k))
    return d ? zetaOf(d) : 1
  }
  let lo = 0
  let hi = 0
  for (const i of pts) {
    const k = LOCUS_K_TOP * (i / LOCUS_SAMPLES) ** 3
    if (zAt(k) <= DESIGN_ZETA) {
      hi = k
      break
    }
    lo = k
  }
  for (let i = 0; i < 80; i++) {
    const mid = (lo + hi) / 2
    if (zAt(mid) > DESIGN_ZETA) lo = mid
    else hi = mid
  }
  return (lo + hi) / 2
}

export const designPoint = (): Cx => dominantPair(closedLoopPoles('none', designGain('none')))!

/** Unit-step response of the unity-feedback loop KGc(s)G(s), by residues at the closed-loop poles. */
export const locusStep = (c: Comp, k0: number): ((t: number) => number) => {
  if (k0 <= 0) return () => 0
  let k = k0
  let roots = closedLoopPoles(c, k)
  const close = (rs: Cx[]) =>
    rs.some((a, i) => rs.some((b, j) => j > i && Math.hypot(a[0] - b[0], a[1] - b[1]) < 1e-4))
  // A repeated root has no simple residue; a gain 0.1% away keeps the curve and avoids it.
  if (close(roots)) {
    k = k0 * 1.001
    roots = closedLoopPoles(c, k)
  }
  const n = polyFromRoots(compZeros(c))
  const q = charPoly(c, k)
  const dq = polyDeriv(q)
  const dc = (k * polyEval(n, [0, 0])[0]) / polyEval(q, [0, 0])[0]
  const res = roots.map(p => cdiv(cmul([k, 0], polyEval(n, p)), cmul(p, polyEval(dq, p))))
  return t => {
    let v = dc
    roots.forEach((p, i) => (v += cmul(res[i], cexp([p[0] * t, p[1] * t]))[0]))
    return v
  }
}

const angleAt = (c: Comp, s: Cx) => {
  let a = 0
  for (const z of compZeros(c)) a += Math.atan2(s[1], s[0] - z)
  for (const p of compPoles(c)) a -= Math.atan2(s[1], s[0] - p)
  let deg = (a * 180) / Math.PI
  while (deg > 0) deg -= 360
  while (deg <= -360) deg += 360
  return deg
}

export const LOCUS_ROWS = ['k', 'poles', 'zeta', 'angle', 'ess', 'stab'] as const

export type LocusView = {
  poles: { cx: number; cy: number; visibility: string }[]
  curve: string
  readout: Record<(typeof LOCUS_ROWS)[number], string>
  aria: string
}

const rootsText = (rs: Cx[]) => {
  const parts: string[] = []
  for (const r of [...rs].sort((a, b) => b[0] - a[0] || b[1] - a[1])) {
    if (r[1] < 0) continue
    parts.push(r[1] > 0 ? `${fmt(r[0])} ± j${fmt(r[1])}` : fmt(r[0]))
  }
  return parts.join(', ')
}

export const locusView = (s: Scale, c: Comp, k: number): LocusView => {
  const roots = closedLoopPoles(c, k)
  const unstable = roots.some(r => r[0] > 1e-9)
  const dom = dominantPair(roots)
  const step = locusStep(c, k)
  const dc = c === 'none' ? k / (20 + k) : 1
  const a = designPoint()
  return {
    poles: Array.from({ length: 4 }, (_, i) => {
      const r = roots[i]
      return r
        ? {
            cx: sx(s, clamp(r[0], s.re[0], s.re[1])),
            cy: sy(s, r[1]),
            visibility: vis(r[0] >= s.re[0]),
          }
        : { cx: 0, cy: 0, visibility: HIDE }
    }),
    curve: curvePath(s, step),
    readout: {
      k: fmt(k, 1),
      poles: rootsText(roots),
      zeta: dom ? fmt(zetaOf(dom), 3) : '–',
      angle: `${fmt(angleAt(c, a), 1)}°`,
      ess: unstable ? '–' : fmt(1 - dc, 3),
      stab: unstable ? 'unstable' : 'stable',
    },
    aria: `Root locus with K = ${fmt(k, 1)}; closed-loop poles ${rootsText(roots)}; ${unstable ? 'unstable' : 'stable'}.`,
  }
}

/** The locus sample nearest an s-plane point, for dragging a closed-loop pole along the locus. */
export const nearestGain = (c: Comp, re: number, im: number): number => {
  let best = 0
  let bd = Infinity
  for (const b of locusBranches(c))
    for (const p of b) {
      if (p.k > K_MAX) break
      const d = Math.hypot(p.re - re, p.im - Math.abs(im))
      if (d < bd) {
        bd = d
        best = p.k
      }
    }
  return best
}
