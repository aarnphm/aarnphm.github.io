import type { Voices } from '../404-crew'
import { BAYER, col, dot, row, type Grid, type Pt, type RGB } from '../404-pixel'
import {
  cell,
  CHAIRS,
  clamp,
  DOWN,
  GAP,
  GATES,
  laneW,
  LIFT_FOOT,
  LIFT_HEAD,
  LIFT_LEN,
  M,
  RUNS,
  seek,
  SHED,
  slid,
  spot,
  stopped,
  type Hill,
  type SkiInk,
} from './hill-state'
import { paintSkiers } from './paint-skiers'

// The runs, the chair and its riders, and everyone on the hill, over the far range. `onRange` says
// where the far range shows between its crest and the mid ridge in front of it; the runs are only
// painted there, and whoever walks in over the mid ridge's crest is hidden until they are over it.
export function paintHill(
  d: Uint8ClampedArray,
  g: Grid,
  ink: SkiInk,
  h: Hill,
  onRange: (i: number, j: number) => boolean,
  dark: boolean,
  heads: Voices['heads'],
) {
  if (col(g, LIFT_FOOT[0] - 30) >= g.bw || col(g, 1474) < 0) return
  for (const who of Object.keys(heads)) if (who.startsWith('ski')) delete heads[who]
  const plot = (i: number, j: number, c: RGB, alpha: number) => {
    if (alpha > BAYER[(j & 3) * 4 + (i & 3)]) dot(d, g, i, j, c)
  }
  const at = ([x, y]: Pt): Pt => [col(g, x), row(g, y)]
  const beat = Math.floor(h.time * 6)

  // The runs: the blue in corduroy, alternate rows catching the light, and the lift line bumped
  // grey, each between two lines of the deepest snow shade, since a run five to nine pixels across
  // reads as one only by its edges against the range's own shading. Where people have skied the
  // cords are polished smooth and bright, where many have it is scraped grey, and a line skied in
  // the last few seconds shows as a groove.
  RUNS.forEach((run, n) => {
    const tracks = h.tracks[n]
    const last = h.last[n]
    let k = 0
    for (
      let j = Math.max(0, row(g, run.y[0]));
      j <= Math.min(g.bh - 1, row(g, run.y[run.n - 1]));
      j++
    ) {
      const y = g.sy[j]
      while (k < run.n - 2 && run.y[k + 1] <= y) k++
      const u = clamp((y - run.y[k]) / (run.y[k + 1] - run.y[k]), 0, 1)
      const cx = run.x[k] + (run.x[k + 1] - run.x[k]) * u
      const half = run.half[k] * M
      const s = run.s[k] + (run.s[k + 1] - run.s[k]) * u
      const i0 = col(g, cx - half)
      const i1 = col(g, cx + half)
      for (let i = i0; i <= i1; i++) {
        if (!onRange(i, j)) continue
        const off = (g.sx[i] - cx) / half
        const c = cell(run, s, off * run.half[k], k)
        const passes = tracks[c]
        const fresh = h.time - last[c] < 12
        const tone =
          i === i0 || i === i1
            ? 0
            : fresh
              ? 1
              : !run.groomed || passes >= 10
                ? 2
                : passes >= 2
                  ? 3
                  : j % 2
                    ? 2
                    : 3
        dot(d, g, i, j, ink.snow[tone])
      }
    }
  })

  // The giant slalom through the summer mornings: gates in alternate colours, knocked as a racer
  // goes by.
  if (h.env.camp)
    GATES.forEach((gate, n) => {
      const [i, j] = at(spot(RUNS[0], gate.s, gate.w))
      if (!onRange(i, j)) return
      const c = ink.gates[n % 2]
      dot(d, g, i, j - 1, c)
      dot(d, g, i + (h.flick[n] > 0 ? Math.sign(gate.w) : 0), j - 2, c)
    })

  // The chair's line, sagging a little between the stations: from across the valley the cables up
  // and down are one thread, and the chairs hang off it like beads.
  const lift = (u: number): Pt => [
    LIFT_FOOT[0] + (LIFT_HEAD[0] - LIFT_FOOT[0]) * u,
    LIFT_FOOT[1] + (LIFT_HEAD[1] - LIFT_FOOT[1]) * u + Math.sin(Math.PI * u) * 6,
  ]
  for (let u = 0; u <= 1; u += 0.01) {
    const [i, j] = at(lift(u))
    dot(d, g, i, j, ink.cable)
  }
  for (const u of [0.36, 0.7]) {
    const [i, j] = at(lift(u))
    for (let k = 1; k <= 4; k++) dot(d, g, i, j + k, ink.steel)
  }
  for (const p of [LIFT_FOOT, LIFT_HEAD]) {
    const [i, j] = at(p)
    for (let di = -1; di <= 1; di++) {
      dot(d, g, i + di, j, ink.rock[1])
      dot(d, g, i + di, j + 1, ink.rock[0])
      dot(d, g, i + di, j - 1, ink.steel)
    }
  }
  // On a wind hold the stopped chairs swing.
  const sway = h.env.hold ? (Math.floor(h.time / 0.9) % 2) * 2 - 1 : 0
  const phase = h.lift % GAP
  for (let n = 0; n < CHAIRS; n++) {
    const p = phase + n * GAP
    const u = p < LIFT_LEN ? p / LIFT_LEN : 2 - p / LIFT_LEN
    const [i, j] = at(lift(u))
    dot(d, g, i + (n % 2 ? sway : 0), j + 1, ink.steel)
  }

  // The groomer: in its shed spot by day; at work, a block with its cab and blade, its tiller
  // combing behind it, its lights ahead and its beacon turning.
  {
    const c = h.cat
    const at0 = h.env.groom
      ? spot(RUNS[0], c.s, laneW(c.lane, RUNS[0].half[seek(RUNS[0], c.s, 0)]))
      : SHED
    const [i, j] = at(at0)
    if (onRange(i, j - 1)) {
      const f = h.env.groom ? (c.dir > 0 ? 1 : -1) : 1
      for (let di = -1; di <= 1; di++) {
        dot(d, g, i + di, j - 1, ink.cat)
        dot(d, g, i + di, j, ink.boot)
      }
      dot(d, g, i, j - 2, ink.cab)
      dot(d, g, i + 2 * f, j, ink.steel)
      if (h.env.groom) {
        dot(d, g, i - 2 * f, j, ink.snow[3])
        if (beat % 3 === 0) dot(d, g, i, j - 3, ink.beacon)
        if (dark) {
          for (let dj = -2; dj <= 2; dj++)
            for (let di = 0; di <= 4; di++) {
              const r = Math.hypot(di, dj * 1.5)
              if (r > 0) plot(i + f * (2 + di), j - 1 + dj, ink.glow, (1 - r / 5) * 0.6)
            }
          dot(d, g, i + 2 * f, j - 1, ink.lamp)
        }
      }
    }
  }

  // Night skiing: lamp posts down the side of each run, each throwing a pool of light on the snow.
  if (dark && h.env.open)
    RUNS.forEach(run => {
      for (let s = run.length * 0.12; s < run.length * 0.95; s += run.length * 0.27) {
        const k = seek(run, s, 0)
        const [i, j] = at(spot(run, s, -run.half[k] - 1))
        if (!onRange(i, j)) continue
        for (let dj = -1; dj <= 1; dj++)
          for (let di = 0; di <= 3; di++) {
            const r = Math.hypot(di * 0.7, dj)
            if (r > 0 && onRange(i + di, j + dj))
              plot(i + di, j + dj, ink.flood, (1 - r / 2.5) * 0.6)
          }
        dot(d, g, i, j, ink.steel)
        dot(d, g, i, j - 1, ink.steel)
        dot(d, g, i, j - 2, ink.lamp)
      }
    })

  // Avalanche control: the flash, the charge's black smoke and the snow it throws up, rising and
  // thinning; where the slope lets go, the slide runs down the fall line under its powder cloud. Its
  // crown and bed stay on the face after the cloud settles. White on a white face reads only by the
  // cloud's shaded underside, and a slide path one pixel wide reads as a run border until the crown
  // crosses its top.
  const cloud = (ci: number, cj: number, r: number, alpha: number) => {
    const n = Math.ceil(r)
    for (let dj = -n; dj <= n; dj++)
      for (let di = -n; di <= n; di++) {
        const q = Math.hypot(di, dj) / r
        if (q > 1 || !onRange(ci + di, cj + dj)) continue
        const under = dj > 0 && q > 0.5
        plot(ci + di, cj + dj, under ? ink.snow[1] : ink.snow[3], alpha * (1 - 0.5 * q * q))
      }
  }
  for (const b of h.blasts) {
    const [bi, bj] = at([b.x, b.y])
    const rise = b.t * 0.35
    if (b.slide) {
      const [, hj] = at([b.x, b.y + slid(b.t, b.slide) * DOWN])
      const settle = Math.exp(-Math.max(0, b.t - stopped(b.slide)) / 6)
      const w = Math.min(2, Math.round(b.slide / 25))
      // The crown bows down at its flanks, as a slab breaks.
      if (b.t > 0.2)
        for (let i = bi - w; i <= bi + w; i++) {
          const cj = bj + (Math.abs(i - bi) === w ? 1 : 0)
          if (onRange(i, cj)) plot(i, cj, ink.snow[0], 1)
        }
      for (let j = bj + 1; j <= hj; j++)
        if (onRange(bi, j)) plot(bi, j, ink.snow[0], 0.5 + 0.4 * settle)
      cloud(bi, hj - 1, 1.8, 0.95 * settle)
    }
    if (b.t >= 40) continue
    cloud(bi, bj - Math.round(rise * 0.5), 1.5 + 3 * (1 - Math.exp(-b.t / 1.5)), Math.exp(-b.t / 8))
    const puff = 1 + 1.5 * (1 - Math.exp(-b.t / 0.6))
    const sooty = b.t < 1 ? 1 : Math.exp(-(b.t - 1) / 2.5)
    for (let dj = -2; dj <= 2; dj++)
      for (let di = -2; di <= 2; di++) {
        const q = Math.hypot(di, dj) / puff
        const [si, sj] = [bi + di, bj - Math.round(rise) + dj]
        if (q <= 1 && onRange(si, sj)) plot(si, sj, ink.smoke, sooty * (1 - 0.4 * q))
      }
    if (b.t < 0.2) {
      for (const [di, dj] of [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ])
        dot(d, g, bi + di, bj + dj, ink.flood)
      dot(d, g, bi, bj, ink.lamp)
    }
  }

  paintSkiers({ d, g, ink, h, onRange, dark, heads, plot, at, beat, lift })
}
