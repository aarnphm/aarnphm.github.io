// The peak's form. On the lit face, spurs run off the ridge down to the meadow and the gully
// between each pair has a side turned to the light and a side turned from it; first light pools
// at the summit and a penumbra softens the edge of the dark face. On the dark faces, the shade
// deepens in steps away from the crest under hatching whose density follows that depth.

import { mulberry } from '../../components/scripts/404-pixel'
import { curve, hand, poly, pressure, stroke, type Pt } from '../../util/mascot/pen'
import { type Face, H, LEAN, ridgeAt } from './scene'

export type Relief = { lit: string; dark: string; glow: string; seam: string; shade: string }

const SKEW = (-Math.atan(LEAN[1]) * 180) / Math.PI
const between = (line: readonly Pt[], x0: number, x1: number) =>
  line.filter(([x]) => x >= x0 && x <= x1)

// Spurs run off the ridge between x0 (the summit) and x1 down to the meadow at `floor`, fanning
// out from the summit. Each spur's flank toward the light (the right) is lit and the flank away
// from it is dark: wedges with their points at the ridge, so the crest is craggy and the slopes
// below smooth out. Drawn under the meadow and the dark face, which cover their feet and the
// summit's fall line.
export function facets(
  ridgePts: readonly Pt[],
  line: readonly Pt[],
  x0: number,
  x1: number,
  floor: readonly Pt[],
  tone: Relief,
  seed: number,
  id: string,
) {
  const rand = mulberry(seed)
  const y0 = ridgeAt(ridgePts, x0)
  const yb = Math.max(...floor.map(p => p[1])) + 8
  const grad = (name: string, fill: string, a: number, b: number, y1: number, y2: number) =>
    `<linearGradient id="${id}-${name}" gradientUnits="userSpaceOnUse" x1="0" y1="${y1}" x2="0" y2="${y2}"><stop offset="0" stop-color="${fill}" stop-opacity="${a}"/><stop offset="1" stop-color="${fill}" stop-opacity="${b}"/></linearGradient>`
  const defs = `<defs>${grad('g', tone.glow, 0.95, 0, y0, y0 + 150)}${grad('v', tone.dark, 0, 0.16, y0 + 40, yb)}${grad('d', tone.dark, 0.6, 0.08, y0, yb)}${grad('l', tone.lit, 0.8, 0.05, y0, yb)}<linearGradient id="${id}-p" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="150" y2="0" gradientTransform="translate(${x0} ${y0}) skewX(${SKEW.toFixed(3)})"><stop offset="0" stop-color="${tone.shade}" stop-opacity="0.45"/><stop offset="1" stop-color="${tone.shade}" stop-opacity="0"/></linearGradient></defs>`

  const face = [...between(line, x0, x1), [x1, yb], [x0 - 60, yb]] as Pt[]
  const fall = (y: number) => x0 - LEAN[1] * (y - y0)
  const under = `<path d="${poly(face)}" fill="url(#${id}-g)"/><path d="${poly(face)}" fill="url(#${id}-v)"/><path d="${poly([...between(line, x0, x0 + 150), [fall(yb) + 150, yb], [fall(yb), yb]])}" fill="url(#${id}-p)"/>`

  // The spurs: bent fall lines leaning further out the farther they stand from the summit, set
  // down at uneven intervals, so narrow gullies sit beside broad buttresses. Some end partway
  // down the slope and some are thrown sideways by a fault and carry on from there.
  type Spur = { pts: Pt[]; end: number; jog: number }
  const spurs: Spur[] = []
  for (
    let x = x0 + 18 + 14 * rand();
    x < x1 - 30;
    x += rand() < 0.4 ? 20 + 16 * rand() : 48 + 70 * rand()
  ) {
    const y = ridgeAt(ridgePts, x) + 1
    const lean = 0.04 + 0.5 * ((x - x0) / (x1 - x0)) ** 1.3 + 0.3 * (rand() - 0.5)
    const xb = x + lean * (yb - y)
    const mid: Pt = [(x + xb) / 2 + 14 * (rand() - 0.5), (y + yb) / 2]
    const pts = hand(curve([x, y], mid, [xb, yb], 24), 1000, 1.4, seed + spurs.length)
    const fate = rand()
    const end = fate < 0.3 ? 8 + Math.floor(rand() * 8) : 23
    const jog = fate > 0.72 ? 7 + Math.floor(rand() * 7) : -1
    if (jog > 0) {
      const dx = (rand() < 0.5 ? -1 : 1) * (10 + 16 * rand())
      for (let j = jog + 1; j < pts.length; j++) pts[j] = [pts[j][0] + dx, pts[j][1]]
    }
    spurs.push({ pts, end, jog })
  }
  // A flank: the spur from its root down to where it ends, across toward its neighbour by `f` of
  // the gap, and a hand line back up to the root.
  const flank = (spur: Spur, nb: Spur, f: number, k: number) => {
    const line = spur.pts.slice(0, spur.end + 1)
    const root = line[0]
    const tip = line[spur.end]
    // A ledge where a spur ends: the line across dips with the ridge, a little past level.
    const dx = (nb.pts[spur.end][0] - tip[0]) * f
    const across: Pt = [tip[0] + dx, tip[1] + (spur.end < 23 ? dx * (0.15 + 0.3 * rand()) : 0)]
    const inner = hand(
      curve(
        across,
        [(across[0] + root[0]) / 2 + 8 * (rand() - 0.5), (across[1] + root[1]) / 2],
        root,
        16,
      ),
      1000,
      1.2,
      seed + 200 + k,
    )
    return [...line, across, ...inner.slice(1, -1)]
  }

  const marks: string[] = []
  for (let i = 0; i < spurs.length; i++) {
    const a = spurs[i].pts
    const k = 0.5 + 0.5 * rand()
    if (i + 1 < spurs.length) {
      const b = spurs[i + 1].pts
      if (rand() < 0.75)
        marks.push(
          `<path d="${poly(flank(spurs[i], spurs[i + 1], 0.15 + 0.45 * rand(), i * 2))}" fill="url(#${id}-l)" opacity="${(k * (0.6 + 0.4 * rand())).toFixed(2)}"/>`,
        )
      if (rand() < 0.85)
        marks.push(
          `<path d="${poly(flank(spurs[i + 1], spurs[i], 0.15 + 0.45 * rand(), i * 2 + 1))}" fill="url(#${id}-d)" opacity="${k.toFixed(2)}"/>`,
        )
      // Strata across the lit flank, dipping with the ridge.
      const dip = (b[0][1] - a[0][1]) / (b[0][0] - a[0][0])
      for (let s = 0, count = Math.floor(rand() * 3); s < count; s++) {
        const j = Math.floor((0.15 + 0.55 * rand()) * 23)
        const span = b[j][0] - a[j][0]
        const from = a[j][0] + span * (0.08 + 0.2 * rand())
        const w = span * (0.25 + 0.45 * rand())
        marks.push(
          `<g opacity="0.5">${stroke(
            hand(
              [
                [from, a[j][1] + (from - a[j][0]) * dip * 0.7],
                [from + w, a[j][1] + (from + w - a[j][0]) * dip * 0.7 + 2 * (rand() - 0.5)],
              ],
              5,
              0.9,
              seed + 80 + i * 4 + s,
            ),
            pressure(seed + 120 + i * 4 + s, 0.9 + 0.9 * rand(), 0.45),
            tone.seam,
          )}</g>`,
        )
      }
    }
    // The spur's own edge, inked down from the ridge for part of its length; a seam stops at a
    // fault, and sometimes picks up again beyond it.
    const { end, jog } = spurs[i]
    if (rand() < 0.8) {
      const pen = (n: number) => pressure(seed + 40 + i + n, 1 + 1.4 * rand(), 0.5)
      const n = Math.min(end < 23 ? end - 3 : 6 + Math.floor(rand() * 12), jog > 0 ? jog + 1 : 24)
      marks.push(`<g opacity="0.7">${stroke(a.slice(0, n), pen(0), tone.seam)}</g>`)
      if (jog > 0 && rand() < 0.6)
        marks.push(
          `<g opacity="0.7">${stroke(a.slice(jog + 1, jog + 4 + Math.floor(rand() * 8)), pen(1), tone.seam)}</g>`,
        )
    }
  }
  return `${defs}${under}${marks.join('')}`
}

// The dark faces: two depth steps away from the crest, then hatching parallel to the fall line in
// broken strokes that thicken and crowd as the shade deepens, with a second cross pass deepest in.
export function shading(
  faces: Face[],
  id: string,
  ink: string,
  deep: string,
  gap: number,
  seed: number,
) {
  const rand = mulberry(seed)
  return faces
    .map((f, k) => {
      const xs = f.pts.map(p => p[0])
      const ys = f.pts.map(p => p[1])
      const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
      const steps = [44, 110]
        .map((d, i) => {
          const edge = hand(
            f.top.map(([x, y]) => [x + f.s * 0.35 * d, y + d] as const),
            10,
            3 + 2 * i,
            seed + 7 + i,
          )
          return `<path d="${poly([...edge, [edge[edge.length - 1][0], H + 200], [edge[0][0], H + 200]])}" fill="${deep}" opacity="${0.4 + 0.15 * i}"/>`
        })
        .join('')
      const run0 = (y1 - y0) * LEAN[1] * f.s
      const [tx0, tx1] = [f.top[0][0], f.top[f.top.length - 1][0]]
      const [xa, xb] = [x0 - Math.max(0, run0), x1 - Math.min(0, run0)]
      const marks: string[] = []
      const pass = (lean: number, g: number, bias: number, floor: number) => {
        for (let x = xa + rand() * g; x < xb; ) {
          const u = f.s < 0 ? (xb - x) / (xb - xa) : (x - xa) / (xb - xa)
          const run = run0 * lean * (1 + 0.24 * (rand() - 0.5))
          const w = 1 + 1.1 * rand()
          for (let t = 0.06 * rand(); t < 1; ) {
            const len = 0.12 + 0.3 * rand()
            const at = (t: number): Pt => [x + t * run, y0 + t * (y1 - y0)]
            // Below the crest, at the dash's middle; a dash whose middle is beyond the face's
            // peak-side edge is clipped away anyway.
            const [px, py] = at(t + len / 2)
            const crest = ridgeAt(f.top, Math.min(Math.max(px, tx0), tx1))
            const depth = Math.min(1, Math.max(0, (py - crest) / 160))
            const d = 0.55 * depth + 0.45 * u
            const p = floor + (1 - floor) * Math.max(0, (d - bias) / (1 - bias))
            if (py >= crest - 4 && rand() < p)
              marks.push(
                stroke(
                  hand([at(t), at(t + len)], 6, 1 + 1.2 * rand(), seed + marks.length),
                  pressure(seed + 500 + marks.length, w * (0.7 + 0.6 * d), 0.5),
                  ink,
                ),
              )
            t += len + 0.01 + 0.05 * rand() * (1.4 - d)
          }
          x += g * (1.3 - 0.7 * u) * (0.7 + 0.6 * rand())
        }
      }
      pass(1, gap, 0, 0.55)
      pass(0.42, gap * 1.5, 0.5, 0)
      return `<clipPath id="${id}${k}"><path d="${poly(f.pts)}"/></clipPath><g clip-path="url(#${id}${k})">${steps}<g opacity="0.5">${marks.join('')}</g></g>`
    })
    .join('')
}
