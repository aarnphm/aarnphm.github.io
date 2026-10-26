// Pen lines for the ranges: a ridge bowed a little between its vertices, then inked in runs, the
// pen lifting now and then and coming down again a little before or after where it left off, its
// pressure swelling where `weight` says the edge is near or in shade.

import { mulberry } from '../../components/scripts/404-pixel'
import { curve, pressure, stroke, type Pt } from '../../util/mascot/pen'

// The polyline with each run bowed by up to `bow` of its length, the corners kept.
export function bowed(pts: readonly Pt[], seed: number, bow = 0.08): Pt[] {
  const rand = mulberry(seed)
  const out: Pt[] = [pts[0]]
  for (let i = 0; i < pts.length - 1; i++) {
    const [a, b] = [pts[i], pts[i + 1]]
    const len = Math.hypot(b[0] - a[0], b[1] - a[1])
    const k = bow * len * (rand() - 0.5) * 2
    const c: Pt = [
      (a[0] + b[0]) / 2 - ((b[1] - a[1]) / len) * k,
      (a[1] + b[1]) / 2 + ((b[0] - a[0]) / len) * k,
    ]
    out.push(...curve(a, c, b, Math.max(2, Math.round(len / 24))).slice(1))
  }
  return out
}

export function inked(
  line: readonly Pt[],
  base: number,
  ink: string,
  seed: number,
  weight: (p: Pt) => number = () => 1,
  // A line against the sky is drawn without lifting: each run starts over the end of the last.
  lifts = true,
) {
  const rand = mulberry(seed)
  const runs: string[] = []
  for (let i = 0; i < line.length - 1; ) {
    const n = Math.min(line.length - 1, i + 24 + Math.floor(rand() * 48))
    const run = line.slice(i, n + 1)
    const w = pressure(seed + runs.length, base, 0.4)
    runs.push(stroke(run, t => w(t) * weight(run[Math.round(t * (run.length - 1))]), ink))
    if (n >= line.length - 1) break
    i = n + (rand() < 0.7 || !lifts ? -1 - Math.floor(rand() * 2) : 1)
  }
  return runs.join('')
}

// Heavier along the crests of the shaded faces, easing in over 30 px either side.
export const shaded =
  (spans: readonly (readonly [number, number])[], by = 0.3) =>
  (p: Pt) => {
    let k = 0
    for (const [x0, x1] of spans) k = Math.max(k, Math.min(1, Math.min(p[0] - x0, x1 - p[0]) / 30))
    return 1 + by * Math.max(0, k)
  }
