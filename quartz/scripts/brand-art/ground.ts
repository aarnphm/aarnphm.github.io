// What lies on the land between the ranges: mist in the valleys and scrub on the near spurs.

import { mulberry } from '../../components/scripts/404-pixel'
import { even, hand, poly, pressure, stroke, type Pt } from '../../util/mascot/pen'
import { H, ridgeAt, W } from './scene'

const foot = (line: readonly Pt[]): Pt[] => [...line, [W + 20, H + 20], [-20, H + 20]]

// Mist on a range: a wash over its body that is full at `level` and thins to nothing above it, with
// thicker pools lying in the dips of its ridge. Clipped to the range, so the sky stays flat.
export function haze(
  line: readonly Pt[],
  ridgePts: readonly Pt[],
  level: number,
  fill: string,
  opacity: number,
  seed: number,
  id: string,
) {
  const rand = mulberry(seed)
  const defs = `<defs><clipPath id="${id}-c"><path d="${poly(foot(line))}"/></clipPath><linearGradient id="${id}-g" gradientUnits="userSpaceOnUse" x1="0" y1="${level - 76}" x2="0" y2="${level + 24}"><stop offset="0" stop-color="${fill}" stop-opacity="0"/><stop offset="0.5" stop-color="${fill}" stop-opacity="${(opacity * 0.45).toFixed(3)}"/><stop offset="1" stop-color="${fill}" stop-opacity="${opacity}"/></linearGradient><filter id="${id}-b" x="-30%" y="-80%" width="160%" height="260%"><feGaussianBlur stdDeviation="9"/></filter></defs>`
  const wash = `<path d="${poly(
    foot(
      hand(
        [
          [-20, level - 70],
          [W + 20, level - 70],
        ],
        40,
        6,
        seed,
      ),
    ),
  )}" fill="url(#${id}-g)"/>`
  const pools = ridgePts
    .slice(1, -1)
    .filter((p, i) => p[1] > ridgePts[i][1] && p[1] > ridgePts[i + 2][1])
    .map(
      ([x, y]) =>
        `<ellipse cx="${(x + 30 * (rand() - 0.5)).toFixed(1)}" cy="${(y + 14).toFixed(1)}" rx="${(70 + 60 * rand()).toFixed(1)}" ry="${(10 + 10 * rand()).toFixed(1)}" fill="${fill}" opacity="${(opacity * (0.35 + 0.3 * rand())).toFixed(3)}" filter="url(#${id}-b)"/>`,
    )
    .join('')
  return `${defs}<g clip-path="url(#${id}-c)">${wash}${pools}</g>`
}

// One tuft of scrub, `s` tall, rooted at (x, y): a few strokes splayed from the root. Four kinds,
// each with its own count and spread, so no two read as the same glyph.
function tuft(
  x: number,
  y: number,
  s: number,
  kind: number,
  ink: string,
  rand: () => number,
  seed: number,
) {
  const blade = (dx: number, dy: number, bend: number, w: (t: number) => number, i: number) =>
    stroke(
      hand(
        [
          [x, y],
          [x + dx * 0.5 + bend, y + dy * 0.55],
          [x + dx, y + dy],
        ],
        3,
        0.4,
        seed + i,
      ),
      w,
      ink,
    )
  const marks: string[] = []
  if (kind === 0)
    for (const k of [-1, 0, 1])
      marks.push(
        blade(
          k * s * (0.45 + 0.4 * rand()),
          -s * (0.7 + 0.4 * rand()),
          2 * (rand() - 0.5),
          even(1.2 + 0.7 * rand()),
          k,
        ),
      )
  else if (kind === 1)
    for (let k = 0; k < 5; k++)
      marks.push(
        blade(
          (k - 2) * s * (0.3 + 0.2 * rand()),
          -s * (0.45 + 0.35 * rand()),
          0,
          even(1 + 0.6 * rand()),
          k,
        ),
      )
  else if (kind === 2) {
    const lean = (rand() < 0.7 ? 1 : -1) * s * (0.4 + 0.3 * rand())
    marks.push(blade(lean, -s * 1.1, lean * 0.4, pressure(seed, 1.8, 0.3), 0))
    marks.push(blade(lean * 0.6, -s * 0.7, lean * 0.3, pressure(seed + 1, 1.4, 0.3), 1))
  } else {
    for (let k = 0; k < 4; k++) {
      const a = Math.PI * (1.08 + 0.84 * (k / 3)) + 0.2 * (rand() - 0.5)
      const l = s * (0.5 + 0.3 * rand())
      marks.push(blade(l * Math.cos(a), l * Math.sin(a), 0, even(1.1 + 0.5 * rand()), k))
    }
    marks.push(
      stroke(
        hand(
          [
            [x - s * 0.5, y + 1],
            [x + s * 0.4, y + 1],
          ],
          3,
          0.3,
          seed + 9,
        ),
        even(1.1),
        ink,
      ),
    )
  }
  return marks.join('')
}

// Scrub along a spur between xa and xb, `dy` below its ridge: `n` clusters of tufts scattered round
// seeded centres, taking root more readily in the ridge's hollows than on its crests and growing
// larger lower down, nearer the viewer.
export function scrubField(
  ridgePts: readonly Pt[],
  xa: number,
  xb: number,
  dy: readonly [number, number],
  n: number,
  ink: string,
  seed: number,
) {
  const rand = mulberry(seed)
  const hollow = (x: number) =>
    ridgeAt(ridgePts, x) - (ridgeAt(ridgePts, x - 80) + ridgeAt(ridgePts, x + 80)) / 2
  const tufts: string[] = []
  for (let c = 0; c < n; c++) {
    const cx = xa + rand() * (xb - xa)
    const count = 2 + Math.floor(rand() * 4)
    for (let k = 0; k < count; k++) {
      const x = cx + 22 * (rand() + rand() + rand() - 1.5)
      if (x < xa || x > xb) continue
      if (rand() > Math.min(1, Math.max(0.25, 0.6 + hollow(x) / 30))) continue
      const y = ridgeAt(ridgePts, x) + dy[0] + rand() * (dy[1] - dy[0])
      const s = (9 + 8 * rand()) * (0.7 + 0.3 * Math.min(1, Math.max(0, (y - 440) / 200)))
      const r = rand()
      const kind = r < 0.4 ? 0 : r < 0.65 ? 1 : r < 0.85 ? 2 : 3
      tufts.push(tuft(x, y, s, kind, ink, rand, seed + tufts.length * 11))
    }
  }
  return tufts.join('')
}
