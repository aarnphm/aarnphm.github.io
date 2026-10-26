// The landscape both social cards share: five ranges lit from the right, a snow cap on the summit and
// pines on the near spurs; relief.ts shapes the peak's faces and ground.ts lays the mist and scrub. Each card picks the tones; the geometry
// and the light are the same, so the day and night cards are twins.

import { mulberry } from '../../components/scripts/404-pixel'
import { even, hand, poly, pressure, stroke, type Pt } from '../../util/mascot/pen'
import { bowed, inked, shaded } from './line'

export const W = 1200
export const H = 630

export const svg = (w: number, h: number, body: string) =>
  Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">${body}</svg>`,
  )

// A closed shape in one colour inside a pen line.
export const shape = (
  pts: readonly Pt[],
  fill: string,
  width: (t: number) => number,
  ink?: string,
) => `<path d="${poly(pts)}" fill="${fill}"/>${stroke(pts, width, ink)}`

// `a` with `t` of `b` mixed in, both #rrggbb: color-mix() for librsvg, which lacks it.
export const mix = (a: string, b: string, t: number) => {
  const c = (h: string) => [1, 3, 5].map(i => Number.parseInt(h.slice(i, i + 2), 16))
  const [x, y] = [c(a), c(b)]
  return `#${x
    .map((v, i) =>
      Math.round(v + (y[i] - v) * t)
        .toString(16)
        .padStart(2, '0'),
    )
    .join('')}`
}

// A ridge's height at x, before the hand wobble, for setting things on it.
export function ridgeAt(pts: readonly Pt[], x: number) {
  for (let i = 0; i < pts.length - 1; i++) {
    const [ax, ay] = pts[i]
    const [bx, by] = pts[i + 1]
    if (x >= ax && x <= bx) return ay + ((by - ay) * (x - ax)) / (bx - ax)
  }
  return H
}

// Back to front. The sun and the moon both stand to the right of the summit, so the summit's left
// flank is the dark face on both cards. The near spurs frame the bottom corners.
export const FAR: Pt[] = [
  [-20, 314],
  [80, 276],
  [170, 296],
  [260, 254],
  [330, 282],
  [450, 242],
  [560, 266],
  [640, 250],
  [720, 282],
  [800, 260],
  [880, 290],
  [960, 246],
  [1060, 276],
  [1140, 254],
  [W + 20, 284],
]
export const BACK: Pt[] = [
  [-20, 348],
  [60, 308],
  [150, 330],
  [230, 294],
  [320, 326],
  [400, 304],
  [470, 338],
  [560, 304],
  [650, 320],
  [740, 290],
  [830, 318],
  [900, 308],
  [980, 274],
  [1060, 304],
  [1140, 290],
  [W + 20, 320],
]
// The summit and its left flank, which both cards keep; each card draws its own right shoulder.
export const SUMMIT: Pt = [430, 228]
export const FLANK: Pt[] = [
  [-20, 530],
  [70, 480],
  [180, 410],
  [250, 346],
  [310, 312],
  [370, 262],
  SUMMIT,
]
export const NEAR_L: Pt[] = [
  [-20, 450],
  [60, 418],
  [140, 436],
  [230, 478],
  [330, 540],
  [430, 596],
  [600, H + 20],
]
export const NEAR_R: Pt[] = [
  [560, H + 20],
  [700, 598],
  [780, 584],
  [870, 560],
  [960, 548],
  [1050, 520],
  [1140, 530],
  [W + 20, 508],
]

export type Tone = { fill: string; shade: string; ink: string; pen: number }

export const ridge = (pts: readonly Pt[], amp: number, seed: number) =>
  hand(bowed(pts, seed), 6, amp, seed)

const foot = (line: readonly Pt[]): Pt[] => [...line, [W + 20, H + 20], [-20, H + 20]]

// The faces the light cannot reach, for a sun or moon low at `lightX`: left of it, every run of
// the ridge that rises to the right; right of it, every run that falls. A face runs from its valley
// to its peak, and the peak's fall line leans away from the light harder than the valley's, so the
// face is a wedge that narrows as it drops and closes where the two meet, if that is above the
// bottom. The fall lines are drawn by hand too, so a face never ends in a ruled edge.
export type Face = { pts: Pt[]; top: Pt[]; s: 1 | -1 }
export const LEAN = [0.2, 0.5] as const
export function shadeFaces(
  pts: readonly Pt[],
  line: readonly Pt[],
  lightX: number,
  minRise = 30,
): Face[] {
  const faces: Face[] = []
  for (let i = 0; i < pts.length - 1; ) {
    const rising = pts[i + 1][1] < pts[i][1]
    let j = i
    while (
      j < pts.length - 1 &&
      pts[j + 1][1] < pts[j][1] === rising &&
      pts[j + 1][1] !== pts[j][1]
    )
      j++
    const [l, r] = [pts[i], pts[j]]
    const mid = (l[0] + r[0]) / 2
    if (j > i && Math.abs(l[1] - r[1]) >= minRise && (rising ? mid < lightX : mid > lightX)) {
      const s = rising ? -1 : 1
      const [ll, lr] = rising ? LEAN : [LEAN[1], LEAN[0]]
      const xAt = (p: Pt, lean: number, y: number) => p[0] + s * lean * (y - p[1])
      const meet = (r[0] - l[0] - s * lr * r[1] + s * ll * l[1]) / (s * (ll - lr))
      const top = line.filter(([x]) => x >= l[0] && x <= r[0])
      const bottom: Pt[] =
        meet > Math.max(l[1], r[1]) && meet < H + 20
          ? [[xAt(l, ll, meet), meet]]
          : [
              [xAt(r, lr, H + 20), H + 20],
              [xAt(l, ll, H + 20), H + 20],
            ]
      const edge = hand([top[top.length - 1], ...bottom, top[0]], 12, 3, 40 + faces.length)
      faces.push({ pts: [...top, ...edge.slice(1, -1)], top, s })
    }
    i = Math.max(j, i + 1)
  }
  return faces
}

// A range: its body, the shaded faces, then `under` (worn by the shade) and `over` (on top of it)
// before the ridge line. `amp` is the hand's wobble: small far away, where the pen is thin too.
export function range(
  pts: readonly Pt[],
  line: readonly Pt[],
  tone: Tone,
  seed: number,
  lightX: number,
  under = '',
  over = '',
  lifts = true,
) {
  const faces = shadeFaces(pts, line, lightX)
  const shade = faces.map(f => `<path d="${poly(f.pts)}" fill="${tone.shade}"/>`).join('')
  const crests = faces.map(f => [f.top[0][0], f.top[f.top.length - 1][0]] as const)
  return `<path d="${poly(foot(line))}" fill="${tone.fill}"/>${under}${shade}${over}${inked(line, tone.pen, tone.ink, seed, shaded(crests), lifts)}`
}

// A band of the body below `top` in another tone: meadow or forest under the rock.
export const band = (top: readonly Pt[], seed: number, fill: string) =>
  `<path d="${poly(foot(hand(top, 8, 3, seed)))}" fill="${fill}"/>`

// The snow cap: the ridge between xL and xR over a ragged snow line, its dark side the faces' dark.
export function snowCap(
  line: readonly Pt[],
  faces: Face[],
  xL: number,
  xR: number,
  fill: string,
  shade: string,
  ink: string,
  seed: number,
) {
  const top = line.filter(([x]) => x >= xL && x <= xR)
  const [l, r] = [top[0], top[top.length - 1]]
  const rand = mulberry(seed)
  const n = 9
  const edge: Pt[] = Array.from({ length: n + 1 }, (_, i) => {
    const t = i / n
    const sag = 10 + 14 * Math.sin(Math.PI * t) + (i % 2 ? 9 : -6) * (0.6 + rand())
    return [r[0] + (l[0] - r[0]) * t, r[1] + (l[1] - r[1]) * t + sag]
  })
  const cap = [...top, ...edge]
  return `<path d="${poly(cap)}" fill="${fill}"/><clipPath id="snow${seed}"><path d="${poly(cap)}"/></clipPath><g clip-path="url(#snow${seed})">${faces.map(f => `<path d="${poly(f.pts)}" fill="${shade}"/>`).join('')}</g>${stroke(hand(edge, 4, 1, seed + 1), even(1.6), ink)}`
}

// A pine in silhouette, `h` tall with its foot at (x, y): tiers of boughs narrowing to the tip.
export function pine(
  x: number,
  y: number,
  h: number,
  seed: number,
  fill: string,
  ink: string,
  pen: number,
) {
  const rand = mulberry(seed)
  const tiers = 4 + Math.floor(rand() * 2)
  const base = y - 0.12 * h
  const left: Pt[] = []
  const right: Pt[] = []
  for (let i = 0; i < tiers; i++) {
    const t = i / tiers
    const yb = base - t * 0.86 * h
    const yt = yb - (0.86 * h * 0.5) / tiers
    const w = 0.3 * h * (1 - t) ** 0.8 + 0.02 * h
    left.push([x - w * (0.85 + 0.3 * rand()), yb], [x - w * 0.45, yt])
    right.push([x + w * (0.85 + 0.3 * rand()), yb], [x + w * 0.45, yt])
  }
  const pts: Pt[] = [...left, [x + 0.02 * h * (rand() - 0.5), y - h], ...right.reverse(), left[0]]
  const trunk = stroke(
    hand(
      [
        [x, y + 1],
        [x, base + 3],
      ],
      3,
      0.3,
      seed,
    ),
    even(pen * 1.3),
    ink,
  )
  return trunk + shape(hand(pts, 3, 0.7, seed + 1), fill, pressure(seed + 2, pen, 0.3), ink)
}

// Paper grain over the land only, so the sky stays the flat ground the press kit bleeds into.
export const grain = (land: readonly Pt[], opacity: number, blend: 'multiply' | 'screen') =>
  `<defs><filter id="grain" x="0" y="0" width="1" height="1"><feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" stitchTiles="stitch"/><feColorMatrix type="saturate" values="0"/></filter><clipPath id="land"><path d="${poly(foot(land))}"/></clipPath></defs><rect width="${W}" height="${H}" filter="url(#grain)" clip-path="url(#land)" opacity="${opacity}" style="mix-blend-mode:${blend}"/>`

// The summit flag: a pole in the snow and a pennant with a fold in its lower half.
export function flag(
  ink: string,
  pink: string,
  fold: string,
  pen: (s: number, b: number) => (t: number) => number,
) {
  const [x, y] = SUMMIT
  const pole = stroke(
    hand(
      [
        [x, y + 2],
        [x + 1, y - 52],
      ],
      3,
      0.4,
      72,
    ),
    pen(72, 2.8),
    ink,
  )
  const tip: Pt[] = [
    [x + 1, y - 50],
    [x + 48, y - 38],
    [x + 1, y - 24],
    [x + 1, y - 50],
  ]
  const crease: Pt[] = [
    [x + 1, y - 38],
    [x + 46, y - 38],
    [x + 1, y - 24],
  ]
  return `${pole}<path d="${poly(hand(tip, 4, 1, 73))}" fill="${pink}"/><path d="${poly(crease)}" fill="${fold}"/>${stroke(hand(tip, 4, 1, 73), pen(74, 2.2), ink)}`
}

// Pines along a spur's ridge, each foot sunk a little into it.
export const pines = (
  ridgePts: readonly Pt[],
  at: readonly (readonly [number, number])[],
  fill: string,
  ink: string,
  pen: number,
  seed: number,
) =>
  at.map(([x, h], i) => pine(x, ridgeAt(ridgePts, x) + 3, h, seed + i * 7, fill, ink, pen)).join('')
