import { fbm, frac, smooth, type Pt } from '../404-pixel'

function valley(x: number, reach: number) {
  return Math.max(0, 1 - Math.abs(x - 800) / reach)
}

export const farRidge = (x: number) =>
  640 - 440 * (1 - valley(x, 820) ** 1.3) + (fbm(x * 0.004, 3.1, 11, 4) - 0.5) * 150
export const midRidge = (x: number) =>
  770 - 360 * (1 - valley(x, 600) ** 0.85) + (fbm(x * 0.006, 7.7, 23, 4) - 0.5) * 110
export const skyline = (x: number) => Math.min(farRidge(x), midRidge(x))

// The island: a broad massif rising out of the water to two rounded summits, the gate sunk into the
// saddle between them, a shoulder on the left where the fallen stone lies and a bench on the right
// flank where the cat has settled. It is modelled the way the ranges are, spurs and gullies down
// its faces, with its crest roughened as theirs is.
export function hillTop(x: number, water: number) {
  const u = Math.max(-1, Math.min(1, (x - 800) / 700))
  const bell = (0.5 + 0.5 * Math.cos(Math.PI * u)) ** 0.7
  const bump = (c: number, w: number, h: number) => h * Math.exp(-(((x - c) / w) ** 2))
  const rise =
    235 * bell +
    bump(610, 110, 40) +
    bump(1010, 100, 45) -
    bump(800, 90, 10) +
    bump(1290, 60, 32) +
    bump(330, 80, 20) +
    (fbm(x * 0.008, 2.3, 19, 3) - 0.5) * 34 * bell
  return water - Math.max(0, rise)
}

// Spurs and gullies run down each face from the crest, bending and spreading as they fall. Across
// the face a spur's crest sits at every whole number of this phase and the gully between two spurs
// at every half, so each spur is two flat planes: one turned to the light and one away from it.
// That is what gives the ranges their bulk. Rock shows along the spur crests.
export const spur = (x: number, down: number, seed: number, freq: number) =>
  x * freq + fbm(x * 0.004, seed, seed, 2) * 1.4 + Math.sin(down * 0.011 + x * 0.002 + seed) * 0.3

// Spurs leave the crest square to it and fall toward the valley, so on either flank of a summit they
// fan out downhill. The ridge's slope is read over 180 units and the drift is capped, which keeps
// neighbouring spurs from crossing.
export const lean = (ridge: (x: number) => number, x: number) =>
  Math.max(-1.2, Math.min(1.2, (ridge(x + 90) - ridge(x - 90)) / 180))
export const spurAt = (back: boolean, x: number, down: number, tilt: number) => {
  const x0 = x - tilt * 110 * Math.tanh(down / 100)
  return back ? spur(x0, down, 17, 0.011) : spur(x0, down, 29, 0.008)
}

// Which way the face turns at a phase, +1 toward the right: sharp over a spur's crest, eased
// across the gully's floor.
export const facing = (u: number) => {
  const f = frac(u)
  return f < 0.5 ? 1 - 2 * smooth(0.4, 0.6, f) : 2 * smooth(0.4, 0.6, 1 - f) - 1
}

// A glacier under the far right crest keeps its snow all year, so the lift above the trail has
// something to run on in August.
export const glacier = (x: number, y: number) => x > 1372 && x < 1500 && y < 322 + (x - 1372) * 0.4

// People on the ranges, a pixel wide and three tall: two beside the cypress on the left ridge, one
// of them waving; someone with a lantern on the right ridge; a trail runner in an orange vest on
// the switchbacks below them; and another party's fire on the far range, which by day is only a
// thread of smoke.
export const FOLK = [
  { x: 246, wave: false, lamp: false },
  { x: 254, wave: true, lamp: false },
  { x: 1318, wave: false, lamp: true },
]
export const CAMP = 1046
// A trail cut into the right ridge, as turns in scene units: it comes out from behind the knoll,
// switchbacks up the face (the leftward legs short, since the crest falls away toward the valley),
// and scrambles out onto the crest by the pines. A trail runner works up it leaning into the grade,
// throws both arms up on top, and runs back down, all day; the triathlete's mountain legs share it.
// The timings are the climb, the summit, the descent and the look at the watch at the bottom, in
// seconds.
export const TRAIL: Pt[] = [
  [1110, 760],
  [1290, 700],
  [1230, 660],
  [1330, 600],
  [1250, 565],
  [1360, 505],
  [1400, 404],
]
export const TRAIL_LEN = TRAIL.slice(1).reduce(
  (n, [x, y], k) => n + Math.hypot(x - TRAIL[k][0], y - TRAIL[k][1]),
  0,
)
export const TRAIL_TIME = { up: 30, summit: 3, down: 18, foot: 2 }

// The point `s` units up the trail, and which way that stretch runs.
export function onTrail(s: number) {
  for (let k = 1; k < TRAIL.length; k++) {
    const [ax, ay] = TRAIL[k - 1]
    const [bx, by] = TRAIL[k]
    const len = Math.hypot(bx - ax, by - ay)
    if (s > len && k < TRAIL.length - 1) {
      s -= len
      continue
    }
    const u = Math.min(1, s / len)
    return { x: ax + (bx - ax) * u, y: ay + (by - ay) * u, dir: Math.sign(bx - ax) }
  }
  return { x: TRAIL[0][0], y: TRAIL[0][1], dir: 1 }
}
