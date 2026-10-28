import type { Pt } from '../404-pixel'
import { slant } from '../404-downhill'
import { DOOR, type Beast, type Herd, type HerdEnv, type Kind } from './herd-state'
import { LANE, onGrass, route, sight, tread } from './pasture'

export const HEEL = 28

// The share of the herd down chewing the cud at an hour: most of the night, some of the middle of
// the day (more in the heat), and few through the morning and evening grazing.
export function lying(env: HerdEnv) {
  const h = env.hour
  if (h < 5 || h >= 21) return 0.75
  if (h >= 10.5 && h < 16) return env.hot ? 0.6 : 0.35
  return 0.12
}

// A beast lies down for five to twenty minutes at a time, and does anything else for about a
// minute before it chooses again. To keep the share of time down at the hour's, the odds of lying
// down at each choice are scaled by those lengths: at the morning's 12% they are under 1%.
export const LIE = { min: 300, span: 900 }
const BOUT = 60
function liesDown(share: number) {
  const lie = LIE.min + LIE.span / 2
  return (share * BOUT) / (share * BOUT + (1 - share) * lie)
}

// Each keeps a little room from the others: about a body length apart along the slope, or three
// rows up or down it, so no two are painted over each other. Lengths are in units.
export const LEN: Record<Kind, number> = { cow: 24, calf: 16, sheep: 16 }
export function roomy(h: Herd, b: Beast | null, kind: Kind, x: number, y: number) {
  for (const o of h.beasts) {
    if (o === b || o.in) continue
    const gap = (LEN[kind] + LEN[o.kind]) / 2 + 4
    // Where it stands, where it is walking, and the end of its way out to the grass.
    const at: Pt[] = [[o.x, o.y]]
    if (o.to) at.push(o.to)
    if (o.path.length) at.push(o.path[o.path.length - 1])
    for (const [ox, oy] of at) if (Math.abs(ox - x) < gap && Math.abs(oy - y) < 12) return false
  }
  return true
}

// Grass with room for a beast within `spread` of a point, on a tread where one runs near and has
// room, sampled wider across the slope than up and down it, as cattle work along the contour; the
// search widens if the near grass is taken. From where a beast stands, only grass it can walk
// straight to.
export function near(
  h: Herd,
  b: Beast | null,
  kind: Kind,
  x: number,
  y: number,
  spread: number,
  from: Pt | null = b && [b.x, b.y],
): Pt | null {
  for (let n = 0; n < 32; n++) {
    const s = spread * (1 + n / 8)
    const px = x + (h.rng() - 0.5) * 2 * s
    const raw = y + (h.rng() - 0.5) * s
    for (const py of [tread(px, raw) ?? raw, raw])
      if (onGrass(kind, px, py) && roomy(h, b, kind, px, py) && (!from || sight(from, [px, py])))
        return [px, py]
  }
  return null
}

const kinScratch: Beast[] = []
export function kin(h: Herd, b: Beast): Beast[] {
  kinScratch.length = 0
  const isSheep = b.kind === 'sheep'
  for (const o of h.beasts) {
    if (!o.in && (o.kind === 'sheep') === isSheep) kinScratch.push(o)
  }
  return kinScratch
}
function centre(group: { x: number; y: number }[]): Pt {
  let x = 0
  let y = 0
  for (const o of group) {
    x += o.x
    y += o.y
  }
  return [x / group.length, y / group.length]
}

export function go(b: Beast, to: Pt | null, run = false) {
  if (!to) return graze(b, 10)
  b.doing = run ? 'run' : 'walk'
  b.to = to
  b.path = []
  b.t = 0
  b.until = 40
}
export function graze(b: Beast, until: number) {
  b.doing = 'graze'
  b.to = null
  b.path = []
  b.t = 0
  b.until = until
}
// Along a way in or out, which nothing turns it from.
export function follow(b: Beast, path: Pt[]) {
  b.spot = -1
  b.doing = 'walk'
  b.path = path
  b.to = path[0]
  b.t = 0
  b.until = 120
  b.at = null
  b.watch = null
}
export const homeward = (b: Beast) => b.path[b.path.length - 1] === DOOR
// The cattle go in and come out in their order: the lead cow with her calf at heel, the other two
// cows, the red cow's calf after her, the black cow, then the ewes, a few seconds apart.
const ORDER = [0, 4, 1, 2, 5, 3, 6, 7, 8]
const GAP = 6
export const turn = (h: Herd, n: number) => h.since >= ORDER.indexOf(n) * GAP
// Out of the doors in the morning to where its own are headed: a calf to her dam's heel, the first
// of the cattle or the ewes to the middle of the pasture or the upper end.
const HERD: Pt = [318, 668]
const FLOCK: Pt = [336, 592]
export function leave(h: Herd, b: Beast) {
  b.in = false
  b.x = DOOR[0]
  b.y = DOOR[1]
  const bound = (o: Beast): Pt => (o.path.length ? o.path[o.path.length - 1] : [o.x, o.y])
  const dam = b.kind === 'calf' && !h.beasts[b.dam].in ? h.beasts[b.dam] : null
  const own = kin(h, b).filter(o => o !== b && o.kind !== 'calf')
  const [x, y]: Pt = dam
    ? [bound(dam)[0] - HEEL * dam.face, bound(dam)[1]]
    : own.length
      ? centre(own.map(o => ({ x: bound(o)[0], y: bound(o)[1] })))
      : b.kind === 'sheep'
        ? FLOCK
        : HERD
  // Nobody stops in the mouth of the lane, where the rest are still coming out.
  const home = b.kind === 'sheep' ? FLOCK : HERD
  const clear = (p: Pt | null) => (p && slant(p[0] - LANE.x0, p[1] - DOOR[1]) > 40 ? p : null)
  const to =
    clear(near(h, b, b.kind, x, y, dam ? 6 : 30, null)) ??
    clear(near(h, b, b.kind, home[0], home[1], 30, null)) ??
    home
  const back = route(to[0], to[1])
  back.pop()
  follow(b, [...back.reverse(), to])
}

// What it does next: down to chew the cud, more readily when the others are down; back to its dam,
// the flock or the herd if it has strayed; otherwise on grazing where it is, off a little way to
// fresh grass, or a look about.
export function next(h: Herd, b: Beast, env: HerdEnv) {
  const r = h.rng
  const group = kin(h, b)
  let downCount = 0
  for (const o of group) if (o.doing === 'lie') downCount++
  const down = downCount / group.length
  b.at = null
  b.watch = null
  b.path = []
  // Cattle lie down together, so the share pulls a third of the way toward how many are down now.
  const share = Math.min(0.9, lying(env) + (b.kind === 'calf' ? 0.2 : 0))
  if (r() < liesDown(share + (down - share) / 3)) {
    b.doing = 'lie'
    b.t = 0
    b.until = LIE.min + r() * LIE.span
    return
  }
  const dam = b.kind === 'calf' && !h.beasts[b.dam].in ? h.beasts[b.dam] : null
  // A calf follows at her heels, a body length behind her.
  const [cx, cy] = dam ? [dam.x - HEEL * dam.face, dam.y] : centre(group)
  const leash = dam ? 14 : b.kind === 'sheep' ? 16 : 44
  if (slant(cx - b.x, cy - b.y) > leash) return go(b, near(h, b, b.kind, cx, cy, dam ? 4 : 12))
  const roll = r()
  if (roll < 0.2)
    return go(b, near(h, b, b.kind, b.x + (r() - 0.5) * 80, b.y + (r() - 0.5) * 20, 8))
  if (roll < 0.28) {
    b.doing = 'look'
    b.t = 0
    b.until = 3 + r() * 6
    b.at = [b.x + b.face * 40, b.y - 8]
    return
  }
  if (r() < 0.25) b.face = -b.face
  graze(b, 20 + r() * 70)
}
