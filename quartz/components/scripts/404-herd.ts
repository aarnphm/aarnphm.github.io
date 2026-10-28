import { winter, type Almanac, type Sky } from './404-almanac'
import { graze, HEEL, LIE, lying, near } from './404-herd/grazing'
import { HOUSE, STOCK, type Beast, type Herd, type HerdEnv } from './404-herd/herd-state'
import { survey, walk } from './404-herd/pasture'
import { mulberry } from './404-pixel'
export { BARN } from './404-herd/herd-state'
export type { Herd, HerdEnv, Passer } from './404-herd/herd-state'
export { TERRACES, onTerrace } from './404-herd/pasture'
export { stepHerd } from './404-herd/traffic'
export { paintBarn, paintBeast, paintFarmer, paintTrough } from './404-herd/paint-herd'
export type { HerdInk } from './404-herd/paint-herd'

export function herdEnv(al: Almanac, sky: Sky): HerdEnv {
  return {
    out: !winter(al) && al.sun.alt > 2,
    hour: al.hour,
    hot: sky.temp !== null && sky.temp > 26,
  }
}

// Out onto the pasture as found when anyone looks: the cattle about one spot, calves by their dams,
// the ewes together above, each at its own point of whatever it is doing; or all in the barn.
function scatter(h: Herd, env: HerdEnv) {
  const r = h.rng
  h.beasts = []
  h.out = env.out
  h.since = Infinity
  h.doors = env.out ? 2 : 0
  h.farmer = { chore: 'home', after: 'home', way: [], x: HOUSE[0], y: HOUSE[1], face: -1, t: 0 }
  h.trough = 0
  h.feeds = 0
  const herd = near(h, null, 'cow', 318, 668, 40) ?? [318, 668]
  const flock = near(h, null, 'sheep', 336, 592, 30) ?? [336, 592]
  for (const s of STOCK) {
    const dam = s.dam >= 0 ? h.beasts[s.dam] : null
    const [x, y] = dam
      ? (near(h, null, 'calf', dam.x - HEEL * dam.face, dam.y, 4) ??
        near(h, null, 'calf', dam.x, dam.y, 30) ??
        near(h, null, 'calf', herd[0], herd[1], 40) ?? [dam.x, dam.y])
      : s.kind === 'sheep'
        ? (near(h, null, 'sheep', flock[0], flock[1], 20) ?? flock)
        : (near(h, null, 'cow', herd[0], herd[1], 40) ?? herd)
    const b: Beast = {
      ...s,
      x,
      y,
      face: r() < 0.5 ? -1 : 1,
      doing: 'graze',
      t: 0,
      until: 0,
      to: null,
      at: null,
      watch: null,
      calm: 0,
      in: !env.out,
      path: [],
      spot: -1,
      fed: 0,
    }
    h.beasts.push(b)
    if (r() < Math.min(0.9, lying(env) + (s.kind === 'calf' ? 0.2 : 0))) {
      b.doing = 'lie'
      b.until = LIE.min + r() * LIE.span
    } else graze(b, 20 + r() * 70)
    b.t = r() * b.until
  }
}

export function makeHerd(seed: number, env: HerdEnv): Herd {
  if (!walk) survey()
  const h = { seed, rng: mulberry(seed ^ 0x5eed), beasts: [] } as unknown as Herd
  scatter(h, env)
  return h
}

// Without motion: the herd as it stands at this hour, the same on every repaint.
export function poseHerd(h: Herd, env: HerdEnv) {
  h.rng = mulberry(h.seed ^ 0x5eed)
  scatter(h, env)
}

// Home from the river with the stock out: the farmer goes in and makes up their feed.
export function feedHerd(h: Herd) {
  const f = h.farmer
  if (!h.out || f.chore !== 'home') return
  f.chore = 'make'
  f.t = 0
}
