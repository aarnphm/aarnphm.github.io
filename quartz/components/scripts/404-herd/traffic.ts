import type { Pt } from '../404-pixel'
import { slant, UNIT } from '../404-downhill'
import { follow, go, graze, homeward, kin, leave, LEN, near, next, roomy, turn } from './grazing'
import {
  ASIDE,
  BEHIND,
  DOOR,
  FRONT,
  HOLDS,
  HOUSE,
  MAKE,
  MOUTH,
  PLACES,
  STRIDE,
  SWING,
  TROUGH,
  type Beast,
  type Chore,
  type Farmer,
  type Herd,
  type HerdEnv,
  type Kind,
  type Passer,
} from './herd-state'
import { cell, home, onGrass, route, sight, tread } from './pasture'

// Ground speeds: a walk, a run, and the creep of grazing.
const PACE: Record<Kind, { walk: number; run: number }> = {
  cow: { walk: 0.9 * UNIT, run: 2.5 * UNIT },
  calf: { walk: 1.1 * UNIT, run: 3 * UNIT },
  sheep: { walk: 0.9 * UNIT, run: 3 * UNIT },
}
const CREEP = 0.12 * UNIT
// How close a bike has to pass for a cow to look up or the ewes to bolt, and how far off the hurt
// the cows will come to stand.
const STARTLE = { cow: 32, calf: 32, sheep: 34 }
const STARE = 150
const STAND_OFF = 26

// On a step toward a point at a ground speed, true on arriving. Every point it walks to is in
// sight over the grass, so the way there stays on it.
function toward(b: { x: number; y: number; face: number }, to: Pt, v: number, dt: number) {
  const dx = to[0] - b.x
  const dy = to[1] - b.y
  const s = Math.min(1, (v * dt) / Math.max(1e-6, slant(dx, dy)))
  b.x += dx * s
  b.y += dy * s
  if (Math.abs(dx) > 0.5) b.face = Math.sign(dx)
  return s >= 1
}

function send(f: Farmer, way: Pt[], after: Chore) {
  f.chore = 'walk'
  f.way = way
  f.after = after
  f.t = 0
}

// Up from the house to open the doors in the morning, or to stand by them while the stock come in
// at night and shut them once the last is in; back down to the house when done.
function stepFarmer(h: Herd, dt: number) {
  const f = h.farmer
  f.t += dt
  switch (f.chore) {
    case 'walk':
      if (!toward(f, f.way[0], STRIDE, dt)) break
      f.way.shift()
      if (!f.way.length) {
        f.chore = f.after
        f.t = 0
      }
      break
    case 'doors': {
      if (f.t < SWING) break
      f.t = 0
      const want = h.out ? 2 : 0
      if (h.doors !== want) h.doors += Math.sign(want - h.doors)
      else if (h.out) send(f, [ASIDE], 'watch')
      else send(f, [ASIDE, HOUSE], 'home')
      break
    }
    case 'make':
      if (f.t > MAKE) send(f, [ASIDE, FRONT, MOUTH, BEHIND], 'pour')
      break
    case 'pour':
      f.face = 1
      if (f.t < 2.5) break
      h.trough = 1
      h.feeds++
      send(f, [MOUTH, FRONT, ASIDE, HOUSE], 'home')
      break
    case 'watch': {
      // Facing the lane until the last is out and clear of its mouth, or in, and a moment after.
      f.face = -1
      const done = h.out
        ? h.beasts.every(b => !b.in && slant(b.x - DOOR[0], b.y - DOOR[1]) > 40)
        : h.beasts.every(b => b.in)
      if (!done) f.t = 0
      else if (f.t > 4) {
        if (h.out) send(f, [HOUSE], 'home')
        else send(f, [FRONT], 'doors')
      }
      break
    }
  }
}

// At the trough: whichever of the cattle that have not eaten from this filling is nearest a free
// place walks over to it, until the trough is empty. A beast that has turned to something else on
// the way gives its place up.
function serve(h: Herd) {
  if (h.trough <= 0) return
  for (const b of h.beasts) if (b.spot >= 0 && b.doing !== 'eat' && b.doing !== 'walk') b.spot = -1
  for (const [k, at] of PLACES.entries()) {
    if (h.beasts.some(b => b.spot === k)) continue
    let best: Beast | null = null
    for (const b of h.beasts) {
      if (b.in || b.kind === 'sheep' || b.fed === h.feeds || b.spot >= 0) continue
      if (b.path.length || b.watch || b.doing === 'run') continue
      if (!best || slant(b.x - at[0], b.y - at[1]) < slant(best.x - at[0], best.y - at[1])) best = b
    }
    if (!best) return
    best.spot = k
    toPlace(h, best)
  }
}
function toPlace(h: Herd, b: Beast) {
  const at = PLACES[b.spot]
  const to = sight([b.x, b.y], at) ? at : near(h, b, b.kind, at[0], at[1], 12)
  // Nowhere to get at it from: it misses this filling.
  if (!to) {
    b.fed = h.feeds
    b.spot = -1
    return
  }
  go(b, to)
}

const hurtScratch: Passer[] = []
export function stepHerd(h: Herd, env: HerdEnv, dt: number, people: Passer[]) {
  if (env.out !== h.out) {
    h.out = env.out
    h.since = 0
    const f = h.farmer
    if (f.chore === 'home') {
      f.x = HOUSE[0]
      f.y = HOUSE[1]
    }
    if (h.out) send(f, [ASIDE, FRONT], 'doors')
    else send(f, [ASIDE], 'watch')
  }
  stepFarmer(h, dt)
  // The stock go through the doors, and their turns count, only while the doors stand open and the
  // farmer is clear of them.
  const open = h.doors === 2 && slant(h.farmer.x - FRONT[0], h.farmer.y - FRONT[1]) > 20
  if (open) h.since += dt
  serve(h)
  hurtScratch.length = 0
  for (const o of people) if (o.hurt) hurtScratch.push(o)
  const hurt = hurtScratch
  for (const [n, b] of h.beasts.entries()) {
    // Called in, each in its turn sets off for the barn from whatever it was doing; called out,
    // each in its turn comes out of the doors once the one before is clear of them.
    if (b.in) {
      if (
        h.out &&
        open &&
        turn(h, n) &&
        h.beasts.every(o => o.in || slant(o.x - DOOR[0], o.y - DOOR[1]) > 30)
      )
        leave(h, b)
      continue
    }
    if (!h.out && turn(h, n) && !homeward(b)) follow(b, route(b.x, b.y))
    b.t += dt
    b.calm = Math.max(0, b.calm - dt)
    const free = !b.path.length
    // A cow comes over to stand and stare at whoever lies hurt within sight, and goes back to the
    // grass once they are gone.
    if (free && b.kind === 'cow') {
      const sore = hurt.find(o => slant(o.x - b.x, o.y - b.y) < STARE)
      if (sore && !b.watch && b.doing !== 'lie') {
        const d = slant(b.x - sore.x, b.y - sore.y) || 1
        const u = STAND_OFF / d
        go(b, near(h, b, 'cow', sore.x + (b.x - sore.x) * u, sore.y + (b.y - sore.y) * u, 20))
        b.watch = [sore.x, sore.y]
      } else if (b.watch && !hurt.some(o => slant(o.x - b.watch![0], o.y - b.watch![1]) < 20))
        next(h, b, env)
    }
    // A bike going by close: a cow looks up, a calf as well; the ewes bolt together, away from it.
    const by =
      !free || b.calm > 0 || b.watch
        ? null
        : people.find(o => o.fast && slant(o.x - b.x, o.y - b.y) < STARTLE[b.kind])
    if (by) {
      b.calm = 8
      if (b.kind === 'sheep' && h.rng() < 0.7) {
        const d = slant(b.x - by.x, b.y - by.y) || 1
        const ux = (b.x - by.x) / d
        const uy = (b.y - by.y) / d
        for (const o of kin(h, b))
          if (!o.path.length && slant(o.x - b.x, o.y - b.y) < 24) {
            o.calm = 8
            o.watch = null
            go(o, near(h, o, 'sheep', o.x + ux * 30, o.y + uy * 15, 5), true)
          }
      } else if (b.kind !== 'sheep' && b.doing === 'graze' && h.rng() < 0.4) {
        b.doing = 'look'
        b.t = 0
        b.until = 3 + h.rng() * 4
        b.at = [by.x, by.y]
      }
    }
    switch (b.doing) {
      case 'graze': {
        // A mouthful, a step along the tread, another; turning back at the edge of the grass, and
        // grazing on the spot when another is in the way.
        const nx = b.x + b.face * CREEP * dt
        const ty = tread(nx, b.y)
        const ny = ty !== null && Math.abs(ty - b.y) < 3 ? ty : b.y
        if (!onGrass(b.kind, nx, ny)) b.face = -b.face
        else if (roomy(h, b, b.kind, nx, ny)) {
          b.x = nx
          b.y = ny
        }
        if (b.t > b.until) next(h, b, env)
        break
      }
      case 'walk':
      case 'run': {
        const v = b.doing === 'run' ? PACE[b.kind].run : PACE[b.kind].walk
        // In file along the lane, in or out: it waits while the one ahead of it, nearer the doors
        // on the way in or further from them on the way out, is within a length.
        const way = homeward(b) ? -1 : 1
        const ahead =
          b.path.length > 0 &&
          h.beasts.some(
            o =>
              o !== b &&
              !o.in &&
              o.path.length > 0 &&
              homeward(o) === homeward(b) &&
              way * (home![cell(o.x, o.y)] - home![cell(b.x, b.y)]) > 0 &&
              slant(o.x - b.x, o.y - b.y) < (LEN[o.kind] + LEN[b.kind]) / 2 + 2,
          )
        if (ahead) {
          if (b.t > b.until) next(h, b, env)
        } else if (toward(b, b.to!, v, dt)) {
          if (b.path.length > 1) {
            b.path.shift()
            b.to = b.path[0]
          } else if (homeward(b)) {
            b.in = true
            b.path = []
            b.to = null
          } else if (b.watch) {
            b.doing = 'look'
            b.at = b.watch
            b.t = 0
            b.until = Infinity
          } else if (b.doing === 'run') {
            b.doing = 'look'
            b.t = 0
            b.until = 3 + h.rng() * 4
            b.at = null
          } else if (b.spot >= 0) {
            const at = PLACES[b.spot]
            if (slant(b.x - at[0], b.y - at[1]) > 1) toPlace(h, b)
            else {
              // Head down in the trough, facing along it.
              b.doing = 'eat'
              b.to = null
              b.t = 0
              b.until = 22 + h.rng() * 14
              b.face = Math.sign(TROUGH[0] - b.x)
            }
          } else graze(b, 20 + h.rng() * 70)
        } else if (b.t > b.until) next(h, b, env)
        break
      }
      case 'look':
        if (b.at) b.face = Math.sign(b.at[0] - b.x) || b.face
        if (b.t > b.until) next(h, b, env)
        break
      case 'lie':
        if (b.t > b.until) next(h, b, env)
        break
      case 'eat':
        h.trough = Math.max(0, h.trough - dt / HOLDS)
        if (b.t < b.until && h.trough > 0) break
        b.fed = h.feeds
        b.spot = -1
        next(h, b, env)
        break
    }
  }
}
