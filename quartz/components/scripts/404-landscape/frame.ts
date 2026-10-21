import type { Layer } from './raster'
import type { LandscapeScope, Mode, Pointer } from './scope'
import { stepAngler } from '../404-angler'
import { stepShed } from '../404-boughs'
import { alongCat, catHead, mealFor, poseCat, stepCat } from '../404-cat'
import { ageBubbles, ask, heard, poseCrew, say, stepCrew } from '../404-crew'
import { feedHerd } from '../404-herd'
import { stepJay } from '../404-jay'
import { stepCurtain } from '../404-overgrowth'
import { arrive, dive, stepPet } from '../404-pet'
import { stepSkyway } from '../404-planes'
import { stepPark } from '../404-riders'
import { stepHill } from '../404-skiers'
import { sighting, stepTriathlete } from '../404-triathlete'
import { stepWalker } from './camp'
import { onTrail, TRAIL_LEN } from './geography'
import { handOver, placePaw, STROKE, told } from './hands'
import { stepHearth } from './hearth'
import { paintFront, paintPortal, paintRanges, paintRuin, paintShore, shedding } from './layers'
import { recheck } from './repaint'
import { stepRift } from './rift-motion'

const OFF: Pointer = { x: -1e4, y: -1e4 }

export function stepLandscape(st: LandscapeScope, dt: number, now: number, pointer: Pointer) {
  if (st.reduce) return
  st.clock = now
  const inside = pointer.x > -1e3
  const mx = inside && !Number.isNaN(st.lastX) ? pointer.x - st.lastX : 0
  const my = inside && !Number.isNaN(st.lastY) ? pointer.y - st.lastY : 0
  const moved = Math.hypot(mx, my)
  st.gust = Math.min(3, st.gust * Math.exp(-dt * 1.2) + Math.min(Math.abs(mx), 200) * 0.004)
  st.wind += dt * (0.04 + 0.14 * st.weather.wind + st.gust * 0.25)
  if ((st.almanacAcc += dt) > 60) {
    st.almanacAcc = 0
    recheck(st)
  }
  // The cursor sheds eddies into the sky, alternating in sign on either side of its path: a
  // Kármán street, one vortex per stride.
  if (inside && pointer.y < 640 && moved / dt > 120) {
    st.shed += moved
    if (st.shed > 70) {
      st.shed = 0
      st.sign = -st.sign
      st.air.vortices.push({
        x: pointer.x - (my / moved) * 16 * st.sign,
        y: pointer.y + (mx / moved) * 16 * st.sign,
        gamma: st.sign * Math.min(moved / dt, 1400) * 32,
        age: 0,
      })
      if (st.air.vortices.length > 8) st.air.vortices.shift()
    }
  }
  st.lastX = inside ? pointer.x : NaN
  st.lastY = inside ? pointer.y : NaN
  let vw = 0
  for (let i = 0; i < st.air.vortices.length; i++) {
    const v = st.air.vortices[i]
    v.age += dt
    v.gamma *= Math.exp(-dt / 2.4)
    v.x += 10 * dt
    if (Math.abs(v.gamma) > 600) st.air.vortices[vw++] = v
  }
  st.air.vortices.length = vw
  st.air.drift += 10 * dt * (0.6 + 1.6 * st.weather.wind)
  st.air.time = now
  st.air.pull += (st.pullTarget - st.air.pull) * Math.min(1, dt * 1.5)

  // The moss waits for the reveal.
  if (st.scene.classList.contains('is-ready')) st.grown += dt
  // The clock's minute impulse, heard at most every 0.8 s while time is warped.
  const m = Math.round(st.dial.minute / 6)
  const impulse = !Number.isNaN(st.minute) && m !== st.minute && now - st.lastImpulse > 0.8
  if (impulse) st.lastImpulse = now
  st.minute = m
  stepRift(st.rift, dt)
  stepCurtain(st.curtain, dt, now, st.rift.heat, st.rift.inflow, st.turn)
  // The cat watches whichever is nearer, the pointer or the triathlete going by.
  stepTriathlete(st.tri, dt)
  const seen = sighting(st.tri, st.strand)
  const [hx, hy] = catHead(st.cat)
  const away = (p: Pointer) => Math.hypot(p.x - hx, p.y - hy)
  // With the button focused, the keyboard's hand is the only one on the cat. It strokes while an
  // arrow is held, and past either end of the line it lifts for a frame and comes down at the
  // other end, so a held arrow gives one long stroke after another.
  let hand = pointer
  if (st.paw.on) {
    hand = OFF
    if (st.paw.dir) {
      st.paw.along += (st.paw.dir * -st.cat.face * dt) / STROKE
      const lifted = st.paw.along < 0 || st.paw.along >= 1
      st.paw.along -= Math.floor(st.paw.along)
      if (!lifted) {
        const [x, y] = alongCat(st.cat, st.paw.along)
        hand = { x, y }
      }
    }
  }
  stepCat(st.cat, dt, {
    pointer: seen && away(seen) < away(pointer) ? seen : pointer,
    hand,
    heat: st.rift.heat,
    impulse,
    ground: st.bank,
    told: what => told(st, what),
  })
  ageBubbles(st.purr, dt)
  handOver(st, !st.paw.on && !!st.cat.petting.hand)
  if (st.paw.on) placePaw(st)
  // The pet hops in once the scene shows, from wherever the site pet last stood on screen.
  if (st.pet.mode === 'away' && st.pets() && st.scene.classList.contains('is-ready'))
    arrive(st.pet, st.spotInScene())
  stepPet(st.pet, dt, inside ? [pointer.x, pointer.y] : null, u => onTrail(u * TRAIL_LEN))
  stepCrew(st.crew, dt, {
    clock: now,
    heat: st.rift.heat,
    inflow: st.rift.inflow,
    impulse,
    turn: st.turn,
    cat: st.cat.x,
  })
  // The farmer throws only to a cat lying in front of the boat and within a throw of it.
  const meal = mealFor(st.cat, st.bank)
  stepAngler(st.angler, dt, {
    fit: st.fair(),
    home: st.park.herd.farmer.chore === 'home',
    swimmer: st.tri.leg === 'swim' ? st.tri.x : null,
    cat: meal && meal[0] < st.angler.x && st.angler.x - meal[0] < 320 ? meal : null,
    say: glyph => say(st.crew, 'angler', glyph),
    splash: (x, y, size) => st.splashes.push({ x, y, size, age: 0 }),
    toss: () => (st.cat.coming = true),
    landed: ([x, y]) => {
      st.cat.coming = false
      st.cat.meal = { x, y, left: 1 }
      // The one who has been holding out a fish to it all along.
      say(st.crew, 'friend', '?')
    },
    back: () => feedHerd(st.park.herd),
  })
  if (st.sky)
    stepSkyway(st.skyway, dt, {
      cover: st.weather.cover,
      day: !st.night,
      flyable: !st.weather.fog && !st.weather.storm,
      drift: st.air.drift,
      left: st.sky.g.sx[0],
      right: st.sky.g.sx[st.sky.g.bw - 1],
    })
  stepPark(st.park, dt, st.far)
  stepHill(st.slopes, dt, st.far)
  ageBubbles(st.far, dt)
  ageBubbles(st.boughs, dt)
  if (st.leaves) stepShed(st.litter, st.canopy.trees, shedding(st, st.leaves.g), dt)
  // The clouds drift left to right, so a positive wind comes from the left.
  stepJay(st.jay, dt, {
    active: !st.night && !st.weather.storm,
    errand:
      st.times.autumn > 0.35 && st.times.bare < 0.45
        ? 'cache'
        : st.times.bare > 0.45 || st.times.ground > 0.3 || st.times.blossom > 0.2
          ? 'retrieve'
          : 'forage',
    wind: Math.min(1, st.weather.wind + st.gust * 0.2),
  })
  stepWalker(st.walker, dt, st.drawn)
  stepHearth(st.hearth, st.air, dt)
  for (let n = st.splashes.length - 1; n >= 0; n--)
    if ((st.splashes[n].age += dt) > 1.6) st.splashes.splice(n, 1)
  paintRuin(st)
  st.riftAcc += dt
  if (st.riftAcc >= 1 / 20) {
    st.riftAcc = 0
    paintPortal(st)
  }
  let due: (typeof st.jobs)[number] | null = null
  for (const job of st.jobs) {
    job.acc += dt
    if (job.acc >= job.every && (!due || job.acc / job.every > due.acc / due.every)) due = job
  }
  if (due) {
    due.acc = 0
    due.run()
  }
}

export function portalLandscape(st: LandscapeScope, mode: Mode) {
  st.pet.hot = mode !== 'idle'
  if (mode === 'hot' && st.pet.mode === 'sit' && st.pets()) say(st.crew, 'pet', '!')
  if (mode === 'enter') {
    let bw = 0
    for (let i = 0; i < st.crew.bubbles.length; i++)
      if (st.crew.bubbles[i].who !== 'pet') st.crew.bubbles[bw++] = st.crew.bubbles[i]
    st.crew.bubbles.length = bw
    dive(st.pet)
  }
  st.rift.target = mode === 'enter' ? 90 : mode === 'hot' ? 26 : 1.1
  st.rift.inflowTarget = mode === 'enter' ? 1.8 : mode === 'hot' ? 0.22 : 0
  st.pullTarget = mode === 'enter' ? 2.5 : mode === 'hot' ? 1 : 0
  st.drawn = mode !== 'idle'
  if (st.reduce) {
    st.rift.heat = st.drawn ? 1 : 0
    poseCat(st.cat, st.rest, st.drawn)
    poseCrew(st.crew, st.drawn)
    st.stance = `${st.cat.alert}${st.cat.look}`
    paintRuin(st)
    paintPortal(st)
    paintShore(st)
  }
}

export function hailLandscape(st: LandscapeScope, x: number, y: number) {
  // Client pixels into each layer's buffer; the layers drift apart under parallax.
  const at = (l: Layer | null, el: HTMLCanvasElement | null) => {
    if (!l || !el) return null
    const r = el.getBoundingClientRect()
    return [((x - r.left) / r.width) * l.g.bw, ((y - r.top) / r.height) * l.g.bh] as const
  }
  const c = at(st.leaves, st.front)
  const n = at(st.glint, st.near)
  const f = at(st.ranges, st.back)
  const tree = c && heard(st.boughs, c[0], c[1], 3, 9)
  const bank = !tree && n ? heard(st.crew, n[0], n[1], 3, 10) : null
  const ridge = !tree && !bank && f ? heard(st.far, f[0], f[1], 2, 4) : null
  if (tree) ask(st.boughs, tree)
  else if (bank) ask(st.crew, bank)
  else if (ridge) ask(st.far, ridge)
  else return false
  paintFront(st)
  paintShore(st)
  paintRanges(st)
  // Without motion nothing ages the bubble, so it is taken down on a timer.
  if (st.reduce) {
    window.clearTimeout(st.hush)
    st.hush = window.setTimeout(() => {
      for (const v of [st.crew, st.far, st.boughs])
        v.bubbles = v.bubbles.filter(b => b.glyph !== '?')
      paintFront(st)
      paintShore(st)
      paintRanges(st)
    }, 1800)
  }
  return true
}
