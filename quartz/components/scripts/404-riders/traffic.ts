import type { Voices } from '../404-crew'
import type { Park, Rider } from './park-state'
import { LIFT, PATHS, UNIT } from '../404-downhill'
import { stepHerd, type Passer } from '../404-herd'
import { next, recruit } from './population'
import { held, mainQ, onCourse, place } from './positions'
import { downFrom, LIMP, OUT, stepMedics, stepWalk } from './rescue'
import { release } from './ride-plan'
import { down, fly, ride } from './rider-motion'

// The drag lift's speed along its track, and the seconds between its bars.
export const LIFT_V = 2.5 * UNIT
export const BARS = 5
// Pushing a bike up the lift track instead: Tobler's hiking function on the track's grade, a
// quarter slower for the bike, turned from pace over the map into pace along the slope.
const HIKE = (() => {
  const [[x0, y0], [x1, y1]] = LIFT
  const rise = (y0 - y1) / Math.hypot(x1 - x0, (y0 - y1) * Math.sqrt(3))
  return 0.75 * (6 / 3.6) * Math.exp(-3.5 * Math.abs(rise + 0.05)) * Math.hypot(1, rise) * UNIT
})()
const WALK = 1.2 * UNIT
const ROLL = 2.5 * UNIT
// Spacing in a queue, along the slope.
export const SLOT = 9

const byTicket = (a: Rider, b: Rider) => a.ticket - b.ticket

// Whom the herd takes notice of: bikes going by fast, and anyone lying hurt.
const passersOut: Passer[] = []
function passers(p: Park): Passer[] {
  passersOut.length = 0
  for (const r of p.riders) {
    const fast = (r.state === 'ride' || r.state === 'air') && r.v > 3 * UNIT
    const hurt = r.state === 'hurt'
    if (!fast && !hurt) continue
    const at = place(r)
    passersOut.push({ x: at.x, y: at.ground, fast, hurt })
  }
  return passersOut
}

const toward = (from: number, to: number, step: number) =>
  from < to ? Math.min(to, from + step) : Math.max(to, from - step)

const liftqScratch: Rider[] = []
const gateqScratch: Rider[] = []

export function stepPark(p: Park, dt: number, voices: Voices) {
  p.time += dt
  p.cheer = Math.max(0, p.cheer - dt)
  p.flash = Math.max(0, p.flash - dt)
  p.go = Math.max(0, p.go - dt)
  p.since += dt
  // The lift runs through its hours, and on until whoever is on it is off.
  const bar = Math.floor(p.lift / BARS)
  if (p.env.open || p.riders.some(r => r.state === 'lift')) p.lift += dt
  let board = p.env.open && Math.floor(p.lift / BARS) !== bar

  let live = 0
  liftqScratch.length = 0
  gateqScratch.length = 0
  for (const r of p.riders) {
    if (r.state !== 'leave') live++
    if (r.state === 'liftq') liftqScratch.push(r)
    else if (r.state === 'link' || r.state === 'gate') gateqScratch.push(r)
  }
  if (live < p.env.crowd && (p.arrive -= dt) <= 0) {
    recruit(p, p.env.crowd - live)
    p.arrive = 6 + p.rng() * 14
  }

  const liftq = liftqScratch.sort(byTicket)
  const gateq = gateqScratch.sort(byTicket)
  const { base, lift, link, runout } = PATHS
  for (const r of p.riders) {
    r.t += dt
    switch (r.state) {
      case 'arrive': {
        r.s += WALK * dt
        const back = base.length - SLOT * liftq.length
        if (p.env.open && !r.hikes && r.s >= back) {
          r.state = 'liftq'
          r.ticket = ++p.tickets
          liftq.push(r)
        } else if (r.s >= base.length) {
          r.state = 'hike'
          r.s = 0
        }
        break
      }
      case 'liftq': {
        const n = liftq.indexOf(r)
        r.s = toward(r.s, base.length - SLOT * n, WALK * dt)
        if (!p.env.open) r.state = 'arrive'
        else if (n === 0 && board && r.s >= base.length - 0.5) {
          board = false
          r.state = 'lift'
          r.s = 0
        }
        break
      }
      case 'lift':
      case 'hike': {
        // Pushing up, in single file a couple of metres apart.
        let room = Infinity
        if (r.state === 'hike')
          for (const o of p.riders)
            if (o.state === 'hike' && (o.s > r.s || (o.s === r.s && o.id < r.id)))
              room = Math.min(room, o.s - 20)
        r.s = Math.max(r.s, Math.min(room, r.s + (r.state === 'lift' ? LIFT_V : HIKE) * dt))
        if (r.s >= lift.length) {
          r.state = 'link'
          r.s = 0
          r.ticket = ++p.tickets
          gateq.push(r)
        }
        break
      }
      case 'link':
      case 'gate': {
        const slot = link.length - SLOT * gateq.indexOf(r)
        r.s = toward(r.s, slot, (r.state === 'link' ? ROLL : WALK) * dt)
        if (r.s >= slot - 0.5) r.state = 'gate'
        break
      }
      case 'ride':
        ride(p, r, dt, voices)
        break
      case 'air':
        fly(p, r, dt, voices)
        break
      case 'down':
      case 'up':
        down(p, r, dt)
        break
      case 'runout':
        r.v = Math.max(ROLL, r.v - 4 * UNIT * dt)
        r.s += r.v * dt
        if (r.s >= runout.length) next(p, r)
        break
      case 'leave':
        r.s -= ROLL * 1.2 * dt
        if (r.s <= -24) r.gone = true
        break
      case 'hurt':
        // Nobody coming out of hours: up in their own time, and down and home.
        if (!p.env.open && !p.medics.some(m => m.who === r) && r.t > 15 + 10 * r.hurt) {
          r.state = 'limp'
          r.t = 0
          r.walk = downFrom(r.line, r.q, LIMP, false)
        }
        break
      case 'limp':
        if (!stepWalk(r.walk!, dt)) break
        if (r.walk!.legs[r.walk!.legs.length - 1].to < 0) r.gone = true
        else Object.assign(r, { state: 'rest', t: 0, walk: null })
        break
      case 'carried':
        if (!stepWalk(r.walk!, dt)) break
        r.gone = true
        for (const m of p.medics)
          if (m.who === r) Object.assign(m, { state: 'away', t: 0, who: null })
        break
      case 'rest':
        // Patched up and sat a while by the hut, then off home on foot.
        if (r.t > 40 + 10 * (r.id % 6)) Object.assign(r, { state: 'limp', t: 0, walk: OUT(WALK) })
        break
    }
  }
  let rw = 0
  for (let i = 0; i < p.riders.length; i++) {
    const r = p.riders[i]
    if (!r.gone) p.riders[rw++] = r
  }
  p.riders.length = rw
  stepMedics(p, dt, voices)
  stepHerd(p.herd, p.env.herd, dt, passers(p))

  // The starter lets the next rider go once whoever went before is well down the line and nobody
  // is down on it: a few seconds between strangers, a couple between friends, who go as a train.
  // Three beeps, and green. After hours there is no starter, and the rider just goes.
  const head = gateq[0]
  const ready = head && head.state === 'gate' && head.s >= link.length - 0.5
  const clear =
    !held(p) &&
    !p.riders.some(
      o =>
        o.state === 'down' ||
        o.state === 'up' ||
        (onCourse(o) && mainQ(o) < (o.party === head?.party ? 30 : 160)),
    )
  if (!ready || !clear || p.since < (head.party === p.last ? 2.2 : 5)) p.count = -1
  else if (!p.env.open) release(p, head)
  else if ((p.count = Math.max(0, p.count) + dt) >= 1.8) release(p, head)
}
