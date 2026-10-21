import type { Pt } from '../404-pixel'
import type { Leg, Medic, Park, Walk } from './park-state'
import { say, type Voices } from '../404-crew'
import { along, path, PATHS, TRACKS, UNIT, type Path } from '../404-downhill'

// The first-aid hut on the grass below the way in, the point on the way in above its door, and the
// trodden path between them.
export const HUT: Pt = [138, 846]
const POST = 117
const DOOR: Path = (() => {
  const at = along(PATHS.base, POST)
  return path([
    [HUT[0] + 14, HUT[1]],
    [at.x, at.y],
  ])
})()
// Worse than this and they stay down for the medic; worse than that and they go out on a board.
export const HURT = 1.5
const BOARD = 2
// On foot: a jog along the flat, a climb up the line, and down it beside someone hurt or carrying
// them on the board. Riders go by anyone on foot on the line at a walk.
const JOG = 2.5 * UNIT
const CLIMB = 1.3 * UNIT
export const LIMP = 0.8 * UNIT
const CARRY = 0.7 * UNIT

// On along a walk; true once at its end.
export function stepWalk(w: Walk, dt: number) {
  let l = w.legs[w.n]
  w.u += l.v * dt
  while (w.u >= Math.abs(l.to - l.from)) {
    if (w.n === w.legs.length - 1) {
      w.u = Math.abs(l.to - l.from)
      return true
    }
    w.u = (w.u - Math.abs(l.to - l.from)) * (w.legs[w.n + 1].v / l.v)
    l = w.legs[++w.n]
  }
  return false
}

// From the hut along the way in, up the run-out, and up the line to just below someone down on it.
const upTo = (line: 0 | 1, q: number, climb: number): Walk => {
  const { base, runout } = PATHS
  const t = TRACKS[line]
  return {
    legs: [
      { on: DOOR, from: 0, to: DOOR.length, v: JOG },
      { on: base, from: POST, to: base.length, v: JOG },
      { on: runout, from: runout.length, to: 0, v: JOG },
      { on: t, from: t.length, to: Math.min(t.length, q + 6), v: climb },
    ],
    n: 0,
    u: 0,
  }
}
// Down the line from there and along the run-out, then back along the way in to the hut's door, or
// on out past it to the car park.
export const downFrom = (line: 0 | 1, q: number, v: number, hut: boolean): Walk => {
  const { base, runout } = PATHS
  const t = TRACKS[line]
  const legs: Leg[] = [
    { on: t, from: q, to: t.length, v },
    { on: runout, from: 0, to: runout.length, v: v * 1.4 },
    { on: base, from: base.length, to: hut ? POST : -30, v: v * 1.4 },
  ]
  if (hut) legs.push({ on: DOOR, from: DOOR.length, to: 0, v: v * 1.4 })
  return { legs, n: 0, u: 0 }
}
export const OUT = (v: number): Walk => ({
  legs: [
    { on: DOOR, from: 0, to: DOOR.length, v },
    { on: PATHS.base, from: POST, to: -30, v },
  ],
  n: 0,
  u: 0,
})

// Someone down for good gets the first medic out of the hut; on a board call a second follows with
// the board. They walk the hurt down to sit by the hut, or with the board carry them out to the car
// park, and later one goes back up for the bike. Out of hours there is nobody in the hut, and the
// hurt get up in their own time and walk themselves down and home.
export function stepMedics(p: Park, dt: number, voices: Voices) {
  // A call already answered gets its board whatever the hour.
  for (const r of p.riders) {
    if (r.state !== 'hurt') continue
    let onCount = 0
    let firstOn: Medic | null = null
    let free: Medic | undefined
    for (const m of p.medics) {
      if (m.who === r) {
        onCount++
        if (!firstOn) firstOn = m
      }
      if (!free && m.state === 'post') free = m
    }
    const second = r.hurt > BOARD && onCount === 1 && firstOn!.state === 'tend'
    if (!free || (onCount ? !second : !p.env.open)) continue
    Object.assign(free, { state: 'go', t: 0, who: r, board: second })
    free.walk = upTo(r.line, r.q, second ? CLIMB * 0.75 : CLIMB)
  }
  p.medics.forEach((m, n) => {
    m.t += dt
    const r = m.who
    switch (m.state) {
      case 'go':
        if (!stepWalk(m.walk!, dt)) break
        m.state = 'tend'
        m.t = 0
        m.until = 40 + 60 * (r!.hurt - HURT) + 30 * p.rng()
        if (!m.board) say(voices, `rider-m${n}`, '?')
        break
      case 'tend': {
        if (!r) break
        const both = p.medics.filter(o => o.who === r && o.state === 'tend')
        if (r.hurt > BOARD) {
          // Strapped to the board once the second medic has been there a while.
          if (both.length < 2 || Math.min(...both.map(o => o.t)) < 15) break
          r.state = 'carried'
          r.t = 0
          r.walk = downFrom(r.line, r.q, CARRY, false)
          p.wrecks.push({ line: r.line, q: r.q, frame: r.kit[2] })
          for (const o of both) Object.assign(o, { state: 'carry', t: 0, walk: null })
        } else if (m.t > m.until) {
          // Up on their feet, and the folk at the top clap.
          r.state = 'limp'
          r.t = 0
          r.walk = downFrom(r.line, r.q, LIMP, true)
          m.state = 'lead'
          m.walk = downFrom(r.line, r.q, LIMP, true)
          m.walk.u = -8
          if (p.rng() < 0.7) say(voices, p.rng() < 0.5 ? 'folk0' : 'folk1', '!')
        }
        break
      }
      case 'lead':
        if (stepWalk(m.walk!, dt)) Object.assign(m, { state: 'post', walk: null, who: null })
        break
      case 'away':
        if (m.t > 25) {
          m.state = 'back'
          m.walk = {
            legs: [
              { on: PATHS.base, from: 0, to: POST, v: 1.5 * UNIT },
              { on: DOOR, from: DOOR.length, to: 0, v: 1.5 * UNIT },
            ],
            n: 0,
            u: 0,
          }
        }
        break
      case 'back': {
        if (!stepWalk(m.walk!, dt)) break
        const wreck = p.wrecks[0]
        if (wreck && !p.medics.some(o => o.state === 'fetch' || o.state === 'wheel')) {
          m.state = 'fetch'
          m.walk = upTo(wreck.line, wreck.q - 6, CLIMB)
        } else Object.assign(m, { state: 'post', walk: null, board: false })
        break
      }
      case 'fetch': {
        if (!stepWalk(m.walk!, dt)) break
        const wreck = p.wrecks[0]
        m.state = 'wheel'
        m.walk = downFrom(wreck.line, wreck.q, 1.2 * UNIT, true)
        break
      }
      case 'wheel':
        if (!stepWalk(m.walk!, dt)) break
        p.leaning.push({ frame: p.wrecks.shift()!.frame, t: 0 })
        Object.assign(m, { state: 'post', walk: null })
        break
    }
  })
  let lw = 0
  for (let i = 0; i < p.leaning.length; i++) {
    const b = p.leaning[i]
    b.t += dt
    if (b.t < 900) p.leaning[lw++] = b
  }
  p.leaning.length = lw
  if (p.leaning.length > 2) p.leaning.splice(0, p.leaning.length - 2)
}
