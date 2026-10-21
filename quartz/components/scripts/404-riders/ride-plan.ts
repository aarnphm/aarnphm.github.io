import type { Park, Rider } from './park-state'
import { DROP_OFF, G, MAIN, TABLE_FLIGHT, TRACKS, UNIT, type Track } from '../404-downhill'
import { noise } from '../404-pixel'

// A bermed turn lends this many g on top of the dirt's own grip.
export const BANK = 0.7

// The berm below the drop, along the main line, and what the dirt holds at a sample of a line.
const WORN: [number, number] = [255, 310]
export const gripAt = (p: Park, t: Track, k: number) =>
  p.env.grip * (t.main[k] > WORN[0] && t.main[k] < WORN[1] ? 1 - p.env.loose : 1)

export const brakeOf = (r: Rider) => G * r.grip * (0.6 + 0.25 * r.skill)
// Six watts a kilo out of the gate (four for a child), capped at what the tyres put down.
export const pedalOf = (r: Rider) => Math.min(25, (10 * (r.kid ? 4 : 6)) / Math.max(1, r.v / UNIT))

// The speed a rider means to carry at each sample of their line this run: as fast as they dare in
// a straight line, round each turn at their own share of what they think the dirt and the berm
// will hold (with the odd misjudged corner, fewer the better they are), at the speed they have
// settled on for the table, and braking back from each of those as hard as they brake.
export function planFor(p: Park, r: Rider) {
  const t = TRACKS[r.line]
  const plan = new Float32Array(t.n)
  const margin = (0.7 + 0.25 * r.skill) * (r.shaken ? 0.85 : 1)
  const top = (8 + 6 * r.skill) * UNIT * (r.kid ? 0.8 : 1)
  const sway = 0.04 + 0.2 * (1 - r.skill)
  const seed = Math.floor(p.rng() * 1e4)
  for (let k = 0; k < t.n; k++) {
    // What the tyres have to give sideways is what is left once they have braked against the hill
    // enough to hold their speed down it.
    const err = 1 + sway * (noise(t.q[k] / 45, 0.5, seed) - 0.5) * 3
    const hold = G * Math.max(0, -t.grade[k] / Math.hypot(1, t.grade[k]))
    const grip = G * (r.grip + BANK * t.bank[k]) * margin * err
    const lat = Math.sqrt(Math.max(0.1 * grip * grip, grip * grip - hold * hold))
    plan[k] = Math.min(top, Math.sqrt(lat / Math.max(1e-4, Math.abs(t.bend[k]))))
  }
  const lip = t.lips[t.lips.length - 1]
  const aim = TABLE_FLIGHT.v * (1 + r.bias + (p.rng() - 0.5) * (0.04 + 0.2 * (1 - r.skill)))
  plan[lip] = Math.min(plan[lip], aim)
  const brake = brakeOf(r) * margin
  for (let k = t.n - 2; k >= 0; k--) {
    const sin = t.grade[k + 1] / Math.hypot(1, t.grade[k + 1])
    const ds = t.q[k + 1] - t.q[k]
    plan[k] = Math.min(plan[k], Math.sqrt(plan[k + 1] ** 2 + 2 * Math.max(8, brake + G * sin) * ds))
    // Off the drop no faster than lands them at a speed they can carry from where they land.
    if (t === MAIN && k === MAIN.lips[0]) {
      let cap = DROP_OFF[0].v
      for (const o of DROP_OFF) if (o.glide <= plan[Math.min(t.n - 1, o.k)]) cap = o.v
      plan[k] = Math.min(plan[k], cap)
    }
  }
  return plan
}

export function release(p: Park, r: Rider) {
  // The drop, or the chicken line round it: nerve and confidence against the drop, and a child
  // only with the skill for it.
  const bold = r.conf * (0.6 + r.skill) + (p.rng() - 0.5) * 0.2
  r.line = bold > 0.52 && !(r.kid && r.skill < 0.45) ? 0 : 1
  r.state = 'ride'
  r.k = 0
  r.q = 0
  r.v = 0
  r.fell = false
  r.shaken = false
  r.swerve = 0
  r.plan = planFor(p, r)
  p.since = 0
  p.last = r.party
  p.count = -1
  p.go = 0.6
}
