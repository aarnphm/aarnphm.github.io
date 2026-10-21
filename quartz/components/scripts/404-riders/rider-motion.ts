import { say, type Voices } from '../404-crew'
import {
  G,
  LANDING,
  seek,
  surface,
  SWEET,
  TABLE_FLIGHT,
  TRACKS,
  UNIT,
  type Track,
} from '../404-downhill'
import { DUR, TRICKS, type Park, type Rider } from './park-state'
import { across, clamp, footQ, mainQ, onCourse } from './positions'
import { HURT } from './rescue'
import { BANK, brakeOf, gripAt, pedalOf, planFor } from './ride-plan'

// Rolling resistance of knobbly tyres in dirt, and air drag per unit of speed squared for a rider
// and bike of 95 kg with 0.55 m² of frontal area.
const CRR = 0.035
const DRAG = (0.5 * 1.2 * 0.55) / 95 / UNIT

// The table is the last lip on either line.
const isTable = (t: Track, lip: number) => lip === t.lips[t.lips.length - 1]

// Past the limit in a turn, the odds the slide goes from under them, per unit past it.
const LOSE = 10

// How badly they are hurt: how far past the limit they came off, times the energy they came off
// with against eight metres a second, times how they happened to land.
function crash(p: Park, r: Rider, hit: number, v: number, voices: Voices) {
  r.state = 'down'
  r.t = 0
  r.v = v * 0.6
  r.hurt = hit * (v / (8 * UNIT)) ** 2 * Math.exp(0.6 * (p.rng() + p.rng() + p.rng() - 1.5))
  r.trick = null
  r.fell = true
  r.conf = clamp(r.conf - 0.2 * hit, 0.05, 1)
  say(voices, `rider${r.id}`, '!')
  if (p.rng() < 0.6) say(voices, p.rng() < 0.5 ? 'folk0' : 'folk1', '!')
}

// Riding the line: the hill pulls, the tyres and the air drag, and the rider brakes or pedals
// toward the lesser of their plan and the speed at which they could still stop behind whoever is
// ahead. Friends ride closer than strangers.
export function ride(p: Park, r: Rider, dt: number, voices: Voices) {
  const t = TRACKS[r.line]
  const me = mainQ(r)
  let gap = Infinity
  let lead: Rider | null = null
  for (const o of p.riders) {
    if (o === r || !onCourse(o)) continue
    const d = mainQ(o) - me
    if ((d > 0 || (d === 0 && o.id < r.id)) && d < gap) {
      gap = d
      lead = o
    }
  }
  const friend = lead?.party === r.party
  const s0 = friend ? 20 : 45
  const tau = friend ? 0.35 : 0.8
  const brake = brakeOf(r)
  // When the plan has got away from them they grab a handful, up to nearly all the grip they
  // believe the dirt has.
  const hard = G * r.grip * (0.8 + 0.15 * r.skill)
  const vl = lead && (lead.state === 'ride' || lead.state === 'air') ? lead.v : 0
  const safe = lead
    ? -brake * tau + Math.sqrt(Math.max(0, (brake * tau) ** 2 + vl * vl + 2 * brake * (gap - s0)))
    : Infinity
  // They ride to the slowest the plan asks of them over the next quarter second, which is how far
  // behind it their hands run, and never past a lip before they have reached it.
  const plan = r.plan!
  const lipAhead = t.lips.find(l => l > r.k) ?? t.n - 1
  const reach = Math.min(lipAhead, seek(t, r.q + r.v * 0.25 + 1, r.k) + 1)
  let want = safe
  for (let k = r.k + 1; k <= reach; k++) want = Math.min(want, plan[k])
  // Past someone hurt on the line, or anyone on foot on it, at a walk.
  for (const o of p.riders) {
    const at =
      o.state === 'hurt'
        ? mainQ(o)
        : o.state === 'limp' || o.state === 'carried'
          ? footQ(o.walk)
          : null
    if (at === null) continue
    if (at - me > -15 && at - me < 90) want = Math.min(want, PASS)
    if (o.state === 'hurt' && at > me && at - me < 100 && r.asked !== o.id) {
      r.asked = o.id
      say(voices, `rider${r.id}`, '?')
    }
  }
  for (const m of p.medics) {
    const at = footQ(m.walk)
    if (at !== null && at - me > -15 && at - me < 90) want = Math.min(want, PASS)
  }
  // A beast grazing at the line's edge, ahead or alongside: they hold their speed and swerve to the
  // far side of the line, the more the closer it stands.
  const [nx, ny] = across(t, r.k)
  let away = 0
  for (const b of p.herd.beasts) {
    const bx = b.x - t.x[r.k]
    const by = b.y - t.y[r.k]
    if (b.in || bx * ny - by * nx < -8) continue
    const lean = -Math.sign(bx * nx + by * ny) * SWERVE * clamp((24 - Math.hypot(bx, by)) / 6, 0, 1)
    if (Math.abs(lean) > Math.abs(away)) away = lean
  }
  r.swerve += clamp(away - r.swerve, -SWERVE * 3 * dt, SWERVE * 3 * dt)
  const gr = t.grade[Math.min(t.n - 1, r.k + 1)]
  const cos = 1 / Math.hypot(1, gr)
  const pull = -G * gr * cos - CRR * G * cos - DRAG * r.v * r.v
  // They brake only as hard as they think the tyres will take on top of the turn.
  const lat = r.v * r.v * Math.abs(t.bend[r.k])
  const felt = G * (r.grip + BANK * t.bank[r.k])
  const room = Math.sqrt(Math.max(0, felt * felt - lat * lat))
  const grab = r.v > want + 0.5 * UNIT ? hard : brake
  const control = clamp((want - r.v) / 0.25 - pull, -Math.min(grab, room), pedalOf(r))
  r.v = Math.max(0, r.v + (pull + control) * dt)
  // Coming up on someone who is down or getting up: are you all right?
  if (lead && (lead.state === 'down' || lead.state === 'up') && gap < 100 && r.asked !== lead.id) {
    r.asked = lead.id
    say(voices, `rider${r.id}`, '?')
  }
  // In a turn the tyres hold what the dirt and the banking give them, braking included. Past it
  // they slide, and the further past and the less skilled they are, the likelier it goes from under
  // them; otherwise a foot down saves it.
  const dirt = gripAt(p, t, r.k)
  const over = Math.hypot(lat, Math.min(0, control)) / (G * (dirt + BANK * t.bank[r.k]))
  if (over > 1.04 && r.v > 20) {
    r.grip += (dirt - r.grip) * 0.6
    r.conf = clamp(r.conf - 0.06, 0.05, 1)
    if (over > 1.18 || p.rng() < (over - 1.04) * LOSE * (1.2 - r.skill))
      return crash(p, r, over, r.v, voices)
    r.v *= 0.82
    r.dab = 0.5
  }
  r.dab = Math.max(0, r.dab - dt)
  r.heavy = Math.max(0, r.heavy - dt)
  const q0 = r.q
  r.q += r.v * cos * dt
  r.k = seek(t, r.q, r.k)
  for (const lip of t.lips)
    if (q0 < t.q[lip] && r.q >= t.q[lip]) return launch(p, r, t, lip, voices)
  if (r.k >= t.n - 1) {
    r.state = 'runout'
    r.s = 0
    r.runs++
    if (!r.fell) r.conf = clamp(r.conf + 0.08, 0.05, 1)
  }
}

// Off a lip along its grade, springing off it as well as they can. Off the drop at a walk the
// front wheel falls first. Off the table, if the flight they expect has room for it and their nerve
// is up, they throw a trick.
function launch(p: Park, r: Rider, t: Track, lip: number, voices: Voices) {
  const a = Math.atan(t.grade[lip])
  r.state = 'air'
  r.lip = lip
  r.launch = r.v
  r.q = t.q[lip]
  r.k = lip
  r.z = t.z[lip]
  r.vh = r.v * Math.cos(a)
  r.vz = r.v * Math.sin(a) + 0.4 * UNIT * r.skill
  r.trick = null
  r.trickT = 0
  if (!isTable(t, lip)) {
    if (r.v < 2.2 * UNIT) crash(p, r, 1.3, r.v, voices)
    return
  }
  // How long the trick takes them this time they only find out in the air: the less skilled, the
  // more it can drag, and a trick still out when the wheels touch is a crash.
  if (!r.tricks.length) return
  const expect = (TABLE_FLIGHT.time * r.v) / TABLE_FLIGHT.v
  const pick = r.tricks[Math.floor(p.rng() * r.tricks.length)]
  if (p.rng() < r.nerve * r.conf && expect > DUR[pick] + 0.3 * (1 - r.nerve)) {
    r.trick = pick
    r.trickT = DUR[pick] * (1 + (p.rng() - 0.3) * 2.4 * (1 - r.skill) ** 1.5)
  }
}

export function fly(p: Park, r: Rider, dt: number, voices: Voices) {
  const t = TRACKS[r.line]
  const steps = Math.ceil(dt * 240)
  const h = dt / steps
  for (let n = 0; n < steps; n++) {
    const up = r.vz > 0
    r.q += r.vh * h
    r.vz -= G * h
    r.z += r.vz * h
    r.k = seek(t, r.q, r.k)
    if (r.trick) {
      r.trickT -= h
      // The photographer fires as the trick peaks.
      if (up && r.vz <= 0 && p.env.open) p.flash = 0.25
    }
    if (r.k >= t.n - 1 || r.z <= surface(t, r.q, r.k)) return land(p, r, t, voices)
  }
}

// Down onto whatever is under them: the speed along the surface carries on, the speed into it
// is what their legs have to take. Short of the table's landing is a case, onto its deck; past it
// is the flat of the hill. They judge from where they came down how far off their speed was.
function land(p: Park, r: Rider, t: Track, voices: Voices) {
  const gl = t.grade[Math.min(t.n - 1, r.k + 1)]
  const norm = Math.hypot(1, gl)
  const impact = -(r.vz - gl * r.vh) / norm
  const glide = Math.max(0, (r.vh + r.vz * gl) / norm)
  r.z = surface(t, r.q, r.k)
  const table = isTable(t, r.lip)
  const at = t.main[r.k] + (r.q - t.q[r.k])
  const cased = table && at < LANDING[0]
  const tol = UNIT * (3.2 + 3 * r.skill) * (r.kid ? 0.85 : 1)
  let hit = impact / tol + (cased ? 0.25 : 0)
  const botched = r.trick !== null && r.trickT > 0
  if (botched) hit = Math.max(hit, 1.3)
  if (table) {
    // How far short or long of the sweet spot they came down, over how much further a flight goes
    // for a little more speed, is the speed they were out by, and they shift what they aim for by
    // some of it. They feel it only roughly, and the new feel it worse.
    const off = (SWEET - at) / TABLE_FLIGHT.reach / TABLE_FLIGHT.v
    r.bias += off * (0.35 + 0.4 * r.skill) + (p.rng() - 0.5) * 0.06 * (1 - r.skill)
  }
  if (hit > 1) {
    // Half the time a botched trick is put away for the day.
    if (botched && p.rng() < 0.5) r.tricks = r.tricks.filter(t => t !== r.trick)
    return crash(p, r, hit, glide, voices)
  }
  r.state = 'ride'
  r.v = glide * (cased ? 0.75 : 1)
  r.heavy = hit > 0.6 ? 0.35 : 0
  if (r.trick) {
    p.cheer = 1.6
    r.conf = clamp(r.conf + 0.05, 0.05, 1)
    if (p.rng() < 0.4) say(voices, 'folk1', '!')
    // Landing the hardest one they have, the nervy try the next one up.
    const n = r.tricks.length
    if (r.trick === r.tricks[n - 1] && n < TRICKS.length && p.rng() < 0.25 * r.nerve)
      r.tricks = TRICKS.slice(0, n + 1)
  } else if (!cased && hit < 0.6) r.conf = clamp(r.conf + 0.03, 0.05, 1)
  r.trick = null
}

// Sliding to a stop, lying there a moment (longer the harder it was), getting up, and riding on
// from a standstill, shaken, at a lesser margin for the rest of the run.
export function down(p: Park, r: Rider, dt: number) {
  const t = TRACKS[r.line]
  if (r.state === 'down') {
    r.v = Math.max(0, r.v - 0.6 * G * dt)
    r.q = Math.min(t.length, r.q + r.v * dt)
    r.k = seek(t, r.q, r.k)
    if (r.t > 1.2 + 1.6 * r.hurt) {
      r.state = r.hurt > HURT ? 'hurt' : 'up'
      r.t = 0
      r.v = 0
    }
  } else if (r.t > 1.4) {
    r.state = 'ride'
    r.v = 0
    r.shaken = true
    r.plan = planFor(p, r)
  }
}

const PASS = 2.5 * UNIT
// The furthest a rider swerves across the line, about two pixels.
const SWERVE = 8
