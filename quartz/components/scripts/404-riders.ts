// Whoever turns up at the bike park on the left-hand hill (404-downhill.ts). Each rider is made up
// on arrival from the day's seed: kit, size, how good they are, how much nerve they have, what they
// throw off a jump, how many runs they came for, and whether they would rather push up than ride
// the lift. On the line they follow a physics trace (gravity down the grade against rolling
// resistance and air, the grip of the dirt and the berms) planned to their own margin, and keep
// their distance from whoever is ahead by Gipps's safe speed. They learn as they go: whoever cased
// the table carries more speed into it next time, whoever slid in a berm trusts the dirt less, and
// whoever keeps getting away with it takes the drop instead of the chicken line. They queue for the
// lift and the gate, friends drop in as a train, and now and then somebody goes down and the next
// rider stops to ask. Whoever goes down hard enough to stay down gets the medic from the hut at the
// bottom, who walks them out, or with a second medic carries them out on a board, while the starter
// holds the gate. The hill's cattle (404-herd.ts) graze between the turns, and riders swerve to the
// far side of the line past any grazing at its edge.

import { type Almanac, type Sky, moonlit, winter } from './404-almanac'
import { type Voices, say } from './404-crew'
import {
  DROP_OFF,
  G,
  LANDING,
  LIFT,
  MAIN,
  PATHS,
  PHOTO,
  type Path,
  SWEET,
  TABLE_FLIGHT,
  TRACKS,
  type Track,
  UNIT,
  along,
  path,
  seek,
  surface,
} from './404-downhill'
import {
  type Herd,
  type HerdEnv,
  type HerdInk,
  type Passer,
  BARN,
  herdEnv,
  makeHerd,
  paintBarn,
  paintBeast,
  paintFarmer,
  paintTrough,
  poseHerd,
  stepHerd,
} from './404-herd'
import { type Grid, type Pt, type RGB, BAYER, col, dot, mulberry, noise, row } from './404-pixel'

// Rolling resistance of knobbly tyres in dirt, and air drag per unit of speed squared for a rider
// and bike of 95 kg with 0.55 m² of frontal area.
const CRR = 0.035
const DRAG = (0.5 * 1.2 * 0.55) / 95 / UNIT
// A bermed turn lends this many g on top of the dirt's own grip.
const BANK = 0.7
// The drag lift's speed along its track, and the seconds between its bars.
const LIFT_V = 2.5 * UNIT
const BARS = 5
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
const SLOT = 9
// Seconds a trick takes in the air, start to finish, easiest first.
const DUR = { nohander: 0.36, whip: 0.42, superman: 0.5 } as const
type Trick = keyof typeof DUR
const TRICKS = Object.keys(DUR) as Trick[]
// The table is the last lip on either line.
const isTable = (t: Track, lip: number) => lip === t.lips[t.lips.length - 1]

type State =
  | 'arrive'
  | 'liftq'
  | 'lift'
  | 'hike'
  | 'link'
  | 'gate'
  | 'ride'
  | 'air'
  | 'down'
  | 'up'
  | 'runout'
  | 'leave'
  // Down for good, waiting for the medic; walking out beside them; out on the board; sat by the
  // hut being seen to.
  | 'hurt'
  | 'limp'
  | 'carried'
  | 'rest'

export type Rider = {
  id: number
  party: number
  kid: boolean
  skill: number
  nerve: number
  tricks: Trick[]
  hikes: boolean
  laps: number
  // Jersey, helmet and frame, as indices into the day's inks.
  kit: [number, number, number]
  // What they have learned: their error on the table's speed as a fraction of it, how much grip
  // they believe the dirt has, and how sure of themselves they are.
  bias: number
  grip: number
  conf: number
  runs: number
  state: State
  // Seconds in this state, and a place in whichever queue they are in.
  t: number
  ticket: number
  // Along a path, or on a line: which line, the sample they are at, distance over the ground and
  // speed along it; in the air, height and the speeds across and up.
  s: number
  line: 0 | 1
  k: number
  q: number
  v: number
  z: number
  vh: number
  vz: number
  lip: number
  launch: number
  plan: Float32Array | null
  trick: Trick | null
  trickT: number
  // A foot put down in a berm, a heavy landing, how hard they last came off, whether they are riding
  // it out shaken, and whom they have already asked after.
  dab: number
  heavy: number
  hurt: number
  shaken: boolean
  fell: boolean
  asked: number
  gone: boolean
  // The way they are walking or being carried out.
  walk: Walk | null
  // How far they have swerved off the middle of the line, across it.
  swerve: number
}

export type ParkEnv = {
  // Whether the lift runs and the starter is at the gate, how many riders the day brings out, how
  // much grip the dirt has, and how much of it the berm below the drop has lost to the day's riding.
  open: boolean
  crowd: number
  grip: number
  loose: number
  herd: HerdEnv
}

export type Park = {
  seed: number
  rng: () => number
  env: ParkEnv
  riders: Rider[]
  ids: number
  parties: number
  tickets: number
  time: number
  arrive: number
  // Seconds the lift has run, which spaces its bars.
  lift: number
  // The gate: seconds since the last rider left it, the countdown under way (or -1), the green
  // light after it, and whose party went last.
  since: number
  count: number
  go: number
  last: number
  // The spectators' cheer, and the photographer's flash.
  cheer: number
  flash: number
  // The two medics, bikes left on the line after a board call, and bikes fetched down that lean
  // on the hut until someone comes for them.
  medics: Medic[]
  wrecks: { line: 0 | 1; q: number; frame: number }[]
  leaning: { frame: number; t: number }[]
  herd: Herd
}

// The day's conditions: the lift runs from nine to seven, solar time, unless the wind or a storm
// stops it; weekends bring twice the riders; drizzle halves them. From the first snow on the ranges
// until it melts the park is shut for the season, and the locals push up on fat bikes and ride the
// packed snow, which holds a studded tyre at about half a g. After hours they come with head
// torches: two or three through the evening until half past ten, one at first light, and through
// the small hours a die-hard, or two under a full moon. Rain and snow shut the line: its steepest pitches run at about 27°, and
// wet dirt holds a tyre at about half a g, less than the tan 27° it would take to stop on them.
// Grip is what knobbly tyres get on the dirt, damp or dry. Everyone off the drop brakes hard into
// the berm below it, and through the day they dig bumps into its entry and push its dirt over the
// top, so by a busy late afternoon it holds a sixth less than when the crew raked it that morning.
export function parkEnv(al: Almanac, sky: Sky, weekend: boolean): ParkEnv {
  const season = !winter(al)
  const shut = sky.storm || sky.fall === 'rain' || sky.fall === 'snow'
  const day = al.hour >= 9 && al.hour < 19
  const open = season && day && !shut && sky.wind < 0.75
  const grip = !season ? 0.55 : sky.fall === 'drizzle' ? 0.65 : sky.fog ? 0.72 : 0.85
  const ridden = clamp(al.hour - 9, 0, 10) * (weekend ? 6 : 3)
  const loose = season ? 0.2 * (1 - Math.exp(-ridden / 24)) : 0
  const awake = al.hour >= 9 && al.hour < 22.5
  const dawn = al.hour >= 6.5 && al.hour < 9
  let crowd = open
    ? weekend
      ? 6
      : 3
    : awake
      ? weekend
        ? 3
        : 2
      : dawn || !moonlit(al, sky)
        ? 1
        : 2
  if (open && sky.fall !== 'none') crowd *= 0.5
  if (sky.temp !== null && (sky.temp < -5 || sky.temp > 32)) crowd *= 0.6
  return {
    open,
    grip,
    loose,
    herd: herdEnv(al, sky),
    crowd: shut ? 0 : Math.max(1, Math.round(crowd)),
  }
}

export function makePark(seed: number, env: ParkEnv): Park {
  return {
    seed,
    rng: mulberry(seed),
    env,
    riders: [],
    ids: 0,
    parties: 0,
    tickets: 0,
    time: 0,
    arrive: 0,
    lift: 0,
    since: 99,
    count: -1,
    go: 0,
    last: -1,
    cheer: 0,
    flash: 0,
    medics: [medic(), medic()],
    wrecks: [],
    leaning: [],
    herd: makeHerd(seed, env.herd),
  }
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

function rider(p: Park, party: number, kid: boolean, skill: number, hikes: boolean, laps: number) {
  const r = p.rng
  const nerve = clamp(r() * 0.7 + skill * 0.4, 0, 1)
  const one: Rider = {
    id: ++p.ids,
    party,
    kid,
    skill,
    nerve,
    // The tricks they have, up the ladder as far as their skill and a little of their nerve go.
    tricks: TRICKS.slice(0, Math.round((skill + 0.4 * nerve) * TRICKS.length)),
    hikes,
    laps,
    kit: [Math.floor(r() * 64), Math.floor(r() * 64), Math.floor(r() * 64)],
    // Nobody knows the table's speed on their first run, least of all the new, and everyone reads
    // the dirt a little wrong, the new more wrong.
    bias: (r() - 0.7) * 0.3 * (1.2 - skill),
    grip: p.env.grip * (1 + (r() - 0.35) * 0.4 * (1 - skill)),
    conf: 0.3 + 0.45 * skill,
    runs: 0,
    state: 'arrive',
    t: 0,
    ticket: 0,
    s: 0,
    line: 1,
    k: 0,
    q: 0,
    v: 0,
    z: 0,
    vh: 0,
    vz: 0,
    lip: -1,
    launch: 0,
    plan: null,
    trick: null,
    trickT: 0,
    dab: 0,
    heavy: 0,
    hurt: 0,
    shaken: false,
    fell: false,
    asked: 0,
    gone: false,
    walk: null,
    swerve: 0,
  }
  return one
}

// A party turns up at the left edge and walks in along the bottom: on its own, a crew of two or
// three friends of about the same level, or a parent with a child who goes first.
function recruit(p: Park, room: number) {
  const r = p.rng
  const kind = room >= 2 && r() < 0.2 ? 'family' : room >= 2 && r() < 0.5 ? 'crew' : 'solo'
  const size = kind === 'family' ? 2 : kind === 'crew' ? Math.min(room, r() < 0.4 ? 3 : 2) : 1
  const party = ++p.parties
  const level = 0.2 + r() * 0.7
  const hikes = !p.env.open || r() < 0.1
  const laps = 2 + Math.floor(r() * 5)
  for (let m = 0; m < size; m++) {
    const kid = kind === 'family' && m === 0
    const skill = clamp(
      kid
        ? 0.15 + r() * 0.35
        : kind === 'crew'
          ? level + (r() - 0.5) * 0.3
          : kind === 'family'
            ? 0.45 + r() * 0.45
            : 0.15 + r() * 0.8,
      0.1,
      0.97,
    )
    const one = rider(p, party, kid, skill, hikes, laps)
    one.s = -m * 12
    p.riders.push(one)
  }
}

const onCourse = (r: Rider) =>
  r.state === 'ride' || r.state === 'air' || r.state === 'down' || r.state === 'up'
const mainQ = (r: Rider) => {
  const t = TRACKS[r.line]
  const k = Math.min(t.n - 2, r.k)
  return t.main[k] + ((r.q - t.q[k]) * (t.main[k + 1] - t.main[k])) / (t.q[k + 1] - t.q[k])
}
const byTicket = (a: Rider, b: Rider) => a.ticket - b.ticket

// The berm below the drop, along the main line, and what the dirt holds at a sample of a line.
const WORN: [number, number] = [255, 310]
const gripAt = (p: Park, t: Track, k: number) =>
  p.env.grip * (t.main[k] > WORN[0] && t.main[k] < WORN[1] ? 1 - p.env.loose : 1)
// Past the limit in a turn, the odds the slide goes from under them, per unit past it.
const LOSE = 10

const brakeOf = (r: Rider) => G * r.grip * (0.6 + 0.25 * r.skill)
// Six watts a kilo out of the gate (four for a child), capped at what the tyres put down.
const pedalOf = (r: Rider) => Math.min(25, (10 * (r.kid ? 4 : 6)) / Math.max(1, r.v / UNIT))

// The speed a rider means to carry at each sample of their line this run: as fast as they dare in
// a straight line, round each turn at their own share of what they think the dirt and the berm
// will hold (with the odd misjudged corner, fewer the better they are), at the speed they have
// settled on for the table, and braking back from each of those as hard as they brake.
function planFor(p: Park, r: Rider) {
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

function release(p: Park, r: Rider) {
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
function ride(p: Park, r: Rider, dt: number, voices: Voices) {
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

function fly(p: Park, r: Rider, dt: number, voices: Voices) {
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
function down(p: Park, r: Rider, dt: number) {
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

// Off to the lift, up the track on foot, or home: home when they have had their runs, when the
// park is thinning out, sometimes after a hard fall, and when the weather shuts it.
function next(p: Park, r: Rider) {
  const done =
    r.runs >= r.laps ||
    p.riders.filter(o => !o.gone && o.state !== 'leave').length > p.env.crowd + 1 ||
    (r.fell && r.hurt > 1 && p.rng() < 0.5) ||
    p.env.crowd === 0
  if (done) {
    r.state = 'leave'
    r.s = PATHS.base.length
  } else if (p.env.open && !r.hikes) {
    r.state = 'liftq'
    r.ticket = ++p.tickets
    r.s = PATHS.base.length
  } else {
    r.state = 'hike'
    r.s = 0
  }
  r.t = 0
}

// The first-aid hut on the grass below the way in, the point on the way in above its door, and the
// trodden path between them.
const HUT: Pt = [138, 846]
const POST = 117
const DOOR: Path = (() => {
  const at = along(PATHS.base, POST)
  return path([
    [HUT[0] + 14, HUT[1]],
    [at.x, at.y],
  ])
})()
// Worse than this and they stay down for the medic; worse than that and they go out on a board.
const HURT = 1.5
const BOARD = 2
// On foot: a jog along the flat, a climb up the line, and down it beside someone hurt or carrying
// them on the board. Riders go by anyone on foot on the line at a walk.
const JOG = 2.5 * UNIT
const CLIMB = 1.3 * UNIT
const LIMP = 0.8 * UNIT
const CARRY = 0.7 * UNIT
const PASS = 2.5 * UNIT
// The furthest a rider swerves across the line, about two pixels.
const SWERVE = 8

// A way on foot: legs along the paths or up and down a line, each from one distance to another at
// its own pace, and how far along the current one the walker has got.
type Leg = { on: Path | Track; from: number; to: number; v: number }
type Walk = { legs: Leg[]; n: number; u: number }
const isTrack = (on: Path | Track): on is Track => 'n' in on

export type Medic = {
  // In the hut, up the line to someone down, knelt by them, walking them down, carrying them out
  // on the board, gone with them to the car park, back from it, up for the bike left on the line,
  // and down with it.
  state: 'post' | 'go' | 'tend' | 'lead' | 'carry' | 'away' | 'back' | 'fetch' | 'wheel'
  t: number
  until: number
  walk: Walk | null
  who: Rider | null
  board: boolean
}
const medic = (): Medic => ({ state: 'post', t: 0, until: 0, walk: null, who: null, board: false })

// On a line at `q`: the point, and which way the line runs there.
// Across a line at a sample, square to the way down it on the screen.
function across(t: Track, k: number): Pt {
  const j = Math.min(t.n - 2, k)
  const dx = t.x[j + 1] - t.x[j]
  const dy = t.y[j + 1] - t.y[j]
  const l = Math.hypot(dx, dy) || 1
  return [-dy / l, dx / l]
}

function onTrack(t: Track, q: number) {
  const k = Math.min(t.n - 2, seek(t, q, 0))
  const u = clamp((q - t.q[k]) / (t.q[k + 1] - t.q[k]), 0, 1)
  return {
    x: t.x[k] + (t.x[k + 1] - t.x[k]) * u,
    y: -surface(t, q, k),
    face: Math.sign(t.x[Math.min(t.n - 1, k + 3)] - t.x[Math.max(0, k - 3)]) || 1,
  }
}

function walkAt(w: Walk) {
  const l = w.legs[w.n]
  const dir = Math.sign(l.to - l.from) || 1
  const s = l.from + dir * w.u
  if (isTrack(l.on)) {
    const at = onTrack(l.on, s)
    return { x: at.x, y: at.y, face: at.face * dir }
  }
  const at = along(l.on, s)
  return { x: at.x, y: at.y, face: at.dir * dir }
}

// Where along the main line someone on foot on a line is, or null when they are off it.
function footQ(w: Walk | null) {
  const l = w?.legs[w.n]
  if (!w || !l || !isTrack(l.on)) return null
  const t = l.on
  const q = l.from + (Math.sign(l.to - l.from) || 1) * w.u
  const k = Math.min(t.n - 2, seek(t, q, 0))
  return t.main[k] + ((q - t.q[k]) * (t.main[k + 1] - t.main[k])) / (t.q[k + 1] - t.q[k])
}

// On along a walk; true once at its end.
function stepWalk(w: Walk, dt: number) {
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
const downFrom = (line: 0 | 1, q: number, v: number, hut: boolean): Walk => {
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
const OUT = (v: number): Walk => ({
  legs: [
    { on: DOOR, from: 0, to: DOOR.length, v },
    { on: PATHS.base, from: POST, to: -30, v },
  ],
  n: 0,
  u: 0,
})

// The line stays shut while anyone hurt is still on it.
const held = (p: Park) =>
  p.riders.some(
    o =>
      o.state === 'hurt' ||
      ((o.state === 'limp' || o.state === 'carried') && footQ(o.walk) !== null),
  )

// Someone down for good gets the first medic out of the hut; on a board call a second follows with
// the board. They walk the hurt down to sit by the hut, or with the board carry them out to the car
// park, and later one goes back up for the bike. Out of hours there is nobody in the hut, and the
// hurt get up in their own time and walk themselves down and home.
function stepMedics(p: Park, dt: number, voices: Voices) {
  // A call already answered gets its board whatever the hour.
  for (const r of p.riders) {
    if (r.state !== 'hurt') continue
    const on = p.medics.filter(m => m.who === r)
    const second = r.hurt > BOARD && on.length === 1 && on[0].state === 'tend'
    const free = p.medics.find(m => m.state === 'post')
    if (!free || (on.length ? !second : !p.env.open)) continue
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
  for (const b of p.leaning) b.t += dt
  p.leaning = p.leaning.filter(b => b.t < 900).slice(-2)
}

// Whom the herd takes notice of: bikes going by fast, and anyone lying hurt.
function passers(p: Park): Passer[] {
  const out: Passer[] = []
  for (const r of p.riders) {
    const fast = (r.state === 'ride' || r.state === 'air') && r.v > 3 * UNIT
    const hurt = r.state === 'hurt'
    if (!fast && !hurt) continue
    const at = place(r)
    out.push({ x: at.x, y: at.ground, fast, hurt })
  }
  return out
}

const toward = (from: number, to: number, step: number) =>
  from < to ? Math.min(to, from + step) : Math.max(to, from - step)

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

  const live = p.riders.filter(r => r.state !== 'leave').length
  if (live < p.env.crowd && (p.arrive -= dt) <= 0) {
    recruit(p, p.env.crowd - live)
    p.arrive = 6 + p.rng() * 14
  }

  const liftq = p.riders.filter(r => r.state === 'liftq').sort(byTicket)
  const gateq = p.riders.filter(r => r.state === 'link' || r.state === 'gate').sort(byTicket)
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
  p.riders = p.riders.filter(r => !r.gone)
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

// Without motion, one moment of a busy afternoon: somebody at the top of their flight over the
// table with the back end thrown out and the photographer's flash going, someone in the gate on the
// second beep with a friend behind, and someone halfway up the lift; on some days someone sat by the
// first-aid hut with an arm in a sling. After hours, locals up the lift track with their head
// torches.
export function posePark(p: Park) {
  // Each resize or change of light poses it again, and the same riders come back in the same kit.
  p.rng = mulberry(p.seed)
  p.ids = 0
  p.parties = 0
  p.riders = []
  p.medics = [medic(), medic()]
  p.wrecks = []
  p.leaning = []
  poseHerd(p.herd, p.env.herd)
  if (!p.env.crowd) return
  if (!p.env.open) {
    for (let n = 0; n < Math.min(2, p.env.crowd); n++) {
      const one = rider(p, ++p.parties, false, 0.8 - n * 0.2, true, 3)
      one.state = 'hike'
      one.s = PATHS.lift.length * (0.45 + n * 0.35)
      p.riders.push(one)
    }
    return
  }
  const air = rider(p, ++p.parties, false, 0.9, false, 4)
  const lip = MAIN.lips[MAIN.lips.length - 1]
  const a = Math.atan(MAIN.grade[lip])
  const half = TABLE_FLIGHT.time * 0.45
  const vz = TABLE_FLIGHT.v * Math.sin(a)
  Object.assign(air, {
    state: 'air',
    line: 0,
    trick: 'whip',
    trickT: DUR.whip / 2,
    q: MAIN.q[lip] + TABLE_FLIGHT.v * Math.cos(a) * half,
    z: MAIN.z[lip] + vz * half - (G * half * half) / 2,
    v: TABLE_FLIGHT.v,
  })
  air.k = seek(MAIN, air.q, lip)
  const party = ++p.parties
  const gate = rider(p, party, false, 0.6, false, 4)
  const friend = rider(p, party, false, 0.55, false, 4)
  const up = rider(p, ++p.parties, false, 0.4, false, 4)
  Object.assign(gate, { state: 'gate', s: PATHS.link.length, ticket: 1 })
  Object.assign(friend, { state: 'gate', s: PATHS.link.length - SLOT, ticket: 2 })
  Object.assign(up, { state: 'lift', s: PATHS.lift.length * 0.55 })
  p.riders.push(air, gate, friend, up)
  if (p.rng() < 0.4) {
    const sore = rider(p, ++p.parties, false, 0.5, false, 4)
    Object.assign(sore, { state: 'rest', hurt: 1.8, fell: true })
    p.riders.push(sore)
  }
  p.count = 0.7
  p.flash = 1
  p.cheer = 1
}

export type RiderInk = {
  skin: RGB
  shorts: RGB
  tyre: RGB
  dust: RGB
  lamp: RGB
  glow: RGB
  steel: RGB
  flash: RGB
  // The gate's beacon: counting down (and held, lit steady), then go.
  beacon: [RGB, RGB]
  jerseys: RGB[]
  helmets: RGB[]
  frames: RGB[]
  // The medics' jerseys, their kit bag and board, and the hut's walls and cross.
  patrol: RGB
  kit: RGB
  hut: RGB
  cross: RGB
  herd: HerdInk
}

// Where on the screen a rider is and which way they face, and, in the air, the ground under them.
function place(r: Rider): { x: number; y: number; face: number; ground: number } {
  const { base, lift, link, runout } = PATHS
  const on = (s: number, p: typeof base) => {
    const at = along(p, s)
    return { x: at.x, y: at.y, face: at.dir, ground: at.y }
  }
  switch (r.state) {
    case 'arrive':
    case 'liftq':
      return { ...on(r.s, base), face: 1 }
    case 'leave':
      return { ...on(r.s, base), face: -1 }
    case 'lift':
      return on(r.s, lift)
    // Beside the lift's track, out of the way of its bars.
    case 'hike': {
      const at = on(r.s, lift)
      return { ...at, x: at.x - 8 }
    }
    case 'link':
    case 'gate':
      return on(r.s, link)
    case 'runout':
      return on(r.s, runout)
    case 'limp':
    case 'carried': {
      const at = walkAt(r.walk!)
      return { ...at, ground: at.y }
    }
    // On the grass beside the hut's door.
    case 'rest':
      return { x: HUT[0] + 24, y: HUT[1], face: -1, ground: HUT[1] }
    default: {
      const t = TRACKS[r.line]
      const at = onTrack(t, r.q)
      const [nx, ny] = across(t, r.k)
      const s = r.state === 'ride' ? r.swerve : 0
      const y = at.y + ny * s
      return { x: at.x + nx * s, y: r.state === 'air' ? -r.z : y, face: at.face, ground: y }
    }
  }
}

// The riders, the lift's bars, the gate's beacon and the photographer, over the ranges. Each rider
// is a pixel or three wide: wheels and frame, then shorts, jersey and helmet; a child a pixel
// shorter. Tricks move those pixels about: the back wheel kicked up and out for a whip, both arms
// out for a no-hander, the body stretched out flat behind the bars for a superman.
export function paintPark(
  d: Uint8ClampedArray,
  g: Grid,
  ink: RiderInk,
  p: Park,
  dark: boolean,
  heads: Voices['heads'],
) {
  for (const who of Object.keys(heads)) if (who.startsWith('rider')) delete heads[who]
  const beat = Math.floor(p.time * 8)
  const plot = (i: number, j: number, c: RGB, alpha: number) => {
    if (alpha > BAYER[(j & 3) * 4 + (i & 3)]) dot(d, g, i, j, c)
  }
  const lamp = (i: number, j: number) => {
    for (let dj = -2; dj <= 2; dj++)
      for (let di = -2; di <= 2; di++) {
        const r = Math.hypot(di, dj)
        if (r > 0) plot(i + di, j + dj, ink.glow, (1 - r / 2.8) * 0.6)
      }
    dot(d, g, i, j, ink.lamp)
  }

  // The lift's bars, climbing the thread while it runs.
  const { lift } = PATHS
  for (let n = Math.floor((p.lift * LIFT_V - lift.length) / (LIFT_V * BARS)); ; n++) {
    const s = p.lift * LIFT_V - n * LIFT_V * BARS
    if (s < 0) break
    if (s > lift.length) continue
    const at = along(lift, s)
    dot(d, g, col(g, at.x) + 1, row(g, at.y) - 1, ink.steel)
  }

  // The beacon on the start hut's roof: three beeps, then green as the rider goes.
  const si = col(g, MAIN.x[0])
  const sj = row(g, MAIN.y[0])
  if (p.go > 0) dot(d, g, si, sj - 6, ink.beacon[1])
  else if (p.count >= 0 && p.count % 0.6 < 0.3) dot(d, g, si, sj - 6, ink.beacon[0])
  else if (p.env.open && held(p)) dot(d, g, si, sj - 6, ink.beacon[0])

  // The photographer, crouched by the table's landing through opening hours, camera up.
  if (p.env.open) {
    const pi = col(g, PHOTO[0])
    const pj = row(g, PHOTO[1])
    dot(d, g, pi, pj - 1, ink.shorts)
    dot(d, g, pi, pj - 2, ink.jerseys[0])
    dot(d, g, pi, pj - 3, ink.skin)
    dot(d, g, pi - 1, pj - 3, p.flash > 0 ? ink.flash : ink.tyre)
    if (p.flash > 0) {
      for (const [di, dj] of [
        [-2, -3],
        [-1, -4],
        [-1, -2],
      ])
        dot(d, g, pi + di, pj + dj, ink.flash)
      if (dark) lamp(pi - 1, pj - 3)
    }
  }

  // The first-aid hut: white under a tin roof with a red cross on its side, a lamp over the door
  // when it is dark, and bikes fetched down off the line against its far wall.
  const hi = col(g, HUT[0])
  const hj = row(g, HUT[1])
  for (let dj = 1; dj <= 4; dj++)
    for (let di = -2; di <= 2; di++) {
      const cross = (di === 0 && dj >= 2) || (dj === 3 && Math.abs(di) === 1)
      dot(d, g, hi + di, hj - dj, cross ? ink.cross : ink.hut)
    }
  for (let di = -3; di <= 3; di++) dot(d, g, hi + di, hj - 5, ink.steel)
  if (dark && p.env.open) lamp(hi + 3, hj - 4)
  // The barn, with the yard light over its doors through the dark, and the farmer at it with a
  // lantern.
  paintBarn(d, g, ink.herd, p.herd)
  if (dark) lamp(col(g, BARN[0]), row(g, BARN[1]) - 4)
  paintFarmer(d, g, ink.herd, p.herd, p.time)
  paintTrough(d, g, ink.herd, p.herd)
  const farmer = p.herd.farmer
  if (dark && farmer.chore !== 'home' && farmer.chore !== 'make')
    lamp(col(g, farmer.x) + farmer.face, row(g, farmer.y) - 2)
  p.leaning.forEach((b, n) => {
    dot(d, g, hi - 3 - n, hj - 1, ink.tyre)
    dot(d, g, hi - 3 - n, hj - 2, ink.frames[b.frame % ink.frames.length])
    dot(d, g, hi - 3 - n, hj - 3, ink.tyre)
  })
  // Bikes left lying on the line after a board call. The first stays on the list until it reaches
  // the hut, so skip it while a medic wheels it down.
  const wheeling = p.medics.some(m => m.state === 'wheel')
  for (const w of p.wrecks) {
    if (wheeling && w === p.wrecks[0]) continue
    const at = onTrack(TRACKS[w.line], w.q)
    const [wi, wj] = [col(g, at.x), row(g, at.y)]
    dot(d, g, wi - 1, wj - 1, ink.tyre)
    dot(d, g, wi, wj - 1, ink.frames[w.frame % ink.frames.length])
    dot(d, g, wi + 1, wj - 1, ink.tyre)
  }

  // A medic: shorts, the patrol jersey with the kit bag on its back, and a dark helmet; knelt, the
  // jersey sits on the knee and a hand reaches out.
  const standing = (i: number, j: number, f: number, walking: boolean) => {
    if (walking) {
      dot(d, g, i - 1, j - 1, ink.shorts)
      dot(d, g, i + 1, j - 1, ink.shorts)
    } else dot(d, g, i, j - 1, ink.shorts)
    dot(d, g, i, j - 2, ink.patrol)
    dot(d, g, i, j - 3, ink.tyre)
    dot(d, g, i - f, j - 2, ink.kit)
  }
  const kneeling = (i: number, j: number, f: number) => {
    dot(d, g, i, j - 1, ink.shorts)
    dot(d, g, i, j - 2, ink.patrol)
    dot(d, g, i, j - 3, ink.tyre)
    dot(d, g, i + f, j - 2, ink.skin)
  }
  const medicHead = (m: Medic, at: Pt) => {
    heads[`rider-m${p.medics.indexOf(m)}`] = at
  }

  // The herd and the riders, nearer over farther.
  const herd = p.herd.beasts
  const beasts = [...herd].sort((a, b) => a.y - b.y)
  let nb = 0
  const order = p.riders.map(r => ({ r, at: place(r) })).sort((a, b) => a.at.ground - b.at.ground)
  for (const { r, at } of order) {
    for (; nb < beasts.length && beasts[nb].y <= at.ground; nb++)
      paintBeast(d, g, ink.herd, beasts[nb], p.time, herd.indexOf(beasts[nb]))
    const jersey = ink.jerseys[r.kit[0] % ink.jerseys.length]
    const helmet = ink.helmets[r.kit[1] % ink.helmets.length]
    const frame = ink.frames[r.kit[2] % ink.frames.length]
    const f = at.face
    const i = col(g, at.x)
    const j = row(g, at.y)
    const tall = r.kid ? 3 : 4
    let head: [number, number] = [i, j - tall]
    const moving = !['gate', 'liftq', 'down', 'hurt', 'carried', 'rest'].includes(r.state)
    switch (r.state) {
      case 'ride':
      case 'air':
      case 'runout':
      case 'leave': {
        // On the bike, weight back, bars forward. A foot down in a slide; a pixel of compression
        // off a heavy landing; the odd root jolting them off the saddle.
        const sink = r.heavy > 0 ? 1 : 0
        const jolt =
          r.state === 'ride' && !sink && Math.sin(r.q * 0.37 + r.id) > 0.93 && r.v > 40 ? 1 : 0
        const y0 = j - 1
        const b = y0 - 1 + sink - jolt
        const whip = r.trick === 'whip'
        dot(d, g, i - f, whip ? y0 - 1 : y0, ink.tyre)
        dot(d, g, i, y0, frame)
        dot(d, g, i + f, y0, ink.tyre)
        if (r.trick === 'superman') {
          // Stretched out flat behind the bars.
          dot(d, g, i + f, b, ink.tyre)
          dot(d, g, i, b, jersey)
          dot(d, g, i - f, b, ink.shorts)
          dot(d, g, i - 2 * f, b, ink.shorts)
          dot(d, g, i + f, b - 1, helmet)
          head = [i + f, b - 1]
        } else {
          if (r.trick !== 'nohander') dot(d, g, i + f, b, ink.tyre)
          if (r.kid) {
            dot(d, g, i, b, jersey)
            dot(d, g, i - f, b - 1, helmet)
            head = [i - f, b - 1]
          } else {
            dot(d, g, i, b, ink.shorts)
            dot(d, g, i - f, b - 1, jersey)
            dot(d, g, i, b - 1, jersey)
            dot(d, g, i - f, b - 2, helmet)
            head = [i - f, b - 2]
          }
          if (r.trick === 'nohander') {
            dot(d, g, head[0] - f, head[1] + 1, ink.skin)
            dot(d, g, head[0] + 2 * f, head[1] + 1, ink.skin)
          }
        }
        if (whip) dot(d, g, i - 2 * f, y0 - 1, frame)
        if (r.dab > 0) dot(d, g, i + f * (beat & 1 ? 1 : 0), y0 + 1, ink.shorts)
        if (r.state === 'air') dot(d, g, i, row(g, at.ground), ink.dust)
        else if (r.state === 'ride') {
          const t = TRACKS[r.line]
          // Roost off the back wheel out of a berm, dust on the straights, a puff off a landing.
          if (t.bank[r.k] > 0.5 && r.v > 50) {
            dot(d, g, i - 2 * f, y0 - (beat & 1), ink.dust)
            dot(d, g, i - 2 * f, y0 - 1 + Math.sign(t.bend[r.k]) * f, ink.dust)
            if (beat & 1) dot(d, g, i - 3 * f, y0 - 1, ink.dust)
          } else if (r.v > 40 && beat & 1) dot(d, g, i - 2 * f, y0, ink.dust)
          if (sink) {
            dot(d, g, i - 2, y0 + 1, ink.dust)
            dot(d, g, i + 2, y0 + 1, ink.dust)
          }
        }
        break
      }
      case 'down':
      case 'hurt': {
        // The bike on its side where it stopped, the rider tumbling past it and then lying there.
        dot(d, g, i - 1, j - 1, ink.tyre)
        dot(d, g, i, j - 1, frame)
        dot(d, g, i + 1, j - 1, ink.tyre)
        const r0 = i + 2 * f
        const fresh = r.state === 'down'
        if (fresh && r.t < 0.35 && beat & 1) {
          dot(d, g, r0, j - 1, helmet)
          dot(d, g, r0, j - 2, jersey)
          dot(d, g, r0, j - 3, ink.shorts)
          head = [r0, j - 3]
        } else {
          dot(d, g, r0, j - 1, ink.shorts)
          dot(d, g, r0 + f, j - 1, jersey)
          dot(d, g, r0 + 2 * f, j - 1, helmet)
          head = [r0 + 2 * f, j - 2]
        }
        if (fresh && r.t < 0.6) {
          plot(i - f, j - 2, ink.dust, 0.9)
          plot(i + 3 * f, j - 3, ink.dust, 0.6)
          plot(i, j - 3, ink.dust, 0.7)
        }
        // A medic knelt at their head with the bag beside; on a board call the second knelt across
        // the bike with the board slid under them.
        for (const m of p.medics) {
          if (m.who !== r || m.state !== 'tend') continue
          if (m.board) {
            kneeling(i - 2 * f, j, f)
            for (let di = 2; di <= 4; di++) dot(d, g, i + di * f, j, ink.kit)
            medicHead(m, [i - 2 * f, j - 3])
          } else {
            kneeling(i + 5 * f, j, -f)
            dot(d, g, i + 6 * f, j - 1, ink.kit)
            medicHead(m, [i + 5 * f, j - 3])
          }
        }
        break
      }
      case 'carried': {
        // Feet first down the hill on the board, a medic at either end.
        for (let di = -2; di <= 2; di++) dot(d, g, i + di, j - 2, ink.kit)
        dot(d, g, i + f, j - 3, ink.shorts)
        dot(d, g, i, j - 3, jersey)
        dot(d, g, i - f, j - 3, helmet)
        head = [i - f, j - 3]
        const step = Math.floor(p.time * 2.2) % 2
        p.medics
          .filter(m => m.state === 'carry')
          .forEach((m, n) => {
            const e = n ? -1 : 1
            const at: Pt = [i + 3 * e * f, j]
            if ((step + n) % 2) {
              dot(d, g, at[0] - 1, j - 1, ink.shorts)
              dot(d, g, at[0] + 1, j - 1, ink.shorts)
            } else dot(d, g, at[0], j - 1, ink.shorts)
            dot(d, g, at[0], j - 2, ink.patrol)
            dot(d, g, at[0], j - 3, ink.tyre)
            if (dark) lamp(at[0], j - 3)
            medicHead(m, [at[0], j - 3])
          })
        break
      }
      case 'rest': {
        // Sat on the grass by the hut with the helmet off beside them, an arm in a white sling, and
        // the bike laid down.
        dot(d, g, i, j - 1, ink.shorts)
        dot(d, g, i + 1, j - 1, ink.shorts)
        dot(d, g, i, j - 2, jersey)
        dot(d, g, i + 1, j - 2, ink.hut)
        dot(d, g, i, j - 3, ink.skin)
        dot(d, g, i - 1, j - 1, helmet)
        dot(d, g, i + 3, j - 1, ink.tyre)
        dot(d, g, i + 4, j - 1, frame)
        dot(d, g, i + 5, j - 1, ink.tyre)
        head = [i, j - 3]
        break
      }
      case 'lift': {
        // Sat on the bike, one hand up on the bar.
        dot(d, g, i - 1, j - 1, ink.tyre)
        dot(d, g, i, j - 1, frame)
        dot(d, g, i + 1, j - 1, ink.tyre)
        dot(d, g, i, j - 2, ink.shorts)
        dot(d, g, i, j - 3, jersey)
        dot(d, g, i, j - 4, helmet)
        dot(d, g, i + 1, j - 4, ink.skin)
        dot(d, g, i + 1, j - 5, ink.steel)
        head = [i, j - 4]
        break
      }
      case 'gate': {
        // Astride the bike, waiting their turn.
        dot(d, g, i - 1, j - 1, ink.tyre)
        dot(d, g, i, j - 1, frame)
        dot(d, g, i + 1, j - 1, ink.tyre)
        if (!r.kid) dot(d, g, i, j - 2, ink.shorts)
        dot(d, g, i, j - tall + 1, jersey)
        dot(d, g, i, j - tall, helmet)
        break
      }
      default: {
        // On foot beside the bike, a hand on the bars, walking or standing: arriving, queueing,
        // pushing up the lift track, or getting up after a fall.
        const walking =
          (r.state === 'arrive' || r.state === 'hike' || r.state === 'limp') &&
          Math.floor(p.time * (r.state === 'limp' ? 1.5 : 2.5) + r.id) % 2
        if (walking) {
          dot(d, g, i - 1, j - 1, ink.shorts)
          dot(d, g, i + 1, j - 1, ink.shorts)
        } else dot(d, g, i, j - 1, ink.shorts)
        dot(d, g, i, j - 2, r.kid ? helmet : jersey)
        if (!r.kid) dot(d, g, i, j - 3, helmet)
        dot(d, g, i + f, j - 2, ink.skin)
        dot(d, g, i + f, j - 1, ink.tyre)
        dot(d, g, i + 2 * f, j - 1, frame)
        dot(d, g, i + 3 * f, j - 1, ink.tyre)
        dot(d, g, i + 2 * f, j - 2, ink.tyre)
        head = [i, j - (r.kid ? 2 : 3)]
      }
    }
    if (dark && moving) lamp(head[0], head[1])
    heads[`rider${r.id}`] = head
  }
  for (; nb < beasts.length; nb++)
    paintBeast(d, g, ink.herd, beasts[nb], p.time, herd.indexOf(beasts[nb]))

  // The medics on their feet: one at the hut's door through opening hours, and whoever is out on a
  // call, the second up the line with the board on their back, the last down it wheeling the bike.
  const door = p.medics.find(m => m.state === 'post')
  for (const m of p.medics) {
    const out = m.walk && ['go', 'lead', 'back', 'fetch', 'wheel'].includes(m.state)
    if (!out && !(m === door && p.env.open)) continue
    const at = out ? walkAt(m.walk!) : { x: HUT[0] + 14, y: HUT[1], face: 1 }
    const i = col(g, at.x)
    const j = row(g, at.y)
    const f = at.face
    standing(i, j, f, !!out && Math.floor(p.time * 2.5 + p.medics.indexOf(m)) % 2 === 1)
    if (m.state === 'go' && m.board)
      for (let dj = 1; dj <= 3; dj++) dot(d, g, i - f, j - dj, ink.kit)
    const wreck = p.wrecks[0]
    if (m.state === 'wheel' && wreck) {
      dot(d, g, i + f, j - 2, ink.skin)
      dot(d, g, i + f, j - 1, ink.tyre)
      dot(d, g, i + 2 * f, j - 1, ink.frames[wreck.frame % ink.frames.length])
      dot(d, g, i + 3 * f, j - 1, ink.tyre)
      dot(d, g, i + 2 * f, j - 2, ink.tyre)
    }
    if (dark && out) lamp(i, j - 3)
    medicHead(m, [i, j - 3])
  }
}

// Seeds the day: the same riders turn up for everyone who looks on a given day, from the same
// arrivals onward.
export function daySeed(now: Date, lon: number) {
  return Math.floor((now.getTime() / 3.6e6 + lon / 15) / 24)
}
