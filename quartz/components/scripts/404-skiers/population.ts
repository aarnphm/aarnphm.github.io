import { say, type Voices } from '../404-crew'
import {
  BOOM,
  clamp,
  free,
  GAP,
  GATES,
  lane,
  LIFT_FOOT,
  M,
  mark,
  QUEUE,
  RUNS,
  seek,
  SHED,
  spot,
  THROW,
  WAY,
  ZONES,
  type Hill,
  type Mode,
  type Role,
  type Skier,
} from './hill-state'

const WALK = 1.1

export function skier(
  h: Hill,
  party: number,
  role: Role,
  board: boolean,
  kid: boolean,
  skill: number,
) {
  const r = h.rng
  const nerve = clamp(r() * 0.7 + skill * 0.4, 0, 1)
  // The speed they would like to hold, and the most they will ever let it run to.
  const top = kid ? 6 : 4 + 14 * skill
  const conf = 0.3 + 0.45 * skill
  const mode: Mode =
    role === 'racer'
      ? 'race'
      : role === 'pupil' || role === 'teach'
        ? 'plough'
        : board
          ? skill < 0.3
            ? 'leaf'
            : skill < 0.7
              ? 'skid'
              : 'carve'
          : skill < 0.3 || (kid && skill < 0.4)
            ? 'plough'
            : skill < 0.65
              ? 'skid'
              : 'carve'
  const one: Skier = {
    id: ++h.ids,
    party,
    role,
    board,
    kid,
    skill,
    nerve,
    kit: [Math.floor(r() * 64), Math.floor(r() * 64), Math.floor(r() * 64)],
    mode,
    pace:
      role === 'racer' ? 16 : mode === 'plough' ? (kid ? 3.2 : 4.5) : top * (0.65 + 0.25 * conf),
    conf,
    runs: 0,
    laps: 3 + Math.floor(r() * 6),
    state: 'arrive',
    t: 0,
    ticket: 0,
    x: WAY[0][0],
    y: WAY[0][1],
    path: [WAY[1]],
    run: 0,
    k: 0,
    s: 0,
    w: 0,
    psi: 0,
    v: 0,
    dir: 1,
    amp: 0.8,
    skid: 0,
    chair: 0,
    seat: 0,
    gear: -1,
    gearW: 0,
    hurt: 0,
    stop: 0,
    asked: 0,
    cell: -1,
    lead: null,
    slot: 0,
    trail: new Float32Array(0),
    patient: null,
    shot: -1,
    gone: false,
  }
  return one
}

// Whoever turns up next, walking in over the crest of the mid ridge: on their own, a crew of
// friends of about the same level on skis and boards, a parent with a child or two, and in the
// mornings a ski-school class of small children behind their instructor.
export function recruit(h: Hill, room: number) {
  const r = h.rng
  const party = ++h.parties
  const add = (one: Skier, m: number) => {
    one.x -= m * 3
    one.y += m * 4
    h.skiers.push(one)
    return one
  }
  if (h.env.school && room >= 4 && r() < 0.3) {
    const teach = add(skier(h, party, 'teach', false, false, 0.95), 0)
    teach.trail = new Float32Array(160)
    const n = 2 + Math.floor(r() * 2)
    for (let m = 1; m <= n; m++) {
      const kid = add(skier(h, party, 'pupil', false, true, 0.1 + r() * 0.2), m)
      kid.lead = teach
      kid.slot = m
      kid.laps = teach.laps
    }
    return
  }
  const kind = room >= 3 && r() < 0.2 ? 'family' : room >= 2 && r() < 0.45 ? 'crew' : 'solo'
  const size =
    kind === 'family' ? 2 + (r() < 0.4 ? 1 : 0) : kind === 'crew' ? 2 + (r() < 0.4 ? 1 : 0) : 1
  const level = 0.2 + r() * 0.75
  const laps = 3 + Math.floor(r() * 6)
  for (let m = 0; m < Math.min(size, room); m++) {
    const kid = kind === 'family' && m > 0
    const skill = clamp(
      kid
        ? 0.15 + r() * 0.5
        : kind === 'crew'
          ? level + (r() - 0.5) * 0.3
          : kind === 'family'
            ? 0.4 + r() * 0.5
            : 0.1 + r() * 0.87,
      0.05,
      0.97,
    )
    const one = add(skier(h, party, 'free', r() < (kid ? 0.2 : 0.35), kid, skill), m)
    one.laps = laps
  }
}

// The summer weekday race camp, four racers and their coach, already out beside the fifth gate.
export function camp(h: Hill) {
  const party = ++h.parties
  const coach = skier(h, party, 'coach', false, false, 0.95)
  const k = seek(RUNS[0], GATES[4].s, 0)
  ;[coach.x, coach.y] = spot(RUNS[0], GATES[4].s, -RUNS[0].half[k] + 1)
  coach.state = 'coach'
  h.skiers.push(coach)
  for (let m = 1; m <= 4; m++) {
    const one = skier(h, party, 'racer', false, h.rng() < 0.3, 0.82 + h.rng() * 0.15)
    one.x -= m * 3
    one.y += m * 4
    one.laps = 99
    h.skiers.push(one)
  }
}

// After hours, a local or two over the crest with skins on their skis or a splitboard, to climb the
// edge of the blue and ride it down once or twice.
export function tourers(h: Hill) {
  const party = ++h.parties
  const n = h.rng() < 0.35 ? 2 : 1
  const laps = h.rng() < 0.4 ? 2 : 1
  const run = RUNS[0]
  for (let m = 0; m < n; m++) {
    const one = skier(h, party, 'tour', h.rng() < 0.4, false, 0.55 + h.rng() * 0.4)
    one.laps = laps
    one.s = run.length - 3 - 4 * m
    one.path = [WAY[1], spot(run, run.length - 3 - 4 * m, lane(run, run.n - 1))]
    one.x -= m * 3
    one.y += m * 4
    h.skiers.push(one)
  }
}

export function skin(o: Skier) {
  const run = RUNS[0]
  o.state = 'skin'
  o.run = 0
  o.s = Math.min(o.s || run.length - 3, run.length - 3)
  o.k = seek(run, o.s, 0)
  o.w = lane(run, o.k)
  o.t = 0
}

// The patrol: called by whoever stopped with a casualty, or after half a minute by someone who saw it
// from the chair, a patroller skis down from the hut by the head station with a toboggan behind,
// spends a while with them, and takes them down to the shed at the foot in the toboggan.
// A powder morning's round: at first light a patroller up the chair with a pack of hand charges,
// before anyone else; looked in on later, somewhere along the crest already.
export function control(h: Hill) {
  const p = skier(h, ++h.parties, 'patrol', false, false, 0.95)
  if (h.time > 1)
    Object.assign(p, { state: 'chair', chair: Math.floor(h.lift / GAP) * GAP, seat: 0, shot: 0 })
  else {
    const shot = Math.floor(h.rng() * ZONES.length)
    const t = h.rng() * (BOOM + 15)
    const [x, y] = t < THROW ? ZONES[shot].stand : ZONES[shot].hide
    Object.assign(p, { state: 'charge', shot, t, x, y })
  }
  h.skiers.push(p)
}
// The round done, or the chair open: back to the top of the blue and down it, first tracks.
export const descend = (o: Skier) => {
  o.state = 'top'
  o.run = 0
  o.path = [spot(RUNS[0], 0, 0)]
  o.t = 0
}

export function patrol(h: Hill, o: Skier) {
  const p = skier(h, ++h.parties, 'patrol', false, false, 0.95)
  Object.assign(p, { state: 'rescue', patient: o, run: o.run, s: 0, k: 0, w: 0, v: 0 })
  h.skiers.push(p)
}

export function rescue(h: Hill, o: Skier, dt: number, voices: Voices) {
  const c = o.patient
  const run = RUNS[o.run]
  o.k = seek(run, o.s, o.k)
  if (!c) return
  if (o.state === 'rescue') {
    // Straight down to them, as fast as the empty toboggan behind lets them: about 20 km/h.
    o.v = Math.min(5.5, o.v + 2 * dt)
    o.s += o.v * dt
    o.w += clamp(c.w - Math.sign(c.w || 1) * 1.5 - o.w, -1.5 * dt, 1.5 * dt)
    mark(h, o, run)
    if (o.s >= c.s - 3) {
      o.state = 'tend'
      o.t = 0
      o.v = 0
      say(voices, `ski${o.id}`, '?')
      // Whoever stayed with them goes on.
      for (const p of h.skiers) if (Math.abs(p.asked) === c.id) p.stop = Math.min(p.stop, 2)
    }
  } else if (o.state === 'tend') {
    if (o.t > 45 + 30 * (c.hurt - 1)) {
      // Into the handles, below the toboggan.
      o.state = 'sled'
      c.state = 'sled'
      o.s = c.s + 2.5
      o.w = c.w
      o.t = 0
    }
  } else if (o.state === 'sled') {
    o.v = Math.min(3, o.v + dt)
    o.s = Math.min(run.length - 4, o.s + o.v * dt)
    c.s = o.s - 2.5
    c.w = o.w
    mark(h, o, run)
    if (o.s >= run.length - 4) {
      o.state = 'haul'
      c.state = 'haul'
      ;[o.x, o.y] = spot(run, o.s, o.w)
      o.path = [[1322, 452], SHED]
    }
  } else if (o.state === 'haul') {
    const ahead = o.path[0]?.[0] ?? o.x
    const there = walk(o, dt * 0.6)
    // The toboggan trails them.
    c.x = o.x + (ahead < o.x ? 3 : -3)
    c.y = o.y
    if (there) {
      c.gone = true
      o.patient = null
      // Back up on the chair to the hut, or into the shed when it has stopped.
      if (h.env.open) {
        o.state = 'walk'
        o.path = [QUEUE(h.skiers.filter(p => p.state === 'queue').length)]
      } else o.gone = true
    }
  }
}

export const walk = (o: Skier, dt: number) => {
  const to = o.path[0]
  if (!to) return true
  const dx = to[0] - o.x
  const dy = to[1] - o.y
  const d = Math.hypot(dx, dy)
  const step = WALK * M * dt
  if (d <= step) {
    o.x = to[0]
    o.y = to[1]
    o.path.shift()
    return !o.path.length
  }
  o.x += (dx / d) * step
  o.y += (dy / d) * step
  return false
}

// At the bottom, a stop across the hill in a spray of snow; then to the queue again, or home.
export function finish(h: Hill, o: Skier) {
  o.runs++
  if (o.hurt < 0) o.conf = clamp(o.conf + 0.08, 0.05, 1)
  if (o.role === 'racer' && o.hurt < 0) h.cheer = 1.5
  const top = o.kid ? 6 : 4 + 14 * o.skill
  if (o.mode !== 'plough' && o.role === 'free') o.pace += 0.06 * (top - o.pace)
  // A board beginner who has got down twice without falling tries linking turns.
  if (o.mode === 'leaf' && o.runs >= 2 && o.conf > 0.45) {
    o.mode = 'skid'
    o.skill = Math.max(o.skill, 0.32)
    o.pace = 4
  }
  const busy = h.skiers.filter(free).length
  const [x, y] = spot(RUNS[o.run], RUNS[o.run].length - 3, o.w)
  o.x = x
  o.y = y
  // The patrol rides straight back up while the zones still want shooting, and goes in once the
  // chair opens.
  if (o.shot >= 0) {
    o.state = h.env.control ? 'walk' : 'leave'
    o.path = h.env.control ? [LIFT_FOOT] : [[1322, 452], SHED]
    o.t = 0
    return
  }
  const home =
    o.role === 'racer'
      ? !h.env.camp
      : o.runs >= o.laps ||
        (o.role === 'tour' ? !h.env.tour : !h.env.open || busy > h.env.crowd + 2)
  if (home) {
    o.state = 'leave'
    o.path = [[1322, 452], ...WAY.slice().reverse()]
  } else if (o.role === 'tour') {
    o.s = RUNS[0].length - 3
    skin(o)
  } else {
    o.state = 'walk'
    o.path = [[1322, 452]]
  }
  o.t = 0
}
