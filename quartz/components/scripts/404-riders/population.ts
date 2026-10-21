import { PATHS } from '../404-downhill'
import { TRICKS, type Park, type Rider } from './park-state'
import { clamp } from './positions'

export function rider(
  p: Park,
  party: number,
  kid: boolean,
  skill: number,
  hikes: boolean,
  laps: number,
) {
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
export function recruit(p: Park, room: number) {
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

// Off to the lift, up the track on foot, or home: home when they have had their runs, when the
// park is thinning out, sometimes after a hard fall, and when the weather shuts it.
export function next(p: Park, r: Rider) {
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
