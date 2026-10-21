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

import { moonlit, winter, type Almanac, type Sky } from './404-almanac'
import { G, MAIN, PATHS, seek, TABLE_FLIGHT } from './404-downhill'
import { herdEnv, makeHerd, poseHerd } from './404-herd'
import { mulberry } from './404-pixel'
import { DUR, medic, type Park, type ParkEnv } from './404-riders/park-state'
import { rider } from './404-riders/population'
import { clamp } from './404-riders/positions'
import { SLOT } from './404-riders/traffic'
export type { Park, RiderInk } from './404-riders/park-state'
export { stepPark } from './404-riders/traffic'
export { paintPark } from './404-riders/paint-park'

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

// Seeds the day: the same riders turn up for everyone who looks on a given day, from the same
// arrivals onward.
export function daySeed(now: Date, lon: number) {
  return Math.floor((now.getTime() / 3.6e6 + lon / 15) / 24)
}
