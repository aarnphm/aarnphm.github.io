import type { Voices } from './404-crew'
import { moonlit, winter as snowSeason, type Almanac, type Sky } from './404-almanac'
import { mulberry } from './404-pixel'
import {
  BOOM,
  COLS,
  GATES,
  lane,
  LIFT_LEN,
  QUEUE,
  ROWS,
  RUNS,
  seek,
  spot,
  ZONES,
  type Hill,
  type HillEnv,
  type Skier,
} from './404-skiers/hill-state'
import { skier, skin } from './404-skiers/population'
import { ski } from './404-skiers/ski-motion'
import { start } from './404-skiers/traffic'
export type { Hill, SkiInk } from './404-skiers/hill-state'
export { stepHill } from './404-skiers/traffic'
export { paintHill } from './404-skiers/paint-hill'

// The season and the hour: through the winter the chair runs from nine, solar time, and on under
// floodlights until nine at night, as the small hills round the lake do; on the summer glacier from
// half past seven until half past twelve, when the snow goes soft. Wind stops a chair long before a
// storm does. Winter weekends bring the most, lessons run in the mornings, and summer weekdays bring
// a race camp to the giant slalom on the blue. Once the chair stops, locals skin up the edge of the
// blue by headlamp: most in the evening until half past ten, a pair at first light, and through the
// small hours a die-hard, or two under a full moon. The groomer goes out after dark and again
// before dawn.
export function hillEnv(al: Almanac, sky: Sky, weekend: boolean): HillEnv {
  const winter = snowSeason(al)
  const [from, to] = winter ? [9, 21] : [7.5, 12.5]
  // A powder morning: a hand's depth of new snow loads the slopes under the crest, and from first
  // light the patrol shoots their start zones with hand charges. The chair opens half an hour late,
  // or up to an hour and a half after a big fall, with the line for first chair already waiting.
  const powder = winter && sky.snow >= 10
  const opens = from + (powder ? Math.min(1.5, 0.5 + (sky.snow - 10) / 30) : 0)
  const hours = al.hour >= opens && al.hour < to
  const open = hours && !sky.storm && sky.wind < 0.6
  const control = powder && al.sun.alt > -6 && al.hour < opens && !sky.storm && sky.wind < 0.75
  const line = control && al.hour >= from - 0.75
  let crowd = winter ? (weekend ? 12 : 6) : 3
  if (winter && al.hour >= 17) crowd *= 0.6
  if (sky.fog) crowd *= 0.5
  if (sky.fall === 'rain' || sky.fall === 'drizzle') crowd *= 0.3
  if (sky.fall === 'snow' && sky.heavy > 0.6) crowd *= 0.6
  if (sky.temp !== null && sky.temp < -20) crowd *= 0.5
  const glide =
    sky.fall === 'snow'
      ? 0.09
      : sky.temp === null
        ? 0.05
        : sky.temp > 0
          ? 0.08
          : sky.temp < -15
            ? 0.07
            : 0.05
  const groomed = al.hour >= to + 2.5 || al.hour < opens
  const uphill =
    al.hour >= to && al.hour < 22.5
      ? weekend
        ? 3
        : 2
      : (al.hour >= 6 && al.hour < from) || moonlit(al, sky)
        ? 2
        : 1
  return {
    open,
    hold: hours && !sky.storm && !open,
    crowd: open || line ? Math.max(1, Math.round(crowd)) : 0,
    glide,
    camp: open && !winter && !weekend,
    school: open && winter && al.hour < 13,
    groom:
      !hours &&
      !control &&
      ((al.hour >= Math.max(to + 0.5, 17) && al.hour < 23.5) ||
        (al.hour >= 4 && al.hour < from - 0.5)),
    fresh: sky.fall === 'snow' ? 0.3 + 0.7 * sky.heavy : 0,
    since: hours ? al.hour - opens : groomed ? 0 : to - opens,
    tour: !hours && !control && !sky.storm && sky.fall !== 'rain' && sky.wind < 0.75 ? uphill : 0,
    control,
  }
}

export function makeHill(seed: number, env: HillEnv): Hill {
  const lines = RUNS.map(() => new Uint8Array(ROWS * COLS)) as [Uint8Array, Uint8Array]
  // Nobody grooms the lift line, so it is always skied out.
  lines[1].fill(6)
  return {
    seed: seed * 7 + 3,
    rng: mulberry(seed * 7 + 3),
    env,
    skiers: [],
    ids: 0,
    parties: 0,
    tickets: 0,
    time: 0,
    arrive: 0,
    lift: 0,
    loaded: 0,
    tracks: lines,
    last: [new Float32Array(ROWS * COLS).fill(-99), new Float32Array(ROWS * COLS).fill(-99)],
    fill: 0,
    cat: { s: 0, lane: 0, dir: -1 },
    flick: new Float32Array(GATES.length),
    cheer: 0,
    blasts: [],
  }
}

// The day so far on the runs: tracks from the hours since the groomer last went over them, as that
// many runs by that many skiers of every level.
export function warmHill(h: Hill) {
  const quiet: Voices = { bubbles: [], heads: {} }
  const runs = Math.round(Math.min(7, h.env.since) * 14)
  const saved = h.skiers
  h.skiers = []
  for (let n = 0; n < runs; n++) {
    const one = skier(h, 0, 'free', h.rng() < 0.35, false, 0.15 + h.rng() * 0.8)
    one.run = one.skill > 0.7 && h.rng() < 0.4 && one.mode !== 'leaf' ? 1 : 0
    start(h, one)
    one.w = (h.rng() - 0.5) * RUNS[one.run].half[0]
    for (let k = 0; k < 900 && one.state === 'ski'; k++) ski(h, one, 0.1, quiet)
  }
  for (const t of h.last) t.fill(-99)
  h.skiers = saved
  h.ids = 0
}

// Without motion, one moment of a busy morning: someone carving across the blue, a boarder sat at
// the top strapping in, a pair on a chair and one in the queue, a ski-school line halfway down;
// on a summer weekday a racer between gates with the coach beside them; on a busy day, now and
// then, the patrol's toboggan on its way down; after hours the groomer halfway down with its lights
// on, someone skinning up the edge and a splitboarder at the top.
export function poseHill(h: Hill) {
  // Each resize or change of light poses it again, and the same people come back in the same kit.
  h.rng = mulberry(h.seed)
  h.ids = 0
  h.parties = 0
  h.skiers = []
  h.blasts = []
  h.cat.s = RUNS[0].length * (h.env.groom ? 0.45 : 1)
  h.cat.dir = h.env.groom ? 1 : -1
  if (h.env.tour) {
    // After hours: one climbing the edge of the blue, and when there are more, a splitboarder at
    // the top putting the board back together.
    const up = skier(h, ++h.parties, 'tour', false, false, 0.8)
    up.s = RUNS[0].length * 0.6
    skin(up)
    h.skiers.push(up)
  }
  if (h.env.tour > 1) {
    const split = skier(h, ++h.parties, 'tour', true, false, 0.7)
    const w = lane(RUNS[0], 0)
    Object.assign(split, {
      state: 'strap',
      w,
      x: spot(RUNS[0], 0, w)[0],
      y: spot(RUNS[0], 0, w)[1],
    })
    h.skiers.push(split)
  }
  if (h.env.control) {
    // A powder morning: the patroller back behind the rib, a charge just gone off under the
    // cornice, the slope letting go below it, last round's slide on the next face, and the line
    // for first chair stood waiting.
    const z = ZONES[1]
    const p = skier(h, ++h.parties, 'patrol', false, false, 0.95)
    Object.assign(p, { state: 'charge', shot: 1, t: BOOM + 1.5, x: z.hide[0], y: z.hide[1] })
    h.skiers.push(p)
    h.blasts.push({ x: z.lands[0], y: z.lands[1], t: 1.5, slide: 45 })
    h.blasts.push({ x: ZONES[2].lands[0], y: ZONES[2].lands[1], t: 900, slide: 35 })
    for (let n = 0; n < h.env.crowd; n++) {
      const o = skier(h, ++h.parties, 'free', h.rng() < 0.35, false, 0.3 + 0.6 * h.rng())
      const [qx, qy] = QUEUE(n)
      Object.assign(o, { state: 'queue', x: qx, y: qy })
      h.skiers.push(o)
    }
    return
  }
  if (!h.env.crowd) return
  const party = () => ++h.parties
  const put = (o: Skier, run: 0 | 1, s: number, w: number, psi: number, v: number) => {
    Object.assign(o, {
      state: 'ski',
      run,
      s,
      w,
      psi,
      v,
      dir: Math.sign(psi) || 1,
      amp: 1,
      skid: 0.3,
    })
    o.k = seek(RUNS[run], s, 0)
    h.skiers.push(o)
    return o
  }
  put(skier(h, party(), 'free', false, false, 0.85), 0, RUNS[0].length * 0.3, 4, 1.1, 11)
  const strap = skier(h, party(), 'free', true, false, 0.5)
  const [sx, sy] = spot(RUNS[0], 0, -RUNS[0].half[0] * 0.6)
  Object.assign(strap, { state: 'strap', x: sx, y: sy })
  h.skiers.push(strap)
  const pair = party()
  for (let seat = 0; seat < 2; seat++) {
    const o = skier(h, pair, 'free', seat === 1, false, 0.6)
    Object.assign(o, { state: 'chair', chair: h.lift - LIFT_LEN * 0.55, seat })
    h.skiers.push(o)
  }
  const wait = skier(h, party(), 'free', true, false, 0.4)
  const [qx, qy] = QUEUE(0)
  Object.assign(wait, { state: 'queue', x: qx, y: qy })
  h.skiers.push(wait)
  if (h.env.camp) {
    const camp = party()
    const coach = skier(h, camp, 'coach', false, false, 0.9)
    const [cx, cy] = spot(RUNS[0], GATES[4].s, -RUNS[0].half[seek(RUNS[0], GATES[4].s, 0)] + 1)
    Object.assign(coach, { state: 'coach', x: cx, y: cy })
    h.skiers.push(coach)
    put(skier(h, camp, 'racer', false, false, 0.9), 0, GATES[2].s + 8, 0, -0.5, 15)
    h.flick[3] = 0.3
  } else if (h.env.school) {
    const school = party()
    const teach = put(
      skier(h, school, 'teach', false, false, 0.95),
      0,
      RUNS[0].length * 0.55,
      7,
      1,
      3.5,
    )
    // Strung out behind along the traverse.
    for (let m = 1; m <= 3; m++) {
      const kid = put(
        skier(h, school, 'pupil', false, true, 0.2),
        0,
        teach.s - 6 * m,
        7 - 5.5 * m,
        1,
        3,
      )
      kid.lead = teach
      kid.slot = m
    }
  }
  // On a busy day, about one moment in three: the patrol bringing someone down in the toboggan.
  if (h.env.crowd >= 6 && h.rng() < 0.35) {
    const fallen = put(
      skier(h, party(), 'free', false, false, 0.5),
      0,
      RUNS[0].length * 0.7,
      -4,
      0,
      0,
    )
    const p = put(skier(h, party(), 'patrol', false, false, 0.95), 0, fallen.s + 2.5, -4, 0, 2.5)
    Object.assign(fallen, { state: 'sled', hurt: 1.5 })
    Object.assign(p, { state: 'sled', patient: fallen })
  }
}
