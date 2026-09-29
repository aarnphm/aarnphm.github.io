// The ski hill on the far right range, above the trail's summit: a double chair up the face, a
// groomed blue run that swings out to the right and back to the lift's foot, and the lift line, the
// steep narrow run straight down under the chair that nobody grooms. The glacier keeps both white
// all year. Whoever skis it is made up on arrival from the day's seed: on skis or a board, how good,
// how much nerve, who they came with. A beginner snowploughs, or on a board slides down sideways
// from one edge of the run to the other; the better carve, and hold their speed down by how far
// across the hill they finish each turn. Pairs share a chair, boarders sit down at the top to strap
// in, parties wait for each other before dropping in, uphill yields to downhill, and a fall on skis
// at speed leaves the skis up the slope for their owner to climb back to. Through the day the runs
// fill with tracks, and after hours a groomer with its lights on combs the blue run flat again.

import type { Almanac, Sky } from './404-almanac'
import { type Voices, say } from './404-crew'
import { type Grid, type Pt, type RGB, BAYER, col, dot, mulberry, row } from './404-pixel'

// The far range stands some eight times as far off as the island, so a metre there is 1.2 scene
// units. Its people are drawn three or four pixels tall all the same, several times their size,
// since a pixel is the least a person can be.
const M = 1.2
const G = 9.81
// Air drag over mass, ½ρC_dA/m, for a 75 kg skier standing up in the thin air at altitude.
const DRAG = (0.5 * 0.9 * 0.5) / 75
// A body sliding on snow after a fall.
const SLIDE = 0.55
// Carving radius of a ski's sidecut and of a board's.
const CARVE = { ski: 13, board: 7.5 }
const WALK = 1.1

export const LIFT_FOOT: Pt = [1322, 450]
export const LIFT_HEAD: Pt = [1398, 282]
// The chair's line along the slope, the speed of an old fixed-grip double, and its chairs: an
// even number round the loop, so the gap between them divides it.
const LIFT_LEN = Math.hypot(
  (LIFT_HEAD[0] - LIFT_FOOT[0]) / M,
  (LIFT_FOOT[1] - LIFT_HEAD[1]) / M / Math.sin((25 * Math.PI) / 180),
)
const LIFT_V = 2.3
const CHAIRS = 14
const GAP = (2 * LIFT_LEN) / CHAIRS
// Where the queue stands, and the way in and out over the mid ridge's crest.
const QUEUE = (n: number): Pt => [1316 - 7 * n, 452]
const WAY: Pt[] = [
  [1300, 492],
  [1310, 457],
]
// The groomer's spot by day, beside the lift's foot.
const SHED: Pt = [1306, 452]

type Run = {
  n: number
  x: Float32Array
  y: Float32Array
  // Metres along the slope from the top, the pitch there, and the run's half-width in metres.
  s: Float32Array
  pitch: Float32Array
  half: Float32Array
  length: number
  groomed: boolean
}

const quad = ([a, b, c]: [Pt, Pt, Pt], u: number): Pt => {
  const v = 1 - u
  return [
    v * v * a[0] + 2 * v * u * b[0] + u * u * c[0],
    v * v * a[1] + 2 * v * u * b[1] + u * u * c[1],
  ]
}

// A run from its centreline on the screen, its pitch at the top, middle and bottom in degrees, and
// its half-width in scene units as it comes toward us. The screen gives the drop in height; the
// pitch gives how far along the slope that drop takes.
function survey(ctrl: [Pt, Pt, Pt], deg: Pt3, half: (u: number) => number, groomed: boolean): Run {
  const n = 160
  const run: Run = {
    n,
    x: new Float32Array(n),
    y: new Float32Array(n),
    s: new Float32Array(n),
    pitch: new Float32Array(n),
    half: new Float32Array(n),
    length: 0,
    groomed,
  }
  for (let k = 0; k < n; k++) {
    const u = k / (n - 1)
    const [x, y] = quad(ctrl, u)
    const v = 1 - u
    const p = ((v * v * deg[0] + 2 * v * u * deg[1] + u * u * deg[2]) * Math.PI) / 180
    run.x[k] = x
    run.y[k] = y
    run.pitch[k] = p
    run.half[k] = half(u) / M
    if (k)
      run.s[k] =
        run.s[k - 1] + Math.hypot((x - run.x[k - 1]) / M, (y - run.y[k - 1]) / M / Math.sin(p))
  }
  run.length = run.s[n - 1]
  return run
}
type Pt3 = [number, number, number]

// The blue swings out right under the head station and back to the lift's foot; the lift line goes
// straight down beside the chair, twice as steep.
const RUNS = [
  survey(
    [
      [1416, 294],
      [1474, 372],
      [1340, 444],
    ],
    [16, 22, 10],
    u => 9 + 9 * u,
    true,
  ),
  survey(
    [
      [1400, 296],
      [1370, 370],
      [1336, 442],
    ],
    [30, 33, 14],
    u => 6 + 2 * u,
    false,
  ),
] as const
const seek = (run: Run, s: number, k: number) => {
  while (k < run.n - 2 && run.s[k + 1] <= s) k++
  while (k > 0 && run.s[k] > s) k--
  return k
}
// Where a point of a run is on the screen: across the run is across the screen.
function spot(run: Run, s: number, w: number): Pt {
  const k = seek(run, s, 0)
  const u = Math.max(0, Math.min(1, (s - run.s[k]) / (run.s[k + 1] - run.s[k])))
  return [
    run.x[k] + (run.x[k + 1] - run.x[k]) * u + w * M,
    run.y[k] + (run.y[k + 1] - run.y[k]) * u,
  ]
}

// Tracks, per run, in a coarse grid of how many have passed and when one last did.
const ROWS = 64
const COLS = 12
const cell = (run: Run, s: number, w: number, k: number) => {
  const r = Math.min(ROWS - 1, Math.max(0, Math.floor((s / run.length) * ROWS)))
  const c = Math.min(COLS - 1, Math.max(0, Math.floor(((w / run.half[k] + 1) / 2) * COLS)))
  return r * COLS + c
}

// The summer race camp's giant slalom on the blue: eight gates twenty-six metres apart, left and
// right, the first a couple of turns out of the start. Slalom poles eleven metres apart would stand
// a pixel apart from across the valley.
const GATES = Array.from({ length: 8 }, (_, n) => ({ s: 22 + 26 * n, w: n % 2 ? 4.5 : -4.5 }))
const FINISH = GATES[GATES.length - 1].s + 8

type Role = 'free' | 'teach' | 'pupil' | 'racer' | 'coach'
type Mode = 'carve' | 'skid' | 'plough' | 'leaf' | 'race'
type State =
  | 'arrive'
  | 'queue'
  | 'chair'
  | 'top'
  | 'strap'
  | 'ready'
  | 'ski'
  | 'down'
  | 'up'
  | 'collect'
  | 'out'
  | 'walk'
  | 'leave'
  | 'coach'

export type Skier = {
  id: number
  party: number
  role: Role
  board: boolean
  kid: boolean
  skill: number
  nerve: number
  // Coat, helmet, and skis or board, as indices into the day's inks.
  kit: [number, number, number]
  mode: Mode
  // What they have learned: the speed they are happy at, and how sure of themselves they are.
  pace: number
  conf: number
  runs: number
  laps: number
  state: State
  t: number
  ticket: number
  // On foot, where they are on the screen and where they are walking to.
  x: number
  y: number
  path: Pt[]
  // On a run: which, how far down it and across it, the heading off the fall line, the speed, which
  // way the turn is going, how far across the hill they finish each turn, and how hard they are
  // skidding or ploughing.
  run: 0 | 1
  k: number
  s: number
  w: number
  psi: number
  v: number
  dir: number
  amp: number
  skid: number
  // On the chair: the length of cable run when they boarded, and which seat.
  chair: number
  seat: number
  // Skis left up the slope after a fall, whom they stopped to ask after, and how long they stop.
  gear: number
  gearW: number
  hurt: number
  stop: number
  asked: number
  cell: number
  // The ski school: whom a pupil follows, their place in the line, and the instructor's track.
  lead: Skier | null
  slot: number
  trail: Float32Array
  gone: boolean
}

export type HillEnv = {
  open: boolean
  // In hours but stopped: the wind is too strong for the chair.
  hold: boolean
  crowd: number
  // Ski on snow: dry cold snow runs fastest, wet or new snow slower.
  glide: number
  camp: boolean
  school: boolean
  groom: boolean
  // Snow falling, filling the tracks in.
  fresh: number
  // Hours of skiing on the runs since the groomer last went over them.
  since: number
}

export type Hill = {
  rng: () => number
  env: HillEnv
  skiers: Skier[]
  ids: number
  parties: number
  tickets: number
  time: number
  arrive: number
  // Metres of cable the lift has run, and the last chair loaded.
  lift: number
  loaded: number
  tracks: [Uint8Array, Uint8Array]
  last: [Float32Array, Float32Array]
  fill: number
  cat: { s: number; lane: number; dir: number }
  flick: Float32Array
  cheer: number
}

// The season and the hour: through the winter the chair runs nine to four, solar time; on the
// summer glacier from half past seven until half past twelve, when the snow goes soft. Wind stops a
// chair long before a storm does. Winter weekends bring the most, lessons run in the mornings, and
// summer weekdays bring a race camp to the giant slalom on the blue. The groomer goes out in the
// evening and again before dawn.
export function hillEnv(al: Almanac, sky: Sky, weekend: boolean): HillEnv {
  const winter = Math.cos((2 * Math.PI * (al.winterDay - 40)) / 365.24) > -0.2
  const [from, to] = winter ? [9, 16] : [7.5, 12.5]
  const hours = al.hour >= from && al.hour < to
  const open = hours && !sky.storm && sky.wind < 0.6
  let crowd = winter ? (weekend ? 12 : 6) : 3
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
  const groomed = al.hour >= to + 3 || al.hour < from
  return {
    open,
    hold: hours && !sky.storm && !open,
    crowd: open ? Math.max(1, Math.round(crowd)) : 0,
    glide,
    camp: open && !winter && !weekend,
    school: open && winter && al.hour < 13,
    groom:
      !hours && ((al.hour >= to + 1.5 && al.hour < 23) || (al.hour >= 4 && al.hour < from - 0.5)),
    fresh: sky.fall === 'snow' ? 0.3 + 0.7 * sky.heavy : 0,
    since: hours ? al.hour - from : groomed ? 0 : to - from,
  }
}

export function makeHill(seed: number, env: HillEnv): Hill {
  const lines = RUNS.map(() => new Uint8Array(ROWS * COLS)) as [Uint8Array, Uint8Array]
  // Nobody grooms the lift line, so it is always skied out.
  lines[1].fill(6)
  return {
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
  }
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

function skier(h: Hill, party: number, role: Role, board: boolean, kid: boolean, skill: number) {
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
    gone: false,
  }
  return one
}

// Whoever turns up next, walking in over the crest of the mid ridge: on their own, a crew of
// friends of about the same level on skis and boards, a parent with a child or two, in the
// mornings a ski-school class of small children behind their instructor, and on a summer weekday
// the race camp and its coach.
function recruit(h: Hill, room: number) {
  const r = h.rng
  const party = ++h.parties
  const add = (one: Skier, m: number) => {
    one.x -= m * 3
    one.y += m * 4
    h.skiers.push(one)
    return one
  }
  if (h.env.camp && !h.skiers.some(o => o.role === 'coach')) {
    // The coach is already out on the hill beside the fifth gate.
    const coach = add(skier(h, party, 'coach', false, false, 0.95), 0)
    const k = seek(RUNS[0], GATES[4].s, 0)
    ;[coach.x, coach.y] = spot(RUNS[0], GATES[4].s, -RUNS[0].half[k] + 1)
    coach.state = 'coach'
    for (let m = 1; m <= 4; m++)
      add(skier(h, party, 'racer', false, r() < 0.3, 0.82 + r() * 0.15), m).laps = 99
    return
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

const onRun = (o: Skier) => o.state === 'ski' || o.state === 'down' || o.state === 'up'
const byTicket = (a: Skier, b: Skier) => a.ticket - b.ticket

function mark(h: Hill, o: Skier, run: Run) {
  const c = cell(run, o.s, o.w, o.k)
  if (c === o.cell) return
  o.cell = c
  const t = h.tracks[o.run]
  t[c] = Math.min(255, t[c] + 1)
  h.last[o.run][c] = h.time
}

function fall(h: Hill, o: Skier, voices: Voices) {
  o.state = 'down'
  o.t = 0
  o.hurt = h.rng()
  o.conf = clamp(o.conf - 0.2, 0.05, 1)
  o.pace *= 0.85
  // Off the skis at speed, the skis and poles stay where it happened.
  if (!o.board && o.v > 7 && h.rng() < 0.6) {
    o.gear = o.s
    o.gearW = o.w
  }
  say(voices, `ski${o.id}`, '!')
  // Whoever is coming down behind stops beside them to ask.
  let near: Skier | null = null
  for (const p of h.skiers)
    if (p !== o && p.state === 'ski' && p.run === o.run && p.s < o.s && o.s - p.s < 50)
      if (!near || p.s > near.s) near = p
  if (near && near.asked !== o.id) {
    near.asked = o.id
    near.stop = 6
  }
}

// Down the run. Gravity along the heading against the ski's glide and the air; turns carved round
// the sidecut, or pivoted and skidded, which bleeds speed; and the speed held down by how far
// across the hill each turn finishes, a little further each time it runs over what they are happy
// with. A snowplough brakes with its wedge; a beginner on a board slides down on the heel edge
// from one side of the run to the other without ever pointing down it; a racer turns round gates.
function ski(h: Hill, o: Skier, dt: number, voices: Voices) {
  const run = RUNS[o.run]
  o.k = seek(run, o.s, o.k)
  const a = run.pitch[o.k]
  const half = run.half[o.k]
  const down = G * Math.sin(a)
  const glide = h.env.glide * G * Math.cos(a)
  // Steeper than a blue, everyone skis it slower than they would like to.
  let want = o.stop > 0 ? 0 : o.pace * (1 - Math.max(0, a - 0.4) * 1.6)
  o.stop = Math.max(0, o.stop - dt)
  if (o.stop > 0 && o.v < 0.6 && o.asked > 0) {
    say(voices, `ski${o.id}`, '?')
    o.asked = -o.asked
  }

  if (o.role === 'racer' && o.s > FINISH) want = Math.min(want, 8)

  // Keep clear of whoever is below: they have the right of way.
  let turnAway = 0
  for (const p of h.skiers) {
    if (p === o || p.run !== o.run || !onRun(p) || p.party === o.party) continue
    const ds = p.s - o.s
    const dw = p.w - o.w
    if (ds > 0 && ds < 14 && Math.abs(dw) < 3.5) {
      if (Math.sign(dw) === Math.sign(Math.sin(o.psi)) || Math.abs(o.psi) < 0.3)
        turnAway = -Math.sign(dw) || 1
      if (ds < 4 && Math.abs(dw) < 2) want = Math.min(want, p.v * 0.8)
    }
  }

  let brake = 0
  let over = 0
  if (o.mode === 'leaf') {
    // Across the run on the heel edge, sliding down as fast as they let it.
    o.psi = o.dir * 1.35
    const edge = Math.abs(o.w) > half - 1.5 && Math.sign(o.w) === o.dir
    if (edge || (turnAway && turnAway !== o.dir)) o.dir = -o.dir
    const slip = clamp(1.5 + 1.2 * o.conf, 1.2, 2.6)
    o.v = Math.hypot(slip, 1.8)
    o.s += slip * dt
    o.w += o.dir * 1.8 * dt
  } else {
    const gate = o.mode === 'race' && o.run === 0 ? GATES.find(g => g.s > o.s + 1) : undefined
    if (gate) {
      // Round the outside of the next gate's pole, the line wider the faster they come at it and
      // the edge set harder to bleed what they cannot hold. Past the finish they ski back easily.
      const wide = gate.w + Math.sign(gate.w) * (1.2 + 0.3 * clamp(o.v - want, 0, 4))
      const aim = Math.atan2(wide - o.w, gate.s - o.s)
      const rate = Math.max(o.v / 18, 0.9)
      o.psi += clamp(aim - o.psi, -rate * dt, rate * dt)
      o.skid = o.v > want ? 1 : 0.1
    } else if (o.lead && o.lead.trail.length) {
      // A pupil skis where the instructor skied, a turn or so behind them.
      const tr = o.lead.trail
      const n = tr.length / 2
      const lag = Math.min(n - 1, o.slot * 18)
      const at = (((Math.floor(h.time * 4) - lag) % n) + n) % n
      // Once the instructor is down, the rest of the way is theirs to find.
      const done = o.lead.state !== 'ski'
      const ts = done ? run.length : tr[2 * at]
      const tw = done ? 0 : tr[2 * at + 1]
      const gap = Math.hypot(ts - o.s, tw - o.w) * (ts < o.s - 1 ? -1 : 1)
      const aim = clamp(Math.atan2(tw - o.w, Math.max(1, ts - o.s)), -1.2, 1.2)
      o.psi += clamp(aim - o.psi, -0.8 * dt, 0.8 * dt)
      want = clamp(o.lead.v + 0.5 * gap, 0, 4.5)
    } else {
      // Finishing each turn further across the hill as the speed runs on past what they want.
      const target = Math.max(0.5, want)
      // An instructor leads the class in long traverses from side to side, holding the speed down
      // with the wedge, so the line behind strings out across the run.
      o.amp =
        o.role === 'teach'
          ? 1
          : clamp(
              o.amp + dt * 1.2 * ((o.v - target) / target),
              0.3,
              o.mode === 'plough' ? 0.7 : 1.75,
            )
      // Heading for the side of the run, they tighten the turn as the room runs out, down to half
      // the sidecut's radius with the ski right up on its edge, and start back before they need to.
      // A wedge turns tighter than any sidecut, and slowly.
      const sidecut = o.mode === 'plough' ? 6 : o.board ? CARVE.board : CARVE.ski
      const out = Math.sign(o.w) === Math.sign(Math.sin(o.psi))
      const room = Math.max(0.1, half - 1 - Math.abs(o.w))
      const need = out ? 1 - Math.cos(o.psi) : 0
      // Well over their speed, they check it: the skis thrown across the hill in a hard skid, a turn
      // that needs no more room than their own length.
      const check = o.mode !== 'plough' && o.v > target * 1.2
      const radius = check ? 2 : need > 0 ? clamp(room / need, sidecut / 2, sidecut) : sidecut
      const rate =
        o.mode === 'plough'
          ? o.role === 'teach'
            ? 0.8
            : 0.5
          : check
            ? 2.5
            : Math.max(o.v / radius, o.mode === 'skid' ? 1.4 : 0.6)
      o.psi += o.dir * rate * dt
      // What the edge holds across a carved turn. Asked for more, the ski lets go and skids, and the
      // further past it, the likelier it catches or washes out. A skidder is sliding already.
      if (o.mode === 'carve' && !check) over = (o.v * rate) / (G * (0.45 + o.skill))
      const edge = out && room < (check ? 1 : sidecut * 0.7) * need + 1
      // The instructor holds each traverse out to the side of the run before turning.
      if (o.role === 'teach' && o.dir * o.psi > o.amp) o.psi = o.dir * o.amp
      if ((o.dir * o.psi >= o.amp && o.role !== 'teach') || edge)
        o.dir = edge ? -Math.sign(o.w) : -o.dir
      if (turnAway && o.dir !== turnAway && o.dir * o.psi > 0.2) o.dir = turnAway
      // A teacher waits for the line to catch up.
      if (o.role === 'teach')
        for (const p of h.skiers) if (p.lead === o && p.state !== 'ski') want = 0
      if (o.mode !== 'plough') o.skid = over > 1 || check ? 1 : clamp((1 - o.skill) * 0.9, 0, 1)
    }
    if (o.mode === 'plough') {
      o.skid = clamp(o.skid + dt * (o.v - want), 0.1, 1)
      brake = 0.45 * G * Math.cos(a) * o.skid
    } else brake = 0.35 * G * o.skid * Math.abs(Math.sin(o.psi))
    const acc = down * Math.cos(o.psi) - glide - DRAG * o.v * o.v - brake
    o.v = Math.max(0, o.v + acc * dt)
    o.s += o.v * Math.cos(o.psi) * dt
    o.w += o.v * Math.sin(o.psi) * dt
  }
  if (Math.abs(o.w) > half) {
    o.w = Math.sign(o.w) * half
    o.v *= 0.9
  }
  o.s = Math.max(0, o.s)
  // Catching an edge: boarders more than skiers, beginners far more, more the faster they go, and a
  // beginner on a board's downhill edge most of all.
  const catchRate =
    (o.board ? 0.008 : 0.004) * (1 - o.skill) ** 2 * Math.min(1.5, o.v / 5) +
    (o.mode === 'leaf' ? 0.003 * (1 - o.conf) : 0) +
    Math.max(0, over - 1.3) * 0.1 * (1 - o.skill)
  if (h.rng() < catchRate * dt) return fall(h, o, voices)
  mark(h, o, run)
  if (o.role === 'teach' && o.trail.length) {
    const n = o.trail.length / 2
    const at = Math.floor(h.time * 4) % n
    o.trail[2 * at] = o.s
    o.trail[2 * at + 1] = o.w
  }
  if (o.mode === 'race' && o.run === 0)
    GATES.forEach((g, n) => {
      if (Math.abs(o.s - g.s) < o.v * dt + 0.2 && Math.abs(o.w - g.w) < 1.6) h.flick[n] = 0.4
    })
  if (o.s >= run.length - 6) {
    o.state = 'out'
    o.t = 0
  }
}

// Getting down and back up: the body slides to a stop, lies there a moment, sits up (a boarder
// rolls over onto their knees first), and goes on across the hill; with the skis up the slope they
// sidestep back up to them first and click in.
function tumble(o: Skier, dt: number) {
  const run = RUNS[o.run]
  o.k = seek(run, o.s, o.k)
  if (o.state === 'down') {
    const a = run.pitch[o.k]
    o.v = Math.max(0, o.v - Math.max(1.5, SLIDE * G * Math.cos(a) - G * Math.sin(a)) * dt)
    o.s = Math.min(run.length - 7, o.s + o.v * dt)
    if (o.t > 1.5 + 2.5 * o.hurt) {
      o.state = 'up'
      o.t = 0
    }
  } else if (o.state === 'up' && o.t > (o.board ? 3 : 2)) {
    o.state = o.gear >= 0 ? 'collect' : 'ski'
    o.t = 0
    o.v = 0
    o.psi = o.dir * 1.2
  } else if (o.state === 'collect') {
    const step = 0.5 * dt
    o.s = Math.max(o.gear, o.s - step)
    o.w += clamp(o.gearW - o.w, -step, step)
    if (o.s <= o.gear + 0.1 && o.t > 2) {
      o.gear = -1
      o.state = 'ski'
      o.v = 0
      o.psi = o.dir * 1.2
    }
  }
}

const walk = (o: Skier, dt: number) => {
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
function finish(h: Hill, o: Skier) {
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
  const busy = h.skiers.filter(p => p.state !== 'leave' && p.role !== 'coach').length
  const [x, y] = spot(RUNS[o.run], RUNS[o.run].length - 3, o.w)
  o.x = x
  o.y = y
  const home = o.runs >= o.laps || !h.env.open || busy > h.env.crowd + 2
  if (home) {
    o.state = 'leave'
    o.path = [[1322, 452], ...WAY.slice().reverse()]
  } else {
    o.state = 'walk'
    o.path = [[1322, 452]]
  }
  o.t = 0
}

export function stepHill(h: Hill, dt: number, voices: Voices) {
  h.time += dt
  h.cheer = Math.max(0, h.cheer - dt)
  for (let n = 0; n < h.flick.length; n++) h.flick[n] = Math.max(0, h.flick[n] - dt)
  const env = h.env

  // The chair runs through its hours, and on until whoever is on it is off; the wind stops it.
  const riding = h.skiers.some(o => o.state === 'chair')
  if (!env.hold && (env.open || riding)) h.lift += LIFT_V * dt
  const slot = Math.floor(h.lift / GAP)
  const board = env.open && slot !== h.loaded
  h.loaded = slot

  const live = h.skiers.filter(o => o.state !== 'leave').length
  if (live < env.crowd && (h.arrive -= dt) <= 0) {
    recruit(h, env.crowd - live)
    h.arrive = 8 + h.rng() * 20
  }

  // Snow falling fills the tracks in.
  if ((h.fill += (env.fresh * dt) / 30) >= 1) {
    h.fill -= 1
    for (const t of h.tracks) for (let c = 0; c < t.length; c++) if (t[c] > 1) t[c]--
  }
  stepCat(h, dt)

  const queue = h.skiers.filter(o => o.state === 'queue').sort(byTicket)
  const steps = Math.ceil(dt * 30)
  for (const o of h.skiers) {
    o.t += dt
    switch (o.state) {
      case 'arrive':
        if (walk(o, dt)) {
          o.state = 'walk'
          o.path = [QUEUE(queue.length)]
        }
        break
      case 'walk':
        if (o.path.length) walk(o, dt)
        else {
          o.state = 'queue'
          o.ticket = ++h.tickets
          queue.push(o)
        }
        break
      case 'queue': {
        const n = queue.indexOf(o)
        o.path = [QUEUE(n)]
        walk(o, dt)
        if (!env.open && !env.hold) {
          o.state = 'leave'
          o.path = WAY.slice().reverse()
        }
        break
      }
      case 'chair':
        if (h.lift - o.chair >= LIFT_LEN) {
          o.state = 'top'
          o.x = LIFT_HEAD[0]
          o.y = LIFT_HEAD[1]
          o.run =
            o.role === 'free' && o.mode !== 'plough' && o.mode !== 'leaf' && choose(h, o) ? 1 : 0
          const [x, y] = spot(RUNS[o.run], 0, o.board ? -RUNS[o.run].half[0] * 0.6 : 0)
          o.path = [[x, y]]
          o.t = 0
        }
        break
      case 'top':
        if (walk(o, dt * 1.5)) {
          o.state = o.board ? 'strap' : 'ready'
          o.t = 0
        }
        break
      case 'strap':
        if (o.t > 5 + 10 * (1 - o.skill)) {
          o.state = 'ready'
          o.t = 0
        }
        break
      case 'ready':
        drop(h, o)
        break
      case 'ski':
        for (let n = 0; n < steps && o.state === 'ski'; n++) ski(h, o, dt / steps, voices)
        break
      case 'down':
      case 'up':
      case 'collect':
        tumble(o, dt)
        break
      case 'out':
        // A hockey stop.
        o.v = Math.max(0, o.v - 0.7 * G * dt)
        if (o.v === 0) finish(h, o)
        break
      case 'leave':
        if (walk(o, dt)) o.gone = true
        break
      case 'coach':
        if (!env.camp && !h.skiers.some(p => p.role === 'racer' && p.state !== 'leave')) {
          o.state = 'leave'
          o.path = [spot(RUNS[0], RUNS[0].length - 3, 0), [1322, 452], ...WAY.slice().reverse()]
        }
        break
    }
  }

  // A chair at the foot: the head of the queue sits down, and whoever is next with them if they
  // came together, or if both came alone.
  if (board && queue.length) {
    const first = queue[0]
    const near = Math.hypot(first.x - QUEUE(0)[0], first.y - QUEUE(0)[1]) < 1
    if (near) {
      const alone = (o: Skier) => !queue.some(p => p !== o && p.party === o.party)
      const second = queue[1]
      const pair =
        second && (second.party === first.party || (alone(first) && alone(second))) ? second : null
      for (const [seat, o] of [first, pair].entries()) {
        if (!o) continue
        o.state = 'chair'
        o.chair = slot * GAP
        o.seat = seat
        o.t = 0
      }
    }
  }
  h.skiers = h.skiers.filter(o => !o.gone)
}

// The lift line or the blue: the steeper the better they are and the surer of themselves.
function choose(h: Hill, o: Skier) {
  const bold = o.conf * (0.5 + o.skill) + (h.rng() - 0.5) * 0.2
  return o.skill > 0.6 && h.rng() < clamp((bold - 0.7) * 2, 0, 0.6)
}

// At the top they wait for the rest of their party (up to a minute), then drop in one after another,
// a few seconds apart; a racer waits until the course below is clear.
function drop(h: Hill, o: Skier) {
  const party = h.skiers.filter(p => p.party === o.party && p.role !== 'coach')
  const behind = party.some(p => p.state === 'chair' || p.state === 'top' || p.state === 'strap')
  if (behind && o.t < 60) return
  const order = party.filter(p => p.state === 'ready' || p.state === 'ski').indexOf(o)
  if (o.role === 'racer') {
    const busy = h.skiers.some(p => p.role === 'racer' && p.state === 'ski' && p.s < FINISH)
    if (busy || o.t < 4) return
  } else if (o.role !== 'pupil' && o.t < 2 + 4 * Math.max(0, order)) return
  if (o.role === 'pupil' && o.lead && o.lead.state !== 'ski' && o.t < 60) return
  start(h, o)
}

function start(h: Hill, o: Skier) {
  o.state = 'ski'
  o.t = 0
  o.k = 0
  o.s = 0
  o.w = spotW(o)
  o.v = 0
  o.dir = h.rng() < 0.5 ? -1 : 1
  o.psi = o.dir * 0.6
  o.amp = o.mode === 'plough' ? 0.5 : 0.9
  o.skid = 0.3
  o.hurt = -1
  o.asked = 0
  o.cell = -1
  // A fresh track for the class to follow, starting where they stand.
  o.trail.fill(0)
}
const spotW = (o: Skier) => (o.board ? -RUNS[o.run].half[0] * 0.6 : 0)

// The groomer: up the blue with its blade raised, then down it combing a lane flat, lane by lane
// across the run, from after the lifts close until late, and again before dawn.
function stepCat(h: Hill, dt: number) {
  const c = h.cat
  const run = RUNS[0]
  if (!h.env.groom) {
    c.s = run.length
    c.dir = -1
    return
  }
  c.s += c.dir * (c.dir < 0 ? 2 : 3) * dt
  if (c.s <= 0) c.dir = 1
  if (c.s >= run.length) {
    c.dir = -1
    c.lane = (c.lane + 1) % 3
  }
  c.s = clamp(c.s, 0, run.length)
  if (c.dir > 0) {
    const k = seek(run, c.s, 0)
    const w = laneW(c.lane, run.half[k])
    for (let d = -2.5; d <= 2.5; d += 1) {
      const n = cell(run, c.s, w + d, k)
      h.tracks[0][n] = 0
      h.last[0][n] = -99
    }
  }
}
const laneW = (lane: number, half: number) => (lane - 1) * (half - 3)

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
// on a summer weekday a racer between gates with the coach beside them; after hours the groomer
// halfway down with its lights on.
export function poseHill(h: Hill) {
  h.skiers = []
  h.cat.s = RUNS[0].length * (h.env.groom ? 0.45 : 1)
  h.cat.dir = h.env.groom ? 1 : -1
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
}

export type SkiInk = {
  snow: RGB[]
  skin: RGB
  pants: RGB
  boot: RGB
  steel: RGB
  cable: RGB
  rock: [RGB, RGB]
  coats: RGB[]
  helmets: RGB[]
  gear: RGB[]
  school: RGB
  gates: [RGB, RGB]
  cat: RGB
  cab: RGB
  lamp: RGB
  glow: RGB
  beacon: RGB
}

// The runs, the chair and its riders, and everyone on the hill, over the far range. `onRange` says
// where the far range shows between its crest and the mid ridge in front of it; the runs are only
// painted there, and whoever walks in over the mid ridge's crest is hidden until they are over it.
export function paintHill(
  d: Uint8ClampedArray,
  g: Grid,
  ink: SkiInk,
  h: Hill,
  onRange: (i: number, j: number) => boolean,
  dark: boolean,
  heads: Voices['heads'],
) {
  if (col(g, LIFT_FOOT[0] - 30) >= g.bw || col(g, 1474) < 0) return
  for (const who of Object.keys(heads)) if (who.startsWith('ski')) delete heads[who]
  const plot = (i: number, j: number, c: RGB, alpha: number) => {
    if (alpha > BAYER[(j & 3) * 4 + (i & 3)]) dot(d, g, i, j, c)
  }
  const at = ([x, y]: Pt): Pt => [col(g, x), row(g, y)]
  const beat = Math.floor(h.time * 6)

  // The runs: the blue in corduroy, alternate rows catching the light, and the lift line bumped
  // grey, each between two lines of the deepest snow shade, since a run five to nine pixels across
  // reads as one only by its edges against the range's own shading. Where people have skied the
  // cords are polished smooth and bright, where many have it is scraped grey, and a line skied in
  // the last few seconds shows as a groove.
  RUNS.forEach((run, n) => {
    const tracks = h.tracks[n]
    const last = h.last[n]
    let k = 0
    for (
      let j = Math.max(0, row(g, run.y[0]));
      j <= Math.min(g.bh - 1, row(g, run.y[run.n - 1]));
      j++
    ) {
      const y = g.sy[j]
      while (k < run.n - 2 && run.y[k + 1] <= y) k++
      const u = clamp((y - run.y[k]) / (run.y[k + 1] - run.y[k]), 0, 1)
      const cx = run.x[k] + (run.x[k + 1] - run.x[k]) * u
      const half = run.half[k] * M
      const s = run.s[k] + (run.s[k + 1] - run.s[k]) * u
      const i0 = col(g, cx - half)
      const i1 = col(g, cx + half)
      for (let i = i0; i <= i1; i++) {
        if (!onRange(i, j)) continue
        const off = (g.sx[i] - cx) / half
        const c = cell(run, s, off * run.half[k], k)
        const passes = tracks[c]
        const fresh = h.time - last[c] < 12
        const tone =
          i === i0 || i === i1
            ? 0
            : fresh
              ? 1
              : !run.groomed || passes >= 10
                ? 2
                : passes >= 2
                  ? 3
                  : j % 2
                    ? 2
                    : 3
        dot(d, g, i, j, ink.snow[tone])
      }
    }
  })

  // The giant slalom through the summer mornings: gates in alternate colours, knocked as a racer
  // goes by.
  if (h.env.camp)
    GATES.forEach((gate, n) => {
      const [i, j] = at(spot(RUNS[0], gate.s, gate.w))
      if (!onRange(i, j)) return
      const c = ink.gates[n % 2]
      dot(d, g, i, j - 1, c)
      dot(d, g, i + (h.flick[n] > 0 ? Math.sign(gate.w) : 0), j - 2, c)
    })

  // The chair's line, sagging a little between the stations: from across the valley the cables up
  // and down are one thread, and the chairs hang off it like beads.
  const lift = (u: number): Pt => [
    LIFT_FOOT[0] + (LIFT_HEAD[0] - LIFT_FOOT[0]) * u,
    LIFT_FOOT[1] + (LIFT_HEAD[1] - LIFT_FOOT[1]) * u + Math.sin(Math.PI * u) * 6,
  ]
  for (let u = 0; u <= 1; u += 0.01) {
    const [i, j] = at(lift(u))
    dot(d, g, i, j, ink.cable)
  }
  for (const u of [0.36, 0.7]) {
    const [i, j] = at(lift(u))
    for (let k = 1; k <= 4; k++) dot(d, g, i, j + k, ink.steel)
  }
  for (const p of [LIFT_FOOT, LIFT_HEAD]) {
    const [i, j] = at(p)
    for (let di = -1; di <= 1; di++) {
      dot(d, g, i + di, j, ink.rock[1])
      dot(d, g, i + di, j + 1, ink.rock[0])
      dot(d, g, i + di, j - 1, ink.steel)
    }
  }
  // On a wind hold the stopped chairs swing.
  const sway = h.env.hold ? (Math.floor(h.time / 0.9) % 2) * 2 - 1 : 0
  const phase = h.lift % GAP
  for (let n = 0; n < CHAIRS; n++) {
    const p = phase + n * GAP
    const u = p < LIFT_LEN ? p / LIFT_LEN : 2 - p / LIFT_LEN
    const [i, j] = at(lift(u))
    dot(d, g, i + (n % 2 ? sway : 0), j + 1, ink.steel)
  }

  // The groomer: in its shed spot by day; at work, a block with its cab and blade, its tiller
  // combing behind it, its lights ahead and its beacon turning.
  {
    const c = h.cat
    const at0 = h.env.groom
      ? spot(RUNS[0], c.s, laneW(c.lane, RUNS[0].half[seek(RUNS[0], c.s, 0)]))
      : SHED
    const [i, j] = at(at0)
    if (onRange(i, j - 1)) {
      const f = h.env.groom ? (c.dir > 0 ? 1 : -1) : 1
      for (let di = -1; di <= 1; di++) {
        dot(d, g, i + di, j - 1, ink.cat)
        dot(d, g, i + di, j, ink.boot)
      }
      dot(d, g, i, j - 2, ink.cab)
      dot(d, g, i + 2 * f, j, ink.steel)
      if (h.env.groom) {
        dot(d, g, i - 2 * f, j, ink.snow[3])
        if (beat % 3 === 0) dot(d, g, i, j - 3, ink.beacon)
        if (dark) {
          for (let dj = -2; dj <= 2; dj++)
            for (let di = 0; di <= 4; di++) {
              const r = Math.hypot(di, dj * 1.5)
              if (r > 0) plot(i + f * (2 + di), j - 1 + dj, ink.glow, (1 - r / 5) * 0.6)
            }
          dot(d, g, i + 2 * f, j - 1, ink.lamp)
        }
      }
    }
  }

  const where = (o: Skier): Pt => {
    if (o.state === 'chair') {
      const u = clamp((h.lift - o.chair) / LIFT_LEN, 0, 1)
      return lift(u)
    }
    if (onRun(o) || o.state === 'collect') return spot(RUNS[o.run], o.s, o.w)
    if (o.state === 'out') return spot(RUNS[o.run], RUNS[o.run].length - 4, o.w)
    return [o.x, o.y]
  }
  const order = h.skiers.map(o => ({ o, p: at(where(o)) })).sort((a, b) => a.p[1] - b.p[1])
  for (const { o, p } of order) {
    let [i, j] = p
    if (o.state === 'chair') i += o.seat
    if (!onRange(i, j - 1) && o.state !== 'chair') continue
    const coat = o.role === 'teach' ? ink.school : ink.coats[o.kit[0] % ink.coats.length]
    const helmet = ink.helmets[o.kit[1] % ink.helmets.length]
    const gear = ink.gear[o.kit[2] % ink.gear.length]
    const tall = o.kid ? 3 : 4
    let head: Pt = [i, j - tall + 1]
    const face = Math.sin(o.psi) >= 0 ? 1 : -1
    switch (o.state) {
      case 'chair':
        // Sat on the chair, skis or board hanging below.
        dot(d, g, i, j, coat)
        dot(d, g, i, j - 1, helmet)
        dot(d, g, i, j + 2, gear)
        head = [i, j - 1]
        break
      case 'ski':
      case 'ready':
      case 'out': {
        const lean =
          o.mode === 'leaf' || o.mode === 'plough' ? 0 : Math.abs(o.psi) < o.amp * 0.8 ? o.dir : 0
        if (o.mode === 'plough') {
          // The wedge, tips together: two feet apart under the body.
          dot(d, g, i - 1, j, gear)
          dot(d, g, i + 1, j, gear)
        } else for (let di = -1; di <= 1; di++) dot(d, g, i + di, j, gear)
        if (o.kid) {
          dot(d, g, i, j - 1, coat)
          dot(d, g, i + lean, j - 2, helmet)
          head = [i + lean, j - 2]
        } else {
          const tuck = o.mode === 'race' && Math.abs(o.psi) < 0.3
          dot(d, g, i, j - 1, ink.pants)
          dot(d, g, i + lean, j - 2, coat)
          if (tuck) {
            dot(d, g, i + lean + face, j - 2, helmet)
            head = [i + lean + face, j - 2]
          } else {
            dot(d, g, i + lean, j - 3, helmet)
            head = [i + lean, j - 3]
          }
          // Arms out for balance sliding sideways on a board.
          if (o.mode === 'leaf') {
            dot(d, g, i - 1, j - 2, ink.skin)
            dot(d, g, i + 1, j - 2, ink.skin)
          } else if (!o.board) dot(d, g, i - face, j - 1, ink.steel)
        }
        // Snow off the outside of a skidded turn or a stop.
        const spray = o.state === 'out' ? 1 : o.v > 3 ? o.skid * Math.abs(Math.sin(o.psi)) : 0
        for (let n = 1; n <= 3; n++)
          if (spray * 1.2 > n / 3 && (beat + n) % 2)
            dot(d, g, i + face * (1 + n), j - (n > 1 ? 1 : 0), ink.snow[3])
        break
      }
      case 'down':
      case 'up':
        if (o.state === 'down' || o.t < 1) {
          // Flat out on the snow.
          dot(d, g, i - 1, j - 1, helmet)
          dot(d, g, i, j - 1, coat)
          dot(d, g, i + 1, j - 1, ink.pants)
          dot(d, g, i + 2, j, o.gear >= 0 ? ink.boot : gear)
          head = [i - 1, j - 1]
        } else {
          dot(d, g, i, j - 1, coat)
          dot(d, g, i, j - 2, helmet)
          dot(d, g, i + 1, j, gear)
          head = [i, j - 2]
        }
        break
      case 'strap':
        // Sat in the snow at the top, strapping in.
        for (let di = -1; di <= 1; di++) dot(d, g, i + di, j, gear)
        dot(d, g, i, j - 1, coat)
        dot(d, g, i, j - 2, helmet)
        head = [i, j - 2]
        break
      case 'coach':
        dot(d, g, i, j, ink.pants)
        dot(d, g, i, j - 1, ink.pants)
        dot(d, g, i, j - 2, coat)
        dot(d, g, i, j - 3, helmet)
        if (h.cheer > 0) {
          dot(d, g, i - 1, j - 3 - (beat & 1), ink.skin)
          dot(d, g, i + 1, j - 4 + (beat & 1), ink.skin)
        }
        head = [i, j - 3]
        break
      default: {
        // On foot: stood in the queue or walking, skis on the shoulder or the board under an arm;
        // or sidestepping back up to lost skis.
        const stride = o.state !== 'queue' && Math.floor(h.time * 2.5 + o.id) % 2
        if (stride) {
          dot(d, g, i - 1, j, ink.boot)
          dot(d, g, i + 1, j, ink.boot)
        } else dot(d, g, i, j, ink.boot)
        if (!o.kid) dot(d, g, i, j - 1, ink.pants)
        dot(d, g, i, j - tall + 2, coat)
        dot(d, g, i, j - tall + 1, helmet)
        if (o.board) dot(d, g, i + 1, j - 1, gear)
        else if (o.state !== 'collect') {
          dot(d, g, i + 1, j - tall + 1, gear)
          dot(d, g, i + 1, j - tall, gear)
        }
      }
    }
    // Skis and poles where they came off.
    if (o.gear >= 0) {
      const [gi, gj] = at(spot(RUNS[o.run], o.gear, o.gearW))
      dot(d, g, gi - 1, gj, gear)
      dot(d, g, gi + 1, gj - 1, gear)
      dot(d, g, gi, gj - 1, ink.steel)
    }
    heads[`ski${o.id}`] = head
  }
}
