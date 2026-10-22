import type { Pt, RGB } from '../404-pixel'

// The far range stands some eight times as far off as the island, so a metre there is 1.2 scene
// units. Its people are drawn three or four pixels tall all the same, several times their size,
// since a pixel is the least a person can be.
export const M = 1.2
export const G = 9.81

export const LIFT_FOOT: Pt = [1322, 450]
export const LIFT_HEAD: Pt = [1398, 282]
// The chair's line along the slope, the speed of an old fixed-grip double, and its chairs: an
// even number round the loop, so the gap between them divides it.
export const LIFT_LEN = Math.hypot(
  (LIFT_HEAD[0] - LIFT_FOOT[0]) / M,
  (LIFT_FOOT[1] - LIFT_HEAD[1]) / M / Math.sin((25 * Math.PI) / 180),
)
export const LIFT_V = 2.3
export const CHAIRS = 14
export const GAP = (2 * LIFT_LEN) / CHAIRS
// Where the queue stands, and the way in and out over the mid ridge's crest.
export const QUEUE = (n: number): Pt => [1316 - 7 * n, 452]
export const WAY: Pt[] = [
  [1300, 492],
  [1310, 457],
]
// The groomer's spot by day, beside the lift's foot.
export const SHED: Pt = [1306, 452]
// The start zones under the crest that the patrol shoots on a powder morning, one left of the head
// station above the lift line and two right of it above the blue: where the patroller stands to
// throw, where the charge lands, and where they wait for it behind a rib.
export const ZONES: { stand: Pt; lands: Pt; hide: Pt }[] = [
  { stand: [1378, 309], lands: [1368, 327], hide: [1390, 297] },
  { stand: [1438, 263], lands: [1447, 281], hide: [1424, 272] },
  { stand: [1468, 245], lands: [1478, 263], hide: [1454, 253] },
]
// A hand charge: the igniter pulled, the charge thrown, and a ninety-second fuse.
export const LIGHT = 2
export const THROW = 4
export const BOOM = LIGHT + 90
// A start zone pitches at about 38°, so a metre down it is 0.74 scene units down the screen. A slide
// gets away at 4 m/s² and runs on at 12 m/s until it runs out of slope.
export const DOWN = M * Math.sin((38 * Math.PI) / 180)
export const slid = (t: number, length: number) =>
  Math.min(length, t < 3 ? 2 * t * t : 18 + 12 * (t - 3))
export const stopped = (length: number) =>
  length <= 18 ? Math.sqrt(length / 2) : 3 + (length - 18) / 12

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
export const RUNS = [
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
export const seek = (run: Run, s: number, k: number) => {
  while (k < run.n - 2 && run.s[k + 1] <= s) k++
  while (k > 0 && run.s[k] > s) k--
  return k
}
// Where a point of a run is on the screen: across the run is across the screen.
export function spot(run: Run, s: number, w: number): Pt {
  const k = seek(run, s, 0)
  const u = Math.max(0, Math.min(1, (s - run.s[k]) / (run.s[k + 1] - run.s[k])))
  return [
    run.x[k] + (run.x[k + 1] - run.x[k]) * u + w * M,
    run.y[k] + (run.y[k + 1] - run.y[k]) * u,
  ]
}

// Tracks, per run, in a coarse grid of how many have passed and when one last did.
export const ROWS = 64
export const COLS = 12
export const cell = (run: Run, s: number, w: number, k: number) => {
  const r = Math.min(ROWS - 1, Math.max(0, Math.floor((s / run.length) * ROWS)))
  const c = Math.min(COLS - 1, Math.max(0, Math.floor(((w / run.half[k] + 1) / 2) * COLS)))
  return r * COLS + c
}

// The summer race camp's giant slalom on the blue: eight gates twenty-six metres apart, left and
// right, the first a couple of turns out of the start. Slalom poles eleven metres apart would stand
// a pixel apart from across the valley.
export const GATES = Array.from({ length: 8 }, (_, n) => ({
  s: 22 + 26 * n,
  w: n % 2 ? 4.5 : -4.5,
}))
export const FINISH = GATES[GATES.length - 1].s + 8

export type Role = 'free' | 'teach' | 'pupil' | 'racer' | 'coach' | 'tour' | 'patrol'
export type Mode = 'carve' | 'skid' | 'plough' | 'leaf' | 'race'
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
  | 'skin'
  | 'hurt'
  | 'rescue'
  | 'tend'
  | 'sled'
  | 'haul'
  | 'route'
  | 'charge'

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
  // A patroller's casualty, or on a powder morning the start zone they are shooting (-1 otherwise).
  patient: Skier | null
  shot: number
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
  // Out of hours, how many locals are skinning up to ride it down.
  tour: number
  // A powder morning before the chair opens: the patrol is shooting the start zones.
  control: boolean
}

// A charge gone off: where, how long ago, and how far the slope it released ran, in metres (0 when
// the slope held).
type Blast = { x: number; y: number; t: number; slide: number }

export type Hill = {
  seed: number
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
  blasts: Blast[]
}

export const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

// The skin track, up the edge of the blue out of everyone's way: the lift line's side, where the
// trees on the mid ridge leave it in view.
export const lane = (run: Run, k: number) => 1.5 - run.half[k]

// The crowd the day brings, apart from the camp and the uphillers, who come on their own terms.
export const free = (o: Skier) =>
  o.state !== 'leave' && (o.role === 'free' || o.role === 'teach' || o.role === 'pupil')

export const onRun = (o: Skier) =>
  o.state === 'ski' ||
  o.state === 'down' ||
  o.state === 'up' ||
  o.state === 'skin' ||
  o.state === 'hurt' ||
  o.state === 'rescue' ||
  o.state === 'tend' ||
  o.state === 'sled'

export function mark(h: Hill, o: Skier, run: Run) {
  const c = cell(run, o.s, o.w, o.k)
  if (c === o.cell) return
  o.cell = c
  const t = h.tracks[o.run]
  t[c] = Math.min(255, t[c] + 1)
  h.last[o.run][c] = h.time
}

export const spotW = (o: Skier) => (o.board ? -RUNS[o.run].half[0] * 0.6 : 0)

export const laneW = (lane: number, half: number) => (lane - 1) * (half - 3)

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
  // Boards, in louder colours than skis, so a boarder reads as one from across the valley.
  boards: RGB[]
  school: RGB
  gates: [RGB, RGB]
  cat: RGB
  cab: RGB
  lamp: RGB
  glow: RGB
  // Floodlit snow: the night glow is a brown that reads as dirt on it.
  flood: RGB
  // The patrol's jackets and its toboggan.
  patrol: RGB
  sled: RGB
  beacon: RGB
  // A charge's smoke, dark against the snow it throws up.
  smoke: RGB
}
