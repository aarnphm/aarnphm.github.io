import type { Path, Track } from '../404-downhill'
import type { Herd, HerdEnv, HerdInk } from '../404-herd'
import type { RGB } from '../404-pixel'

// Seconds a trick takes in the air, start to finish, easiest first.
export const DUR = { nohander: 0.36, whip: 0.42, superman: 0.5 } as const
type Trick = keyof typeof DUR
export const TRICKS = Object.keys(DUR) as Trick[]

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

// A way on foot: legs along the paths or up and down a line, each from one distance to another at
// its own pace, and how far along the current one the walker has got.
export type Leg = { on: Path | Track; from: number; to: number; v: number }
export type Walk = { legs: Leg[]; n: number; u: number }

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
export const medic = (): Medic => ({
  state: 'post',
  t: 0,
  until: 0,
  walk: null,
  who: null,
  board: false,
})

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
