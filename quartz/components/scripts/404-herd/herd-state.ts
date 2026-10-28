import type { Pt } from '../404-pixel'
import { UNIT } from '../404-downhill'

export type HerdEnv = {
  // Out on the pasture (not in winter, and only while the sun is up), the solar hour, and whether
  // it is hot enough to lie up through the day.
  out: boolean
  hour: number
  hot: boolean
}

export type Kind = 'cow' | 'calf' | 'sheep'
type Doing = 'graze' | 'walk' | 'look' | 'lie' | 'run' | 'eat'

export type Beast = {
  kind: Kind
  // Coat and head as indices into the inks, and which pixels of the body carry the second colour.
  coat: number
  head: number
  spots: number
  // A calf's dam, by its place in the herd.
  dam: number
  x: number
  y: number
  face: number
  doing: Doing
  t: number
  until: number
  to: Pt | null
  // What it is looking at; someone hurt it has come over to stand and stare at; seconds before a
  // passing bike can startle it again.
  at: Pt | null
  watch: Pt | null
  calm: number
  // In the barn, and the rest of its way into it or out of it to the grass.
  in: boolean
  path: Pt[]
  // The place at the trough it is headed for or eating at, and the last feed it had there.
  spot: number
  fed: number
}

// The farmer: at the house, on the way to a point and what to do there, at the barn's doors swinging
// them a leaf at a time, stood clear of them while the stock go out or come in, making up feed in
// the house, or tipping it into the trough.
export type Chore = 'home' | 'walk' | 'doors' | 'watch' | 'make' | 'pour'
export type Farmer = {
  chore: Chore
  after: Chore
  way: Pt[]
  x: number
  y: number
  face: number
  t: number
}

// Whether the herd has been called out or in, the stock's seconds since through open doors, how
// many leaves of the doors stand open, the farmer, and the feed in the trough (1 full) and how many
// times it has been filled.
export type Herd = {
  seed: number
  rng: () => number
  beasts: Beast[]
  out: boolean
  since: number
  doors: number
  farmer: Farmer
  trough: number
  feeds: number
}

// Someone on the hill as the herd sees them: where they are, whether they are going by fast on a
// bike, and whether they are lying hurt.
export type Passer = { x: number; y: number; fast: boolean; hurt: boolean }

// Who the herd is: two black-and-white cows, a red one with a white face, a black one, a calf each
// for the first and the red one, and the three ewes. Coats are white, black, red.
export const STOCK: Pick<Beast, 'kind' | 'coat' | 'head' | 'spots' | 'dam'>[] = [
  { kind: 'cow', coat: 0, head: 1, spots: 0b10000110, dam: -1 },
  { kind: 'cow', coat: 0, head: 1, spots: 0b00111001, dam: -1 },
  { kind: 'cow', coat: 2, head: 0, spots: 0, dam: -1 },
  { kind: 'cow', coat: 1, head: 1, spots: 0, dam: -1 },
  { kind: 'calf', coat: 0, head: 1, spots: 0b010, dam: 0 },
  { kind: 'calf', coat: 2, head: 0, spots: 0, dam: 2 },
  { kind: 'sheep', coat: 0, head: 1, spots: 0, dam: -1 },
  { kind: 'sheep', coat: 0, head: 1, spots: 0, dam: -1 },
  { kind: 'sheep', coat: 0, head: 1, spots: 0, dam: -1 },
]

// The barn on the open slope right of the tall cypress (the middle of its front, at the foot of the
// wall), and where a beast stands at its doors to go in or come out.
export const BARN: Pt = [460, 674]
export const DOOR: Pt = [460, 678]
// The farmer's way: from the house along the slope below the island's brow, where the island hides
// a standing figure at any parallax, to stand clear of the doors on their right, and in front of
// them. A leaf takes this many seconds to swing.
export const HOUSE: Pt = [548, 686]
export const ASIDE: Pt = [484, 680]
export const FRONT: Pt = [460, 680]
export const STRIDE = 1.1 * UNIT
export const SWING = 0.8
// The trough by the mouth of the lane, left of the cypress, with a place for a beast at either
// end; the way out to it along the lane, where the farmer stands behind it to tip the feed in; the
// seconds it takes to make the feed up, and the beast-seconds of eating a trough holds.
export const TROUGH: Pt = [356, 686]
export const PLACES: Pt[] = [
  [TROUGH[0] - 20, TROUGH[1]],
  [TROUGH[0] + 20, TROUGH[1]],
]
export const MOUTH: Pt = [400, 682]
export const BEHIND: Pt = [TROUGH[0], TROUGH[1] - 5]
export const MAKE = 12
export const HOLDS = 110
