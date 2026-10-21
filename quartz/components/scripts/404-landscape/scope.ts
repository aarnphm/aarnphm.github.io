import type { Palette } from '../../../util/scene-palette'
import type { TrailSeason } from '../../../util/visitor-trail-weather'
import type { Season } from '../../../util/visitor-world'
import type { Almanac, Site, Sky } from '../404-almanac'
import type { Angler, AnglerInk } from '../404-angler'
import type { Bough, Shed } from '../404-boughs'
import type { Cat, CatInk } from '../404-cat'
import type { Crew, CrewInk, Voices } from '../404-crew'
import type { Jay } from '../404-jay'
import type { Curtain, Greens, Overgrowth } from '../404-overgrowth'
import type { Pet, PetInk } from '../404-pet'
import type { Pt } from '../404-pixel'
import type { PlaneInk, Skyway } from '../404-planes'
import type { Park } from '../404-riders'
import type { Hill } from '../404-skiers'
import type { Triathlete, TriInk } from '../404-triathlete'
import type { Walker } from './camp'
import type { Dial, Rim } from './clock-light'
import type { Cypress } from './cypress'
import type { Hearth } from './hearth'
import type { Masonry } from './masonry'
import type { Meadow } from './meadow'
import type { Layer } from './raster'
import type { Relief } from './relief'
import type { Rift } from './rift-motion'
import type { Air, SkyStar } from './sky'
import type { Canopy } from './trees'
import type { Ripple, Splash } from './water'

export type Pointer = { x: number; y: number }

export type Mode = 'idle' | 'hot' | 'enter'

export type Landscape = {
  step(dt: number, clock: number, pointer: Pointer): void
  portal(mode: Mode): void
  splash(x: number, y: number, size: number): void
  // Repaints the layers that carry the clock, for when nothing else is animating them.
  refresh(): void
  // Where the visitor is and their weather, once the worker has said.
  place(site: Site): void
  // Reduced motion: poses the cat and the crew for where the pointer is, repainting only on a change.
  still(pointer: Pointer): void
  // Whoever is under the click (client pixels) asks "?"; true if someone was there.
  hail(x: number, y: number): boolean
  // A press (client pixels) on the cat pats it; true if it was there to be patted.
  pat(x: number, y: number): boolean
  // Whether a hand at a client pixel is on the cat while it can be stroked.
  onCat(x: number, y: number): boolean
  // The keyboard's hand: on the cat while the button over it has focus, stroking it while an arrow
  // is held (-1 left on screen, 1 right, 0 none).
  paw(on: boolean, dir: number): void
  // A press of that button pats the cat where the hand is; true if it was there to be patted.
  patPaw(): boolean
  // Buffer pixel pitch and origin in scene units, for snapping vector sprites onto the grid.
  pixel(): { px: number; x0: number; y0: number } | null
  dispose(): void
}

// The scene's state, shared by setupLandscape and the phases it hands each frame, tick and repaint
// to. setupLandscape fills it in order, so a phase reads only fields set before it first runs.
export type LandscapeScope = {
  scene: HTMLElement
  reduce: boolean
  water: number
  dial: Dial
  site: Site
  skyCanvas: HTMLCanvasElement
  back: HTMLCanvasElement
  riftCanvas: HTMLCanvasElement
  hill: HTMLCanvasElement
  grain: HTMLCanvasElement
  near: HTMLCanvasElement
  front: HTMLCanvasElement
  temple: SVGSVGElement
  rift: Rift
  air: Air
  walker: Walker
  hearth: Hearth
  ripples: Ripple[]
  splashes: Splash[]
  bank: (x: number) => number
  overgrowth: Overgrowth
  curtain: Curtain
  cat: Cat
  crew: Crew
  pet: Pet
  pets: () => boolean
  spotInScene: () => Pt | null
  tri: Triathlete
  far: Voices
  boughs: Voices
  litter: Shed
  bough: Bough | null
  hush: number
  camp: readonly [number, number]
  strand: (x: number) => number
  turn: (r: number) => number
  inOpening: (i: number, j: number) => boolean
  pines: Cypress[]
  pal: Palette
  dark: boolean
  lume: number
  seasonOverride: TrailSeason | null
  al: Almanac
  weather: Sky
  times: Season
  weekend: () => boolean
  day: () => { park: Park; slopes: Hill }
  park: Park
  slopes: Hill
  fair: () => boolean
  angler: Angler
  skyway: Skyway
  relief: Relief
  rim: Rim
  night: boolean
  lightKey: string
  hung: string
  almanacAcc: number
  ranges: Layer | null
  ruin: Layer | null
  sky: Layer | null
  field: Layer | null
  glint: Layer | null
  well: Layer | null
  leaves: Layer | null
  masonry: Masonry | null
  greens: Greens | null
  catInk: CatInk | null
  crewInk: CrewInk | null
  petInk: PetInk | null
  triInk: TriInk | null
  anglerInk: AnglerInk | null
  planeInk: PlaneInk | null
  figures: Uint8ClampedArray<ArrayBuffer>
  people: Uint8ClampedArray<ArrayBuffer>
  landBase: Uint8ClampedArray<ArrayBuffer>
  ruinBase: Uint8ClampedArray<ArrayBuffer>
  skyBase: Uint8ClampedArray<ArrayBuffer>
  hillBase: Uint8ClampedArray<ArrayBuffer>
  ridge: Float32Array<ArrayBuffer>
  ground: Meadow | null
  stars: SkyStar[]
  canopy: Canopy
  canopyKey: string
  jay: Jay
  gust: number
  wind: number
  clock: number
  pullTarget: number
  drawn: boolean
  riftAcc: number
  grown: number
  lastGlow: number
  lastGrowth: number
  minute: number
  rest: Pointer
  stance: string
  purr: Voices
  purred: boolean
  stroking: boolean
  purring: number
  petter: HTMLButtonElement | null
  says: HTMLElement | null
  paw: { on: boolean; dir: number; along: number }
  placed: string
  said: string
  lastImpulse: number
  lastX: number
  lastY: number
  shed: number
  sign: number
  jobs: { run: () => void; every: number; acc: number }[]
}
