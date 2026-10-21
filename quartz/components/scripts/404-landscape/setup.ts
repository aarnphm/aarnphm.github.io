import type { Dial } from './clock-light'
import type { Landscape, LandscapeScope } from './scope'
import { readPetSpot } from '../../../util/pet-spot'
import { LIGHT } from '../../../util/scene-palette'
import { readTrailSeason, subscribeTrailSeason } from '../../../util/trail-season'
import { season } from '../../../util/visitor-world'
import { fireOut } from '../../../util/world-schedule'
import { almanac, readSky, winter, type Site } from '../404-almanac'
import { makeAngler } from '../404-angler'
import { alongCat, makeCat, partUnder, poseCat } from '../404-cat'
import { makeCrew, type Voices } from '../404-crew'
import { GATE } from '../404-gate'
import { makeJay } from '../404-jay'
import { makeCurtain, makeOvergrowth } from '../404-overgrowth'
import { makePet } from '../404-pet'
import { mulberry, smooth, type Pt } from '../404-pixel'
import { makeSkyway } from '../404-planes'
import { daySeed, makePark, parkEnv, posePark, stepPark } from '../404-riders'
import { hillEnv, makeHill, poseHill, stepHill, warmHill } from '../404-skiers'
import { makeTriathlete } from '../404-triathlete'
import { hailLandscape, portalLandscape, stepLandscape } from './frame'
import { hillTop, midRidge } from './geography'
import { handOver, onNear, patAt, placePaw, SAYS, tell } from './hands'
import { stepHearth } from './hearth'
import { paintAir, paintFront, paintMeadow, paintPortal, paintRanges, paintShore } from './layers'
import { CREAM } from './light-marks'
import { moveLandscape } from './place'
import { recheck, repaint } from './repaint'
import { makeRift, orbit } from './rift-motion'
import { makeRipples } from './water'

const TICK = 1 / 10

export function setupLandscape(
  scene: HTMLElement,
  reduce: boolean,
  water: number,
  dial: Dial,
  site: Site,
): Landscape | null {
  const canvas = (cls: string) => scene.querySelector<HTMLCanvasElement>(cls)
  const skyCanvas = canvas('.nf-sky')
  const back = canvas('.nf-land')
  const riftCanvas = canvas('.nf-rift')
  const hill = canvas('.nf-hill')
  const grain = canvas('.nf-grain')
  const near = canvas('.nf-near')
  const front = canvas('.nf-canopy')
  const temple = scene.querySelector<SVGSVGElement>('.nf-temple')
  if (!skyCanvas || !back || !riftCanvas || !hill || !grain || !near || !front || !temple)
    return null
  const st = {
    scene,
    reduce,
    water,
    dial,
    site,
    skyCanvas,
    back,
    riftCanvas,
    hill,
    grain,
    near,
    front,
    temple,
  } as LandscapeScope

  const pole: Pt = [GATE.x, GATE.y]
  st.rift = makeRift(pole)
  st.air = { drift: 0, time: 0, pull: 0, pole, vortices: [], body: null, cover: 0.3 }
  st.walker = { x: 900, dir: 1, pause: 1.5, next: 9, stride: 0, walking: false }
  // Start with the fire already smoking, so the column is standing on the first frame.
  st.hearth = { smoke: [], embers: [], spawn: 0, spark: 0, doused: false }
  for (let n = 0; n < 105; n++) stepHearth(st.hearth, st.air, 1 / 15)
  st.ripples = makeRipples(st.water)
  st.splashes = []
  st.bank = (x: number) => hillTop(x, st.water)
  st.overgrowth = makeOvergrowth()
  st.curtain = makeCurtain()
  st.cat = makeCat(1250, st.bank(1250))
  st.crew = makeCrew()
  st.pet = makePet(st.reduce)
  // The site pet's on/off switch (pet.inline.ts) stands on <body>.
  st.pets = () => document.body.dataset.pets !== 'off'
  st.spotInScene = (): Pt | null => {
    const spot = readPetSpot()
    const g = st.glint?.g
    if (!spot || !g || !st.near) return null
    const r = st.near.getBoundingClientRect()
    if (spot.x < 0 || spot.y < 0 || spot.x > window.innerWidth || spot.y > window.innerHeight)
      return null
    return [
      g.sx[0] + ((spot.x - r.left) / r.width) * g.bw * g.px,
      g.sy[0] + ((spot.y - r.top) / r.height) * g.bh * g.px,
    ]
  }
  st.tri = makeTriathlete(st.reduce)
  // The people on the far ridges talk in their own layer, so their bubbles travel with them.
  st.far = { bubbles: [], heads: {} }
  // And the two up the left-hand tree in theirs.
  st.boughs = { bubbles: [], heads: {} }
  st.litter = { leaves: [], spawn: 0 }
  st.bough = null
  st.hush = 0
  // The triathlete keeps just behind the camp's footing, and follows the bank down to the water
  // where the hill runs out. Past the camp itself they swing up the bank, far enough that their
  // feet clear the tops of the tent and the campers, so the fire is all that crosses them.
  st.camp = [0, 0]
  st.strand = (x: number) =>
    Math.min(st.water - 3, Math.max(844, st.bank(x) + 4)) -
    28 * smooth(st.camp[0] - 60, st.camp[0], x) * (1 - smooth(st.camp[1], st.camp[1] + 60, x))
  // The rift's angular speed at radius r, for whatever it carries round.
  st.turn = (r: number) => (orbit(r) * st.rift.rate * Math.PI) / 180
  st.inOpening = (i: number, j: number) => {
    const o = st.rift.opening
    return (
      !!o && i >= o.i0 && i <= o.i1 && j >= o.j0 && j <= o.j1 && o.mask[j * o.grid.bw + i] === 255
    )
  }

  st.pines = []
  const r = mulberry(612)
  for (const [x0, x1, n] of [
    [140, 460, 4],
    [1140, 1480, 5],
  ] as const)
    for (let k = 0; k < n; k++) {
      const x = x0 + r() * (x1 - x0)
      st.pines.push({ x, base: midRidge(x) + 8, h: 36 + r() * 40, w: 9 + r() * 6 })
    }

  st.pal = LIGHT
  st.dark = false
  st.lume = 0
  st.seasonOverride = readTrailSeason(st.scene.ownerDocument.defaultView)
  const seasonSignal = new AbortController()
  // The visitor's sky: where the sun and moon stand, the weather, the season, and how the land is
  // lit as a result. Recomputed on every repaint and checked once a minute.
  st.al = almanac(new Date(), st.site)
  st.weather = readSky(st.site.weather)
  st.hearth.doused = fireOut(st.weather)
  st.times = season(st.al, st.weather, st.seasonOverride)
  // The bike park's day: who turns up comes from the date, so everyone looking on a given day sees
  // the same riders. With motion it has been open a while by the time anyone looks.
  st.weekend = () => {
    const day = new Date(Date.now() + (st.site.lon / 15) * 3.6e6).getUTCDay()
    return day === 0 || day === 6
  }
  // The ski hill's the same way, with the tracks of the day so far already on its runs; its chair
  // takes two and a half minutes, so it has been running ten by the time anyone looks. Both start
  // again once the worker says where the visitor is: the guess from the time zone can be an hour
  // and more of sun time out, enough to have warmed them up shut.
  st.day = () => {
    const park = makePark(
      daySeed(new Date(), st.site.lon),
      parkEnv(st.al, st.weather, st.weekend()),
    )
    const slopes = makeHill(
      daySeed(new Date(), st.site.lon) + 1,
      hillEnv(st.al, st.weather, st.weekend()),
    )
    warmHill(slopes)
    if (st.reduce) {
      posePark(park)
      poseHill(slopes)
    } else {
      const quiet: Voices = { bubbles: [], heads: {} }
      for (let n = 0; n < 1800; n++) stepPark(park, 1 / 15, quiet)
      for (let n = 0; n < 3000; n++) stepHill(slopes, 1 / 5, quiet)
    }
    return { park, slopes }
  }
  ;({ park: st.park, slopes: st.slopes } = st.day())
  // The farmer's fishing weather: a fair day out of the winter, the sun well up, and the stock out.
  st.fair = () => !winter(st.al) && st.al.sun.alt > 10 && !st.weather.storm && st.park.herd.out
  st.angler = makeAngler(st.fair(), st.reduce)
  st.skyway = makeSkyway()
  st.relief = {
    side: 1,
    contrast: 1,
    snowline: st.times.snowline,
    dark: st.dark,
    frost: st.times.frost,
    thaw: st.times.thaw,
  }
  st.rim = { side: 1, color: CREAM, contrast: 1, night: false, fire: true }
  st.night = false
  st.lightKey = ''
  st.hung = ''
  st.almanacAcc = 0
  st.ranges = null
  st.ruin = null
  st.sky = null
  st.field = null
  st.glint = null
  st.well = null
  st.leaves = null
  st.masonry = null
  st.greens = null
  st.catInk = null
  st.crewInk = null
  st.petInk = null
  st.triInk = null
  st.anglerInk = null
  st.planeInk = null
  st.figures = new Uint8ClampedArray(0)
  st.people = new Uint8ClampedArray(0)
  st.landBase = new Uint8ClampedArray(0)
  st.ruinBase = new Uint8ClampedArray(0)
  st.skyBase = new Uint8ClampedArray(0)
  st.hillBase = new Uint8ClampedArray(0)
  st.ridge = new Float32Array(0)
  st.ground = null
  st.stars = []
  st.canopy = { trees: [], roots: [] }
  st.canopyKey = ''
  // The homepage's jay, in an edge tree, and the woodpecker that drums there while it is away.
  st.jay = makeJay()
  st.gust = 0
  st.wind = 0
  st.clock = 0
  st.pullTarget = 0
  st.drawn = false
  st.riftAcc = 0
  st.grown = st.reduce ? Infinity : 0
  st.lastGlow = -1
  st.lastGrowth = -1
  st.minute = NaN
  st.rest = { x: -1e4, y: -1e4 }
  st.stance = ''
  // The cat purrs a heart over its head. The friend who has been holding out a fish to it all along
  // asks about the first one, and calls out at a swat. Over a cat that can be stroked, the cursor
  // is a hand.
  st.purr = { bubbles: [], heads: {} }
  st.purred = false
  st.stroking = false
  st.purring = 0
  // The keyboard's hand, from the button over the cat (404.tsx). While the button has focus the
  // hand rests `along` the cat's line (alongCat), and a held arrow moves it toward the tail or the
  // head. The button's focus ring follows it, and its status line says what the cat does.
  st.petter = st.scene.querySelector<HTMLButtonElement>('.nf-pet')
  st.says = st.scene.querySelector<HTMLElement>('.nf-pet-says')
  st.paw = { on: false, dir: 0, along: 0 }
  st.placed = ''
  st.said = ''
  st.lastImpulse = -Infinity
  st.lastX = NaN
  st.lastY = NaN
  st.shed = 0
  st.sign = 1

  const resize = () => repaint(st)
  resize()
  window.addEventListener('resize', resize)
  subscribeTrailSeason(
    st.scene.ownerDocument,
    next => {
      if (next === st.seasonOverride) return
      st.seasonOverride = next
      repaint(st)
    },
    seasonSignal.signal,
  )

  // At most one stop-motion layer repaints per animation frame, whichever is most overdue, so the
  // ticks interleave instead of landing on the same frame.
  st.jobs = [
    { run: () => paintAir(st), every: 1 / 8, acc: 0 },
    { run: () => paintMeadow(st), every: TICK, acc: 0.03 },
    { run: () => paintShore(st), every: TICK, acc: 0.06 },
    { run: () => paintFront(st), every: TICK, acc: 0.09 },
    { run: () => paintRanges(st), every: TICK, acc: 0.05 },
  ]

  return {
    step(dt, now, pointer) {
      stepLandscape(st, dt, now, pointer)
    },
    portal(mode) {
      portalLandscape(st, mode)
    },
    splash(x, y, size) {
      st.splashes.push({ x, y, size, age: 0 })
    },
    still(pointer) {
      st.rest = pointer
      handOver(st, !!partUnder(st.cat, pointer.x, pointer.y))
      poseCat(st.cat, pointer, st.drawn)
      const next = `${st.cat.alert}${st.cat.look}`
      if (next === st.stance) return
      st.stance = next
      paintShore(st)
    },
    refresh() {
      if ((st.almanacAcc += 1) >= 60) {
        st.almanacAcc = 0
        recheck(st)
      }
      paintPortal(st)
      paintShore(st)
    },
    place(next) {
      moveLandscape(st, next)
    },
    hail(x, y) {
      return hailLandscape(st, x, y)
    },
    pat(x, y) {
      const spot = onNear(st, x, y)
      return !!spot && patAt(st, spot)
    },
    paw(on, dir) {
      // Each new focus starts on its cheek with a fresh status line.
      if (on && !st.paw.on) {
        st.paw.along = 0
        st.said = ''
        if (st.says) st.says.textContent = ''
      }
      st.paw.on = on
      st.paw.dir = on ? dir : 0
      if (!on) return
      // Without motion nothing strokes, and an arrow moves the hand to the next of its face, its
      // back and its haunch.
      if (st.reduce && dir)
        st.paw.along = Math.max(0, Math.min(1, st.paw.along + dir * -st.cat.face * 0.5))
      placePaw(st)
    },
    patPaw() {
      if (patAt(st, alongCat(st.cat, st.paw.along))) return true
      tell(st, SAYS.busy)
      return false
    },
    onCat(x, y) {
      const spot = onNear(st, x, y)
      return !!spot && !!partUnder(st.cat, ...spot)
    },
    pixel() {
      const g = st.glint?.g
      return g ? { px: g.px, x0: g.sx[0] - g.px / 2, y0: g.sy[0] - g.px / 2 } : null
    },
    dispose() {
      seasonSignal.abort()
      window.clearTimeout(st.hush)
      window.clearTimeout(st.purring)
      window.removeEventListener('resize', resize)
      st.ranges = st.ruin = st.sky = st.field = st.glint = st.well = st.leaves = null
      st.masonry = null
      st.ground = null
      st.figures =
        st.people =
        st.landBase =
        st.ruinBase =
        st.skyBase =
        st.hillBase =
          new Uint8ClampedArray(0)
      st.ridge = new Float32Array(0)
    },
  }
}
