import type { Layer } from './raster'
import type { LandscapeScope } from './scope'
import { paintAngler } from '../404-angler'
import { paintBough, paintShed, workFor, type BoughInk } from '../404-boughs'
import { catHead, paintCat } from '../404-cat'
import { paintBubbles, paintCrew } from '../404-crew'
import { paintJay, type JayInk } from '../404-jay'
import { GROWN, paintCurtain, paintOvergrowth } from '../404-overgrowth'
import { paintPet, paintSpeck } from '../404-pet'
import { col, INK, mix, putRgb, row, smooth, type Grid } from '../404-pixel'
import { paintPlanes } from '../404-planes'
import { headPixel, paintAscent, paintTriathlete } from '../404-triathlete'
import { CAMPERS, FIGURE, FIRE, paintCamp, paintPool, SITTER } from './camp'
import { paintCanopy, paintDapple } from './canopy'
import { paintHands, paintSpill, relight } from './clock-light'
import { paintFolk } from './folk'
import { onTrail, TRAIL_LEN } from './geography'
import { paintHearth } from './hearth'
import { CREAM, GOLDEN } from './light-marks'
import { paintField } from './meadow'
import { paintRift } from './rift-paint'
import { cloudAt, hang, makeStars, paintBolt, paintSky, paintSkyBase } from './sky'
import { paintFall, paintGlitter, paintRings, paintSplashes, paintWater } from './water'

function flush(l: Layer) {
  return l.ctx.putImageData(l.img, 0, 0)
}

export function paintRanges(st: LandscapeScope) {
  if (!st.ranges) return
  const d = st.ranges.img.data
  paintFolk(
    d,
    st.ranges.g,
    st.pal,
    st.landBase,
    st.clock,
    st.dark,
    st.far.heads,
    st.park,
    st.slopes,
  )
  st.far.heads.athlete =
    (st.triInk &&
      paintAscent(d, st.ranges.g, st.tri, st.triInk, st.clock, u => onTrail(u * TRAIL_LEN))) ||
    undefined
  st.far.heads.pet =
    (st.petInk && st.pets() && paintSpeck(d, st.ranges.g, st.pet, st.petInk, st.clock)) || undefined
  paintBubbles(d, st.ranges.g, st.far, CREAM, INK)
  flush(st.ranges)
}

// The printed ruin only changes while the moss grows or the spill steps up or down.
export function paintRuin(st: LandscapeScope) {
  if (!st.ruin || !st.masonry) return
  const glow = Math.round(st.rift.heat * 4) / 4
  const growth = Math.min(Math.floor(st.grown * 8) / 8, GROWN)
  if (glow === st.lastGlow && growth === st.lastGrowth) return
  st.lastGlow = glow
  st.lastGrowth = growth
  const d = st.ruin.img.data
  d.set(st.ruinBase)
  paintSpill(d, st.masonry, glow, st.dark)
  if (st.greens) paintOvergrowth(d, st.ruin.g, st.overgrowth, growth, st.greens)
  flush(st.ruin)
}

export function paintAir(st: LandscapeScope) {
  if (!st.sky) return
  paintSky(st.sky.img.data, st.sky.g, st.pal, st.air, st.skyBase, st.ridge, st.stars, st.clock)
  if (st.planeInk)
    paintPlanes(
      st.sky.img.data,
      st.sky.g,
      st.skyway,
      st.planeInk,
      st.clock,
      st.night,
      st.air.drift,
      st.weather.cover,
      (x, y) => cloudAt(st.air, x, y)[0],
    )
  if (st.weather.storm && !st.reduce) paintBolt(st.sky.img.data, st.sky.g, st.clock, st.ridge)
  flush(st.sky)
}

export function paintMeadow(st: LandscapeScope) {
  if (!st.field || !st.ground) return
  paintField(
    st.field.img.data,
    st.field.g,
    st.pal,
    st.water,
    st.ground,
    st.hillBase,
    st.clock,
    st.gust,
  )
  flush(st.field)
}

// Water, the figures, camp and smoke. The cat, the crew and the triathlete are painted apart
// first, so the water can reflect them with everything else as it all stands right now. The
// swimmer lies below the waterline, where the reflection never samples, and comes in with the
// rest of the figures over the water.
export function paintShore(st: LandscapeScope) {
  if (!st.glint) return
  const { g } = st.glint
  const d = st.glint.img.data
  const ink = st.masonry?.print.ink ?? st.pal.figure.K
  const same = (l: Layer | null) => (l && l.g.bw === g.bw && l.g.bh === g.bh ? l.img.data : null)
  st.figures.fill(0)
  if (st.catInk) paintCat(st.figures, g, st.cat, st.catInk, st.clock, st.bank)
  const [catX, catY] = catHead(st.cat)
  st.purr.heads.cat = [col(g, catX), row(g, catY - 34)]
  st.crew.heads.pet =
    (st.petInk && st.pets() && paintPet(st.figures, g, st.pet, st.petInk, st.clock)) || undefined
  // The people are lit apart from the cat, which is stone and carries its own modelling.
  st.people.fill(0)
  if (st.crewInk)
    paintCrew(st.people, g, st.crew, st.crewInk, st.clock, st.rift.heat, st.cat.alert, st.bank)
  if (st.triInk) paintTriathlete(st.people, g, st.tri, st.triInk, st.strand, st.water)
  st.crew.heads.angler =
    (st.anglerInk && paintAngler(st.people, g, st.angler, st.anglerInk, st.clock, st.bank)) ??
    undefined
  relight(st.people, g, st.rim, st.rift.heat)
  for (let k = 3; k < st.people.length; k += 4)
    if (st.people[k] === 255)
      putRgb(st.figures, k - 3, st.people[k - 3], st.people[k - 2], st.people[k - 1])
  d.fill(0)
  paintPool(d, g, st.pal, st.water, st.clock, st.dark, st.hearth.doused)
  const layers = [st.figures, same(st.ruin), same(st.field), same(st.well), same(st.ranges)]
  paintWater(d, g, st.pal, st.water, ink, layers, st.ripples, st.clock, st.dark, st.hearth.doused)
  paintGlitter(d, g, st.pal, st.water, st.rift.heat, st.clock, st.air.body)
  paintRings(d, g, st.water, st.weather, st.clock, mix(st.pal.water[2], st.pal.glint[0], 0.6))
  paintSplashes(d, g, st.splashes, ink)
  for (let k = 3; k < d.length; k += 4)
    if (st.figures[k] === 255)
      putRgb(d, k - 3, st.figures[k - 3], st.figures[k - 2], st.figures[k - 1])
  paintCamp(d, g, st.pal, st.water, st.walker, st.clock, st.dark, st.hearth.doused)
  paintHearth(d, g, st.pal, st.hearth, st.clock)
  // Everyone else on the bank can be spoken to as well.
  const foot = row(g, FIRE.y)
  CAMPERS.forEach((c, n) => (st.crew.heads[`camper${n}`] = [col(g, c.x), foot - SITTER.length + 1]))
  st.crew.heads.walker = [
    col(g, st.walker.x),
    row(g, Math.max(848, st.bank(st.walker.x) + 8)) - FIGURE.length,
  ]
  st.crew.heads.athlete = headPixel(st.tri, g, st.strand) ?? undefined
  paintBubbles(d, g, st.crew, CREAM, INK)
  paintBubbles(d, g, st.purr, CREAM, INK)
  flush(st.glint)
}

// Only the opening is ever painted, and the underpainting covers all of it each frame.
export function paintPortal(st: LandscapeScope) {
  if (!st.well) return
  paintRift(st.well.img.data, st.well.g, st.rift, st.clock)
  if (st.masonry) paintHands(st.well.img.data, st.well.g, st.dial, st.masonry.print, st.lume)
  paintCurtain(st.well.img.data, st.well.g, st.curtain, st.grown, st.inOpening)
  flush(st.well)
}

export function paintFront(st: LandscapeScope) {
  if (!st.leaves) return
  const d = st.leaves.img.data
  d.fill(0)
  paintCanopy(
    d,
    st.leaves.g,
    st.canopy,
    st.pal,
    st.wind,
    st.clock * 0.05,
    st.relief.side,
    st.times,
    st.weather,
  )
  paintShed(
    d,
    st.leaves.g,
    st.litter,
    [st.pal.autumn[1], st.pal.autumn[2], st.pal.autumn[3], st.pal.crown[2]],
    mix(st.pal.snow[3], st.pal.poppy, 0.25),
  )
  st.boughs.heads = {}
  if (st.bough)
    paintBough(d, st.leaves.g, st.bough, workFor(st.times), boughInk(st), st.clock, st.boughs.heads)
  paintJay(d, st.leaves.g, st.jay, jayInk(st), st.clock, !st.night && !st.weather.storm)
  paintDapple(d, st.leaves.g, st.pal, st.canopy.roots, st.wind * 0.6, st.clock, st.water)
  const streak = st.dark
    ? mix(st.pal.water[2], st.pal.glint[0], 0.35)
    : mix(st.pal.cloud[0], st.pal.far[0], 0.4)
  paintFall(d, st.leaves.g, st.weather, st.clock, streak, st.dark ? st.pal.snow[3] : CREAM)
  paintBubbles(d, st.leaves.g, st.boughs, CREAM, INK)
  flush(st.leaves)
}

function boughInk(st: LandscapeScope): BoughInk {
  return {
    skin: st.pal.figure.S,
    hat: st.pal.crew[3],
    coats: [st.pal.crew[1], st.pal.crew[2]],
    trousers: st.pal.figure.D,
    boots: st.pal.figure.K,
    harness: st.pal.crew[0],
    paper: CREAM,
    apple: st.pal.poppy,
    steel: mix(st.pal.rock[2], CREAM, 0.3),
    rope: st.pal.rope,
    bark: st.pal.bark[0],
    dust: st.pal.bark[2],
    lights: [GOLDEN, st.pal.poppy, st.pal.cornflower, CREAM],
    glow: st.pal.glow[1],
    steam: st.pal.smoke[1],
    dark: st.dark,
  }
}

function jayInk(st: LandscapeScope): JayInk {
  return {
    blue: st.pal.cornflower,
    bar: st.pal.figure.D,
    ink: st.pal.figure.K,
    white: st.pal.figure.C,
    belly: st.pal.smoke[2],
    nape: st.pal.poppy,
    acorn: st.pal.autumn[0],
  }
}

// What comes down out of the trees: the leaves that are left in autumn, blossom in spring, and
// whatever the arborist shakes loose. It blows off the edge of the view or falls out of the bottom.
export function shedding(st: LandscapeScope, g: Grid) {
  return {
    leaves: st.times.autumn * (1 - st.times.bare),
    blossom: st.times.blossom * (1 - st.times.bare),
    wind: Math.min(1, st.weather.wind + st.gust * 0.2),
    shake: st.bough && workFor(st.times) === 'shake' ? st.bough.anchor : null,
    box: [g.sx[0], g.sy[0], g.sx[g.bw - 1], g.sy[g.bh - 1] + 10] as [
      number,
      number,
      number,
      number,
    ],
  }
}

// The sun by day, the moon by night if it is up, hung over the sky layer, with the sky's base
// coat and stars under it. Cheap, so the body can move on without a full repaint.
export function hangBody(st: LandscapeScope, force: boolean) {
  if (!st.sky) return
  const sunUp = st.al.day > 0.2
  const up = sunUp ? st.al.sun : st.al.moon.alt > -3 ? st.al.moon : null
  const r = sunUp ? 46 : 40
  const veil = st.weather.fog ? 1 : smooth(0.55, 0.95, st.weather.cover)
  const phase = sunUp ? 0.5 : st.al.moon.phase
  st.air.body = up ? { ...hang(up, st.sky.g, r), r, phase, mirror: st.al.south, veil } : null
  st.air.cover = st.weather.cover
  const b = st.air.body
  const at = b ? `${col(st.sky.g, b.x)},${row(st.sky.g, b.y)}` : ''
  if (!force && at === st.hung) return false
  st.hung = at
  let floor = 0
  for (const y of st.ridge) floor = Math.max(floor, y)
  const len = st.sky.img.data.length
  if (st.skyBase.length !== len) st.skyBase = new Uint8ClampedArray(len)
  else st.skyBase.fill(0)
  paintSkyBase(st.skyBase, st.sky.g, st.pal, floor + 30, b)
  st.stars = st.night ? makeStars(b, st.weather.cover) : []
  return true
}
