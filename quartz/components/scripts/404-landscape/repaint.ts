import type { LandscapeScope } from './scope'
import { lighting } from '../../../util/scene-palette'
import { season } from '../../../util/visitor-world'
import { fireOut } from '../../../util/world-schedule'
import { almanac, readSky, type Almanac } from '../404-almanac'
import { layAngler } from '../404-angler'
import { findBough, stepShed } from '../404-boughs'
import { perchJay } from '../404-jay'
import { INK, mix, smooth, type Pt, type RGB, type Tones } from '../404-pixel'
import { parkEnv, posePark } from '../404-riders'
import { hillEnv, poseHill } from '../404-skiers'
import { layCourse } from '../404-triathlete'
import { campSpan } from './camp'
import { skyline, TRAIL } from './geography'
import { paintLand } from './land'
import {
  hangBody,
  paintAir,
  paintFront,
  paintMeadow,
  paintPortal,
  paintRanges,
  paintRuin,
  paintShore,
  shedding,
} from './layers'
import { CREAM, FLAME, GOLDEN, ROSE } from './light-marks'
import { rasterRuin } from './masonry'
import { meadow, paintHillBase } from './meadow'
import { layer } from './raster'
import { growTree } from './trees'

// Everything about the light that needs the layers repainted: the palette's step through the
// day, the twilight glow, which side the light comes from, the snowline, and the frost's retreat.
function keyFor(st: LandscapeScope, a: Almanac) {
  const source = a.day > 0.2 ? a.sun : a.moon
  const s = season(a, st.weather, st.seasonOverride)
  return [
    Math.round(a.day * 16),
    Math.round(a.dusk * 8),
    source.across >= 0,
    a.day > 0.2,
    Math.round(s.snowline / 20),
    s.frost > 0.02 ? Math.round(s.thaw * 12) : -1,
  ].join('|')
}

export function repaint(st: LandscapeScope) {
  st.al = almanac(new Date(), st.site)
  st.weather = readSky(st.site.weather)
  st.hearth.doused = fireOut(st.weather)
  st.times = season(st.al, st.weather, st.seasonOverride)
  st.park.env = parkEnv(st.al, st.weather, st.weekend())
  st.slopes.env = hillEnv(st.al, st.weather, st.weekend())
  if (st.reduce) {
    posePark(st.park)
    poseHill(st.slopes)
  }
  st.lightKey = keyFor(st, st.al)
  const [lit, day] = lighting(st.al, st.times, st.weather)
  st.pal = lit
  st.dark = day < 0.5
  st.lume = 1 - smooth(0.25, 0.6, day)
  st.night = st.al.day < 0.35
  st.scene.dataset.night = String(st.night)
  const sunUp = st.al.day > 0.2
  const side = (sunUp ? st.al.sun : st.al.moon).across >= 0 ? 1 : -1
  const over = smooth(0.55, 1, st.weather.cover) + (st.weather.fog ? 0.3 : 0)
  const moonlit = 1 - Math.abs(st.al.moon.phase - 0.5) * 2
  const contrast = Math.max(0.25, 1 - 0.65 * over) * (sunUp ? 1 : 0.35 + 0.4 * moonlit)
  st.relief = {
    side,
    contrast,
    snowline: st.times.snowline,
    dark: st.dark,
    frost: st.times.frost,
    thaw: st.times.thaw,
  }
  st.rim = {
    side,
    color: mix(st.pal.sun[0], CREAM, 0.5),
    contrast,
    night: !sunUp,
    fire: !st.hearth.doused,
  }
  st.ranges = layer(st.back, st.ranges)
  if (st.ranges) {
    paintLand(st.ranges.img.data, st.ranges.g, st.pal, st.pines, st.relief)
    if (st.landBase.length !== st.ranges.img.data.length) st.landBase = st.ranges.img.data.slice()
    else st.landBase.set(st.ranges.img.data)
  }
  st.ruin = layer(st.grain, st.ruin)
  // The stones stay paper-coloured by day and go grey under the moon.
  const stone = 0.5 + 0.5 * day
  st.masonry = st.ruin && rasterRuin(st.ruin.img.data, st.ruin.g, st.temple, st.bank, side, stone)
  if (st.ruin) {
    if (st.ruinBase.length !== st.ruin.img.data.length) st.ruinBase = st.ruin.img.data.slice()
    else st.ruinBase.set(st.ruin.img.data)
  }
  if (st.masonry) {
    const { print } = st.masonry
    const tones = (stops: RGB[]): Tones => ({
      ink: print.ink,
      light: stops[stops.length - 1],
      plate: stops[stops.length - 2],
      shade: stops[stops.length - 3],
    })
    const flowers = [print.rose[3], CREAM]
    st.greens = {
      moss: tones(st.pal.moss),
      bark: tones(st.pal.bark),
      // The small tree on the gate turns early, being exposed up there.
      crown: tones(st.times.autumn > 0.45 ? st.pal.autumn : st.pal.crown),
      flowers,
      hair: st.pal.bark[st.dark ? 2 : 0],
    }
    // A British Blue with copper eyes: the ruin's stone with a slate cast, so it greys with the
    // ruin under the moon.
    const slate = ([r, g, b]: RGB): RGB => [r * 0.68, g * 0.765, b * 0.91]
    st.catInk = {
      stone: tones(print.stone.map(slate)),
      moss: tones(st.pal.moss),
      eye: mix(GOLDEN, st.pal.poppy, 0.4),
      nose: print.rose[2],
      ear: print.rose[st.dark ? 1 : 2],
      flowers,
      fish: st.pal.fish,
    }
  }
  // The pet keeps the page's paper and the print's ink, as the site draws it.
  const petLine = st.masonry?.print.ink ?? INK
  st.petInk = {
    body: {
      ink: petLine,
      light: CREAM,
      plate: mix(CREAM, st.pal.rock[1], 0.12),
      shade: mix(CREAM, petLine, 0.3),
    },
    ink: petLine,
  }
  st.crewInk = {
    skin: st.pal.figure.S,
    trousers: st.pal.figure.D,
    boots: st.pal.figure.K,
    hat: st.pal.ochre,
    coats: st.pal.crew,
    wood: st.pal.rope,
    paper: st.pal.figure.C,
    gold: GOLDEN,
    rose: ROSE,
    fish: st.pal.fish,
    flame: FLAME[2],
    glow: st.pal.glow[2],
    ink: INK,
    dark: st.dark,
  }
  st.triInk = {
    skin: st.pal.figure.S,
    hair: st.pal.bark[1],
    shoes: st.pal.figure.K,
    paper: st.pal.figure.C,
    suits: st.pal.crew,
    caps: [FLAME[2], GOLDEN, ROSE, CREAM],
    disc: mix(st.pal.figure.K, st.pal.figure.C, 0.45),
    rack: st.pal.rope,
    deep: mix(st.pal.water[0], INK, st.dark ? 0.5 : 0.3),
    foam: st.pal.glint[0],
    wake: st.dark ? mix(st.pal.water[2], st.pal.glint[0], 0.4) : mix(st.pal.water[0], INK, 0.25),
    flag: st.pal.poppy,
    dust: mix(st.pal.mid[2], CREAM, st.dark ? 0.3 : 0.6),
  }
  // The farmer as on the hill, and the boat in the barn's red.
  const hull = mix(st.pal.poppy, st.pal.autumn[0], 0.5)
  st.anglerInk = {
    hat: GOLDEN,
    skin: st.pal.figure.S,
    shirt: st.pal.figure.C,
    bib: st.pal.figure.B,
    trousers: st.pal.figure.D,
    boots: st.pal.figure.K,
    hull,
    keel: mix(hull, INK, 0.35),
    wood: st.pal.rope,
    rod: st.pal.bark[0],
    line: st.dark ? mix(st.pal.glint[0], st.pal.water[2], 0.5) : mix(CREAM, st.pal.rock[1], 0.35),
    float: st.pal.poppy,
    cork: CREAM,
    fish: st.pal.fish,
    deep: st.triInk.deep,
    foam: st.pal.glint[0],
  }
  st.planeInk = {
    body: mix(st.pal.sky[0], INK, 0.45),
    glint: CREAM,
    trail: st.pal.cloud[st.pal.cloud.length - 1],
    red: st.pal.poppy,
    green: mix(st.pal.leaf[st.pal.leaf.length - 1], CREAM, 0.3),
    strobe: CREAM,
  }
  st.lastGlow = st.lastGrowth = -1
  st.sky = layer(st.skyCanvas, st.sky)
  if (st.sky) {
    const sx = st.sky.g.sx
    if (st.ridge.length !== sx.length) st.ridge = Float32Array.from(sx, skyline)
    else for (let i = 0; i < sx.length; i++) st.ridge[i] = skyline(sx[i])
    hangBody(st, true)
  }
  st.field = layer(st.hill, st.field)
  if (st.field) {
    st.ground = meadow(st.field.g, st.water, st.relief, st.times)
    const len = st.field.img.data.length
    if (st.hillBase.length !== len) st.hillBase = new Uint8ClampedArray(len)
    else st.hillBase.fill(0)
    paintHillBase(st.hillBase, st.field.g, st.pal, st.water, st.ground)
  }
  st.glint = layer(st.near, st.glint)
  if (st.glint) {
    const len = st.glint.img.data.length
    if (st.figures.length !== len) {
      st.figures = new Uint8ClampedArray(len)
      st.people = new Uint8ClampedArray(len)
    }
    // The mountain legs run only when the flag on the crest is on screen.
    const { g } = st.glint
    st.camp = campSpan(g.px)
    layCourse(
      st.tri,
      g,
      st.water,
      st.camp,
      TRAIL[TRAIL.length - 1][0] + 8 * g.px < g.sx[g.bw - 1] - 48,
    )
    layAngler(st.angler, g, st.water)
  }
  st.well = layer(st.riftCanvas, st.well)
  // The tree folk's heads and speech are in the old grid's pixels, and a resize may leave no one
  // in the tree to hail.
  st.bough = null
  st.boughs.heads = {}
  st.boughs.bubbles = []
  st.leaves = layer(st.front, st.leaves)
  if (st.leaves) {
    // Repoussoir trees hug whatever edges the slice fit leaves visible.
    const left = st.leaves.g.sx[0]
    const span = st.leaves.g.sx[st.leaves.g.bw - 1] - left
    const nextCanopyKey = `${st.leaves.g.bw}x${st.leaves.g.bh}:${left.toFixed(1)}:${span.toFixed(1)}`
    if (st.canopyKey !== nextCanopyKey) {
      st.canopyKey = nextCanopyKey
      const roots = [left + span * 0.04, left + span * 0.97]
      const k = Math.max(0.45, Math.min(1, span / 1500))
      st.canopy = {
        roots,
        trees: [
          growTree(4040, roots[0], 1080, -0.04, 46 * k, 90 * k),
          growTree(4041, roots[1], 1080, 0.02, 36 * k, 80 * k),
        ],
      }
    }
    // The layers bleed past the viewport for the parallax, so the tree folk are placed within
    // what the viewport shows; and the departures board sits over the top-left corner, so nobody
    // climbs up behind it.
    const g = st.leaves.g
    const fr = st.front!.getBoundingClientRect()
    const toScene = (x: number, y: number): Pt => [
      g.sx[0] + ((x - fr.left) / fr.width) * g.bw * g.px,
      g.sy[0] + ((y - fr.top) / fr.height) * g.bh * g.px,
    ]
    const vr = st.front!.parentElement!.getBoundingClientRect()
    const [x0, y0] = toScene(vr.left, vr.top)
    const [x1, y1] = toScene(vr.right, vr.bottom)
    const board = st.scene.querySelector('.nf-board')?.getBoundingClientRect()
    const edge = board && toScene(board.right + 4, board.bottom + 4)
    const behindBoard = (x: number, y: number) => !!edge && x < edge[0] && y < edge[1]
    st.bough = findBough(st.canopy.trees[0], g, [x0, y0, x1, y1], behindBoard)
    const folk = st.bough ? [st.bough.seat, st.bough.root, st.bough.end, st.bough.anchor] : []
    perchJay(
      st.jay,
      [st.canopy.trees[1], st.canopy.trees[0]],
      g,
      [x0, y0, x1, y1],
      behindBoard,
      folk,
    )
    if (!st.litter.leaves.length)
      for (let n = 0; n < 120; n++) stepShed(st.litter, st.canopy.trees, shedding(st, g), 1 / 12)
  }
  paintRanges(st)
  paintRuin(st)
  paintAir(st)
  paintMeadow(st)
  paintPortal(st)
  paintShore(st)
  paintFront(st)
}

// Once a minute: a repaint when the light has moved on far enough to change the palette or the
// modelling, otherwise only the sun or the moon creeping along its arc.
export function recheck(st: LandscapeScope) {
  const next = almanac(new Date(), st.site)
  // The park and the lift open and close on their hours whether or not the light has changed,
  // and the groomer goes out on its own.
  const env = parkEnv(next, st.weather, st.weekend())
  const snow = hillEnv(next, st.weather, st.weekend())
  const shut =
    env.open !== st.park.env.open ||
    env.crowd !== st.park.env.crowd ||
    snow.open !== st.slopes.env.open ||
    snow.groom !== st.slopes.env.groom ||
    snow.tour !== st.slopes.env.tour ||
    snow.control !== st.slopes.env.control
  st.park.env = env
  st.slopes.env = snow
  if (keyFor(st, next) !== st.lightKey) return repaint(st)
  if (shut && st.reduce) {
    posePark(st.park)
    poseHill(st.slopes)
    paintRanges(st)
  }
  st.al = next
  if (hangBody(st, false)) paintAir(st)
}
