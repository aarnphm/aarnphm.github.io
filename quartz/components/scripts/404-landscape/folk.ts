import type { Palette } from '../../../util/scene-palette'
import type { Voices } from '../404-crew'
import { BAYER, clampCol, col, dot, hash, INK, mix, row, type Grid, type RGB } from '../404-pixel'
import { paintPark, type Park, type RiderInk } from '../404-riders'
import { paintHill, type Hill, type SkiInk } from '../404-skiers'
import { CAMP, farRidge, FOLK, midRidge, onTrail, TRAIL_LEN, TRAIL_TIME } from './geography'
import { CREAM, FLAME, GOLDEN, LUME } from './light-marks'

export function paintFolk(
  d: Uint8ClampedArray,
  g: Grid,
  pal: Palette,
  base: Uint8ClampedArray,
  clock: number,
  dark: boolean,
  heads: Voices['heads'],
  park: Park,
  hill: Hill,
) {
  d.set(base)
  const beat = Math.floor(clock * 5)
  const plot = (i: number, j: number, c: RGB, alpha: number) => {
    if (alpha > BAYER[(j & 3) * 4 + (i & 3)]) dot(d, g, i, j, c)
  }
  // First row of ground under a column, so feet land on the ridge rather than near it.
  const ground = (i: number, ridge: (x: number) => number) =>
    Math.ceil((ridge(g.sx[clampCol(g, i)]) - g.sy[0]) / g.px)
  const glow = (i: number, j: number, flick: number) => {
    for (let dj = -2; dj <= 2; dj++)
      for (let di = -2; di <= 2; di++) {
        const r = Math.hypot(di, dj)
        if (r > 0) plot(i + di, j + dj, pal.glow[1], (1 - r / 2.8) * (0.55 + 0.25 * flick))
      }
  }
  FOLK.forEach((f, n) => {
    const i = col(g, f.x)
    const top = ground(i, midRidge)
    heads[`folk${n}`] = [i, top - 3]
    if (f.lamp && dark) glow(i + 1, top - 2, hash(beat, 3, 41))
    for (let k = 1; k <= 3; k++) dot(d, g, i, top - k, pal.figure.K)
    // The two by the cypress throw both arms up when a rider lands a trick off the table.
    if (park.cheer > 0 && !f.lamp) {
      const up = Math.floor(clock / 0.2) % 2
      dot(d, g, i - 1, top - 3 - up, pal.figure.K)
      dot(d, g, i + 1, top - 4 + up, pal.figure.K)
    } else if (f.wave) dot(d, g, i + 1, top - (Math.floor(clock / 0.45) % 2 ? 4 : 3), pal.figure.K)
    if (f.lamp) dot(d, g, i + 1, top - 2, hash(beat, 3, 41) < 0.3 ? FLAME[2] : FLAME[1])
  })

  // A lap starts on the summit, so a still frame (reduced motion) finds them there, arms up.
  const { up, summit, down } = TRAIL_TIME
  let lap = (clock + up) % (up + summit + down + TRAIL_TIME.foot)
  let s = 0
  let pose: 'up' | 'summit' | 'down' | 'foot' = 'foot'
  if (lap < up) {
    s = (TRAIL_LEN * lap) / up
    pose = 'up'
  } else if ((lap -= up) < summit) {
    s = TRAIL_LEN
    pose = 'summit'
  } else if ((lap -= summit) < down) {
    s = TRAIL_LEN * (1 - lap / down)
    pose = 'down'
  }
  const at = onTrail(s)
  const face = pose === 'down' ? -at.dir : at.dir
  const ri = col(g, at.x)
  const rt = row(g, at.y)
  // Head, vest and the pack behind it; the head leans a pixel into the climb.
  heads.runner = [ri + (pose === 'up' ? face : 0), rt - 3]
  dot(d, g, ri + (pose === 'up' ? face : 0), rt - 3, pal.figure.S)
  dot(d, g, ri, rt - 2, FLAME[2])
  dot(d, g, ri - face, rt - 2, pal.crew[1])
  const stride =
    pose === 'up' ? Math.floor(clock * 3) % 2 : pose === 'down' ? Math.floor(clock * 6) % 2 : 0
  if (stride) {
    dot(d, g, ri - 1, rt - 1, pal.figure.K)
    dot(d, g, ri + 1, rt - 1, pal.figure.K)
  } else dot(d, g, ri, rt - 1, pal.figure.K)
  if (pose === 'summit') {
    dot(d, g, ri - 1, rt - 4, pal.figure.S)
    dot(d, g, ri + 1, rt - 4, pal.figure.S)
  }
  if (pose === 'foot') dot(d, g, ri + face, rt - 2, pal.figure.S)

  const ci = col(g, CAMP)
  const top = ground(ci, farRidge)
  const flick = hash(beat, 7, 43)
  if (dark) glow(ci, top - 1, flick)
  dot(d, g, ci, top - 1, flick < 0.5 ? FLAME[2] : FLAME[1])
  // Its smoke leans off with the wind and thins as it climbs.
  for (let s = 1; s <= 12; s++)
    plot(
      ci + Math.round(s * 0.35 + Math.sin(s * 0.7 - clock * 1.3) * 0.6),
      top - 1 - s,
      pal.smoke[dark ? 2 : 1],
      (1 - s / 13) * (dark ? 0.5 : 0.8),
    )
  // The ski hill shows only where the far range does, under its crest and above the mid ridge.
  const onRange = (i: number, j: number) => {
    if (i < 0 || j < 0 || i >= g.bw || j >= g.bh) return false
    const y = g.sy[j]
    return y > farRidge(g.sx[i]) + g.px && y < midRidge(g.sx[i]) - g.px
  }
  paintHill(d, g, skiInk(pal), hill, onRange, dark, heads)
  paintPark(d, g, riderInk(pal), park, dark, heads)
}

// Its riders in the crew's colours under bright helmets, on frames of any of them; the gate's
// beacon counts down in poppy and goes in sage. Its medics wear the ski patrol's flame, and their
// hut is the figures' white with a poppy cross. The herd on its grass is white, black and the
// autumn's deepest red, with the figures' skin for a muzzle; the ewes white with black faces. Their
// barn is a brick red between the poppy and the red cow's coat, so she shows against it at the
// doors, under the hut's steel, its doors painted the figures' white. The farmer wears the figures'
// denim under a golden straw hat.
function riderInk(pal: Palette): RiderInk {
  return {
    skin: pal.figure.S,
    shorts: pal.figure.D,
    tyre: pal.figure.K,
    dust: mix(pal.bark[2], CREAM, 0.65),
    lamp: CREAM,
    glow: pal.glow[1],
    steel: mix(pal.rock[0], INK, 0.4),
    flash: CREAM,
    beacon: [pal.poppy, LUME],
    jerseys: pal.crew,
    helmets: [GOLDEN, pal.poppy, CREAM, pal.cornflower, pal.crew[1], pal.crew[3]],
    frames: [pal.crew[2], pal.crew[3], pal.crew[0], CREAM, pal.crew[5], pal.rock[1]],
    patrol: FLAME[3],
    kit: FLAME[2],
    hut: pal.figure.C,
    cross: pal.poppy,
    herd: {
      coats: [pal.figure.C, pal.figure.K, pal.autumn[0]],
      muzzle: pal.figure.S,
      leg: pal.bark[0],
      fleece: pal.figure.C,
      face: pal.figure.K,
      boards: mix(pal.poppy, pal.autumn[0], 0.5),
      roof: mix(pal.rock[0], INK, 0.4),
      trim: pal.figure.C,
      dark: pal.figure.K,
      hat: GOLDEN,
      skin: pal.figure.S,
      bib: pal.figure.B,
      trousers: pal.figure.D,
      pail: mix(pal.rock[2], CREAM, 0.3),
      feed: mix(GOLDEN, pal.bark[2], 0.3),
    },
  }
}

// The ski hill in the ranges' colours: coats from the crew's under bright helmets, skis and boards
// of any colour, the instructor in a poppy bib, the slalom's gates poppy and cornflower, and the
// groomer golden with a flame-coloured beacon.
function skiInk(pal: Palette): SkiInk {
  return {
    snow: pal.snow,
    skin: pal.figure.S,
    pants: pal.figure.D,
    boot: pal.figure.K,
    steel: mix(pal.rock[0], INK, 0.4),
    cable: mix(pal.far[0], INK, 0.4),
    rock: [pal.rock[0], pal.rock[1]],
    coats: pal.crew,
    helmets: [GOLDEN, pal.poppy, CREAM, pal.cornflower, pal.crew[1], pal.crew[3]],
    gear: [pal.figure.K, pal.crew[2], mix(pal.rock[0], INK, 0.4), CREAM],
    boards: [GOLDEN, pal.poppy, pal.cornflower, pal.crew[1], CREAM],
    school: pal.poppy,
    gates: [pal.poppy, pal.cornflower],
    cat: GOLDEN,
    cab: mix(pal.far[0], INK, 0.4),
    lamp: CREAM,
    glow: pal.glow[1],
    flood: mix(CREAM, GOLDEN, 0.4),
    patrol: FLAME[3],
    sled: FLAME[2],
    beacon: FLAME[2],
    smoke: mix(pal.smoke[0], INK, 0.5),
  }
}
