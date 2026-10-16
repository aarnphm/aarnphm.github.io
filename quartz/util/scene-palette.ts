import type { Almanac, Sky } from '../components/scripts/404-almanac'
import type { Season } from './visitor-world'
import { INK, type RGB, hex, mix, smooth } from '../components/scripts/404-pixel'

export type Ink = 'C' | 'S' | 'B' | 'D' | 'K'
export type Palette = {
  sky: RGB[]
  cloud: RGB[]
  swirl: [RGB, RGB]
  sun: RGB[]
  far: RGB[]
  mid: RGB[]
  hill: RGB[]
  pine: RGB[]
  bark: RGB[]
  leaf: RGB[]
  // The broadleaf crowns, which turn with the season, and what the season turns things toward.
  crown: RGB[]
  autumn: RGB[]
  dun: RGB[]
  snow: RGB[]
  rock: RGB[]
  gold: RGB
  poppy: RGB
  cornflower: RGB
  ochre: RGB
  star: RGB
  glint: RGB[]
  figure: Record<Ink, RGB>
  shade: [...RGB, number]
  fringe: [...RGB, number]
  water: RGB[]
  smoke: RGB[]
  glow: RGB[]
  coats: [RGB, RGB]
  tent: [RGB, RGB]
  moss: RGB[]
  // Crest lines: far ridge, mid ridge, hill.
  contour: [RGB, RGB, RGB]
  crew: RGB[]
  rope: RGB
  fish: RGB
}

export const LIGHT: Palette = {
  sky: ['#b8d9d2', '#c9e1d9', '#dae9e0', '#ecf1e6', '#fffcf0'].map(hex),
  cloud: ['#a6abbd', '#bcbfcc', '#d2d3da'].map(hex),
  swirl: [hex('#9fcbc2'), hex('#fffcf0')],
  sun: ['#fbe3d6', '#fdc4b4', '#fdb2a2'].map(hex),
  far: ['#b3c6d2', '#c7d6de', '#dde6e6'].map(hex),
  mid: ['#879578', '#a2ad8c', '#c3c9a6'].map(hex),
  hill: ['#56652a', '#74853a', '#97a652', '#bcc47e'].map(hex),
  pine: ['#5d6b4c', '#6d7b5b', '#83906c'].map(hex),
  bark: ['#3e3128', '#5e4b3b', '#7e6a55'].map(hex),
  leaf: ['#2f3d0c', '#4d6212', '#7d902a', '#b4bf6a'].map(hex),
  crown: ['#2f3d0c', '#4d6212', '#7d902a', '#b4bf6a'].map(hex),
  autumn: ['#6b2c0b', '#bc5215', '#da702c', '#d0a215'].map(hex),
  dun: ['#6b6a45', '#8a875c', '#aaa57a', '#c9c39c'].map(hex),
  snow: ['#98a9bf', '#bfcad8', '#dfe5ea', '#fffcf0'].map(hex),
  rock: ['#57534b', '#7f796c', '#a39c8c', '#cdc6b2'].map(hex),
  gold: hex('#e0b43c'),
  poppy: hex('#e0604a'),
  cornflower: hex('#4f72b0'),
  ochre: hex('#d8bd55'),
  star: hex('#fffcf0'),
  glint: ['#fffcf0', '#fdd6c9'].map(hex),
  figure: {
    C: hex('#fffcf0'),
    S: hex('#e2ab91'),
    B: hex('#3f6db0'),
    D: hex('#28497e'),
    K: hex('#100f0f'),
  },
  shade: [...hex('#34435e'), 0.2],
  fringe: [...hex('#d9a93a'), 0.32],
  water: ['#8fc9be', '#a4d5ca', '#bbe0d6'].map(hex),
  smoke: ['#8f909c', '#a9aab4', '#c3c3ca'].map(hex),
  glow: ['#a8984a', '#c9a557', '#e8bf6a'].map(hex),
  coats: [hex('#bc5215'), hex('#24837b')],
  tent: [hex('#d98f7e'), hex('#fdb2a2')],
  moss: ['#4d6212', '#74853a', '#97a652', '#c6cc84'].map(hex),
  contour: [hex('#8199ab'), hex('#5f6c50'), hex('#56652a')],
  crew: ['#bc5215', '#24837b', '#205ea6', '#ad8301', '#5e409d', '#a02f6f'].map(hex),
  rope: hex('#7e6a55'),
  fish: hex('#6f8fa8'),
}

export const DARK: Palette = {
  sky: ['#121726', '#141a28', '#161b25', '#141619', '#100f0f'].map(hex),
  cloud: ['#1b2233', '#212940', '#29324b'].map(hex),
  swirl: [hex('#0d1019'), hex('#33405e')],
  sun: ['#2a2622', '#8f8a7c', '#cecdc3'].map(hex),
  far: ['#1b1f25', '#20252b', '#262c33'].map(hex),
  mid: ['#191e14', '#20271a', '#293221'].map(hex),
  hill: ['#12160a', '#1a200e', '#232b13', '#2e3819'].map(hex),
  pine: ['#161b11', '#1e2517', '#262f1d'].map(hex),
  bark: ['#0d0b09', '#1c1814', '#2c251f'].map(hex),
  leaf: ['#0b0f04', '#151d07', '#24300c', '#3d4c10'].map(hex),
  crown: ['#0b0f04', '#151d07', '#24300c', '#3d4c10'].map(hex),
  autumn: ['#1e0d04', '#3d1a07', '#5c2a0c', '#6e5406'].map(hex),
  dun: ['#15150e', '#1e1d14', '#29281c', '#363426'].map(hex),
  snow: ['#1f2533', '#2b3342', '#3c4556', '#58627a'].map(hex),
  rock: ['#0e0d0b', '#171613', '#22201b', '#302d25'].map(hex),
  gold: hex('#5a4714'),
  poppy: hex('#8c3a2e'),
  cornflower: hex('#27365a'),
  ochre: hex('#4d4519'),
  star: hex('#e6e4d9'),
  glint: ['#cecdc3', '#6f6a5e'].map(hex),
  figure: {
    C: hex('#cecdc3'),
    S: hex('#8c6b5c'),
    B: hex('#2d4c7c'),
    D: hex('#1d3356'),
    K: hex('#050505'),
  },
  shade: [...hex('#fffcf0'), 0.05],
  fringe: [...hex('#d0a215'), 0.14],
  water: ['#132726', '#193230', '#203d3a'].map(hex),
  smoke: ['#2e323d', '#3b3f4b', '#4a4d58'].map(hex),
  glow: ['#2e2412', '#5c3514', '#9a5019'].map(hex),
  coats: [hex('#7a3a12'), hex('#1b5953')],
  tent: [hex('#4a302b'), hex('#7d4f45')],
  moss: ['#1a220a', '#2c3a10', '#3d4c10', '#5a6b22'].map(hex),
  contour: [hex('#39414b'), hex('#3a4530'), hex('#3d4a22')],
  crew: ['#7a3a12', '#1b5953', '#1a3f6e', '#6e5406', '#3c2a66', '#6a1f4a'].map(hex),
  rope: hex('#8f8a7c'),
  fish: hex('#4f6b82'),
}

// Twilight: the sky warms from the horizon up, the far range and the water go violet.
export const DUSK = {
  sky: ['#4a5478', '#7d7a9c', '#c98f98', '#f0a58c', '#fdc9a0'].map(hex),
  cloud: ['#6d6285', '#9a7f98', '#d59a96'].map(hex),
  sun: ['#fdd2a0', '#f59a6e', '#e0604a'].map(hex),
  far: ['#7e7496', '#978aa8', '#b3a3b8'].map(hex),
  water: ['#8a7f9c', '#b095a6', '#d8aca4'].map(hex),
  glint: ['#fdd6a0', '#f0a58c'].map(hex),
} satisfies Partial<Palette>

// Colour by colour, part way from one palette (or any nest of colours) to another.
export function blend<T>(a: T, b: T, t: number): T {
  if (t <= 0) return a
  if (t >= 1) return b
  if (typeof a === 'number') return (a + ((b as number) - a) * t) as T
  if (Array.isArray(a)) return a.map((v, k) => blend(v, (b as unknown[])[k], t)) as T
  return Object.fromEntries(
    Object.entries(a as object).map(([k, v]) => [
      k,
      blend(v, (b as Record<string, unknown>)[k], t),
    ]),
  ) as T
}

export const tint = (a: RGB[], b: RGB[], t: number) =>
  t <= 0 ? a : a.map((c, k) => mix(c, b[k], t))

// The day or the night palette, turned for the season and greyed by the weather.
export function weathered(p: Palette, s: Season, sky: Sky): Palette {
  const over = smooth(0.55, 1, sky.cover) * 0.6 + (sky.fog ? 0.3 : 0)
  const overcast = [p.cloud[1], p.cloud[1], p.cloud[2], p.cloud[2], p.cloud[2]]
  const haze = p.sky.slice(2)
  let hill = tint(p.hill, p.autumn, s.autumn * 0.15)
  hill = tint(hill, p.dun, s.bare * 0.45)
  return {
    ...p,
    sky: tint(p.sky, overcast, over),
    sun: tint(p.sun, p.cloud, over * 0.6),
    far: tint(p.far, haze, sky.fog ? 0.8 : 0),
    mid: tint(p.mid, haze, sky.fog ? 0.5 : 0),
    hill: tint(hill, p.snow, Math.min(0.95, s.ground * 1.15)),
    // What leaves hang on through the winter are dead ones.
    crown: tint(p.crown, p.dun, s.bare * 0.85),
    autumn: tint(p.autumn, p.dun, s.bare * 0.5),
    // Smoke shares the snow's grey; over lying snow it takes the dark of wet woodsmoke so the
    // column still reads.
    smoke: p.smoke.map(c => mix(c, INK, Math.min(1, s.ground * 1.15) * 0.45)),
  }
}

// The palette for this moment: night and day blended by how much daylight there is, then warmed
// through twilight.
export function lighting(al: Almanac, s: Season, sky: Sky): [Palette, number] {
  // A storm's anvil or a downpour takes a good part of the daylight with it.
  const gloom = sky.storm
    ? 0.45
    : sky.fall === 'rain' || sky.fall === 'drizzle'
      ? 0.12 + 0.25 * sky.heavy
      : 0.15 * smooth(0.8, 1, sky.cover)
  const day = al.day * (1 - gloom)
  const night = weathered(DARK, s, sky)
  const pal = blend(night, weathered(LIGHT, s, sky), day)
  const glow = al.dusk * 0.75 * (1 - smooth(0.55, 1, sky.cover) * 0.6)
  if (glow <= 0) return [pal, day]
  const out = { ...pal }
  for (const key of Object.keys(DUSK) as (keyof typeof DUSK)[])
    out[key] = blend(pal[key], blend(night[key], DUSK[key], 0.4 + 0.6 * day), glow)
  return [out, day]
}
