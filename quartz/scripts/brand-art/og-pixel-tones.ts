// The pixel cards' tones: the homepage trail's rock, field, tree and campsite colours, resolved to
// hex from their stylesheet formulas (trail-mountains.scss, shared-trail-frame.scss, trail-pet.scss)
// against the light theme by day and the dark theme by night, since librsvg has no color-mix().

import { INK, PAPER, PINK, SAGE } from '../../util/mascot/pen'
import { mix } from './scene'

export type Rock = 'rock' | 'shade' | 'light' | 'snow' | 'cap' | 'fracture' | 'lichen'
export type Range = { fill: string; shade: string; light: string }
type Leaf = { lit: string; mid: string; shade: string }
export type Palette = {
  sky: string
  // Sky bands from the top of the card down to the horizon, dithered into one another.
  dawn: string[]
  far: Range
  back: Range
  peak: Record<Rock, string>
  field: string[]
  tuft: string
  trail: string
  flag: string
  pole: string
  orb: string[]
  star: string[]
  css: string
}

// color-mix(in srgb, a p%, b)
const cm = (a: string, p: number, b: string) => mix(b, a, p / 100)

// The tree tones for one leaf set (sage), each hazed toward the sky by `haze` per cent for a far belt.
const trees = (scope: string, leaf: Leaf, lightgray: string, sky: string, haze: number) => {
  const h = (c: string) => cm(sky, haze, c)
  const dark = cm(leaf.shade, 35, '#65724f')
  const tone = {
    crown: cm(leaf.mid, 50, '#84916b'),
    light: cm(leaf.lit, 60, '#b6b58a'),
    dark,
    gold: cm(leaf.lit, 45, '#b49a62'),
    trunk: cm(leaf.shade, 75, '#80624a'),
    bark: leaf.lit,
    hollow: dark,
  }
  return (
    Object.entries(tone)
      .map(([k, c]) => `${scope} .trail-tree-${k}{fill:${h(c)}}`)
      .join('') +
    `${scope} .trail-tree-shadow{fill:${lightgray};opacity:.65}${scope} .trail-tree-flip{fill:${h(cm(tone.light, 82, leaf.lit))}}`
  )
}
// The pet's eyes a row down from the brim, so they do not read as a fringe under the hat.
const hidden =
  '.trail-tree-limbs,.trail-tree-sprout,.trail-tree-bud,.trail-tree-snow,.trail-winter-kit,.trail-fair-weather-kit,.trail-camp-drift,.trail-fire-thaw,.trail-smoulder,.trail-camp-shadow,.trail-firelight-halo,.trail-smoke,.camp>.trail-log,[data-part]{display:none}[data-part=eyes-down]{display:inline}'

// The pet as trail-pet.scss prints it (the 404's tones) by day, and by night a dark body in a
// moonlight keyline as the night card's camper is drawn.
const pet = (t: Record<string, string>) =>
  Object.entries(t)
    .map(([k, c]) => `.trail-pet-${k}{fill:${c}}`)
    .join('')

export function day(): Palette {
  const [light, lightgray, darkgray] = [PAPER, '#e6e4d9', '#6f6e69']
  const [cool, warm] = ['#8ea0ad', '#8c7f6c']
  const leaf: Leaf = { lit: lightgray, mid: '#a3a196', shade: '#4f4e4a' }
  return {
    sky: PAPER,
    // Paper at the top, where the press kit's page meets the card, warming to peach at the horizon.
    dawn: [PAPER, mix(PAPER, PINK, 0.14), mix(PAPER, PINK, 0.3), mix(PAPER, '#fcc192', 0.46)],
    // A value ladder the paper sky cannot give: far ranges in cool haze, the peak in warm rock
    // darker than both, the meadow between, the near ground and the trees darkest.
    far: { fill: cm(cool, 30, light), shade: cm(cool, 42, light), light: cm(cool, 18, light) },
    back: { fill: cm(cool, 48, light), shade: cm(cool, 62, light), light: cm(cool, 34, light) },
    peak: {
      rock: cm(warm, 58, light),
      shade: cm(warm, 88, light),
      light: cm(warm, 36, light),
      snow: cm(warm, 10, light),
      cap: cm(warm, 64, light),
      fracture: cm('#5a5146', 72, light),
      lichen: cm('#8a9a6c', 55, light),
    },
    field: [cm('#9c9a82', 40, light), cm('#8a846a', 54, light), cm('#6f6e69', 66, light)],
    tuft: cm(darkgray, 75, light),
    trail: mix(darkgray, INK, 0.3),
    flag: PINK,
    pole: INK,
    orb: [mix(PINK, '#d14d41', 0.2), PINK],
    star: [],
    css: `${hidden}${pet({ ink: '#1c1b1a', hand: '#1c1b1a', body: PAPER, plate: '#f4ecd6', shade: '#bbb8b0', hat: '#d8c08e', 'hat-light': '#ecd9ae', band: '#e9a090', panel: SAGE })}${trees('.near', leaf, lightgray, PAPER, 0)}${trees('.far', leaf, lightgray, PAPER, 38)}`,
  }
}

export function night(): Palette {
  const [sky, moon, ember] = ['#100f0f', '#cecdc3', '#fcc192']
  const [light, lightgray, gray, darkgray] = [sky, '#282726', '#575653', '#878580']
  const lit = (t: number) => mix(sky, moon, t)
  const leaf: Leaf = { lit: '#7a8392', mid: '#525a68', shade: '#2b2f38' }
  const camp = `.trail-hearth{fill:${cm(ember, 25, light)}}.trail-log{fill:${cm(darkgray, 70, '#745439')}}.trail-log-lit{fill:#a36a40}.trail-flame-core{fill:#fff1c4}.trail-flame-gold{fill:#f1d67e}.trail-flame-orange{fill:#da702c}.trail-flame-tip{fill:#af3029}.trail-rocks{fill:${gray}}.trail-ember{fill:#da702c}`
  return {
    sky,
    dawn: [sky],
    far: { fill: lit(0.12), shade: lit(0.08), light: lit(0.18) },
    back: { fill: lit(0.17), shade: lit(0.12), light: lit(0.24) },
    peak: {
      rock: lit(0.22),
      shade: lit(0.12),
      light: lit(0.34),
      snow: lit(0.56),
      cap: lit(0.36),
      fracture: lit(0.07),
      lichen: lit(0.28),
    },
    field: [mix(sky, SAGE, 0.14), mix(sky, SAGE, 0.19), mix(sky, SAGE, 0.24)],
    tuft: mix(sky, SAGE, 0.4),
    trail: lit(0.3),
    flag: PINK,
    pole: moon,
    orb: [moon, lit(0.72)],
    star: [lit(0.6), lit(0.35)],
    css: `${hidden}${pet({ ink: moon, hand: moon, body: '#3a3836', plate: '#2e2c2a', shade: '#232221', hat: '#8a7a55', 'hat-light': '#a8956a', band: PINK, panel: SAGE })}${camp}${trees('.near', leaf, lightgray, sky, 0)}${trees('.far', leaf, lightgray, sky, 30)}`,
  }
}
