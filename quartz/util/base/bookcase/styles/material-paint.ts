import type { Face, ShelfMaterial, Tone } from './style-model'
import { COS30 } from '../geometry'
import { FINISHES, GRAIN_ASSETS, GRAINS, type DefNode } from './grain-recipes'
import { LADDERS, materialLadder, mixHex } from './material-palettes'

const TONES: readonly Tone[] = ['top', 'face', 's3', 's2', 's1', 's0']

const gradientId = (material: ShelfMaterial, tone: Tone) => `bc-g-${material}-${tone}`
const stopVar = (material: ShelfMaterial, tone: Tone, i: number) =>
  `--bc-g-${material}-${tone}-${i}`

/**
 * Colour of one gradient stop. Paper stays on the theme tokens, so its stops are mixes of the
 * `--bc-*` the scene inherits; every other material gets its ladder's hex for the theme.
 */
function stopColour(
  material: ShelfMaterial,
  theme: 'light' | 'dark',
  tone: Tone,
  w: number,
): string {
  const pct = Math.round(Math.abs(w) * 100)
  if (material === 'paper')
    return `color-mix(in srgb, var(--bc-${tone}) ${100 - pct}%, ${w > 0 ? 'white' : 'var(--bc-halo)'})`
  const ladder = LADDERS[theme][material]
  return mixHex(ladder[tone], w > 0 ? ladder.sheen : ladder.halo, Math.abs(w))
}

/** Reference for a case face: its tone's gradient, run the way the face is lit. */
export function faceFill(material: ShelfMaterial, face: Face): string {
  const id = gradientId(material, face.tone)
  const o = face.o ?? 'front'
  if (o === 'top') return `url(#${id}-t)`
  const across = (o === 'front') === !face.tall
  return `url(#${id}${across ? '' : '-h'})`
}

/**
 * Pattern transforms that lay a tile flat on each face. The projection is parallel, so a face is
 * an affine image of its plane: tile x goes along the grain (the face's long axis), tile y across
 * it. World +x projects to (cos30, ½), +y to (cos30, −½) and up to (0, −1).
 */
const GRAIN_MAPS: Record<string, string> = {
  // front face, grain along the run
  fa: `${COS30} 0.5 0 1`,
  // front face, grain up
  fu: `0 -1 ${COS30} 0.5`,
  // side face, grain along the depth
  sa: `${COS30} -0.5 0 1`,
  // side face, grain up
  su: `0 -1 ${COS30} -0.5`,
  // top face, grain along the run
  ta: `${COS30} 0.5 ${COS30} -0.5`,
  // top face, grain along the depth
  td: `${COS30} -0.5 ${COS30} 0.5`,
}

/** Reference for the grain over a case face, null for a material with none. */
export function grainFill(material: ShelfMaterial, face: Face): string | null {
  const grain = FINISHES[material].grain
  if (!grain) return null
  if (!GRAINS[grain].directed) return `url(#bc-grain-${material})`
  const o = face.o ?? 'front'
  const dir = o === 'top' ? (face.tall ? 'td' : 'ta') : `${o[0]}${face.tall ? 'u' : 'a'}`
  return `url(#bc-grain-${material}-${dir})`
}

/** True when a scene with this material needs grain overlays. */
export const hasGrain = (material: ShelfMaterial): boolean => FINISHES[material].grain !== null

/**
 * Gradient, filter and pattern defs for the materials in a scene, plus the spine sheen every book
 * shares. Emit them once in the scene's `<defs>`.
 */
export function materialDefs(
  materials: Iterable<ShelfMaterial>,
  assets: string = GRAIN_ASSETS,
): DefNode[] {
  const defs: DefNode[] = [
    {
      tag: 'linearGradient',
      attrs: { id: 'bc-g-sheen', x1: 0, y1: 0, x2: 1, y2: 0 },
      children: [
        { tag: 'stop', attrs: { offset: 0, 'stop-color': '#FFFFFF', 'stop-opacity': 0.16 } },
        { tag: 'stop', attrs: { offset: 0.35, 'stop-color': '#FFFFFF', 'stop-opacity': 0 } },
        { tag: 'stop', attrs: { offset: 0.7, 'stop-color': '#000000', 'stop-opacity': 0 } },
        { tag: 'stop', attrs: { offset: 1, 'stop-color': '#000000', 'stop-opacity': 0.2 } },
      ],
    },
    // A flat book's spine is lit along its top edge.
    {
      tag: 'linearGradient',
      attrs: { id: 'bc-g-sheen-v', href: '#bc-g-sheen', x1: 0, y1: 0, x2: 0, y2: 1 },
    },
  ]
  for (const material of new Set(materials)) {
    const finish = FINISHES[material]
    for (const tone of TONES) {
      const id = gradientId(material, tone)
      defs.push(
        {
          tag: 'linearGradient',
          attrs: { id, x1: 0, y1: 0, x2: 0, y2: 1 },
          children: finish.stops.map(([offset], i) => ({
            tag: 'stop',
            attrs: { offset, style: `stop-color:var(${stopVar(material, tone, i)})` },
          })),
        },
        {
          tag: 'linearGradient',
          attrs: { id: `${id}-h`, href: `#${id}`, x1: 0, y1: 0, x2: 1, y2: 0 },
        },
        {
          tag: 'linearGradient',
          attrs: { id: `${id}-t`, href: `#${id}`, x1: 0, y1: 1, x2: 1, y2: 0 },
        },
      )
    }
    if (!finish.grain) continue
    const grain = GRAINS[finish.grain]
    const [w, h] = grain.tile
    const id = `bc-grain-${material}`
    if ('image' in grain) {
      const image = {
        tag: 'image',
        attrs: { href: `${assets}${finish.image}`, width: w, height: h },
      }
      defs.push(
        {
          tag: 'pattern',
          attrs: { id, patternUnits: 'userSpaceOnUse', width: w, height: h },
          children: [image],
        },
        {
          tag: 'pattern',
          attrs: {
            id: `${id}-edge`,
            patternUnits: 'userSpaceOnUse',
            width: w,
            height: h * grain.edge,
          },
          children: [image],
        },
      )
    } else {
      defs.push(
        {
          tag: 'filter',
          attrs: {
            id: `${id}-noise`,
            x: 0,
            y: 0,
            width: '100%',
            height: '100%',
            'color-interpolation-filters': 'sRGB',
          },
          children: grain.noise,
        },
        {
          tag: 'pattern',
          attrs: { id, patternUnits: 'userSpaceOnUse', width: w, height: h },
          children: [
            {
              tag: 'rect',
              attrs: { width: w, height: h, opacity: grain.opacity, filter: `url(#${id}-noise)` },
            },
          ],
        },
      )
    }
    if (grain.directed)
      for (const [dir, m] of Object.entries(GRAIN_MAPS)) {
        const base = 'image' in grain && dir.startsWith('s') ? `${id}-edge` : id
        defs.push({
          tag: 'pattern',
          attrs: { id: `${id}-${dir}`, href: `#${base}`, patternTransform: `matrix(${m} 0 0)` },
        })
      }
  }
  return defs
}

/**
 * Tone tokens of each material under `[data-material]`, and every gradient stop colour on the
 * scene root, for both themes. `scope` is the scene's selector and `dark` the prefix of a rule
 * that applies in the dark theme.
 */
export function materialCss(
  materials: Iterable<ShelfMaterial>,
  scope = '.bookcase-scene',
  dark = ":root[saved-theme='dark'] ",
): string {
  const rules: string[] = []
  const set = new Set(materials)
  for (const theme of ['light', 'dark'] as const) {
    const root = theme === 'dark' ? dark : ''
    const stops: string[] = []
    for (const material of set) {
      if (material !== 'paper') {
        const ladder = materialLadder(material, theme)
        const tokens = Object.entries(ladder)
          .filter(([, value]) => value)
          .map(([key, value]) => `--bc-${key}:${value}`)
          .join(';')
        rules.push(`${root}${scope} [data-material='${material}']{${tokens}}`)
      }
      for (const tone of TONES)
        FINISHES[material].stops.forEach(([, w], i) =>
          stops.push(`${stopVar(material, tone, i)}:${stopColour(material, theme, tone, w)}`),
        )
    }
    if (stops.length > 0) rules.push(`${root}${scope}{${stops.join(';')}}`)
  }
  return rules.join('')
}
