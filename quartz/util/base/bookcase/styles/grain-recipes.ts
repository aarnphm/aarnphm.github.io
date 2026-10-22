import type { ShelfMaterial } from './style-model'
import { hexRgb } from './material-palettes'

// ---------------------------------------------------------------------------------------------
// Finishes. Every case face is a gradient of its tone, from a lit edge to a shaded one, and the
// grained materials lay a tiled noise over the face. Both are SVG defs the renderers emit once per
// scene; `materialDefs` describes them, `materialCss` sets the stop colours per theme, and
// `faceFill` / `grainFill` give each face its reference. Gradients stay in the face's own box, so a
// tone step still reads as a step; the grain tiles in user space and is rasterised once per tile.

export type Grain = 'wood' | 'brush' | 'cast' | 'patina'

interface Finish {
  /** Gradient stops from the lit edge: offset, and a weight toward the sheen (+) or the halo (−). */
  stops: readonly (readonly [number, number])[]
  grain: Grain | null
  /** Tile file under the grain assets folder, for a grain painted from an image. */
  image?: string
}

export const FINISHES: Record<ShelfMaterial, Finish> = {
  paper: {
    stops: [
      [0, 0.06],
      [1, -0.09],
    ],
    grain: null,
  },
  'birch-ply': {
    stops: [
      [0, 0.14],
      [1, -0.1],
    ],
    grain: 'wood',
    image: 'birch-ply.webp',
  },
  walnut: {
    stops: [
      [0, 0.2],
      [0.55, 0],
      [1, -0.18],
    ],
    grain: 'wood',
    image: 'walnut.webp',
  },
  // Gloss: a reflection band along the lit edge, then a fall-off into the dark.
  'black-lacquer': {
    stops: [
      [0, 0.34],
      [0.16, 0.1],
      [0.5, -0.02],
      [1, -0.22],
    ],
    grain: null,
  },
  // Brushed: the highlight smears along the brushing, so brightness bands across the face.
  steel: {
    stops: [
      [0, 0.12],
      [0.22, -0.1],
      [0.42, 0.3],
      [0.58, 0.06],
      [0.8, -0.12],
      [1, 0.04],
    ],
    grain: 'brush',
  },
  // Antiqued: darker at both edges, a soft burnish inside, so a cube reads as one wrapped sheet.
  copper: {
    stops: [
      [0, -0.18],
      [0.3, 0.3],
      [0.62, 0],
      [1, -0.36],
    ],
    grain: 'patina',
  },
  concrete: {
    stops: [
      [0, 0.05],
      [1, -0.08],
    ],
    grain: 'cast',
  },
}

/** Where `Finish.image` tiles are served from; the plugin passes its own base. */
export const GRAIN_ASSETS = '/static/bookcase-grain/'

type GrainDef = { tile: [number, number]; directed: boolean } & (
  | { image: true; edge: number }
  | { opacity: number; noise: DefNode[] }
)

export const GRAINS: Record<Grain, GrainDef> = {
  // Veneer: an RGBA tile grown by docs/verification/bookcase-materials/gen-grain.py, dark where the
  // wood is dark and pale where it is pale, transparent at the mean. The image is 2048×1024 px laid
  // over 1024×512 units, so it stays sharp at 2× zoom. Its top 28% is the straight-grained zone far
  // from the pith; edge faces sample that band alone.
  wood: { tile: [1024, 512], directed: true, image: true, edge: 0.28 },
  // Brushing: fine lines along the tile with the brightness varying across them.
  brush: {
    tile: [480, 48],
    opacity: 0.36,
    directed: true,
    noise: [
      {
        tag: 'feTurbulence',
        attrs: {
          type: 'fractalNoise',
          baseFrequency: '0.012 0.5',
          numOctaves: 2,
          seed: 7,
          stitchTiles: 'stitch',
          result: 'n',
        },
      },
      {
        tag: 'feColorMatrix',
        attrs: { in: 'n', type: 'matrix', values: matrix('#000000', 2.2), result: 'd' },
      },
      {
        tag: 'feColorMatrix',
        attrs: { in: 'n', type: 'matrix', values: matrix('#FFFFFF', 2.2), result: 'l' },
      },
      merge(['d', 'l']),
    ],
  },
  // Faux-concrete panel: a faint cloud, trowel streaks that run up the face, a fine speckle.
  cast: {
    tile: [256, 256],
    opacity: 0.24,
    directed: false,
    noise: [
      {
        tag: 'feTurbulence',
        attrs: {
          type: 'fractalNoise',
          baseFrequency: '0.012',
          numOctaves: 2,
          seed: 5,
          stitchTiles: 'stitch',
          result: 'c',
        },
      },
      {
        tag: 'feColorMatrix',
        attrs: { in: 'c', type: 'matrix', values: matrix('#000000', 1.2), result: 'cd' },
      },
      {
        tag: 'feTurbulence',
        attrs: {
          type: 'fractalNoise',
          baseFrequency: '0.09 0.016',
          numOctaves: 3,
          seed: 13,
          stitchTiles: 'stitch',
          result: 'v',
        },
      },
      {
        tag: 'feColorMatrix',
        attrs: { in: 'v', type: 'matrix', values: matrix('#FFFFFF', 1.1), result: 'vl' },
      },
      {
        tag: 'feTurbulence',
        attrs: {
          type: 'turbulence',
          baseFrequency: '0.8',
          numOctaves: 1,
          seed: 9,
          stitchTiles: 'stitch',
          result: 't',
        },
      },
      {
        tag: 'feColorMatrix',
        attrs: {
          in: 't',
          type: 'matrix',
          values: '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.3 0.3 0.3 0 -0.14',
          result: 's',
        },
      },
      merge(['cd', 'vl', 's']),
    ],
  },
  // Antiqued copper: brown clouds and gold burnish at hand scale, smeared a little sideways.
  patina: {
    tile: [512, 512],
    opacity: 0.5,
    directed: false,
    noise: [
      {
        tag: 'feTurbulence',
        attrs: {
          type: 'fractalNoise',
          baseFrequency: '0.006 0.009',
          numOctaves: 4,
          seed: 3,
          stitchTiles: 'stitch',
          result: 'n',
        },
      },
      {
        tag: 'feColorMatrix',
        attrs: { in: 'n', type: 'matrix', values: matrix('#2B1A0C', 2.1), result: 'd' },
      },
      {
        tag: 'feColorMatrix',
        attrs: { in: 'n', type: 'matrix', values: matrix('#F2D8A6', 2.1), result: 'l' },
      },
      {
        tag: 'feTurbulence',
        attrs: {
          type: 'fractalNoise',
          baseFrequency: '0.02 0.05',
          numOctaves: 2,
          seed: 17,
          stitchTiles: 'stitch',
          result: 'w',
        },
      },
      {
        tag: 'feColorMatrix',
        attrs: { in: 'w', type: 'matrix', values: matrix('#5A3D22', 1.0), result: 'wd' },
      },
      merge(['d', 'l', 'wd']),
    ],
  },
}

/** A plain element tree a renderer turns into SVG: hast in Quartz, a string in the plugin. */
export interface DefNode {
  tag: string
  attrs: Record<string, string | number>
  children?: DefNode[]
}

/**
 * Colour matrix that paints `colour` with the noise's brightness as alpha: a dark colour where the
 * noise is bright, a light colour where it is dark, so the two halves of one noise tint both ways.
 */
function matrix(colour: string, gain: number): string {
  const [r, g, b] = hexRgb(colour).map(c => c / 255)
  const dark = r + g + b < 1.5
  const k = (dark ? gain : -gain) / 3
  return `0 0 0 0 ${r} 0 0 0 0 ${g} 0 0 0 0 ${b} ${k} ${k} ${k} 0 ${dark ? -gain / 2 : gain / 2}`
}

function merge(inputs: string[]): DefNode {
  return {
    tag: 'feMerge',
    attrs: {},
    children: inputs.map(i => ({ tag: 'feMergeNode', attrs: { in: i } })),
  }
}
