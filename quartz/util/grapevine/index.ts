import type { Element } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import { s } from 'hastscript'
import type { Pt } from './glyph'
import { GRAPES } from './grapes'
import { blockedMask } from './raster'
import { drawItems } from './svg'
import { composePreset } from './vendor/alembic-ornaments/internal/presets.js'
import { render } from './vendor/alembic-ornaments/internal/render.js'
import { createRng } from './vendor/alembic-ornaments/internal/rng.js'
import { Field } from './vendor/initialis/field.js'
import { register } from './vendor/initialis/kinds.js'
import { grow, ornSizes, type Items, type OrnOptions } from './vendor/initialis/ornament.js'
import { hash, makeNoise, rng } from './vendor/initialis/util.js'

// A grapevine grown around a capital at build time by the INITIĀLIS ornament engine
// (vendor/initialis, MIT, Azlen Elza) run headless: the letter's outline becomes the clearance
// field the stems grow through, and the grown items are written out as SVG in the site's inks.
// The chapter separator (`separator:` in frontmatter) is either one grapevine from the same engine,
// spreading from its middle and thinning to both ends, or alembic-ornaments' running vine
// (vendor/alembic-ornaments, MIT, Lauri Paronen), a woodcut rinceau between two hairlines in
// currentColor. Seeds are the page and the letter or chapter, so a page always prints the same
// ornaments and nothing moves after paint.

register(GRAPES)

export const TILE = 1000
const LETTER_GAP = 16
export const RULE_KINDS = ['grapevine', 'alembic'] as const
export type RuleKind = (typeof RULE_KINDS)[number]

// The grapevine separator's space: a lens the page wide, so the vine has most of its room in the
// middle and runs out of it toward the ends. The page's width is unknown at build time, so the
// vine is grown at three lengths (about 4.5 units to the pixel at the middle of each range) and
// column.scss shows the one the page's width calls for; the others never paint.
export const SPRIG_WIDTHS = { s: 5000, m: 7500, l: 11000 } as const
export type SprigSize = keyof typeof SPRIG_WIDTHS
const SPRIG_H = 280
const SPRIG_TIP = 40
// alembic's vine is 96 units tall; at the rule's 60px this width covers a 2600px viewport before
// `slice` would have to scale it, and narrower screens see the band's middle.
const RULE_W = 4200
// Outline tolerance in the band's units: 0.5 is 0.3 px at the rule's height and halves the SVG.
const RULE_TOL = 0.5
const CELL = 3
const DETAIL_W = 1.4

// Settings in initialis's own terms (PARAMS in its render/src/params.js); a Vitis look: broad
// five-lobed notched leaves, curling ends, bunches of grapes where a stem has room to hang one.
const VINE: OrnOptions = {
  margin: 9,
  stemW: 9,
  taper: 0.55,
  curl: 0.65,
  turns: 1.1,
  undulate: 0,
  wander: 0.3,
  thin: 0.5,
  branch: 0.55,
  maxDepth: 3,
  branchAngle: (40 * Math.PI) / 180,
  length: 1,
  leaf: 'ivy',
  leafSize: 1.6,
  leafEvery: 0.45,
  leafW: 0.65,
  lobes: 5,
  lobeDepth: 0.45,
  serr: 0,
  leafCurl: 0.4,
  leafAngle: (45 * Math.PI) / 180,
  leafSide: 'alt',
  flower: 'grapes',
  flowerChance: 0.45,
  flowerSize: 1.8,
  petals: 6,
  berries: 0.15,
  berrySize: 1.2,
  dots: 0.12,
  beads: 0,
  terminal: 'flower',
  density: 0.6,
  sprout: 0.8,
  mains: 3,
  fillLeaves: true,
  detail: 'vein',
  budget: 4000,
  settings: {},
}

// The separator vine: two canes from one root in the middle, undulating (depth-0 stems do not
// curl) and tapering to their ends; branches curl into tendrils and end in a leaf or a bunch. No
// gap filling, dots or beads, so every mark hangs from the cane.
export const SPRIG: OrnOptions = {
  ...VINE,
  margin: 7,
  stemW: 9,
  taper: 0.8,
  curl: 1,
  turns: 1.2,
  undulate: 0.3,
  wander: 0.3,
  thin: 0.6,
  branch: 0.5,
  maxDepth: 2,
  branchAngle: (50 * Math.PI) / 180,
  leafSize: 1.2,
  leafEvery: 0.7,
  flowerChance: 0.7,
  flowerSize: 1.6,
  berries: 0,
  dots: 0,
  terminal: 'flower',
  density: 0,
  sprout: 0,
  mains: 0,
  fillLeaves: false,
}

const SVG = { xmlns: 'http://www.w3.org/2000/svg', ariaHidden: 'true', focusable: 'false' }

function field(
  width: number,
  height: number,
  o: OrnOptions,
  inset: number,
  blocked: Pt[][] = [],
  gap = LETTER_GAP,
) {
  const grid = { nx: Math.ceil(width / CELL) + 1, ny: Math.ceil(height / CELL) + 1, cell: CELL }
  const f = new Field(width, height, CELL, ornSizes(o).cap)
  f.setBlocked(blockedMask(width, height, grid, inset, blocked, gap))
  return f
}

/** The lens the separator vine grows in, as the two blocked regions outside it. */
function lens(width: number): Pt[][] {
  const cy = SPRIG_H / 2
  const half = cy - 6
  const x0 = SPRIG_TIP
  const x1 = width - SPRIG_TIP
  const edge: Pt[] = []
  for (let k = 0; k <= 72; k++) {
    const u = k / 72
    edge.push([x0 + (x1 - x0) * u, half * Math.sqrt(Math.sin(Math.PI * u))])
  }
  const pad = 20
  const top: Pt[] = [
    [-pad, -pad],
    [width + pad, -pad],
    [width + pad, cy],
    ...edge.map(([x, h]): Pt => [x, cy - h]).reverse(),
    [-pad, cy],
  ]
  const bottom: Pt[] = [
    [-pad, SPRIG_H + pad],
    [-pad, cy],
    ...edge.map(([x, h]): Pt => [x, cy + h]),
    [width + pad, cy],
    [width + pad, SPRIG_H + pad],
  ]
  return [top, bottom]
}

/** The grown items of one vine `width` units long; exported for the sample sheets and probes. */
export function grapevineItems(seed: string, width: number, o: OrnOptions = SPRIG): Items {
  const h = hash(seed, 'rule')
  const R = rng(h)
  const noise = makeNoise(h ^ 0x5bd1e995)
  const opts = { ...o, maxLen: width }
  const f = field(width, SPRIG_H, o, 0, lens(width), 4)
  const cy = SPRIG_H / 2
  // Two canes from one root, each long enough to reach its end of the lens (the lens stops it);
  // the second needs grace past the first's stamp at the root or it is blocked where it starts.
  const root = { x: width / 2, y: cy, len: (width / 2 / 650) * 1.25, grace: o.margin + o.stemW * 2 }
  const items = grow(f, opts, R, noise, [
    { ...root, dir: (R() - 0.5) * 0.3 },
    { ...root, dir: Math.PI + (R() - 0.5) * 0.3 },
  ])
  // A cane that wanders into the lens wall stops early. Carry it on from where it stopped, as
  // thin as it ended and angled back toward the middle, until it is near its tip.
  for (const dir of [0, Math.PI]) {
    const toward = dir === 0 ? 1 : -1
    const tip = dir === 0 ? width - SPRIG_TIP : SPRIG_TIP
    for (let tries = 0; tries < 4; tries++) {
      const canes = items.stems.filter(
        st => st.depth === 0 && Math.sign(st.pts[st.pts.length - 1][0] - width / 2) === toward,
      )
      const far = canes.reduce(
        (a, b) =>
          (b.pts[b.pts.length - 1][0] - width / 2) * toward >
          (a.pts[a.pts.length - 1][0] - width / 2) * toward
            ? b
            : a,
        canes[0],
      )
      if (!far) break
      const [ex, ey] = far.pts[far.pts.length - 1]
      if (Math.abs(tip - ex) < width * 0.06) break
      const inward = (ey > cy ? -1 : 1) * toward * 0.35
      const more = grow(f, opts, R, noise, [
        {
          x: ex,
          y: ey,
          dir: dir + inward,
          w: Math.max(0.4, far.ws[far.ws.length - 1] / o.stemW),
          len: (Math.abs(tip - ex) / 650) * 1.3,
          grace: o.margin + o.stemW,
        },
      ])
      for (const k of Object.keys(items) as (keyof Items)[])
        (items[k] as unknown[]).push(...(more[k] as unknown[]))
      if (!more.stems.some(st => st.depth === 0)) break
    }
  }
  return items
}

/** One vine `width` units long; exported with its options open for the sample sheets. */
export function grapevineRule(seed: string, width: number, o: OrnOptions = SPRIG): Element {
  return s(
    'svg',
    { ...SVG, class: 'gv-rule', viewBox: `0 0 ${width} ${SPRIG_H}` },
    // About 4.5 units to the pixel on the page, so 3 units of outline error is under a pixel.
    drawItems(grapevineItems(seed, width, o), DETAIL_W * 0.8, 3),
  )
}

function alembicRule(seed: string): Element {
  const key = `vine:${seed}`
  const { svg } = render(composePreset('vine', createRng(key), RULE_W), key, RULE_TOL)
  const el = fromHtml(svg, { fragment: true }).children.find(
    (n): n is Element => n.type === 'element',
  )!
  delete el.properties.width
  delete el.properties.height
  el.properties = {
    ...el.properties,
    ...SVG,
    class: 'gv-rule',
    preserveAspectRatio: 'xMidYMid slice',
  }
  return el
}

/**
 * The chapter separator. `grapevine` is one vine from the engine above, grown at each of
 * SPRIG_WIDTHS (data-size) for the page to pick from, sized by its own ratio. `alembic` is
 * alembic-ornaments' running vine cut RULE_W wide, parsed from its SVG string and kept at its
 * own scale (xMidYMid slice) across the full page.
 */
export function vineRule(seed: string, kind: RuleKind): Element[] {
  if (kind === 'alembic') return [alembicRule(seed)]
  return (Object.keys(SPRIG_WIDTHS) as SprigSize[]).map(size => {
    const el = grapevineRule(seed, SPRIG_WIDTHS[size])
    el.properties.dataSize = size
    return el
  })
}
