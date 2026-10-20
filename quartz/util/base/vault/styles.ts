// Materials of the film vault: tone ladders per theme, gradient finishes, grain tiles and the fixed
// label colours. Every gradient and pattern is a shared def; the scene's <style> sets the stop
// colours for both themes, so no element carries its own filter or colour.
import type { Element } from 'hast'
import { s } from 'hastscript'
import { COS30, rgbToHex } from '../bookcase/geometry'
import { labelArc } from './layout'

export const VAULT_MATERIALS = ['rack-steel', 'black-lacquer', 'tin', 'hammertone', 'poly'] as const
export type VaultMaterial = (typeof VAULT_MATERIALS)[number]
export type VaultTone = 'top' | 'face' | 's2'

interface Ladder {
  top: string
  face: string
  s3: string
  s2: string
  s1: string
  s0: string
  mid: string
  faint: string
  lo: string
  hi: string
  halo: string
}

const ladderOf = (row: string, halo: string): Ladder => {
  const [top, face, s3, s2, s1, s0, mid, faint, lo, hi] = row.split(' ')
  return { top, face, s3, s2, s1, s0, mid, faint, lo, hi, halo }
}

const LADDERS: Record<'light' | 'dark', Record<VaultMaterial, Ladder>> = {
  light: {
    'rack-steel': ladderOf(
      '#E3E1D9 #D2D0C7 #C4C2B9 #B3B1A8 #A19F97 #8F8D86 #5C5B56 #B8B6AE #8A8880 #282726',
      '#3A3936',
    ),
    'black-lacquer': ladderOf(
      '#3C3B38 #2C2B29 #1C1B1A #181716 #151413 #100F0F #8E8C85 #4E4D49 #696863 #FFFCF0',
      '#000000',
    ),
    tin: ladderOf(
      '#D7DADA #C3C7C8 #B4B8B9 #A3A8A9 #929798 #818687 #4A4F51 #AEB2B3 #868B8C #22272A',
      '#2E3336',
    ),
    hammertone: ladderOf(
      '#9A9B97 #80817D #747571 #676864 #5B5C58 #4F504C #DCDAD1 #86877F #6C6D68 #FFFCF0',
      '#1F1F1D',
    ),
    poly: ladderOf(
      '#EDEAE0 #DEDBD0 #D1CEC3 #C2BFB4 #B2AFA5 #A29F95 #5F5E58 #C7C4B9 #9B998F #282726',
      '#45443F',
    ),
  },
  dark: {
    'rack-steel': ladderOf(
      '#6B6A66 #5A5955 #504F4B #454440 #3B3A37 #31302D #B5B3AB #6E6D68 #8A8882 #E6E4DA',
      '#000000',
    ),
    'black-lacquer': ladderOf(
      '#454441 #33322F #252423 #1C1B1A #171616 #100F0F #A4A29A #575652 #777570 #FFFCF0',
      '#000000',
    ),
    tin: ladderOf(
      '#8E9394 #767B7C #6A6F70 #5C6162 #4F5354 #424647 #C9CDCE #7F8485 #9A9FA0 #EEF1F1',
      '#000000',
    ),
    hammertone: ladderOf(
      '#767773 #616260 #565755 #4B4C4A #414240 #373836 #C4C2B9 #6B6C68 #83847F #E6E4DA',
      '#000000',
    ),
    poly: ladderOf(
      '#7D7B74 #6B6963 #5F5D57 #52504B #46443F #3A3834 #C4C2B9 #78766F #918F87 #E6E4DA',
      '#000000',
    ),
  },
}

/** Gradient stops from the lit edge: offset, and a weight toward white (+) or the halo (−). */
const FINISH_STOPS: Record<VaultMaterial, readonly (readonly [number, number])[]> = {
  'rack-steel': [
    [0, 0.08],
    [1, -0.1],
  ],
  'black-lacquer': [
    [0, 0.34],
    [0.16, 0.1],
    [0.5, -0.02],
    [1, -0.22],
  ],
  tin: [
    [0, 0.26],
    [0.3, 0.04],
    [0.55, -0.12],
    [1, -0.42],
  ],
  hammertone: [
    [0, 0.2],
    [0.35, 0],
    [1, -0.38],
  ],
  poly: [
    [0, 0.1],
    [1, -0.16],
  ],
}

/** Seamless RGBA tiles from docs/verification/film-vault/gen-can.py, and their world size. */
const GRAINS: Partial<Record<VaultMaterial, { image: string; size: number }>> = {
  tin: { image: 'spangle.webp', size: 96 },
  hammertone: { image: 'dimple.webp', size: 48 },
}

export const GRAIN_ASSETS = '/static/vault-grain/'

export const hasGrain = (material: VaultMaterial) => material in GRAINS

/** Fixed object colours: label stock, ink and stickers. Labels dim a little in the dark theme. */
export const LABELS = {
  vinyl: { fill: '#FFFCF0', ink: '#100F0F', dark: '#EBE8DC' },
  tape: { fill: '#F1E9D2', ink: '#26231F', dark: '#DED6BF' },
  stamp: '#AF3029',
  dotHigh: '#FDB2A2',
  dotGood: '#CDD597',
} as const

function hexRgb(hex: string): [number, number, number] {
  return [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16)) as [number, number, number]
}

function mixHex(a: string, b: string, t: number): string {
  const [ar, ag, ab] = hexRgb(a)
  const [br, bg, bb] = hexRgb(b)
  return rgbToHex(ar + (br - ar) * t, ag + (bg - ag) * t, ab + (bb - ab) * t)
}

const TONES: readonly VaultTone[] = ['top', 'face', 's2']
const stopVar = (m: VaultMaterial, tone: VaultTone, i: number) => `--cn-${m}-${tone}-${i}`

/** Fill of a face of `material` in `tone`. Lids and box tops run diagonally, the rest top down. */
export const toneFill = (material: VaultMaterial, tone: VaultTone) =>
  `url(#cn-g-${material}-${tone})`

/** Shared defs: gradients, grain patterns, the lid rings and the two label arcs. */
export function vaultDefs(materials: Iterable<VaultMaterial>, assets = GRAIN_ASSETS): Element {
  const defs: Element[] = []
  for (const material of new Set(materials)) {
    for (const tone of TONES) {
      const diagonal = tone !== 'face'
      defs.push(
        s(
          'linearGradient',
          { id: `cn-g-${material}-${tone}`, x1: 0, y1: 0, x2: diagonal ? 1 : 0, y2: 1 },
          FINISH_STOPS[material].map(([offset], i) =>
            s('stop', { offset, style: `stop-color:var(${stopVar(material, tone, i)})` }),
          ),
        ),
      )
    }
    const grain = GRAINS[material]
    if (grain) {
      // The tile lies in the rim's plane: tile x along the run, tile y down the can.
      defs.push(
        s(
          'pattern',
          {
            id: `cn-grain-${material}`,
            patternUnits: 'userSpaceOnUse',
            width: grain.size,
            height: grain.size,
            patternTransform: `matrix(${COS30.toFixed(4)} 0.5 0 1 0 0)`,
          },
          [
            s('image', {
              href: `${assets}${grain.image}`,
              width: grain.size,
              height: grain.size,
              preserveAspectRatio: 'none',
            }),
          ],
        ),
      )
    }
  }
  // Embossed lid: concentric pressed rings, alternately catching and losing the light.
  defs.push(
    s(
      'radialGradient',
      { id: 'cn-lid-rings' },
      Array.from({ length: 14 }, (_, i) =>
        s('stop', {
          offset: Math.round((0.18 + (i / 13) * 0.8) * 1000) / 1000,
          'stop-color': i % 2 ? '#000000' : '#FFFFFF',
          'stop-opacity': i === 0 || i === 13 ? 0 : 0.08,
        }),
      ),
    ),
    s('path', { id: 'cn-arc-190', d: labelArc(190) }),
    s('path', { id: 'cn-arc-135', d: labelArc(135) }),
  )
  return s('defs', defs)
}

/** Edge tokens per material and every gradient stop colour, for both themes. */
export function vaultCss(
  materials: Iterable<VaultMaterial>,
  scope = '.vault-scene',
  dark = ":root[saved-theme='dark'] ",
): string {
  const set = new Set(materials)
  const rules: string[] = []
  for (const theme of ['light', 'dark'] as const) {
    const root = theme === 'dark' ? dark : ''
    const stops: string[] = []
    for (const material of set) {
      const ladder = LADDERS[theme][material]
      rules.push(
        `${root}${scope} .cn-m-${material}{--cn-hi:${ladder.hi};--cn-mid:${ladder.mid};--cn-faint:${ladder.faint};--cn-s3:${ladder.s3}}`,
      )
      for (const tone of TONES)
        FINISH_STOPS[material].forEach(([, w], i) => {
          const toward = w > 0 ? '#FFFFFF' : ladder.halo
          // The dark theme keeps its sheen lower so a lit can does not glare.
          const k = Math.abs(w) * (theme === 'dark' && w > 0 ? 0.55 : 1)
          stops.push(`${stopVar(material, tone, i)}:${mixHex(ladder[tone], toward, k)}`)
        })
    }
    const { vinyl, tape } = LABELS
    const labels =
      theme === 'dark'
        ? `--cn-vinyl:${vinyl.dark};--cn-tape:${tape.dark}`
        : `--cn-vinyl:${vinyl.fill};--cn-tape:${tape.fill}`
    stops.push(
      labels,
      `--cn-vinyl-ink:${vinyl.ink};--cn-tape-ink:${tape.ink};--cn-stamp:${LABELS.stamp}`,
      `--cn-dot-high:${LABELS.dotHigh};--cn-dot-good:${LABELS.dotGood}`,
    )
    rules.push(`${root}${scope}{${stops.join(';')}}`)
  }
  return rules.join('')
}
