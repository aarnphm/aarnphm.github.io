// Materials of the film vault: flat tone ladders per theme and the fixed label colours. Faces take
// a tone class that the scene's <style> fills for both themes. Paint servers are kept to the one
// handwheel: about 700 gradient fills cost the rack 30 times the raster time of flat ones.
import type { Element } from 'hast'
import { s } from 'hastscript'
import { labelArc } from './layout'

export const VAULT_MATERIALS = ['rack-steel', 'black-lacquer', 'poly'] as const
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
}

const ladderOf = (row: string): Ladder => {
  const [top, face, s3, s2, s1, s0, mid, faint, lo, hi] = row.split(' ')
  return { top, face, s3, s2, s1, s0, mid, faint, lo, hi }
}

const LADDERS: Record<'light' | 'dark', Record<VaultMaterial, Ladder>> = {
  light: {
    'rack-steel': ladderOf(
      '#E3E1D9 #D2D0C7 #C4C2B9 #B3B1A8 #A19F97 #8F8D86 #5C5B56 #B8B6AE #8A8880 #282726',
    ),
    'black-lacquer': ladderOf(
      '#3C3B38 #2C2B29 #1C1B1A #181716 #151413 #100F0F #8E8C85 #4E4D49 #696863 #FFFCF0',
    ),
    // Moulded grey polypropylene: a mid-grey lid, a lighter skirt and lighter ribs.
    poly: ladderOf(
      '#9A9A98 #8A8A88 #7E7E7C #727270 #686866 #5E5E5C #474745 #8E8E8C #5A5A58 #3A3A39',
    ),
  },
  dark: {
    'rack-steel': ladderOf(
      '#6B6A66 #5A5955 #504F4B #454440 #3B3A37 #31302D #B5B3AB #6E6D68 #8A8882 #E6E4DA',
    ),
    'black-lacquer': ladderOf(
      '#454441 #33322F #252423 #1C1B1A #171616 #100F0F #A4A29A #575652 #777570 #FFFCF0',
    ),
    poly: ladderOf(
      '#76767A #68686B #5E5E61 #545457 #4B4B4E #424245 #2C2C2E #6E6E71 #4A4A4D #C8C8CB',
    ),
  },
}

/** Fixed object colours: label stock, ink and stickers. Labels dim a little in the dark theme. */
export const LABELS = {
  vinyl: { fill: '#FFFCF0', ink: '#100F0F', dark: '#EBE8DC' },
  tape: { fill: '#F1E9D2', ink: '#26231F', dark: '#DED6BF' },
  stamp: '#AF3029',
  dotHigh: '#FDB2A2',
  dotGood: '#CDD597',
} as const

const TONES: readonly VaultTone[] = ['top', 'face', 's2']

/** Class that fills a face of `material` in `tone`. */
export const toneClass = (material: VaultMaterial, tone: VaultTone) => `cn-f-${material}-${tone}`

/** Shared defs: the handwheel's finishes and the two label arcs. */
export function vaultDefs(): Element {
  const defs: Element[] = []
  // Lathe rings on the handwheel's turned faces, alternately catching and losing the light.
  defs.push(
    s(
      'radialGradient',
      { id: 'cn-turning' },
      Array.from({ length: 14 }, (_, i) =>
        s('stop', {
          offset: Math.round((0.18 + (i / 13) * 0.8) * 1000) / 1000,
          'stop-color': i % 2 ? '#000000' : '#FFFFFF',
          'stop-opacity': i === 0 || i === 13 ? 0 : 0.08,
        }),
      ),
    ),
    // Handwheel finishes: enamel lit from the upper left, turned steel with a bright band.
    s('linearGradient', { id: 'cn-crank-enamel', x1: 0, y1: 0, x2: 1, y2: 1 }, [
      s('stop', { offset: 0, style: 'stop-color:var(--cn-crank-hi)' }),
      s('stop', { offset: 0.45, style: 'stop-color:var(--cn-crank-mid)' }),
      s('stop', { offset: 1, style: 'stop-color:var(--cn-crank-lo)' }),
    ]),
    s('linearGradient', { id: 'cn-crank-steel', x1: 1, y1: 0, x2: 0, y2: 1 }, [
      s('stop', { offset: 0, style: 'stop-color:var(--cn-steel-lo)' }),
      s('stop', { offset: 0.3, style: 'stop-color:var(--cn-steel-hi)' }),
      s('stop', { offset: 0.55, style: 'stop-color:var(--cn-steel-mid)' }),
      s('stop', { offset: 1, style: 'stop-color:var(--cn-steel-lo)' }),
    ]),
    s('radialGradient', { id: 'cn-crank-face', cx: 0.38, cy: 0.34, r: 0.75 }, [
      s('stop', { offset: 0, style: 'stop-color:var(--cn-steel-hi)' }),
      s('stop', { offset: 0.6, style: 'stop-color:var(--cn-steel-mid)' }),
      s('stop', { offset: 1, style: 'stop-color:var(--cn-steel-lo)' }),
    ]),
    s('path', { id: 'cn-arc-190', d: labelArc(190) }),
    s('path', { id: 'cn-arc-135', d: labelArc(135) }),
  )
  return s('defs', defs)
}

/** Edge tokens and face fills per material, and the label colours, for both themes. */
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
        `${root}${scope} .cn-m-${material}{--cn-hi:${ladder.hi};--cn-mid:${ladder.mid};--cn-faint:${ladder.faint};--cn-s3:${ladder.s3};--cn-top:${ladder.top}}`,
      )
      for (const tone of TONES)
        rules.push(`${root}${scope} .${toneClass(material, tone)}{fill:${ladder[tone]}}`)
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
