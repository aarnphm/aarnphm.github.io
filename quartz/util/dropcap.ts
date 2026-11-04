import type { Element, ElementContent } from 'hast'
import { h } from 'hastscript'
import { initialSettings, resolvePreset } from './grapevine/presets'

// Drop-cap faces and the markup of one initial, shared by the Profile transformer (ledes,
// chapter openers and ```dropcap specimen fences on note pages) and the latent reader, which
// shows the same specimens in a note's passages. Styles: quartz/styles/pages/newspaper/dropcap.scss.

/** Accepted `dropcap:` faces; other faces must appear in dropcap.scss. `none` disables them.
 * `procedural` picks a preset from the page. `procedural-<id>` uses that share, or any other
 * suffix as a seed that lands on one of the presets. */
const DROPCAPS = ['garamond', 'ebgaramond', 'goudy', 'yinit', 'dezs'] as const
type NamedDropcap = (typeof DROPCAPS)[number]

export interface DropcapChoice {
  face: NamedDropcap | 'procedural' | 'none'
  variant?: string
}

export function dropcapOf(value: unknown): DropcapChoice {
  if (value === 'none' || value === false) return { face: 'none' }
  const raw = typeof value === 'string' ? value.trim() : 'garamond'
  const name = raw.toLowerCase()
  if (name === 'procedural') return { face: 'procedural' }
  if (name.startsWith('procedural-')) return { face: 'procedural', variant: raw.slice(11) }
  return (DROPCAPS as readonly string[]).includes(name)
    ? { face: name as NamedDropcap }
    : { face: 'garamond' }
}

function fieldsOf(meta: string): Record<string, string> {
  const fields: Record<string, string> = {}
  for (const match of meta.matchAll(/([A-Za-z]+)=(?:"([^"]*)"|'([^']*)'|(\S+))/g)) {
    fields[match[1].toLowerCase()] = (match[2] ?? match[3] ?? match[4] ?? '').trim()
  }
  return fields
}

function choiceFromMeta(meta: string): DropcapChoice {
  const fields = fieldsOf(meta)
  const type = fields.type ?? 'garamond'
  if (fields.seed && type.toLowerCase().startsWith('procedural')) {
    return dropcapOf(`procedural-${fields.seed}`)
  }
  return dropcapOf(type)
}

export function initialNode(
  letter: string,
  choice: DropcapChoice,
  seed: string,
  kind: 'lede' | 'section',
): Element {
  const shown = letter.toUpperCase()
  const procedural = choice.face === 'procedural' ? proceduralInitial(shown, choice, seed) : null
  return h(
    'span.mag-initial',
    {
      dataLetter: shown,
      dataKind: kind,
      dataFace: choice.face === 'garamond' ? undefined : choice.face,
      ...(procedural
        ? {
            dataVariant: procedural.variant,
            style: `--mag-ink-a:${procedural.ink};--mag-paper:${procedural.paper}`,
          }
        : {}),
    },
    [shown, ...(procedural ? [procedural.canvas] : [])],
  )
}

// A procedural specimen is captioned by its seed alone, in italics: the one the fence names,
// or the preset the page seed lands on.
function specimenLabel(choice: DropcapChoice, pageSeed: string): ElementContent {
  if (choice.face !== 'procedural') return { type: 'text', value: choice.face }
  return h('i', choice.variant ?? resolvePreset(pageSeed).id)
}

function proceduralInitial(letter: string, choice: DropcapChoice, seed: string) {
  const key = choice.variant ? choice.variant : seed.replace(/:\d+$/, '')
  const preset = resolvePreset(key)
  const settings = initialSettings(preset, letter)
  const ink = typeof settings.inkA === 'string' ? settings.inkA : '#1a1614'
  const paper = typeof settings.paper === 'string' ? settings.paper : '#f3eee3'
  return {
    variant: preset.id,
    ink,
    paper,
    canvas: h('canvas.gv-initial', { dataSettings: JSON.stringify(settings), ariaHidden: 'true' }),
  }
}

/**
 * A ```dropcap fence (`type=goudy`, `type=procedural seed=…`) as the specimen the note page
 * shows: the fence's first letter as an initial over its caption. Null when it holds no letter.
 */
export function dropcapSpecimen(body: string, meta: string, pageSeed: string): Element | null {
  const letter = /[A-Za-z]/.exec(body)?.[0]
  if (!letter) return null
  const choice = choiceFromMeta(meta)
  return h('figure.dropcap-specimen', [
    initialNode(letter, choice, pageSeed, 'section'),
    h('figcaption', [specimenLabel(choice, pageSeed)]),
  ])
}
