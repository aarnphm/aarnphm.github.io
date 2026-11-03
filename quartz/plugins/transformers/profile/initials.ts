import type { Element, ElementContent, Root } from 'hast'
import type { Code, Root as MdastRoot } from 'mdast'
import { toHtml } from 'hast-util-to-html'
import { h } from 'hastscript'
import { SKIP, visit } from 'unist-util-visit'
import { RULE_KINDS, type RuleKind } from '../../../util/grapevine'
import { initialSettings, resolvePreset } from '../../../util/grapevine/presets'

const classes = (node: Element) =>
  Array.isArray(node.properties.className) ? node.properties.className.map(String) : []

const INITIAL = /^(["'\u201C\u2018]?)([A-Za-z])/
// Small caps: whole words, max 28 chars.
const LEADIN = /^[^\s]*(?:\s+[^\s]+){0,3}/
const LEADIN_MAX = 28

/** `separator: grapevine | alembic` sets the ornament above each chapter; anything else, none. */
export function separatorOf(value: unknown, seed: string) {
  const name = typeof value === 'string' ? value.toLowerCase() : ''
  return (RULE_KINDS as readonly string[]).includes(name)
    ? { kind: name as RuleKind, seed }
    : undefined
}

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

/** Wrap the lede and chapter openers in `span.mag-initial`; `::first-letter` cannot carry an
 * ornament. Chapters opening a testimonial band stay plain. The lede is the paragraph after a
 * callout (the dating page's tl;dr) or, on a page without one, its first paragraph before any
 * chapter head (an essay that opens straight into prose, its breaks marked by rules).
 */
export function initials(tree: Root, choice: DropcapChoice, seed: string) {
  if (choice.face === 'none') return
  let n = 0
  const els = tree.children.filter((c): c is Element => c.type === 'element')
  const firstHead = els.findIndex(el => el.tagName === 'h2')
  const head = firstHead === -1 ? els : els.slice(0, firstHead)
  const callout = head.some(el => el.tagName === 'blockquote' && classes(el).includes('callout'))
  const opening = callout ? undefined : head.find(el => el.tagName === 'p')
  if (opening) wrapInitial(opening, 'lede', choice, `${seed}:${n++}`)
  let seenHeading = false
  els.forEach((el, i) => {
    // A float between head and text leaves that paragraph the opener.
    const skip = classes(els[i + 1] ?? el).some(c => c.startsWith('float-')) ? 1 : 0
    const next = els[i + 1 + skip]
    const after = els[i + 2 + skip]
    const lede = !seenHeading && el.tagName === 'blockquote' && classes(el).includes('callout')
    const opener =
      el.tagName === 'h2' && !(after !== undefined && classes(after).includes('profile-pulls'))
    if (el.tagName === 'h2') seenHeading = true
    if (next?.tagName !== 'p' || !(lede || opener)) return
    wrapInitial(next, lede ? 'lede' : 'section', choice, `${seed}:${n++}`)
  })
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

function initialNode(
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

export function dropcapFences(tree: MdastRoot, pageSeed: string) {
  visit(tree, 'code', (node: Code, index, parent) => {
    if (node.lang !== 'dropcap' || index === undefined || !parent) return
    const letter = /[A-Za-z]/.exec(node.value)?.[0]
    if (!letter) return
    const choice = choiceFromMeta(node.meta ?? '')
    const specimen = h('figure.dropcap-specimen', [
      initialNode(letter, choice, pageSeed, 'section'),
      h('figcaption', [specimenLabel(choice, pageSeed)]),
    ])
    parent.children[index] = { type: 'html', value: toHtml(specimen) }
    return SKIP
  })
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

function wrapInitial(
  block: Element,
  kind: 'lede' | 'section',
  choice: DropcapChoice,
  seed: string,
): boolean {
  const first = block.children[0]
  if (!first) return false
  if (first.type === 'element') {
    return (
      ['em', 'strong', 'a', 'span'].includes(first.tagName) &&
      wrapInitial(first, kind, choice, seed)
    )
  }
  if (first.type !== 'text') return false
  const match = INITIAL.exec(first.value)
  if (!match) return false
  const [, quote, letter] = match
  const rest = first.value.slice(match[0].length)
  // End on a whole word within budget; skip if the first word exceeds it. CSS sets small caps.
  let leadin = LEADIN.exec(rest)?.[0] ?? ''
  while (leadin.length > LEADIN_MAX && /\s/.test(leadin))
    leadin = leadin.slice(0, leadin.search(/\s+\S+$/))
  const wrapped: ElementContent[] = [initialNode(letter, choice, seed, kind)]
  if (quote) wrapped.unshift(h('span.mag-initial-punct', quote))
  if (leadin.length > 0 && leadin.length <= LEADIN_MAX) {
    wrapped.push(h('span.mag-leadin', leadin))
    first.value = rest.slice(leadin.length)
  } else first.value = rest
  block.children.splice(0, 1, ...wrapped, first)
  return true
}
