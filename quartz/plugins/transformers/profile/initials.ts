import type { Element, ElementContent, Root } from 'hast'
import { h } from 'hastscript'
import { RULE_KINDS, vineInitial, type RuleKind } from '../../../util/grapevine'

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

/** Accepted `dropcap:` faces; other faces must appear in dropcap.scss. `none` disables them. */
const DROPCAPS = ['garamond', 'ebgaramond', 'goudy', 'yinit', 'dezs', 'procedural'] as const
type Dropcap = (typeof DROPCAPS)[number] | 'none'

export function dropcapOf(value: unknown): Dropcap {
  if (value === 'none' || value === false) return 'none'
  const name = typeof value === 'string' ? value.toLowerCase() : 'garamond'
  return (DROPCAPS as readonly string[]).includes(name) ? (name as Dropcap) : 'garamond'
}

/** Wrap the lede and chapter openers in `span.mag-initial`; `::first-letter` cannot carry an
 * ornament. Chapters opening a testimonial band stay plain.
 */
export function initials(tree: Root, face: Dropcap, seed: string) {
  if (face === 'none') return
  let n = 0
  const els = tree.children.filter((c): c is Element => c.type === 'element')
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
    wrapInitial(next, lede ? 'lede' : 'section', face, `${seed}:${n++}`)
  })
}

function wrapInitial(
  block: Element,
  kind: 'lede' | 'section',
  face: Dropcap,
  seed: string,
): boolean {
  const first = block.children[0]
  if (!first) return false
  if (first.type === 'element') {
    return (
      ['em', 'strong', 'a', 'span'].includes(first.tagName) && wrapInitial(first, kind, face, seed)
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
  const wrapped: ElementContent[] = [
    h(
      'span.mag-initial',
      {
        dataLetter: letter.toUpperCase(),
        dataKind: kind,
        dataFace: face === 'garamond' ? undefined : face,
      },
      // The procedural face draws its tile here, seeded by page and position, so it is final at
      // first paint and the same on every build.
      [letter, ...(face === 'procedural' ? [vineInitial(letter, seed)] : [])].filter(
        c => c !== null,
      ),
    ),
  ]
  if (quote) wrapped.unshift(h('span.mag-initial-punct', quote))
  if (leadin.length > 0 && leadin.length <= LEADIN_MAX) {
    wrapped.push(h('span.mag-leadin', leadin))
    first.value = rest.slice(leadin.length)
  } else first.value = rest
  block.children.splice(0, 1, ...wrapped, first)
  return true
}
