import type { Element, ElementContent, Root } from 'hast'
import type { Code, Root as MdastRoot } from 'mdast'
import { toHtml } from 'hast-util-to-html'
import { h } from 'hastscript'
import { SKIP, visit } from 'unist-util-visit'
import { type DropcapChoice, dropcapOf, dropcapSpecimen, initialNode } from '../../../util/dropcap'
import { RULE_KINDS, type RuleKind } from '../../../util/grapevine'

export { type DropcapChoice, dropcapOf }

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

export function dropcapFences(tree: MdastRoot, pageSeed: string) {
  visit(tree, 'code', (node: Code, index, parent) => {
    if (node.lang !== 'dropcap' || index === undefined || !parent) return
    const specimen = dropcapSpecimen(node.value, node.meta ?? '', pageSeed)
    if (!specimen) return
    parent.children[index] = { type: 'html', value: toHtml(specimen) }
    return SKIP
  })
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
