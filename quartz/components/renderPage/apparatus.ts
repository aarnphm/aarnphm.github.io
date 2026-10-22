import type { Element, ElementContent, Root } from 'hast'
import { toString } from 'hast-util-to-string'
import { h } from 'hastscript'

// The margin apparatus, Locher style, shared by /fr (fr-margins.ts) and technical notes
// (technical-apparatus.ts): each prose block that carries apparatus opens with its number `(n)` and one
// group of `§ label (n)` entries. The `fr-*` classes are the contract with fr-margins.inline.ts, which
// hangs the groups left of the text or folds them into the block, and with fr-margins.scss.

// A margin entry hangs from a prose block; a header cell is no place for one.
// a quotation's first paragraph carries the number, so it opens a line of text, never its own
export const ANCHORS = new Set(['p', 'li', 'td'])

export const classesOf = (node: Element): string[] => {
  const value = node.properties?.className
  return Array.isArray(value) ? value.map(String) : typeof value === 'string' ? [value] : []
}

/** Depth-first walk that skips sidenote bodies, so margin text never counts as page text. */
export function walk(
  node: Element | Root,
  fn: (el: Element, parent: Element | Root) => void | false,
) {
  for (const child of node.children) {
    if (child.type !== 'element') continue
    if (classesOf(child).includes('sidenote-content')) continue
    if (fn(child, node) === false) continue
    walk(child, fn)
  }
}

/** First prose block of a container, if it has one. */
export function firstAnchor(container: Element | Root): Element | undefined {
  let found: Element | undefined
  walk(container, el => {
    if (found) return false
    if (ANCHORS.has(el.tagName) && toString(el).trim()) found = el
  })
  return found
}

/** First prose block of a container, or the container itself when it has none. */
export const anchorOf = (container: Element): Element => firstAnchor(container) ?? container

export interface ApparatusEntry {
  kind: string
  label: string
  body: ElementContent
}

/** Entries waiting for their block, collected in any order; `hang` numbers them. */
export class Apparatus {
  readonly pending = new Map<Element, ApparatusEntry[]>()

  put(block: Element, entry: ApparatusEntry) {
    const list = this.pending.get(block)
    if (list) list.push(entry)
    else this.pending.set(block, [entry])
  }

  /**
   * Numbers the blocks in document order, whichever pass found the block first, and opens each with
   * its number and its group, entries sorted by `rank`. Everything inside a block is a span: a <div>
   * would end the paragraph when the browser parses it.
   */
  hang(tree: Root, rank: (kind: string) => number) {
    const ordered: Element[] = []
    walk(tree, el => {
      if (this.pending.has(el)) ordered.push(el)
    })
    ordered.forEach((block, i) => {
      const n = i + 1
      const entries = this.pending.get(block)!.sort((a, b) => rank(a.kind) - rank(b.kind))
      block.properties = {
        ...block.properties,
        className: [...classesOf(block), 'fr-anchor'],
        dataFrN: String(n),
      }
      // the number and the groups open the block: beside it in the margin, or in parentheses before
      // the text where there is no margin
      // one column, left of the text: the side the page grid leaves room for from its desktop width
      block.children.unshift(
        h('span.fr-n', { ariaHidden: 'true' }, `(${n})`),
        h(
          'span.fr-group',
          { dataN: String(n) },
          entries.map(entry =>
            h('span.fr-entry', { dataFrKind: entry.kind }, [
              h(
                'button.fr-entry-label',
                { type: 'button', ariaExpanded: 'false' },
                `§ ${entry.label} (${n})`,
              ),
              h('span.fr-entry-body', [entry.body]),
            ]),
          ),
        ),
      )
    })
  }
}
