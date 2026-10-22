import type { Element, ElementContent, Root } from 'hast'
import { h } from 'hastscript'
import { ANCHORS, Apparatus, classesOf } from './apparatus'

// content a <p> may hold; anything else ends the paragraph when the browser parses the page
const PHRASING = new Set(
  'a abbr b bdi bdo br button cite code data del dfn em i img input ins kbd label mark q s samp small span strong sub sup time u var wbr audio video'.split(
    ' ',
  ),
)
// foreign content: its children are not HTML
const FOREIGN = new Set(['svg', 'math'])

export const isElement = (node: { type: string }): node is Element => node.type === 'element'

/** A footnote's paragraphs as spans, or undefined when it holds blocks a paragraph cannot. */
function phrasing(nodes: ElementContent[]): ElementContent[] | undefined {
  const out: ElementContent[] = []
  for (const node of nodes) {
    if (!isElement(node)) {
      out.push(node)
      continue
    }
    if (node.properties?.dataFootnoteBackref != null) continue
    if (FOREIGN.has(node.tagName)) {
      out.push(node)
      continue
    }
    const children = phrasing(node.children)
    if (!children) return undefined
    if (node.tagName === 'p') out.push(h('span.fr-para', children))
    else if (PHRASING.has(node.tagName)) out.push({ ...node, children })
    else return undefined
  }
  // a trailing space before a removed back reference
  const last = out.at(-1)
  if (last?.type === 'text') last.value = last.value.trimEnd()
  return out
}

interface Mark {
  /** the element that opens the note in the text, and the chain of its ancestors */
  node: Element
  path: Element[]
}

/** Footnote references and sidenotes in document order, outside the footnote list itself. */
function marksOf(tree: Root): Mark[] {
  const out: Mark[] = []
  const visit = (node: Element | Root, path: Element[]) => {
    for (const child of node.children) {
      if (!isElement(child)) continue
      const classes = classesOf(child)
      if (classes.includes('sidenote-content')) continue
      if (child.tagName === 'section' && child.properties?.dataFootnotes != null) continue
      if (child.properties?.dataFootnoteRef != null || classes.includes('sidenote')) {
        out.push({ node: child, path })
        continue
      }
      visit(child, [...path, child])
    }
  }
  visit(tree, [])
  return out
}

export const anchorIn = (path: Element[]) => path.findLast(el => ANCHORS.has(el.tagName))

function footnotesOf(tree: Root): Map<string, { item: Element; list: Element }> {
  const found = new Map<string, { item: Element; list: Element }>()
  const visit = (node: Element | Root) => {
    for (const child of node.children) {
      if (!isElement(child)) continue
      if (child.tagName === 'section' && child.properties?.dataFootnotes != null) {
        for (const list of child.children.filter(isElement))
          if (list.tagName === 'ol')
            for (const item of list.children.filter(isElement))
              if (item.properties?.id) found.set(String(item.properties.id), { item, list })
        continue
      }
      visit(child)
    }
  }
  visit(tree)
  return found
}

export const remove = (parent: Element, node: Element) => {
  const at = parent.children.indexOf(node)
  if (at >= 0) parent.children.splice(at, 1)
}

export interface NotesEntry {
  kind: string
  label: string
  /** also take the page's sidenotes; /fr keeps its sidenotes as boxes in the margin */
  sidenotes?: boolean
}

/**
 * Notes of the page, each in the group of the block that calls it: footnote text (the marker leaves
 * the text, the block's number is the only number) and, when asked, sidenotes (the label stays as
 * words in the sentence). A footnote whose text holds blocks a paragraph cannot, such as a listing
 * or a list, keeps its marker and its place at the end of the page.
 */
export function takeNotes(tree: Root, apparatus: Apparatus, entry: NotesEntry) {
  const footnotes = footnotesOf(tree)
  const taken = new Set<Element>()
  const notes = new Map<Element, ElementContent[]>()
  const add = (block: Element, note: ElementContent) => {
    const list = notes.get(block)
    if (list) list.push(note)
    else notes.set(block, [note])
  }
  for (const { node, path } of marksOf(tree)) {
    const block = anchorIn(path)
    const parent = path.at(-1)
    if (!block || !parent) continue
    if (node.properties?.dataFootnoteRef != null) {
      const target = footnotes.get(String(node.properties.href ?? '').replace(/^#/, ''))
      const body = target && phrasing(target.item.children)
      if (!target || !body) continue
      // the marker is <sup><a/></sup>; both leave the text
      const sup = parent.tagName === 'sup' ? parent : undefined
      if (sup && path.at(-2)) remove(path.at(-2)!, sup)
      else remove(parent, node)
      add(
        block,
        h(
          'span.fr-gloss',
          { id: taken.has(target.item) ? undefined : target.item.properties?.id },
          body,
        ),
      )
      taken.add(target.item)
      continue
    }
    if (!entry.sidenotes || node.properties?.dataForceInline != null) continue
    const id = String(node.properties?.id ?? '')
    const content = parent.children.find(
      (c): c is Element =>
        isElement(c) &&
        classesOf(c).includes('sidenote-content') &&
        String(c.properties?.dataSidenoteFor) === id,
    )
    if (!content) continue
    const label = node.children.find(
      (c): c is Element => isElement(c) && classesOf(c).includes('sidenote-label'),
    )
    const words =
      label && label.properties?.dataAuto == null
        ? label.children.filter(c => !(isElement(c) && classesOf(c).includes('sidenote-arrow')))
        : []
    const at = parent.children.indexOf(node)
    parent.children.splice(at, 1, ...words)
    remove(parent, content)
    add(
      block,
      h('span.fr-gloss', [
        ...(words.length > 0 ? [h('b', words), ' · '] : []),
        ...content.children,
      ]),
    )
  }
  for (const { item, list } of footnotes.values()) if (taken.has(item)) remove(list, item)
  // an emptied footnote list leaves with its section
  const prune = (node: Element | Root) => {
    node.children = node.children.filter(child => {
      if (!isElement(child)) return true
      if (child.tagName === 'section' && child.properties?.dataFootnotes != null)
        return child.children.some(
          c => isElement(c) && c.tagName === 'ol' && c.children.some(isElement),
        )
      prune(child)
      return true
    }) as typeof node.children
  }
  if (taken.size > 0) prune(tree)
  for (const [block, items] of notes)
    apparatus.put(block, { kind: entry.kind, label: entry.label, body: h('span.fr-margin', items) })
}
