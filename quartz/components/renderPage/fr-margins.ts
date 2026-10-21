import type { Element, ElementContent, Root } from 'hast'
import { toString } from 'hast-util-to-string'
import { h } from 'hastscript'
import { buildSidenoteHast } from '../../plugins/transformers/sidenotes'
import { QuartzComponentProps } from '../../types/component'
import { renderDataFor } from '../../util/ctx'
import { parseFlashcards } from '../../util/flashcards'
import { flashcardsSlug } from '../../util/flashcards-path'
import { assignCards, foldText, FrSection, lessonLabel } from '../../util/fr-margins'
import { FullSlug, resolveRelative, simplifySlug } from '../../util/path'

type Ctx = QuartzComponentProps['ctx']

const SIDENOTE_ID_BASE = 9000
const GLOSS_HEADING = 'comprendre-les-expressions'
const BLOCKS = new Set(['p', 'li', 'td', 'th', 'blockquote'])

const classesOf = (node: Element): string[] => {
  const value = node.properties?.className
  return Array.isArray(value) ? value.map(String) : typeof value === 'string' ? [value] : []
}

const levelOf = (node: Element): number => Number(node.properties?.dataLevel ?? 0)

const isSection = (node: ElementContent, level?: number): node is Element =>
  node.type === 'element' &&
  node.tagName === 'section' &&
  classesOf(node).includes('collapsible-header') &&
  (level === undefined || levelOf(node) === level)

/** Depth-first walk that skips sidenote bodies, so margin text never counts as page text. */
function walk(node: Element | Root, fn: (el: Element, parent: Element | Root) => void | false) {
  for (const child of node.children) {
    if (child.type !== 'element') continue
    if (classesOf(child).includes('sidenote-content')) continue
    if (fn(child, node) === false) continue
    walk(child, fn)
  }
}

function sectionsOf(tree: Root): { node: Element; info: FrSection; body: Element }[] {
  const out: { node: Element; info: FrSection; body: Element }[] = []
  walk(tree, el => {
    if (!isSection(el, 2)) return
    let body: Element | undefined
    walk(el, inner => {
      if (!body && classesOf(inner).includes('collapsible-header-content') && levelOf(inner) === 2)
        body = inner
    })
    if (!body) return
    const headingIds: string[] = []
    walk(el, inner => {
      if (isSection(inner) && levelOf(inner) > 2) headingIds.push(String(inner.properties?.id))
    })
    out.push({
      node: el,
      body,
      info: { id: String(el.properties?.id), headingIds, text: foldText(toString(body)) },
    })
    return false
  })
  return out
}

function anchored(
  label: string,
  kind: 'rev' | 'gloss' | 'cite',
  content: ElementContent[],
  baseId: string,
  serial: number,
): ElementContent[] {
  const [span, body] = buildSidenoteHast(
    [],
    false,
    kind === 'rev',
    kind !== 'rev',
    content,
    SIDENOTE_ID_BASE + serial,
    baseId,
  )
  span.properties = { ...span.properties, dataFr: kind, ariaLabel: label }
  return [span, body]
}

function reviewMargin(deck: FullSlug, section: string, ids: string[]): Element {
  return h(
    'div.fr-margin.fr-rev',
    { dataFrRev: '', dataDeck: deck, dataIds: ids.join(','), dataSection: section },
    [
      h('span.fr-label', [
        'révision',
        ' ',
        h('span.fr-count', { dataFrCount: '' }, String(ids.length)),
      ]),
      h('div.fr-card', { dataFrCard: '', hidden: true }),
    ],
  )
}

function notesMargin(section: string): Element {
  return h('div.fr-margin.fr-notes', { dataFrNotes: '', dataSection: section }, [
    h('span.fr-label', 'mes notes'),
    h('div.fr-note-list', { dataFrNoteList: '' }),
    h('button.fr-btn', { type: 'button', dataFrAct: 'add' }, 'ajouter'),
  ])
}

interface Gloss {
  term: string
  sense: string
  usage: string
}

function glossesOf(tree: Root): Gloss[] {
  let table: Element | undefined
  walk(tree, el => {
    if (table) return false
    if (isSection(el) && String(el.properties?.id).startsWith(GLOSS_HEADING)) {
      walk(el, inner => {
        if (!table && inner.tagName === 'table') table = inner
      })
      return false
    }
  })
  if (!table) return []
  const rows: Gloss[] = []
  walk(table, tr => {
    if (tr.tagName !== 'tr') return
    const cells = tr.children.filter(
      (c): c is Element => c.type === 'element' && (c.tagName === 'td' || c.tagName === 'th'),
    )
    if (cells.length < 2 || cells[0].tagName === 'th') return
    rows.push({
      term: toString(cells[0]).trim(),
      sense: toString(cells[1]).trim(),
      usage: cells[2] ? toString(cells[2]).trim() : '',
    })
    return false
  })
  return rows
}

const stripTail = (value: string) => value.replace(/[.!?…]+$/, '').trim()

/** First block outside the gloss table that uses `term`, in document order. */
function firstUse(tree: Root, terms: Gloss[]): Map<Gloss, Element> {
  const found = new Map<Gloss, Element>()
  const needles = terms.map(gloss => ({ gloss, needle: foldText(stripTail(gloss.term)) }))
  walk(tree, el => {
    if (isSection(el) && String(el.properties?.id).startsWith(GLOSS_HEADING)) return false
    if (!BLOCKS.has(el.tagName)) return
    const body = foldText(toString(el))
    for (const { gloss, needle } of needles) {
      if (needle.length >= 3 && !found.has(gloss) && body.includes(needle)) found.set(gloss, el)
    }
  })
  return found
}

// A gloss lands inside the block that first uses its term, often a <p>; a <div> there would end the
// paragraph when the browser parses it, leaving the sidenote empty and the gloss in the prose.
function glossMargin(items: Gloss[]): Element {
  return h(
    'span.fr-margin.fr-glosses',
    items.map(item =>
      h('span.fr-gloss', [
        h('b', stripTail(item.term)),
        ' · ',
        item.sense,
        item.usage ? h('span.fr-meta', item.usage) : null,
      ]),
    ),
  )
}

function citedByMargin(links: { slug: FullSlug; label: string }[], from: FullSlug): Element {
  return h('div.fr-margin.fr-cited', [
    h('span.fr-label', 'cité par'),
    ...links.map(link =>
      h('div.fr-gloss', [
        h(
          'a.internal',
          { href: resolveRelative(from, link.slug), dataSlug: link.slug },
          link.label,
        ),
      ]),
    ),
  ])
}

/**
 * Left margin per `##`: reviewing the cards of the twin deck that cite the page, plus the
 * reader's notes. Right margin: the page's "comprendre les expressions" rows at the first use of
 * each expression, and the lessons that cite the page. Decks are only visible at render time
 * (`ctx.decks`), so this runs here rather than in a transformer.
 */
export function renderFrMargins(tree: Root, componentData: QuartzComponentProps, ctx: Ctx) {
  const { fileData, allFiles } = componentData
  const slug = fileData.slug as FullSlug | undefined
  if (!slug || !slug.startsWith('fr/') || fileData.flashcards) return
  const sections = sectionsOf(tree)
  if (sections.length === 0) return

  let serial = 0
  const baseId = (kind: string, key: string) =>
    `sidenote-fr-${kind}-${key.replace(/[^A-Za-z0-9_-]/g, '-')}`

  const deck = ctx.decks?.find(d => d.flashcards?.sourceSlug === slug)
  const deckSlug = flashcardsSlug(slug)
  const cards = deck ? parseFlashcards(deck.rawMarkdownSource ?? '').cards : []
  const assigned = assignCards(
    cards,
    slug,
    sections.map(s => s.info),
  )

  for (const { body, info } of sections) {
    const ids = assigned.get(info.id) ?? []
    const content: Element[] = []
    if (ids.length > 0) content.push(reviewMargin(deckSlug, info.id, ids))
    content.push(notesMargin(info.id))
    body.children.unshift(
      ...anchored('marge gauche', 'rev', content, baseId('rev', info.id), serial++),
    )
  }

  const glosses = glossesOf(tree)
  if (glosses.length > 0) {
    const uses = firstUse(tree, glosses)
    const heading = sections
      .flatMap(s => {
        const found: Element[] = []
        walk(s.node, el => {
          if (isSection(el) && String(el.properties?.id).startsWith(GLOSS_HEADING)) found.push(el)
        })
        return found
      })
      .at(0)
    const groups = new Map<Element, Gloss[]>()
    for (const gloss of glosses) {
      const block = uses.get(gloss) ?? heading
      if (!block) continue
      const bucket = groups.get(block)
      if (bucket) bucket.push(gloss)
      else groups.set(block, [gloss])
    }
    for (const [block, items] of groups) {
      // a section stands in for its heading's content: its glosses sit at the top of the body
      let target = block
      if (isSection(block)) {
        walk(block, el => {
          if (target === block && classesOf(el).includes('collapsible-header-content')) target = el
        })
      }
      target.children.unshift(
        ...anchored(
          'marge droite',
          'gloss',
          [glossMargin(items)],
          baseId('gloss', `${serial}`),
          serial++,
        ),
      )
    }
  }

  const backlinks = renderDataFor(ctx, allFiles).backlinksBySlug.get(simplifySlug(slug)) ?? []
  const lessons = backlinks
    .map(file => ({ slug: file.slug as FullSlug, label: lessonLabel(file.slug as string) }))
    .filter((link): link is { slug: FullSlug; label: string } => Boolean(link.label))
    .sort((a, b) => a.slug.localeCompare(b.slug))
  if (lessons.length > 0) {
    const first = sections[0].node
    const parent = findParent(tree, first)
    if (parent) {
      parent.children.splice(
        parent.children.indexOf(first),
        0,
        ...anchored(
          'cité par',
          'cite',
          [citedByMargin(lessons, slug)],
          baseId('cite', slug),
          serial++,
        ),
      )
    }
  }
}

function findParent(root: Root | Element, target: Element): Element | Root | undefined {
  for (const child of root.children) {
    if (child === target) return root
    if (child.type === 'element') {
      const found = findParent(child, target)
      if (found) return found
    }
  }
  return undefined
}
