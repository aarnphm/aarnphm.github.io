import type { Element, ElementContent, Root } from 'hast'
import { toString } from 'hast-util-to-string'
import { h } from 'hastscript'
import { QuartzComponentProps } from '../../types/component'
import { renderDataFor } from '../../util/ctx'
import { parseFlashcards } from '../../util/flashcards'
import { flashcardsSlug } from '../../util/flashcards-path'
import { assignCards, foldText, FrSection, lessonLabel } from '../../util/fr-margins'
import { FullSlug, resolveRelative, simplifySlug } from '../../util/path'

type Ctx = QuartzComponentProps['ctx']

const GLOSS_HEADING = 'comprendre-les-expressions'
const BLOCKS = new Set(['p', 'li', 'td', 'th', 'blockquote'])
// A margin entry hangs from a prose block; a header cell is no place for one.
// a quotation's first paragraph carries the number, so it opens a line of text, never its own
const ANCHORS = new Set(['p', 'li', 'td'])

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

type EntryKind = 'rev' | 'notes' | 'lex' | 'cite'

const LABELS: Record<EntryKind, string> = {
  rev: 'révision',
  notes: 'notes',
  lex: 'lexique',
  cite: 'cité par',
}

interface Entry {
  kind: EntryKind
  body: Element
}

// Everything inside a prose block is a span: a <div> would end the paragraph when the browser parses it.
function reviewMargin(deck: FullSlug, section: string, ids: string[]): Element {
  return h(
    'span.fr-margin.fr-rev',
    { dataFrRev: '', dataDeck: deck, dataIds: ids.join(','), dataSection: section },
    [
      h('span.fr-label', [h('span.fr-count', { dataFrCount: '' }, String(ids.length)), ' cartes']),
      h('span.fr-card', { dataFrCard: '', hidden: true }),
    ],
  )
}

function notesMargin(section: string): Element {
  return h('span.fr-margin.fr-notes', { dataFrNotes: '', dataSection: section }, [
    h('span.fr-note-list', { dataFrNoteList: '' }),
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
  return h(
    'span.fr-margin.fr-cited',
    links.map(link =>
      h('span.fr-gloss', [
        h(
          'a.internal',
          { href: resolveRelative(from, link.slug), dataSlug: link.slug },
          link.label,
        ),
      ]),
    ),
  )
}

/** First prose block of a section body, or the body itself when the section has none. */
function anchorOf(container: Element): Element {
  let found: Element | undefined
  walk(container, el => {
    if (found) return false
    if (ANCHORS.has(el.tagName) && toString(el).trim()) found = el
  })
  return found ?? container
}

const bodyOf = (section: Element): Element => {
  let body: Element | undefined
  walk(section, el => {
    if (!body && classesOf(el).includes('collapsible-header-content')) body = el
  })
  return body ?? section
}

/**
 * One margin column, Locher style: each prose block that carries margin work opens with a number
 * `(n)`, and the column holds `§ label (n)` entries for it. Review of the cards that cite the page
 * and the reader's notes sit at the first block of each `##`; the page's "comprendre les expressions"
 * rows sit at the first use of each expression; the lessons that cite the page sit at the first
 * block. The browser hangs the group beside the block, its label level with the number, or folds it
 * under the block on narrow screens. Decks are only visible at render time
 * (`ctx.decks`), so this runs here rather than in a transformer.
 */
export function renderFrMargins(tree: Root, componentData: QuartzComponentProps, ctx: Ctx) {
  const { fileData, allFiles } = componentData
  const slug = fileData.slug as FullSlug | undefined
  if (!slug || !slug.startsWith('fr/') || fileData.flashcards) return
  const sections = sectionsOf(tree)
  if (sections.length === 0) return

  const deck = ctx.decks?.find(d => d.flashcards?.sourceSlug === slug)
  const deckSlug = flashcardsSlug(slug)
  const cards = deck ? parseFlashcards(deck.rawMarkdownSource ?? '').cards : []
  const assigned = assignCards(
    cards,
    slug,
    sections.map(s => s.info),
  )

  const pending = new Map<Element, Entry[]>()
  const put = (block: Element, entry: Entry) => {
    const list = pending.get(block)
    if (list) list.push(entry)
    else pending.set(block, [entry])
  }

  const firstBlock = anchorOf(sections[0].body)
  const backlinks = renderDataFor(ctx, allFiles).backlinksBySlug.get(simplifySlug(slug)) ?? []
  const lessons = backlinks
    .map(file => ({ slug: file.slug as FullSlug, label: lessonLabel(file.slug as string) }))
    .filter((link): link is { slug: FullSlug; label: string } => Boolean(link.label))
    .sort((a, b) => a.slug.localeCompare(b.slug))

  for (const { body, info } of sections) {
    const block = anchorOf(body)
    const ids = assigned.get(info.id) ?? []
    if (ids.length > 0) put(block, { kind: 'rev', body: reviewMargin(deckSlug, info.id, ids) })
    put(block, { kind: 'notes', body: notesMargin(info.id) })
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
      const found = uses.get(gloss)
      const block = found ?? (heading ? anchorOf(bodyOf(heading)) : undefined)
      if (!block) continue
      const bucket = groups.get(block)
      if (bucket) bucket.push(gloss)
      else groups.set(block, [gloss])
    }
    for (const [block, items] of groups) put(block, { kind: 'lex', body: glossMargin(items) })
  }

  if (lessons.length > 0) put(firstBlock, { kind: 'cite', body: citedByMargin(lessons, slug) })

  // numbers follow document order, whichever pass found the block first
  const ordered: Element[] = []
  walk(tree, el => {
    if (pending.has(el)) ordered.push(el)
  })
  ordered.forEach((block, i) => {
    const n = i + 1
    const entries = pending.get(block)!.sort((a, b) => ORDER[a.kind] - ORDER[b.kind])
    block.properties = {
      ...block.properties,
      className: [...classesOf(block), 'fr-anchor'],
      dataFrN: String(n),
    }
    // the number and the groups open the block: beside it in the margin, or in parentheses before
    // the text where there is no margin
    const lead: Element[] = [h('span.fr-n', { ariaHidden: 'true' }, `(${n})`)]
    // Locher's three columns: review and notes hang left of the text, glosses and citing lessons right
    for (const side of ['left', 'right'] as const) {
      const mine = entries.filter(entry => SIDE[entry.kind] === side)
      if (mine.length === 0) continue
      lead.push(
        h(
          'span.fr-group',
          { dataN: String(n), dataSide: side },
          mine.map(entry =>
            h('span.fr-entry', { dataFrKind: entry.kind }, [
              h(
                'button.fr-entry-label',
                { type: 'button', ariaExpanded: 'false' },
                `§ ${LABELS[entry.kind]} (${n})`,
              ),
              h('span.fr-entry-body', [entry.body]),
            ]),
          ),
        ),
      )
    }
    block.children.unshift(...lead)
  })
}

const ORDER: Record<EntryKind, number> = { rev: 0, notes: 1, lex: 2, cite: 3 }

const SIDE: Record<EntryKind, 'left' | 'right'> = {
  rev: 'left',
  notes: 'left',
  lex: 'right',
  cite: 'right',
}
