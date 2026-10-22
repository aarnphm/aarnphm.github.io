import type { Element, ElementContent, Root } from 'hast'
import { toString } from 'hast-util-to-string'
import { h } from 'hastscript'
import { QuartzComponentProps } from '../../types/component'
import { renderDataFor } from '../../util/ctx'
import { parseFlashcards } from '../../util/flashcards'
import { flashcardsSlug } from '../../util/flashcards-path'
import { assignCards, foldText, FrSection, lessonLabel } from '../../util/fr-margins'
import { FullSlug, resolveRelative, simplifySlug } from '../../util/path'
import { anchorOf, Apparatus, classesOf, walk } from './apparatus'
import { takeNotes } from './apparatus-notes'

type Ctx = QuartzComponentProps['ctx']

const GLOSS_HEADING = 'comprendre-les-expressions'
const BLOCKS = new Set(['p', 'li', 'td', 'th', 'blockquote'])

const levelOf = (node: Element): number => Number(node.properties?.dataLevel ?? 0)

const isSection = (node: ElementContent, level?: number): node is Element =>
  node.type === 'element' &&
  node.tagName === 'section' &&
  classesOf(node).includes('collapsible-header') &&
  (level === undefined || levelOf(node) === level)

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

type EntryKind = 'rev' | 'notes' | 'rem' | 'lex' | 'cite'

const LABELS: Record<EntryKind, string> = {
  rev: 'révision',
  notes: 'notes',
  rem: 'remarque',
  lex: 'lexique',
  cite: 'cité par',
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

const bodyOf = (section: Element): Element => {
  let body: Element | undefined
  walk(section, el => {
    if (!body && classesOf(el).includes('collapsible-header-content')) body = el
  })
  return body ?? section
}

/**
 * One margin column left of the text, Locher style: each prose block that carries margin work opens
 * with a number `(n)`, and the column holds `§ label (n)` entries for it. Review of the cards that cite the page
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

  const apparatus = new Apparatus()
  const put = (block: Element, entry: { kind: EntryKind; body: Element }) =>
    apparatus.put(block, { ...entry, label: LABELS[entry.kind] })

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

  // the page's footnotes hang beside the block that calls them; its sidenotes keep their boxes
  takeNotes(tree, apparatus, { kind: 'rem', label: LABELS.rem })

  if (lessons.length > 0) put(firstBlock, { kind: 'cite', body: citedByMargin(lessons, slug) })

  apparatus.hang(tree, kind => ORDER[kind as EntryKind])
}

const ORDER: Record<EntryKind, number> = { rev: 0, notes: 1, rem: 2, lex: 3, cite: 4 }
