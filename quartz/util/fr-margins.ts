import type { Card } from './flashcards'
import { stripSpeechMarkup } from '../extensions/micromark-extension-speech/source'

/** One `##` section of a page: its own id, the ids of every heading inside it, and its text. */
export interface FrSection {
  id: string
  headingIds: string[]
  text: string
}

const WIKILINK = /\[\[([^\]|#]*)(#[^\]|]*)?(?:\|[^\]]*)?\]\]/g

/** Folds case, apostrophes and markup so a card answer and page text compare equal. */
export function foldText(text: string): string {
  return stripSpeechMarkup(text)
    .normalize('NFC')
    .toLowerCase()
    .replace(/[’‘`]/g, "'")
    .replace(/[{}*_[\]()«».,;:!?…"“”]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

/** Heading anchor a card note cites on `page`, `''` for the page itself, `undefined` if not cited. */
export function citedAnchor(card: Card, page: string): string | undefined {
  const note = card.note
  if (!note) return undefined
  for (const match of note.matchAll(WIKILINK)) {
    const target = match[1].replace(/\.md$/, '').replace(/^\.?\//, '')
    if (target === page) return (match[2] ?? '').slice(1)
  }
  return undefined
}

const fragments = (text: string): string[] =>
  foldText(text)
    .split(/ ?[.!?;] ?/)
    .filter(part => part.length >= 6)
    .sort((a, b) => b.length - a.length)

/**
 * Puts each card that cites `page` into one section. A cited heading anchor wins; otherwise the
 * section whose text contains the answer; otherwise the section of the previous card, since a deck
 * follows its page's order. Returns section id → card ids in deck order.
 */
export function assignCards(
  cards: readonly Card[],
  page: string,
  sections: readonly FrSection[],
): Map<string, string[]> {
  const out = new Map<string, string[]>()
  if (sections.length === 0) return out
  const byHeading = new Map<string, string>()
  for (const section of sections) {
    for (const id of section.headingIds) byHeading.set(id, section.id)
    byHeading.set(section.id, section.id)
  }
  let previous = 0
  for (const card of cards) {
    const anchor = citedAnchor(card, page)
    if (anchor === undefined) continue
    let at = -1
    const named = anchor ? byHeading.get(anchor) : undefined
    if (named) at = sections.findIndex(section => section.id === named)
    if (at < 0) {
      const needles = [foldText(card.back), ...fragments(card.back)].filter(n => n.length >= 4)
      for (const needle of needles) {
        // the section that uses the answer most is the one that teaches it; ties keep deck order
        let best = 0
        sections.forEach((section, i) => {
          const uses = section.text.split(needle).length - 1
          if (uses > best || (uses === best && uses > 0 && i >= previous && at < previous)) {
            best = uses
            at = i
          }
        })
        if (at >= 0) break
      }
    }
    if (at < 0) at = previous
    previous = at
    const id = sections[at].id
    const bucket = out.get(id)
    if (bucket) bucket.push(card.id)
    else out.set(id, [card.id])
  }
  return out
}

const MONTHS = [
  'janv.',
  'févr.',
  'mars',
  'avr.',
  'mai',
  'juin',
  'juil.',
  'août',
  'sept.',
  'oct.',
  'nov.',
  'déc.',
]

/** `fr/lessons/261006` → `leçon du 6 oct.`; other slugs return `undefined`. */
export function lessonLabel(slug: string): string | undefined {
  const match = /^fr\/lessons\/(\d{2})(\d{2})(\d{2})$/.exec(slug)
  if (!match) return undefined
  const month = MONTHS[Number(match[2]) - 1]
  return month ? `leçon du ${Number(match[3])} ${month}` : undefined
}
