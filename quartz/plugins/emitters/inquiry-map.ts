import type { QuartzEmitterPlugin } from '../../types/plugin'
import { contentDataFor, deckDataFor } from '../../util/ctx'
import { parseFlashcards } from '../../util/flashcards'
import {
  buildInquiryMap,
  inquiriesSlug,
  inquirySpecOf,
  type InquiryDeck,
  type InquiryPage,
} from '../../util/inquiry-map'
import { write } from './helpers'

/** Emits `<page>/inquiries.json` for every page that declares `inquiries` in its frontmatter. */
export const InquiryMap: QuartzEmitterPlugin = () => ({
  name: 'InquiryMap',
  async *emit(ctx, content) {
    const files = contentDataFor(content)
    const declaring = files.flatMap(file => {
      const spec = inquirySpecOf(file.frontmatter?.inquiries)
      return file.slug && spec ? [{ slug: file.slug, spec }] : []
    })
    if (declaring.length === 0) return

    const pages: InquiryPage[] = files.flatMap(file =>
      file.slug && file.rawMarkdownSource !== undefined
        ? [
            {
              slug: file.slug,
              title: String(file.frontmatter?.title ?? file.slug),
              source: file.rawMarkdownSource,
            },
          ]
        : [],
    )
    const decks: InquiryDeck[] = deckDataFor(content).flatMap(deck => {
      const slug = deck.flashcards?.sourceSlug
      if (!slug) return []
      const cards = parseFlashcards(deck.rawMarkdownSource ?? '').cards
      return [{ slug, notes: cards.map(c => c.note), probes: cards.map(c => c.back || c.front) }]
    })
    for (const { slug, spec } of declaring) {
      yield write({
        ctx,
        slug: inquiriesSlug(slug),
        ext: '.json',
        content: JSON.stringify(buildInquiryMap(spec, pages, decks)),
      })
    }
  },
})
