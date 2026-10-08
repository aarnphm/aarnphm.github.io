import { Root, RootContent } from 'mdast'
import { fromMarkdown } from 'mdast-util-from-markdown'
import { mathFromMarkdown } from 'mdast-util-math'
import { math } from 'micromark-extension-math'
import { wikilink, wikilinkFromMarkdown } from '../../extensions/micromark-extension-ofm-wikilinks'
import { speech, speechFromMarkdown } from '../../extensions/micromark-extension-speech'
import { QuartzTransformerPlugin } from '../../types/plugin'
import { CardKind, parseFlashcards } from '../../util/flashcards'
import { flashcardsSlug, isFlashcardPath, sourceSlugForDeck } from '../../util/flashcards-path'
import { FullSlug } from '../../util/path'

const html = (value: string): RootContent => ({ type: 'html', value })

export const Flashcards: QuartzTransformerPlugin = () => ({
  name: 'Flashcards',
  markdownPlugins(ctx) {
    const allSlugs = new Set(ctx?.allSlugs ?? [])
    const hasSlug = (slug: string) => allSlugs.has(slug as FullSlug)
    const extensions = [wikilink(), math(), speech()]
    const mdastExtensions = [
      wikilinkFromMarkdown({ hasSlug }),
      mathFromMarkdown(),
      speechFromMarkdown(),
    ]
    const parseFace = (src: string) =>
      fromMarkdown(src, { extensions, mdastExtensions }).children as RootContent[]

    return [
      () => (tree: Root, file) => {
        if (!isFlashcardPath(file.data.relativePath ?? file.data.filePath ?? '')) return
        const deck = parseFlashcards(file.data.rawMarkdownSource ?? '')
        const sourceSlug = sourceSlugForDeck(file.data.slug!) as FullSlug
        // the merged drill lifts cards out of their deck page, so each card names its deck
        const deckSlug = flashcardsSlug(sourceSlug)
        file.data.flashcards = {
          sourceSlug,
          cards: deck.cards.map(card => ({ id: card.id, kind: card.kind, groupId: card.groupId })),
        }

        const children: RootContent[] = []
        if (deck.errors.length > 0) {
          const items = deck.errors
            .map(error => `<li>line ${error.line}: ${error.message}</li>`)
            .join('')
          children.push(
            html(
              `<blockquote class="callout" data-callout="warning"><div class="callout-title"><div class="callout-title-inner">flashcard parse errors</div></div><div class="callout-content"><ul>${items}</ul></div></blockquote>`,
            ),
          )
        }
        for (const card of deck.cards) {
          const group = card.groupId ? ` data-group="${card.groupId}"` : ''
          children.push(
            html(
              `<section class="flashcard" data-card-id="${card.id}" data-kind="${card.kind}" data-deck="${deckSlug}"${group}>`,
            ),
            html(`<div class="flashcard-face flashcard-front" data-face="front">`),
            ...parseFace(card.front),
            html(`</div>`),
            html(`<div class="flashcard-face flashcard-back" data-face="back" hidden>`),
            ...parseFace(card.back),
            ...(card.note
              ? [html(`<div class="flashcard-note">`), ...parseFace(card.note), html(`</div>`)]
              : []),
            html(`</div>`),
            html(`</section>`),
          )
        }
        tree.children = children
      },
    ]
  },
})

declare module 'vfile' {
  interface DataMap {
    flashcards: { sourceSlug: FullSlug; cards: { id: string; kind: CardKind; groupId?: string }[] }
  }
}
