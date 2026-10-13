import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { courseViews, unitManifest } from '../util/course'
import { flashcardsSlug } from '../util/flashcards-path'
import { htmlToJsx } from '../util/jsx'
import { FullSlug, resolveRelative } from '../util/path'
// @ts-ignore
import drillScript from './scripts/flashcards.inline'
import style from './styles/flashcards.scss'

interface Options {
  /** Render an empty drill that pulls due cards from every deck under this slug prefix. */
  merge?: { prefix: string; exit: FullSlug }
}

interface Shell {
  deckSlug: FullSlug
  cards: ReturnType<typeof htmlToJsx>
  count: number
  deckTitle: string
  sourceHref: string
  data: Record<string, string>
}

function mergedShell(
  componentData: QuartzComponentProps,
  merge: NonNullable<Options['merge']>,
): Shell {
  const { ctx, fileData, allFiles } = componentData
  const slug = fileData.slug as FullSlug
  const decks = courseViews(allFiles, ctx.decks)
    .flatMap(unitManifest)
    .filter(deck => deck.deck.startsWith(merge.prefix))
  return {
    deckSlug: slug,
    cards: undefined,
    count: 0,
    deckTitle: 'review',
    sourceHref: resolveRelative(slug, merge.exit),
    data: { 'data-prefix': merge.prefix, 'data-decks': JSON.stringify(decks) },
  }
}

function deckShell(componentData: QuartzComponentProps): Shell {
  const { fileData } = componentData
  const { filePath } = fileData
  const sourceSlug = fileData.flashcardRoute?.sourceSlug ?? fileData.flashcards!.sourceSlug
  const deckSlug = flashcardsSlug(fileData.flashcards!.sourceSlug)
  const pageSlug = fileData.flashcardRoute ? fileData.slug! : deckSlug
  const cards = htmlToJsx(filePath!, componentData.tree)
  const count = fileData.flashcards!.cards.length
  const deckTitle = fileData.frontmatter?.title ?? 'flashcards'
  const sourceHref = resolveRelative(pageSlug, sourceSlug)
  return { deckSlug, cards, count, deckTitle, sourceHref, data: { 'data-source': sourceSlug } }
}

export default ((opts?: Options) => {
  const FlashcardsContent: QuartzComponent = (componentData: QuartzComponentProps) => {
    const { deckSlug, cards, count, deckTitle, sourceHref, data } = opts?.merge
      ? mergedShell(componentData, opts.merge)
      : deckShell(componentData)

    return (
      <div class="flashcards-root" data-deck={deckSlug} data-total={count} {...data}>
        <div class="flashcards-main">
          <header class="flashcards-bar">
            <a
              href={sourceHref}
              class="flashcards-exit internal"
              data-slug={sourceHref}
              data-no-popover
              aria-label="back to note"
            >
              ←
            </a>
            <div
              class="flashcards-progress"
              role="progressbar"
              aria-label="drill progress"
              aria-valuemin={0}
              aria-valuemax={count}
              aria-valuenow={0}
            >
              <div class="flashcards-progress-fill" />
            </div>
            <span class="flashcards-status" aria-live="polite">
              0 / {count}
            </span>
            <a class="flashcards-sync" data-router-ignore data-no-popover hidden />
            <button
              type="button"
              class="fc-btn fc-notes"
              data-action="notes"
              aria-pressed="false"
              aria-controls="flashcards-notes"
            >
              notes
              <kbd class="fc-key" aria-hidden="true">
                n
              </kbd>
            </button>
          </header>
          <div class="flashcards-stage">
            <article class="flashcards-card">
              <div class="flashcards-card-head">
                <h1>{deckTitle}</h1>
              </div>
              <div class="flashcards-card-body">
                <div class="flashcards-deck" role="list">
                  {cards}
                </div>
              </div>
            </article>
          </div>
          <nav class="flashcards-controls" aria-label="drill controls">
            <button type="button" class="fc-btn fc-undo" data-action="undo" disabled>
              undo
              <kbd class="fc-key" aria-hidden="true">
                u
              </kbd>
            </button>
            <span class="fc-spacer" />
            <button type="button" class="fc-btn fc-reveal" data-action="reveal">
              reveal
              <kbd class="fc-key" aria-hidden="true">
                space
              </kbd>
            </button>
            <span class="fc-grades" hidden>
              <button type="button" class="fc-btn fc-grade" data-action="grade" data-grade="1">
                forgot
                <kbd class="fc-key" aria-hidden="true">
                  1
                </kbd>
              </button>
              <button type="button" class="fc-btn fc-grade" data-action="grade" data-grade="2">
                hard
                <kbd class="fc-key" aria-hidden="true">
                  2
                </kbd>
              </button>
              <button type="button" class="fc-btn fc-grade" data-action="grade" data-grade="3">
                good
                <kbd class="fc-key" aria-hidden="true">
                  3
                </kbd>
              </button>
              <button type="button" class="fc-btn fc-grade" data-action="grade" data-grade="4">
                easy
                <kbd class="fc-key" aria-hidden="true">
                  4
                </kbd>
              </button>
            </span>
            <span class="fc-spacer" />
            <button type="button" class="fc-btn fc-end" data-action="end">
              end
              <kbd class="fc-key" aria-hidden="true">
                e
              </kbd>
            </button>
          </nav>
          <section class="flashcards-finished" aria-live="polite" hidden>
            <h1>session complete</h1>
            <p class="flashcards-summary" />
          </section>
        </div>
        <aside class="flashcards-notes" id="flashcards-notes" aria-label="source note" hidden>
          <header class="flashcards-notes-head">
            <a class="flashcards-notes-title internal" data-no-popover>
              notes
            </a>
          </header>
          <div class="flashcards-notes-body">
            <p class="flashcards-notes-empty">the current card's note appears here.</p>
          </div>
        </aside>
      </div>
    )
  }
  FlashcardsContent.css = style
  FlashcardsContent.afterDOMLoaded = drillScript
  return FlashcardsContent
}) satisfies QuartzComponentConstructor
