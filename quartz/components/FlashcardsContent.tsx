import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { courseViews, unitManifest } from '../util/course'
import { flashcardsSlug } from '../util/flashcards-path'
import { frManifest, frView } from '../util/fr'
import { htmlToJsx } from '../util/jsx'
import { FullSlug, resolveRelative } from '../util/path'
// @ts-ignore
import drillScript from './scripts/flashcards.inline'
import style from './styles/flashcards.scss'

interface Options {
  /** Render an empty drill that pulls due cards from every deck under this slug prefix. */
  merge?: { prefix: string; exit: FullSlug }
  /** The drill's chrome in French: the labels, the grades and the title of a merged drill. */
  lang?: 'fr'
}

const LABELS = {
  en: {
    title: 'review',
    back: 'back to note',
    progress: 'drill progress',
    notes: 'notes',
    undo: 'undo',
    reveal: 'reveal',
    grades: ['forgot', 'hard', 'good', 'easy'],
    end: 'end',
    controls: 'drill controls',
    done: 'session complete',
    source: 'source note',
    empty: "the current card's note appears here.",
  },
  fr: {
    title: 'révision',
    back: 'retour à la note',
    progress: 'progression',
    notes: 'notes',
    undo: 'annuler',
    reveal: 'montrer',
    grades: ['encore', 'difficile', 'bien', 'facile'],
    end: 'finir',
    controls: 'commandes',
    done: 'séance finie',
    source: 'note source',
    empty: 'la note de la carte apparaît ici.',
  },
} as const

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
  title: string,
): Shell {
  const { ctx, fileData, allFiles } = componentData
  const slug = fileData.slug as FullSlug
  const decks = [
    ...courseViews(allFiles, ctx.decks).flatMap(unitManifest),
    ...frManifest(frView(allFiles, ctx.decks)),
  ].filter(deck => deck.deck.startsWith(merge.prefix))
  return {
    deckSlug: slug,
    cards: undefined,
    count: 0,
    deckTitle: title,
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
    const t = LABELS[opts?.lang ?? 'en']
    const { deckSlug, cards, count, deckTitle, sourceHref, data } = opts?.merge
      ? mergedShell(componentData, opts.merge, t.title)
      : deckShell(componentData)

    return (
      <div
        class="flashcards-root"
        data-deck={deckSlug}
        data-total={count}
        data-lang={opts?.lang}
        {...data}
      >
        <div class="flashcards-main">
          <header class="flashcards-bar">
            <a
              href={sourceHref}
              class="flashcards-exit internal"
              data-slug={sourceHref}
              data-no-popover
              aria-label={t.back}
            >
              ←
            </a>
            <div
              class="flashcards-progress"
              role="progressbar"
              aria-label={t.progress}
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
              {t.notes}
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
          <nav class="flashcards-controls" aria-label={t.controls}>
            <button type="button" class="fc-btn fc-undo" data-action="undo" disabled>
              {t.undo}
              <kbd class="fc-key" aria-hidden="true">
                u
              </kbd>
            </button>
            <span class="fc-spacer" />
            <button type="button" class="fc-btn fc-reveal" data-action="reveal">
              {t.reveal}
              <kbd class="fc-key" aria-hidden="true">
                space
              </kbd>
            </button>
            <span class="fc-grades" hidden>
              <button type="button" class="fc-btn fc-grade" data-action="grade" data-grade="1">
                {t.grades[0]}
                <kbd class="fc-key" aria-hidden="true">
                  1
                </kbd>
              </button>
              <button type="button" class="fc-btn fc-grade" data-action="grade" data-grade="2">
                {t.grades[1]}
                <kbd class="fc-key" aria-hidden="true">
                  2
                </kbd>
              </button>
              <button type="button" class="fc-btn fc-grade" data-action="grade" data-grade="3">
                {t.grades[2]}
                <kbd class="fc-key" aria-hidden="true">
                  3
                </kbd>
              </button>
              <button type="button" class="fc-btn fc-grade" data-action="grade" data-grade="4">
                {t.grades[3]}
                <kbd class="fc-key" aria-hidden="true">
                  4
                </kbd>
              </button>
            </span>
            <span class="flashcards-notes-status" role="status" aria-live="polite" />
            <button type="button" class="fc-btn fc-end" data-action="end">
              {t.end}
              <kbd class="fc-key" aria-hidden="true">
                e
              </kbd>
            </button>
          </nav>
          <section class="flashcards-finished" aria-live="polite" hidden>
            <h1>{t.done}</h1>
            <p class="flashcards-summary" />
          </section>
        </div>
        <aside class="flashcards-notes" id="flashcards-notes" aria-label={t.source} hidden>
          <header class="flashcards-notes-head">
            <a class="flashcards-notes-title internal" data-no-popover>
              {t.notes}
            </a>
          </header>
          <div class="flashcards-notes-body">
            <p class="flashcards-notes-empty">{t.empty}</p>
          </div>
        </aside>
      </div>
    )
  }
  FlashcardsContent.css = style
  FlashcardsContent.afterDOMLoaded = drillScript
  return FlashcardsContent
}) satisfies QuartzComponentConstructor
