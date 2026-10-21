import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { fileBySlug } from '../util/course'
import { cardsWord, frCount, frView, motsLevels, type FrEpisode } from '../util/fr'
import { classNames } from '../util/lang'
import { FullSlug, resolveRelative } from '../util/path'
import { concatenateResources } from '../util/resources'
import { FR_CALENDAR_ID, FrCalendarView, type FrCalendarData } from './course-calendar/fr'
import { FR_REVIEW } from './FrUnit'
// @ts-ignore
import canvasScript from './scripts/canvas.inline'
// @ts-ignore
import script from './scripts/course.inline'
import canvasStyle from './styles/canvas.scss'
import studyStyle from './styles/course-study.scss'
import style from './styles/course.scss'
import frStyle from './styles/fr.scss'

const CANVAS = 'fr/parcours' as FullSlug
const HORAIRE = '/fr/horaire.json'

// JSON inside <script> must not close the element or open a comment.
const scriptJson = (value: unknown) =>
  JSON.stringify(value).replace(/</g, '\\u003c').replace(/>/g, '\\u003e')

const pad = (n: number) => String(n).padStart(2, '0')

const longDate = (date: string | undefined) => {
  if (!date) return ''
  return new Date(`${date}T00:00:00`).toLocaleDateString('fr-CA', {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  })
}

export default (() => {
  const FrHome: QuartzComponent = (props: QuartzComponentProps) => {
    const { fileData, allFiles, ctx, displayClass } = props
    const from = fileData.slug!
    const view = frView(allFiles, ctx.decks)
    const href = (slug: FullSlug) => resolveRelative(from, slug)
    const allIds = view.decks.flatMap(item => item.deck.flashcards!.cards.map(card => card.id))
    const levels = motsLevels(fileBySlug('fr/mots', allFiles))
    const calendar: FrCalendarData = {
      notes: [...view.lessons, ...view.practices].map(note => ({
        date: note.date,
        kind: note.kind,
        href: href(note.slug),
        episodes: note.episodes,
      })),
      horaire: HORAIRE,
      prefix: 'fr/',
      cardIds: allIds,
    }
    const latestPractice = view.practices.at(-1)
    const since = longDate(String(fileData.frontmatter?.date ?? '') || undefined)

    const episode = (item: FrEpisode) => (
      <a
        class={classNames(undefined, 'fr-ep', 'internal')}
        href={href(item.slug)}
        data-slug={item.slug}
        data-frontier={item.frontier ? '' : undefined}
        aria-current={item.frontier ? 'step' : undefined}
        title={item.title}
      >
        <span class="course-num">{pad(item.n)}</span>
        <span class="fr-ep-title">{item.title}</span>
        <span class="course-count">{item.cardIds.length || ''}</span>
      </a>
    )

    return (
      <div
        class={classNames(displayClass, 'course-home', 'fr-home')}
        data-course-prefix="fr/"
        data-course-lang="fr"
      >
        <header class="course-masthead">
          <h1 class="course-masthead-title">français</h1>
          <p class="course-masthead-meta">
            Par ici A1<span class="course-sep">·</span>québécois
            {since ? (
              <>
                <span class="course-sep">·</span>depuis le {since}
              </>
            ) : null}
          </p>
        </header>

        <nav class="fr-shelf" aria-label="accès">
          <a class="internal" href={href('tags/fr' as FullSlug)} data-slug="tags/fr">
            {view.notes} notes
          </a>
          <span class="course-count">
            {view.decks.length} paquets · {frCount(view.cards)} cartes
          </span>
          <a
            class="internal"
            href={href(FR_REVIEW)}
            data-slug={FR_REVIEW}
            data-no-popover
            data-course-action="review"
          >
            réviser ce qui est dû
          </a>
          <span class="course-count" data-cards={allIds.join(' ')} data-unit="all">
            <span data-due="">–</span>
          </span>
          <a class="internal" href={href(CANVAS)} data-slug={CANVAS}>
            carte des notes
          </a>
        </nav>

        <section aria-labelledby="fr-saisons-h">
          <h2 id="fr-saisons-h" class="fr-h">
            saisons
          </h2>
          <div class="fr-saisons">
            {view.seasons.map(season => (
              <div class="fr-saison">
                <span class="fr-saison-name">{season.name}</span>
                {season.episodes.map(episode)}
              </div>
            ))}
          </div>
        </section>

        <section class="fr-calendar" aria-labelledby="fr-calendrier-h">
          <h2 id="fr-calendrier-h" class="fr-h">
            calendrier
          </h2>
          <script
            type="application/json"
            data-fr-calendar-data
            dangerouslySetInnerHTML={{ __html: scriptJson(calendar) }}
          />
          <div data-fr-calendar-mount>
            <FrCalendarView id={FR_CALENDAR_ID} data={calendar} />
          </div>
        </section>

        <section aria-labelledby="fr-parcours-h">
          <h2 id="fr-parcours-h" class="fr-h">
            carte des notes
          </h2>
          <div
            class="canvas-embed-container fr-canvas"
            tabindex={0}
            role="group"
            aria-label="carte des notes"
            data-canvas="/fr/parcours.canvas"
            data-meta="/fr/parcours.meta.json"
            data-cfg={JSON.stringify({ zoom: true, showPreviewOnHover: false })}
            data-embed="live"
          >
            <div class="canvas-loading">…</div>
          </div>
        </section>

        {view.mots ? (
          <section aria-labelledby="fr-mots-h">
            <h2 id="fr-mots-h" class="fr-h">
              <a class="internal" href={href(view.mots.slug)} data-slug={view.mots.slug}>
                mots
              </a>{' '}
              <span class="course-count">{cardsWord(view.mots.cards)}</span>
            </h2>
            {levels.length > 0 ? (
              <p class="fr-shelf fr-shelf--levels">
                {levels.map(item => (
                  <span>
                    {item.level} <span class="course-count">{item.themes.length}</span>
                  </span>
                ))}
              </p>
            ) : null}
          </section>
        ) : null}

        <section aria-labelledby="fr-reperes-h">
          <h2 id="fr-reperes-h" class="fr-h">
            repères
          </h2>
          <p class="fr-shelf">
            {view.reperes.map(item => (
              <a class="internal" href={href(item.slug)} data-slug={item.slug}>
                {item.title}
              </a>
            ))}
            {latestPractice ? (
              <a class="internal" href={href(latestPractice.slug)} data-slug={latestPractice.slug}>
                pratiques
              </a>
            ) : null}
            <a
              class="internal"
              href={href('fr/books/index' as FullSlug)}
              data-slug="fr/books/index"
            >
              livres
            </a>
          </p>
        </section>
      </div>
    )
  }
  FrHome.displayName = 'FrHome'
  FrHome.css = concatenateResources(style, studyStyle, frStyle, canvasStyle)
  FrHome.afterDOMLoaded = concatenateResources(script, canvasScript)
  return FrHome
}) satisfies QuartzComponentConstructor
