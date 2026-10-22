import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { flashcardsSlug } from '../util/flashcards-path'
import {
  FR_EPISODES,
  cardIdsOf,
  cardsWord,
  dateWords,
  episodeLabel,
  frEpisodeOf,
  frView,
  isFrNote,
} from '../util/fr'
import { classNames } from '../util/lang'
import { FullSlug, resolveRelative } from '../util/path'
import { concatenateResources } from '../util/resources'
// @ts-ignore
import script from './scripts/course.inline'
import studyStyle from './styles/course-study.scss'
import style from './styles/course.scss'
import frStyle from './styles/fr.scss'

export const FR_HOME = 'fr/index' as FullSlug
export const FR_REVIEW = 'fr/révision' as FullSlug

const pad = (n: number) => String(n).padStart(2, '0')

/**
 * The course unit bar on one line, in French: the crumb, the facts, the review links. An episode
 * shows its number, season and lessons; any other /fr note shows its title, and its cards when it
 * has a deck.
 */
export const FrUnitBar = (() => {
  const Bar: QuartzComponent = (props: QuartzComponentProps) => {
    const from = props.fileData.slug!
    const n = frEpisodeOf(from)
    if (n === null && !isFrNote(props.fileData)) return null
    const view = frView(props.allFiles, props.ctx.decks)
    const episode = n === null ? undefined : view.episodes[n - 1]
    const ids = episode
      ? episode.cardIds
      : cardIdsOf(view.decks.find(item => item.note.slug === from)?.deck)
    const deck = flashcardsSlug(from)
    const drill = ids.join(' ')
    const dates = episode ? dateWords(episode.lessons.map(lesson => lesson.date)) : ''
    // only an episode is a course card: a note keeps its own popover and search preview
    const card = episode ? { 'data-course-card': true } : {}
    return (
      <nav
        class={classNames(props.displayClass, 'course-unit-bar', 'fr-unit-bar')}
        aria-label={episode ? 'épisode' : 'français'}
        data-course-prefix="fr/"
        data-course-lang="fr"
        {...card}
      >
        <span class="fr-unit-facts">
          <a class="internal" href={resolveRelative(from, FR_HOME)} data-slug={FR_HOME}>
            français
          </a>
          <span class="course-unit-sep">/</span>
          {episode ? (
            <>
              <span class="course-num" title={`épisode ${n} sur ${FR_EPISODES}`}>
                épisode {episodeLabel(episode.n)}
              </span>
              <span class="course-unit-sep">·</span>
              <span>{episode.season}</span>
            </>
          ) : (
            <span aria-current="page">
              {String(props.fileData.frontmatter?.title ?? from.slice(3))}
            </span>
          )}
          {ids.length > 0 ? (
            <>
              <span class="course-unit-sep">·</span>
              <a
                class="internal"
                href={resolveRelative(from, deck)}
                data-slug={deck}
                data-no-popover
              >
                {cardsWord(ids.length)}
              </a>
              <span class="course-unit-sep">·</span>
              <span>
                à revoir{' '}
                <span data-due="" data-cards={drill} data-unit={episode ? String(n) : undefined}>
                  –
                </span>
              </span>
            </>
          ) : null}
          {dates ? (
            <>
              <span class="course-unit-sep">·</span>
              <span>leçons {dates}</span>
            </>
          ) : null}
        </span>
        {episode || ids.length > 0 ? (
          <span class="course-unit-actions">
            <a
              class="course-action internal"
              href={resolveRelative(from, FR_REVIEW)}
              data-slug={FR_REVIEW}
              data-no-popover
            >
              tout réviser
            </a>
            {ids.length > 0 ? (
              <a
                class="course-action internal"
                href={resolveRelative(from, deck)}
                data-slug={deck}
                data-no-popover
                data-course-action="drill"
                data-cards={drill}
                data-unit={episode ? String(n) : undefined}
              >
                réviser
              </a>
            ) : null}
          </span>
        ) : null}
      </nav>
    )
  }
  Bar.displayName = 'FrUnitBar'
  Bar.css = concatenateResources(style, studyStyle, frStyle)
  Bar.afterDOMLoaded = script
  return Bar
}) satisfies QuartzComponentConstructor

/** Previous and next episode under the note. */
export const FrUnitFoot = (() => {
  const Foot: QuartzComponent = (props: QuartzComponentProps) => {
    const n = frEpisodeOf(props.fileData.slug)
    if (n === null) return null
    const from = props.fileData.slug!
    const { episodes } = frView(props.allFiles, props.ctx.decks)
    const neighbour = (m: number, rel: 'prev' | 'next') => {
      const episode = episodes[m - 1]
      if (!episode?.note) return <span aria-hidden="true" />
      return (
        <a rel={rel} class="internal" href={resolveRelative(from, episode.slug)}>
          <span class="course-num">{pad(m)}</span>
          <span class="course-pager-title">{episode.title}</span>
        </a>
      )
    }
    return (
      <footer class={classNames(props.displayClass, 'course-unit-foot')}>
        <nav class="course-unit-pager" aria-label="épisodes">
          {neighbour(n - 1, 'prev')}
          {neighbour(n + 1, 'next')}
        </nav>
      </footer>
    )
  }
  Foot.displayName = 'FrUnitFoot'
  Foot.css = concatenateResources(style, frStyle)
  return Foot
}) satisfies QuartzComponentConstructor
