import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { flashcardsSlug } from '../util/flashcards-path'
import { FR_EPISODES, cardsWord, dateWords, episodeLabel, frEpisodeOf, frView } from '../util/fr'
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

/** The course unit bar on one line, in French: the crumb, the episode's facts, the review links. */
export const FrUnitBar = (() => {
  const Bar: QuartzComponent = (props: QuartzComponentProps) => {
    const n = frEpisodeOf(props.fileData.slug)
    if (n === null) return null
    const from = props.fileData.slug!
    const view = frView(props.allFiles, props.ctx.decks)
    const episode = view.episodes[n - 1]
    const deck = flashcardsSlug(episode.slug)
    const ids = episode.cardIds.join(' ')
    const dates = dateWords(episode.lessons.map(lesson => lesson.date))
    return (
      <nav
        class={classNames(props.displayClass, 'course-unit-bar', 'fr-unit-bar')}
        aria-label="épisode"
        data-course-prefix="fr/"
        data-course-lang="fr"
        data-course-card
      >
        <span class="fr-unit-facts">
          <a class="internal" href={resolveRelative(from, FR_HOME)} data-slug={FR_HOME}>
            français
          </a>
          <span class="course-unit-sep">/</span>
          <span class="course-num" title={`épisode ${n} sur ${FR_EPISODES}`}>
            épisode {episodeLabel(n)}
          </span>
          <span class="course-unit-sep">·</span>
          <span>{episode.season}</span>
          {episode.cardIds.length > 0 ? (
            <>
              <span class="course-unit-sep">·</span>
              <a
                class="internal"
                href={resolveRelative(from, deck)}
                data-slug={deck}
                data-no-popover
              >
                {cardsWord(episode.cardIds.length)}
              </a>
              <span class="course-unit-sep">·</span>
              <span>
                à revoir{' '}
                <span data-due="" data-cards={ids} data-unit={String(n)}>
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
        <span class="course-unit-actions">
          <a
            class="course-action internal"
            href={resolveRelative(from, FR_REVIEW)}
            data-slug={FR_REVIEW}
            data-no-popover
          >
            tout réviser
          </a>
          {episode.cardIds.length > 0 ? (
            <a
              class="course-action internal"
              href={resolveRelative(from, deck)}
              data-slug={deck}
              data-no-popover
              data-course-action="drill"
              data-cards={ids}
              data-unit={String(n)}
            >
              réviser
            </a>
          ) : null}
        </span>
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
