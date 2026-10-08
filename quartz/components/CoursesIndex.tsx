import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import { courseDirOf, courseView, CourseView } from '../util/course'
import { classNames } from '../util/lang'
import { FullSlug, resolveRelative } from '../util/path'
import { CourseBar } from './CourseSpine'
// @ts-ignore
import script from './scripts/course.inline'
import style from './styles/course.scss'

const homeRe = /^courses\/[^/]+\/index$/

export default (() => {
  const CoursesIndex: QuartzComponent = ({
    ctx,
    fileData,
    allFiles,
    displayClass,
  }: QuartzComponentProps) => {
    const from = fileData.slug!
    const homes = allFiles
      .filter(file => homeRe.test(file.slug ?? ''))
      .map(home => ({ home, view: courseView(home, allFiles, ctx.decks) }))
      .sort((a, b) => {
        const an = a.view?.record.number ?? a.home.slug ?? ''
        const bn = b.view?.record.number ?? b.home.slug ?? ''
        return an.localeCompare(bn, undefined, { numeric: true })
      })
    const views = homes.map(h => h.view).filter((v): v is CourseView => v !== null)
    const total = views.reduce((sum, v) => sum + v.units.length, 0)
    const notes = views.reduce((sum, v) => sum + v.units.filter(u => u.note).length, 0)
    const cardIds = views.flatMap(v => v.units.flatMap(u => u.cardIds))
    const hasDecks = cardIds.length > 0

    return (
      <section
        class={classNames(displayClass, 'course-index')}
        data-course-prefix="courses/"
        aria-labelledby="course-index-h"
      >
        <CourseBar
          from={from}
          heading={`${homes.length} ${homes.length === 1 ? 'course' : 'courses'}`}
          headingId="course-index-h"
          notes={notes}
          total={total}
          cards={cardIds.length}
          cardIds={cardIds}
          reviewSlug={'courses/review' as FullSlug}
          hasDecks={hasDecks}
        />
        {homes.length === 0 ? (
          <p class="course-count">
            no courses vendored. run <code>pnpm vendor:ocw &lt;url&gt;</code>.
          </p>
        ) : (
          <ol class="course-index-list">
            {homes.map(({ home, view }) => {
              const dir = courseDirOf(home.slug ?? '') ?? ''
              const title = view?.record.name || String(home.frontmatter?.title ?? dir)
              // homes without a study block still carry the number in their title
              const number =
                view?.record.number ??
                /\b\d+\.\d+[a-z]?\b/i.exec(String(home.frontmatter?.title ?? ''))?.[0] ??
                dir
              const term = [view?.record.term, view?.record.instructors[0]]
                .filter(Boolean)
                .join(' · ')
              const ids = view?.units.flatMap(u => u.cardIds) ?? []
              const noted = view?.units.filter(u => u.note).length
              return (
                <li class="course-row">
                  <a
                    class="course-row-link internal"
                    href={resolveRelative(from, home.slug as FullSlug)}
                    data-slug={home.slug}
                    data-no-popover
                  >
                    <span class="course-num">{number}</span>
                    <span class="course-row-title">{title}</span>
                    <span class="course-row-term">{term || '–'}</span>
                    <span class="course-stats" data-cards={ids.join(' ')} data-unit={dir}>
                      notes {view ? `${noted}/${view.units.length}` : '–'} · cards{' '}
                      {ids.length > 0 ? ids.length : '–'} · due <span data-due="">–</span> · last{' '}
                      <span data-last="" data-for={dir}>
                        –
                      </span>
                    </span>
                  </a>
                </li>
              )
            })}
          </ol>
        )}
      </section>
    )
  }
  CoursesIndex.css = style
  CoursesIndex.afterDOMLoaded = script
  return CoursesIndex
}) satisfies QuartzComponentConstructor
