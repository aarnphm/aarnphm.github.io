import { JSX } from 'preact'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import {
  CourseLink,
  CourseUnitView,
  courseLinks,
  courseResources,
  courseView,
  fileBySlug,
  formatCount,
} from '../util/course'
import { descriptionToPlainText } from '../util/description'
import { classNames } from '../util/lang'
import { FullSlug, resolveRelative } from '../util/path'
import { concatenateResources } from '../util/resources'
import { psetEntries, psetRows, solvedSets, type Slot, unitRow, unitSlots } from './CourseHome/rows'
import { CoursesCrumb } from './CourseResource'
import { courseStatements, statementsSlug } from './CourseStatements'
// @ts-ignore
import script from './scripts/course.inline'
import studyStyle from './styles/course-study.scss'
import style from './styles/course.scss'

const roman = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']

export { psetNumber } from '../util/course'

/** Bar shared by the schedule and the courses index: counts, then one review link. */
export function CourseBar({
  from,
  heading,
  headingId,
  read,
  notes,
  total,
  cards,
  cardIds,
  reviewSlug,
  hasDecks,
}: {
  from: FullSlug
  heading: string
  headingId: string
  read: number
  notes: number
  total: number
  cards: number
  cardIds: string[]
  reviewSlug: FullSlug
  hasDecks: boolean
}): JSX.Element {
  return (
    <header class="course-bar">
      <h2 id={headingId}>{heading}</h2>
      <p class="course-count" data-cards={cardIds.join(' ')} data-unit="all">
        read {read}/{total} · notes {notes}/{total} · cards {hasDecks ? formatCount(cards) : '–'} ·
        due <span data-due="">–</span> · last{' '}
        <span data-last="" data-for="all">
          –
        </span>{' '}
        · recall{' '}
        <span data-retention="" data-for="all">
          –
        </span>
      </p>
      {hasDecks ? (
        <a
          class="course-action internal"
          href={resolveRelative(from, reviewSlug)}
          data-slug={reviewSlug}
          data-no-popover
          data-course-action="review"
          data-cards={cardIds.join(' ')}
          data-unit="all-action"
        >
          review due <span data-due="">–</span>
        </a>
      ) : (
        <span class="course-action" aria-disabled="true">
          no decks yet
        </span>
      )}
    </header>
  )
}

function railList(from: FullSlug, links: CourseLink[]) {
  return (
    <ul>
      {links.map(link => (
        <li>
          <a class="internal" href={resolveRelative(from, link.slug)} data-slug={link.slug}>
            {link.title}
          </a>
          {link.count !== undefined ? <span class="course-count"> {link.count}</span> : null}
        </li>
      ))}
    </ul>
  )
}

const str = (value: unknown): string => (typeof value === 'string' ? value : '')

/** Tags that name the mirror rather than the subject. */
const mirrorTags = new Set(['course', 'mit', 'ocw'])

export default (() => {
  const CourseHome: QuartzComponent = ({
    ctx,
    fileData,
    allFiles,
    displayClass,
  }: QuartzComponentProps) => {
    const view = courseView(fileData, allFiles, ctx.decks)
    if (!view) return null
    const from = fileData.slug!
    const { record } = view
    const total = view.units.length
    const notes = view.units.filter(unit => unit.note).length
    const read = view.units.filter(unit => unit.read).length
    const cardIds = view.units.flatMap(unit => unit.cardIds)
    const hasDecks = view.units.some(unit => unit.deck)
    // the frontier is the first unit not yet read
    const frontierIdx = view.units.findIndex(u => !u.read)
    const resources = courseResources(view.dir, allFiles)

    // src slots: the union over the course, so the column lines up down the table
    const slots = new Set<Slot>()
    for (const unit of view.units) {
      for (const link of unitSlots(view, unit, from, resources)) if (link.href) slots.add(link.slot)
    }
    if (view.psetText.size > 0) slots.add('tex')
    if (psetEntries(view, resources).length > 0) slots.add('pdf')

    const parts = new Map<number | undefined, CourseUnitView[]>()
    for (const unit of view.units) {
      const key = record.parts.length > 0 ? unit.record.part : undefined
      const list = parts.get(key) ?? []
      list.push(unit)
      parts.set(key, list)
    }
    const psets = psetRows(view, from, slots, resources)
    const kindPlural = `${record.unitKind}s`

    const links = courseLinks(view.dir, allFiles)
    const railResources: CourseLink[] = [
      ...(links.notes ? [links.notes] : []),
      ...links.resources,
      { title: 'all resources', slug: `courses/${view.dir}/resources/index` as FullSlug },
    ]
    // the authored study pages: the dated log and the statement index
    const statementsPage = fileBySlug(statementsSlug(view.dir), allFiles)
    const railStudy: CourseLink[] = [
      ...(view.log ? [{ title: 'study log', slug: view.log.slug as FullSlug, count: read }] : []),
      ...(statementsPage
        ? [
            {
              title: 'statements',
              slug: statementsPage.slug as FullSlug,
              count: courseStatements(view).length,
            },
          ]
        : []),
    ]
    const frontmatter = fileData.frontmatter
    // OCW descriptions carry markdown emphasis; the masthead prints them as plain text
    const description = descriptionToPlainText(
      str(frontmatter?.description),
      fileData.slug,
    ).replace(/(\*{1,2}|_{1,2})(\S[^*_]*?)\1/g, '$2')
    const license = str(frontmatter?.license).toLowerCase()
    const licenseUrl = str(frontmatter?.license_url)
    const topics = (frontmatter?.tags ?? []).filter(tag => !mirrorTags.has(tag))
    const meta = [record.term, record.level, ...record.instructors].filter(Boolean)
    const columns = 8

    return (
      <div
        class={classNames(displayClass, 'course-home')}
        data-course-prefix={`courses/${view.dir}/`}
        data-marks-prefix={`courses/${view.dir}/`}
      >
        <header class="course-masthead" data-course-card>
          <p class="course-masthead-num course-num course-crumb">
            <CoursesCrumb from={fileData.slug!} />
            <span>{record.number}</span>
          </p>
          <h1 class="course-masthead-title course-card-title">{record.name}</h1>
          <p class="course-masthead-meta">
            {meta.map((item, i) => (
              <>
                {i > 0 ? <span class="course-sep">·</span> : null}
                <span>{item}</span>
              </>
            ))}
            {record.source ? (
              <>
                <span class="course-sep">·</span>
                <a href={record.source} target="_blank" rel="noopener noreferrer">
                  mit opencourseware
                </a>
              </>
            ) : null}
          </p>
          {description ? <p class="course-masthead-desc">{description}</p> : null}
        </header>
        <div class="course-body">
          <nav class="course-rail" aria-label="course pages">
            {links.pages.length > 0 ? (
              <section>
                <h2>pages</h2>
                {railList(from, links.pages)}
              </section>
            ) : null}
            {railStudy.length > 0 ? (
              <section>
                <h2>study</h2>
                {railList(from, railStudy)}
              </section>
            ) : null}
            <section>
              <h2>resources</h2>
              {railList(from, railResources)}
            </section>
            <section aria-label="course metadata">
              <dl>
                {topics.length > 0 ? (
                  <>
                    <dt>topics</dt>
                    <dd>{topics.join(', ')}</dd>
                  </>
                ) : null}
                {license ? (
                  <>
                    <dt>license</dt>
                    <dd>
                      {licenseUrl ? (
                        <a href={licenseUrl} target="_blank" rel="noopener noreferrer">
                          {license}
                        </a>
                      ) : (
                        license
                      )}
                    </dd>
                  </>
                ) : null}
                <dt>mirror</dt>
                <dd>
                  <a
                    class="internal"
                    href={resolveRelative(from, 'courses/index' as FullSlug)}
                    data-slug="courses/index"
                  >
                    all courses
                  </a>
                </dd>
              </dl>
            </section>
          </nav>
          <section class="course-main" aria-labelledby="course-spine-h">
            <CourseBar
              from={from}
              heading={kindPlural}
              headingId="course-spine-h"
              read={read}
              notes={notes}
              total={total}
              cards={cardIds.length}
              cardIds={cardIds}
              reviewSlug={`courses/review?in=${view.dir}` as FullSlug}
              hasDecks={hasDecks}
            />
            <table class="course-spine-table" data-course-rows>
              <thead>
                <tr>
                  <th scope="col" class="course-c-n">
                    #
                  </th>
                  <th scope="col">topic</th>
                  <th scope="col" class="course-c-state" title="r read · p problems">
                    state
                  </th>
                  <th scope="col" class="course-c-src">
                    src
                  </th>
                  <th scope="col" class="course-c-marks">
                    marks
                  </th>
                  <th scope="col" class="course-c-cards">
                    cards
                  </th>
                  <th scope="col" class="course-c-due">
                    due
                  </th>
                  <th scope="col" class="course-c-last">
                    last
                  </th>
                </tr>
              </thead>
              {Array.from(parts.entries()).map(([part, units]) => (
                <tbody class="course-part">
                  {part !== undefined && record.parts[part - 1] ? (
                    <tr class="course-part-head">
                      <th scope="rowgroup" colSpan={columns}>
                        {roman[part - 1] ?? part}. {record.parts[part - 1]}
                        <span class="course-count">
                          read {units.filter(u => u.read).length}/{units.length} · notes{' '}
                          {units.filter(u => u.note).length}/{units.length}
                        </span>
                      </th>
                    </tr>
                  ) : null}
                  {units.map(unit =>
                    unitRow(
                      view,
                      unit,
                      from,
                      view.units.indexOf(unit) === frontierIdx,
                      slots,
                      resources,
                    ),
                  )}
                </tbody>
              ))}
              {psets.length > 0 ? (
                <tbody class="course-part course-part--psets">
                  <tr class="course-part-head">
                    <th scope="rowgroup" colSpan={columns}>
                      problem sets
                      <span class="course-count">
                        solved {solvedSets(view)}/{psets.length}
                      </span>
                    </th>
                  </tr>
                  {psets}
                </tbody>
              ) : null}
            </table>
            {record.trailing.length > 0 ? (
              <p class="course-spine-trailing">{record.trailing.join(' · ')}</p>
            ) : null}
          </section>
        </div>
      </div>
    )
  }
  CourseHome.css = concatenateResources(style, studyStyle)
  CourseHome.afterDOMLoaded = script
  return CourseHome
}) satisfies QuartzComponentConstructor
