import { JSX } from 'preact'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import {
  CourseLink,
  CourseResourceRef,
  CourseUnitView,
  CourseView,
  courseFileSlug,
  courseLinks,
  courseResources,
  courseView,
  formatCount,
  formatDuration,
  psetNumber,
  psetTextSlug,
  resourceSlug,
  unitDeckSlug,
  unitPdf,
  unitSources,
} from '../util/course'
import { classNames } from '../util/lang'
import { FullSlug, resolveRelative } from '../util/path'
import { concatenateResources } from '../util/resources'
import { CoursesCrumb } from './CourseResource'
import { courseStatements, statementsSlug } from './CourseStatements'
import { StudySlots } from './CourseStudy'
// @ts-ignore
import script from './scripts/course.inline'
import studyStyle from './styles/course-study.scss'
import style from './styles/course.scss'

const roman = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']

const pad = (n: number) => String(n).padStart(2, '0')

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

type Slot = 'pdf' | 'tex' | 'video' | 'txt'

/** `1:18` for the fixed-width slot: hours and minutes, or minutes and seconds under an hour. */
function slotDuration(seconds: number | undefined): string | undefined {
  const full = formatDuration(seconds)
  if (!full) return undefined
  const parts = full.split(':')
  return parts.length === 3 ? `${parts[0]}:${parts[1]}` : full
}
const slotOrder: Slot[] = ['pdf', 'tex', 'video', 'txt']

interface SlotLink {
  slot: Slot
  href?: string
  text: string
  slug?: FullSlug
  title?: string
}

/** The `src` cell: fixed slots in course order, `–` where the unit has nothing for a slot. */
function unitSlots(
  view: CourseView,
  unit: CourseUnitView,
  from: FullSlug,
  resources: Map<string, CourseResourceRef>,
): SlotLink[] {
  const sources = unitSources(unit, resources)
  const pdfUnit = unitPdf(unit.record)
  const pdfSlug = sources.pdf?.slug ?? (pdfUnit ? resourceSlug(view.dir, pdfUnit.id) : undefined)
  const length = slotDuration(sources.video?.record.duration)
  return [
    {
      slot: 'pdf',
      text: 'pdf',
      ...(pdfSlug ? { href: resolveRelative(from, pdfSlug), slug: pdfSlug } : {}),
    },
    {
      slot: 'tex',
      text: 'tex',
      ...(unit.text
        ? {
            href: resolveRelative(from, unit.text.slug as FullSlug),
            slug: unit.text.slug as FullSlug,
          }
        : {}),
    },
    {
      slot: 'video',
      text: length ?? 'video',
      ...(sources.video
        ? {
            href: resolveRelative(from, sources.video.slug),
            slug: sources.video.slug,
            ...(sources.videos.length > 1 ? { title: `${sources.videos.length} videos` } : {}),
          }
        : {}),
    },
    {
      slot: 'txt',
      text: 'txt',
      ...(sources.transcript
        ? { href: resolveRelative(from, sources.transcript.slug), slug: sources.transcript.slug }
        : {}),
    },
  ]
}

function slotCell(links: SlotLink[], slots: Set<Slot>, className: string) {
  return (
    <span class={className}>
      {slotOrder
        .filter(slot => slots.has(slot))
        .map(slot => {
          const link = links.find(entry => entry.slot === slot)!
          return link.href ? (
            <a
              class="internal course-slot"
              href={link.href}
              data-slug={link.slug}
              title={link.title}
            >
              {link.text}
            </a>
          ) : (
            <span class="course-slot course-slot--empty" aria-hidden="true">
              –
            </span>
          )
        })}
    </span>
  )
}

function unitRow(
  view: CourseView,
  unit: CourseUnitView,
  from: FullSlug,
  frontier: boolean,
  slots: Set<Slot>,
  resources: Map<string, CourseResourceRef>,
) {
  const links = unitSlots(view, unit, from, resources)
  const pdf = links.find(link => link.slot === 'pdf')
  const video = links.find(link => link.slot === 'video')
  // a new unit's topic falls back note → pdf → watch page
  const topicHref =
    unit.state === 'new' ? (pdf?.href ?? video?.href) : resolveRelative(from, unit.slug)
  const pdfUnit = unitPdf(unit.record)
  const marksSrc = pdfUnit?.file ? courseFileSlug(view.dir, pdfUnit.file) : undefined
  const dates = unit.record.keyDates.map((date, i) => {
    const dueId = unit.record.due[i]
    return dueId ? (
      <a class="internal" href={resolveRelative(from, resourceSlug(view.dir, dueId))}>
        {date}
      </a>
    ) : (
      <span>{date}</span>
    )
  })
  return (
    <tr
      class="course-unit-row"
      data-state={unit.state}
      data-frontier={frontier ? 'true' : undefined}
    >
      <td class="course-num course-c-n">{pad(unit.record.n)}</td>
      <td>
        {topicHref ? (
          <a class="internal course-unit-title" href={topicHref}>
            {unit.record.title}
          </a>
        ) : (
          <span class="course-unit-title">{unit.record.title}</span>
        )}
        {unit.record.sessions ? (
          <span class="course-unit-dates">{unit.record.sessions}</span>
        ) : null}
        {dates.length > 0 ? <span class="course-unit-dates">{dates}</span> : null}
        {slotCell(links, slots, 'course-unit-src')}
      </td>
      <td class="course-c-state">
        <StudySlots unit={unit} />
      </td>
      <td class="course-c-src">{slotCell(links, slots, 'course-slots')}</td>
      <td
        class="course-num course-c-marks"
        data-marks-src={marksSrc}
        title={marksSrc ? 'marks · questions in the reader' : undefined}
      >
        –
      </td>
      <td class="course-c-cards">
        {unit.deck ? (
          <a
            class="internal"
            href={resolveRelative(from, unitDeckSlug(view.dir, unit.record.n))}
            data-no-popover
          >
            {unit.cardIds.length}
          </a>
        ) : (
          '–'
        )}
      </td>
      <td
        class="course-c-due"
        data-due=""
        data-cards={unit.cardIds.length > 0 ? unit.cardIds.join(' ') : undefined}
        data-unit={String(unit.record.n)}
      >
        –
      </td>
      <td class="course-c-last" data-last="" data-for={String(unit.record.n)}>
        –
      </td>
    </tr>
  )
}

/**
 * Problem sets in due order: those the calendar names at a unit, then the rest of the course's
 * `pset` resources (18.156 and 18.100B list problem sets with no schedule), one per number.
 */
export function psetEntries(
  view: CourseView,
  resources: Map<string, CourseResourceRef>,
): { id: string; label: string; due?: number }[] {
  const entries: { id: string; label: string; due?: number }[] = []
  const seen = new Set<string>()
  for (const unit of view.units) {
    unit.record.due.forEach((id, i) => {
      if (seen.has(id)) return
      seen.add(id)
      entries.push({ id, label: unit.record.keyDates[i] ?? id, due: unit.record.n })
    })
  }
  const numbers = new Set(entries.map(entry => psetNumber(entry.id)))
  // the shortest id is the term's own set; `pset1-1` style ids are older terms
  const undated = Array.from(resources.values())
    .filter(ref => ref.record.kind === 'pset' && ref.record.n !== undefined)
    .sort((a, b) => a.record.n! - b.record.n! || a.id.length - b.id.length)
  for (const ref of undated) {
    if (numbers.has(ref.record.n!)) continue
    numbers.add(ref.record.n!)
    entries.push({ id: ref.id, label: `problem set ${ref.record.n}` })
  }
  return entries
}

function psetRows(
  view: CourseView,
  from: FullSlug,
  slots: Set<Slot>,
  resources: Map<string, CourseResourceRef>,
) {
  return psetEntries(view, resources).map(({ id, label, due }) => {
    const n = psetNumber(id)
    const solution = n === undefined ? undefined : view.psets.get(n)
    const text = n === undefined ? undefined : view.psetText.get(n)
    const ref = resources.get(id)
    const pdfSlug = resourceSlug(view.dir, id)
    const marksSrc = ref?.record.file ? courseFileSlug(view.dir, ref.record.file) : undefined
    const links: SlotLink[] = [
      { slot: 'pdf', text: 'pdf', href: resolveRelative(from, pdfSlug), slug: pdfSlug },
      {
        slot: 'tex',
        text: 'tex',
        ...(text
          ? {
              href: resolveRelative(from, psetTextSlug(view.dir, n!)),
              slug: psetTextSlug(view.dir, n!),
            }
          : {}),
      },
      { slot: 'video', text: 'video' },
      { slot: 'txt', text: 'txt' },
    ]
    return (
      <tr class="course-unit-row" data-state={solution ? 'notes' : 'new'}>
        <td class="course-num course-c-n">{n === undefined ? '–' : pad(n)}</td>
        <td>
          <a class="internal course-unit-title" href={resolveRelative(from, pdfSlug)}>
            {label}
          </a>
          <span class="course-unit-dates">
            {due === undefined ? 'no due date' : `due at ${view.record.unitKind} ${pad(due)}`}
          </span>
          {slotCell(links, slots, 'course-unit-src')}
        </td>
        <td class="course-c-state" />
        <td class="course-c-src">{slotCell(links, slots, 'course-slots')}</td>
        <td class="course-num course-c-marks" data-marks-src={marksSrc}>
          –
        </td>
        <td class="course-c-cards" colSpan={3}>
          {solution ? (
            <a class="internal" href={resolveRelative(from, solution.slug as FullSlug)}>
              our solutions
            </a>
          ) : (
            '–'
          )}
        </td>
      </tr>
    )
  })
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
    const statementsPage = allFiles.find(file => file.slug === statementsSlug(view.dir))
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
    const description = str(frontmatter?.description).replace(
      /(\*{1,2}|_{1,2})(\S[^*_]*?)\1/g,
      '$2',
    )
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
                        solved {view.psets.size}/{psets.length}
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
