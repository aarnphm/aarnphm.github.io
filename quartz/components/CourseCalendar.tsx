import type { ComponentChildren } from 'preact'
import type { CourseCalendarData } from './course-calendar/model'
import { QuartzComponentProps } from '../types/component'
import {
  courseCadence,
  courseOf,
  coursePageOf,
  courseResources,
  resourceSlug,
  unitPdf,
  unitSources,
} from '../util/course'
import { FullSlug, resolveRelative } from '../util/path'
import { COURSE_CALENDAR_ID, CourseCalendarView } from './course-calendar/app'
import { psetNumber } from './CourseHome'
// @ts-ignore
import script from './scripts/course-calendar.inline'

/** The vendored calendar's units and problem sets, laid out by the `course-calendar` bundle. */
function calendarData(props: QuartzComponentProps, dir: string): CourseCalendarData | null {
  const { fileData, allFiles, ctx } = props
  const { view } = courseOf(dir, allFiles, ctx.decks)
  if (!view || view.units.length === 0) return null
  const from = fileData.slug!
  const href = (slug: FullSlug | undefined) => (slug ? resolveRelative(from, slug) : undefined)
  const resources = courseResources(dir, allFiles)
  const { record } = view
  const cadence = courseCadence(dir, allFiles)
  const seen = new Set<string>()
  const psets: CourseCalendarData['psets'] = []
  for (const unit of view.units) {
    unit.record.due.forEach((id, i) => {
      if (seen.has(id)) return
      seen.add(id)
      const n = psetNumber(id)
      const solution = n === undefined ? undefined : view.psets.get(n)
      psets.push({
        id,
        label: n === undefined ? (unit.record.keyDates[i] ?? id) : `problem set ${n}`,
        href: href(resourceSlug(dir, id)),
        unit: unit.record.n,
        ...(solution ? { solutionHref: href(solution.slug as FullSlug) } : {}),
      })
    })
  }
  return {
    number: record.number,
    unitKind: record.unitKind,
    total: view.units.length,
    cadence: cadence
      ? { ...cadence, source: 'syllabus' }
      : { sessions: 2, minutes: 90, source: 'default' },
    units: view.units.map(unit => {
      const sources = unitSources(unit, resources)
      const pdf = unitPdf(unit.record)
      // a unit without notes opens its pdf, then its video, as on the course home
      const target =
        unit.state === 'new'
          ? (sources.pdf?.slug ?? (pdf ? resourceSlug(dir, pdf.id) : sources.video?.slug))
          : unit.slug
      const part = unit.record.part ? record.parts[unit.record.part - 1] : undefined
      return {
        n: unit.record.n,
        title: unit.record.title,
        ...(part ? { part } : {}),
        ...(unit.record.sessions ? { sessions: unit.record.sessions } : {}),
        ...(target ? { href: href(target) } : {}),
        state: unit.state,
        ...(unit.read ? { read: unit.read } : {}),
        ...(unit.problems ? { problems: unit.problems } : {}),
        keyDates: unit.record.keyDates,
        cards: unit.cardIds.length,
      }
    }),
    psets,
    undated: record.trailing,
    ...(view.log ? { logHref: href(view.log.slug as FullSlug) } : {}),
    prefix: `courses/${dir}/`,
    cardIds: view.units.flatMap(unit => unit.cardIds),
  }
}

// JSON inside <script> must not close the element or open a comment.
const scriptJson = (value: unknown) =>
  JSON.stringify(value).replace(/</g, '\\u003c').replace(/>/g, '\\u003e')

/**
 * A course's vendored calendar page as a month calendar. The build renders that day's plan; the
 * `course-calendar` bundle re-renders it for the reader's today and adds the owner's due cards.
 */
export function CourseCalendar({
  props,
  children,
}: {
  props: QuartzComponentProps
  children: ComponentChildren
}) {
  const page = coursePageOf(props.fileData.slug)
  const data = page?.name === 'calendar' ? calendarData(props, page.dir) : null
  if (!data) return <>{children}</>
  return (
    <section class="course-calendar" aria-label={`${data.number} calendar`}>
      <script
        type="application/json"
        data-course-calendar-data
        dangerouslySetInnerHTML={{ __html: scriptJson(data) }}
      />
      <div class="course-calendar-mount" data-course-calendar-mount>
        <CourseCalendarView id={COURSE_CALENDAR_ID} data={data} />
      </div>
    </section>
  )
}

export const courseCalendarScript: string = script
