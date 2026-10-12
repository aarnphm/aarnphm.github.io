import { JSX } from 'preact'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import {
  CourseUnitView,
  CourseView,
  StudyKey,
  courseOf,
  courseDirOf,
  studyLogSlug,
  unitLabel,
} from '../util/course'
import { courseStudyEndpoint } from '../util/course-study'
import { classNames } from '../util/lang'
import { FullSlug, resolveRelative } from '../util/path'
import { concatenateResources } from '../util/resources'
import { CoursesCrumb } from './CourseResource'
// @ts-ignore
import script from './scripts/course.inline'
import studyStyle from './styles/course-study.scss'
import style from './styles/course.scss'

const pad = (n: number) => String(n).padStart(2, '0')
const keys: StudyKey[] = ['read', 'problems']

/**
 * Read and problems dates for one unit. In dev the two are toggles that post to the study
 * endpoint, followed by a `log` link; elsewhere a set date links to the study log and an unset one
 * is omitted. The row never grows with save feedback: the toggle label carries the date.
 */
export function StudyState({
  view,
  unit,
  from,
  endpoint,
}: {
  view: CourseView
  unit: CourseUnitView
  from: FullSlug
  endpoint?: string
}): JSX.Element | null {
  const n = unit.record.n
  const log = view.log ? studyLogSlug(view.dir) : undefined
  const logLink = (label: string) =>
    log ? (
      <a class="internal" href={resolveRelative(from, log)} data-slug={log} data-no-popover>
        {label}
      </a>
    ) : (
      <span>{label}</span>
    )
  if (!endpoint) {
    const set = keys.filter(key => unit[key])
    if (set.length === 0) return null
    return (
      <span class="course-study">
        {set.map((key, i) => (
          <>
            {i > 0 ? <span class="course-sep">·</span> : null}
            {logLink(`${key} ${unit[key]}`)}
          </>
        ))}
      </span>
    )
  }
  return (
    <span
      class="course-study"
      data-study-endpoint={endpoint}
      data-study-dir={view.dir}
      data-study-n={String(n)}
    >
      {keys.map(key => {
        const date = unit[key]
        return (
          <button
            type="button"
            class="course-study-toggle"
            aria-pressed={date ? 'true' : 'false'}
            data-study-key={key}
            data-study-date={date}
            title={date ? `${key} ${date} · click to clear` : `mark ${key} today`}
          >
            {key}
          </button>
        )
      })}
      {log ? logLink('log') : null}
      <span class="course-study-status" role="status" aria-live="polite" />
    </span>
  )
}

/** Schedule cell: two fixed slots, `r` (read) and `p` (problems), `–` where unset. */
export function StudySlots({ unit }: { unit: CourseUnitView }): JSX.Element {
  return (
    <span class="course-state">
      {keys.map(key => {
        const date = unit[key]
        return date ? (
          <abbr class="course-state-slot" title={`${key} ${date}`}>
            {key[0]}
          </abbr>
        ) : (
          <span class="course-state-slot course-slot--empty" title={`not ${key}`}>
            –
          </span>
        )
      })}
    </span>
  )
}

/** `courses/<dir>/lectures/index.md`: the study log, one row per dated unit, newest first. */
export default (() => {
  const CourseStudyLog: QuartzComponent = (props: QuartzComponentProps) => {
    const { ctx, fileData, allFiles, displayClass } = props
    const from = fileData.slug!
    const dir = courseDirOf(from)
    const view = dir ? courseOf(dir, allFiles, ctx.decks).view : null
    if (!view) return null
    const rows = view.units
      .flatMap(unit => keys.filter(key => unit[key]).map(key => ({ unit, key, date: unit[key]! })))
      .sort(
        (a, b) =>
          b.date.localeCompare(a.date) ||
          b.unit.record.n - a.unit.record.n ||
          a.key.localeCompare(b.key),
      )
    const total = view.units.length
    const read = view.units.filter(unit => unit.read).length
    const problems = view.units.filter(unit => unit.problems).length
    const endpoint = courseStudyEndpoint(ctx.argv)
    const title = String(fileData.frontmatter?.title ?? `${view.record.number} study log`)
    return (
      <div class={classNames(displayClass, 'course-home', 'course-study-log')}>
        <header class="course-masthead" data-course-card>
          <p class="course-masthead-num course-num course-crumb">
            <CoursesCrumb from={from} />
            <a class="internal" href={resolveRelative(from, view.slug)} data-slug={view.slug}>
              {view.record.number}
            </a>
            <span class="course-unit-sep">/</span>
            <span>study log</span>
          </p>
          <h1 class="course-masthead-title course-card-title">{title}</h1>
          <p class="course-count">
            read {read}/{total} · problems {problems}/{total} · source{' '}
            <code class="course-path">lectures/index.md</code>
            {endpoint ? ' · dev: the read and problems toggles write here' : ''}
          </p>
        </header>
        {rows.length === 0 ? (
          <p class="course-count">
            nothing dated yet. mark a {view.record.unitKind} read from its bar or its pdf.
          </p>
        ) : (
          <table class="course-spine-table course-log-table" data-course-rows>
            <thead>
              <tr>
                <th scope="col" class="course-c-date">
                  date
                </th>
                <th scope="col" class="course-c-n">
                  #
                </th>
                <th scope="col">{view.record.unitKind}</th>
                <th scope="col" class="course-c-what">
                  done
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map(({ unit, key, date }) => {
                const target: FullSlug | undefined = unit.note
                  ? unit.slug
                  : unit.text
                    ? (unit.text.slug as FullSlug)
                    : undefined
                return (
                  <tr class="course-unit-row">
                    <td class="course-num course-c-date">{date}</td>
                    <td class="course-num course-c-n">{pad(unit.record.n)}</td>
                    <td>
                      {target ? (
                        <a
                          class="internal course-unit-title"
                          href={resolveRelative(from, target)}
                          data-slug={target}
                        >
                          {unit.record.title}
                        </a>
                      ) : (
                        <span class="course-unit-title">{unit.record.title}</span>
                      )}
                      <span class="course-unit-dates">
                        {unitLabel(view.record.unitKind, unit.record.n, total)}
                      </span>
                    </td>
                    <td class="course-c-what">{key}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    )
  }
  CourseStudyLog.css = concatenateResources(style, studyStyle)
  CourseStudyLog.afterDOMLoaded = script
  return CourseStudyLog
}) satisfies QuartzComponentConstructor
