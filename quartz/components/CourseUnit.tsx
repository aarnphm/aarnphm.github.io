import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import {
  CourseUnitView,
  CourseView,
  courseView,
  resourceSlug,
  unitDeckSlug,
  unitOf,
  unitPdf,
  unitSlug,
  unitVideo,
} from '../util/course'
import { classNames } from '../util/lang'
import { FullSlug, resolveRelative } from '../util/path'
import { pdfReaderPath } from '../util/pdf-marks'
import { psetNumber } from './CourseSpine'
// @ts-ignore
import script from './scripts/course.inline'
import style from './styles/course.scss'

const roman = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
const pad = (n: number) => String(n).padStart(2, '0')

interface UnitContext {
  view: CourseView
  kind: 'lectures' | 'psets'
  n: number
  unit?: CourseUnitView
  index: number
}

function unitContext({ ctx, fileData, allFiles }: QuartzComponentProps): UnitContext | null {
  const at = unitOf(fileData.slug ?? '')
  if (!at) return null
  const home = allFiles.find(file => file.slug === `courses/${at.dir}/index`)
  const view = home ? courseView(home, allFiles, ctx.decks) : null
  if (!view) return null
  const index = view.units.findIndex(unit => unit.record.n === at.n)
  return {
    view,
    kind: at.kind,
    n: at.n,
    unit: at.kind === 'lectures' ? view.units[index] : undefined,
    index,
  }
}

/** Where a problem set falls in the calendar: the unit that lists it as due. */
function dueUnit(view: CourseView, n: number): CourseUnitView | undefined {
  return view.units.find(unit => unit.record.due.some(id => psetNumber(id) === n))
}

export const CourseUnitBar = (() => {
  const Bar: QuartzComponent = (props: QuartzComponentProps) => {
    const ctx = unitContext(props)
    if (!ctx) return null
    const { view, kind, n, unit } = ctx
    const from = props.fileData.slug!
    const homeHref = resolveRelative(from, view.slug)
    const part = unit?.record.part
    const partName = part ? view.record.parts[part - 1] : undefined
    const label =
      kind === 'psets'
        ? `problem set ${pad(n)}`
        : `${view.record.unitKind} ${pad(n)} of ${view.units.length}`
    const due = kind === 'psets' ? dueUnit(view, n) : undefined
    const prefix = `courses/${view.dir}/${kind}/${pad(n)}/`
    return (
      <nav
        class={classNames(props.displayClass, 'course-unit-bar')}
        aria-label="unit"
        data-course-prefix={prefix}
      >
        <a class="internal" href={homeHref} data-slug={view.slug} data-no-popover>
          {view.record.number}
        </a>
        {partName ? (
          <>
            <span class="course-unit-sep">/</span>
            <span>
              {roman[part! - 1] ?? part}. {partName}
            </span>
          </>
        ) : null}
        <span class="course-unit-sep">/</span>
        <span class="course-num">{label}</span>
        {unit && unit.record.keyDates.length > 0 ? (
          <span class="course-unit-dates">{unit.record.keyDates.join(' · ')}</span>
        ) : null}
        {due ? (
          <span class="course-unit-dates">
            due at {view.record.unitKind} {pad(due.record.n)}
          </span>
        ) : null}
        {unit?.read ? <span class="course-unit-dates">read {unit.read}</span> : null}
        {unit?.deck ? (
          <a
            class="course-action internal"
            href={resolveRelative(from, unitDeckSlug(view.dir, n))}
            data-slug={unitDeckSlug(view.dir, n)}
            data-no-popover
            data-course-action="drill"
            data-cards={unit.cardIds.join(' ')}
            data-unit={String(n)}
          >
            due <span data-due="">–</span> of {unit.cardIds.length}
          </a>
        ) : (
          <span class="course-action" aria-disabled="true">
            no deck
          </span>
        )}
      </nav>
    )
  }
  Bar.displayName = 'CourseUnitBar'
  Bar.css = style
  Bar.afterDOMLoaded = script
  return Bar
}) satisfies QuartzComponentConstructor

export const CourseUnitFoot = (() => {
  const Foot: QuartzComponent = (props: QuartzComponentProps) => {
    const ctx = unitContext(props)
    if (!ctx) return null
    const { view, kind, n, unit, index } = ctx
    const from = props.fileData.slug!
    const record = unit?.record
    const pdf = record ? unitPdf(record) : undefined
    const video = record ? unitVideo(record) : undefined
    const pdfPath = pdf?.file ? `courses/${view.dir}/${pdf.file}` : undefined
    const title = record
      ? `${view.record.unitKind} ${pad(n)}: ${record.title}`
      : `problem set ${pad(n)}`

    const dueHere = record
      ? record.due.map(id => {
          const psetN = psetNumber(id)
          const solution = psetN === undefined ? undefined : view.psets.get(psetN)
          return (
            <li>
              <a class="internal" href={resolveRelative(from, resourceSlug(view.dir, id))}>
                {psetN === undefined ? id : `problem set ${psetN}`}
              </a>
              {' · '}
              {solution ? (
                <a class="internal" href={resolveRelative(from, solution.slug as FullSlug)}>
                  our solutions
                </a>
              ) : (
                <span>
                  solutions: <code>psets/{pad(psetN ?? 0)}.md</code>
                </span>
              )}
            </li>
          )
        })
      : []

    const neighbour = (unit: CourseUnitView | undefined, rel: 'prev' | 'next') => {
      if (!unit) return <span aria-hidden="true" />
      const target = unit.note ? unit.slug : undefined
      const resource = unitPdf(unit.record)
      const href = target
        ? resolveRelative(from, target)
        : resource
          ? resolveRelative(from, resourceSlug(view.dir, resource.id))
          : undefined
      const body = (
        <>
          <span class="course-num">{pad(unit.record.n)}</span>
          <span>{unit.record.title}</span>
        </>
      )
      return href ? (
        <a rel={rel} class="internal" href={href}>
          {body}
        </a>
      ) : (
        <span>{body}</span>
      )
    }

    return (
      <footer class={classNames(props.displayClass, 'course-unit-foot')}>
        {pdfPath ? (
          <section aria-labelledby="course-src-h">
            <h2 id="course-src-h">source</h2>
            <p class="course-count">
              <a href={pdfReaderPath(pdfPath)} data-router-ignore>
                open in reader
              </a>
              {' · '}
              <a href={`/${pdfPath}`} target="_blank" rel="noopener noreferrer">
                pdf
              </a>
              {video?.youtube ? (
                <>
                  {' · '}
                  <a href={video.youtube} target="_blank" rel="noopener noreferrer">
                    video
                  </a>
                </>
              ) : null}
              {' · mit ocw cc by-nc-sa'}
            </p>
            <div
              class="internal-embed pdf-embed"
              data-pdf-src={`/${pdfPath}`}
              data-pdf-title={title}
              tabIndex={0}
            >
              <span class="pdf-embed-loading">loading pdf…</span>
            </div>
          </section>
        ) : null}
        {dueHere.length > 0 ? (
          <section aria-labelledby="course-due-h">
            <h2 id="course-due-h">due here</h2>
            <ul>{dueHere}</ul>
          </section>
        ) : null}
        {kind === 'lectures' && index >= 0 ? (
          <nav class="course-unit-pager" aria-label="units">
            {neighbour(view.units[index - 1], 'prev')}
            {neighbour(view.units[index + 1], 'next')}
          </nav>
        ) : null}
        {kind === 'psets' ? (
          <nav class="course-unit-pager" aria-label="units">
            {view.psets.get(n - 1) ? (
              <a
                rel="prev"
                class="internal"
                href={resolveRelative(from, unitSlug(view.dir, n - 1, 'psets'))}
              >
                <span class="course-num">{pad(n - 1)}</span>
                <span>problem set</span>
              </a>
            ) : (
              <span aria-hidden="true" />
            )}
            {view.psets.get(n + 1) ? (
              <a
                rel="next"
                class="internal"
                href={resolveRelative(from, unitSlug(view.dir, n + 1, 'psets'))}
              >
                <span class="course-num">{pad(n + 1)}</span>
                <span>problem set</span>
              </a>
            ) : (
              <span aria-hidden="true" />
            )}
          </nav>
        ) : null}
      </footer>
    )
  }
  Foot.displayName = 'CourseUnitFoot'
  Foot.css = style
  Foot.afterDOMLoaded = script
  return Foot
}) satisfies QuartzComponentConstructor
