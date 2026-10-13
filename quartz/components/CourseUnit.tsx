import { JSX } from 'preact'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import {
  CourseResourceRef,
  CourseUnitView,
  CourseView,
  LatexRecord,
  courseFileSlug,
  courseResources,
  courseView,
  formatDuration,
  latexOf,
  lecturerName,
  psetRefs,
  psetTextSlug,
  resourceSlug,
  unitDeckSlug,
  unitOf,
  unitPdf,
  unitSlug,
  unitSources,
  unitVideo,
} from '../util/course'
import { courseStudyEndpoint } from '../util/course-study'
import { classNames } from '../util/lang'
import { FullSlug, resolveRelative } from '../util/path'
import { pdfReaderPath } from '../util/pdf-marks'
import { concatenateResources } from '../util/resources'
import { CourseDesk, solutionTerm } from './CourseDesk'
import { psetNumber } from './CourseHome'
import { CoursesCrumb } from './CourseResource'
import { StudyState } from './CourseStudy'
// @ts-ignore
import workspaceScript from './scripts/course-workspace.inline'
// @ts-ignore
import script from './scripts/course.inline'
import studyStyle from './styles/course-study.scss'
import workspaceStyle from './styles/course-workspace.scss'
import style from './styles/course.scss'

const roman = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
const pad = (n: number) => String(n).padStart(2, '0')
const str = (value: unknown): string => (typeof value === 'string' ? value : '')
const psetNotesRe = /^courses\/([^/]+)\/notes\/pset-(\d{2,3})$/

type UnitKind = 'lectures' | 'psets' | 'notes' | 'pset-notes'

interface UnitContext {
  view: CourseView
  kind: UnitKind
  n: number
  unit?: CourseUnitView
  index: number
  latex?: LatexRecord
  resources: Map<string, CourseResourceRef>
}

function unitContext({ ctx, fileData, allFiles }: QuartzComponentProps): UnitContext | null {
  const slug = fileData.slug ?? ''
  const at = unitOf(slug)
  const psetNotes = at ? null : psetNotesRe.exec(slug)
  const dir = at?.dir ?? psetNotes?.[1]
  if (!dir) return null
  const home = allFiles.find(file => file.slug === `courses/${dir}/index`)
  const view = home ? courseView(home, allFiles, ctx.decks) : null
  if (!view) return null
  const latex = latexOf(fileData) ?? undefined
  const kind: UnitKind = at ? at.kind : 'pset-notes'
  const n = at ? (kind === 'notes' ? (latex?.unit ?? at.n) : at.n) : Number(psetNotes![2])
  const index = view.units.findIndex(unit => unit.record.n === n)
  const unit = kind === 'lectures' || kind === 'notes' ? view.units[index] : undefined
  return {
    view,
    kind,
    n,
    ...(unit ? { unit } : {}),
    index,
    ...(latex ? { latex } : {}),
    resources: courseResources(dir, allFiles),
  }
}

/** Where a problem set falls in the calendar: the unit that lists it as due. */
function dueUnit(view: CourseView, n: number): CourseUnitView | undefined {
  return view.units.find(unit => unit.record.due.some(id => psetNumber(id) === n))
}

/** The `.tex` resource page for a LaTeX note: the full notes source or the pset with the same n. */
function texRef(
  resources: Map<string, CourseResourceRef>,
  latex: LatexRecord,
): CourseResourceRef | undefined {
  for (const ref of resources.values()) {
    if (ref.record.kind !== 'tex') continue
    if (latex.pset !== undefined ? ref.record.n === latex.pset : ref.record.full) return ref
  }
  return undefined
}

// the bar keeps one row beside the title, so its crumb is short: the lecturer and the unit's
// full count move to titles
function notesCrumb(view: CourseView, from: FullSlug) {
  const notesSlug = `courses/${view.dir}/notes/index` as FullSlug
  const who = lecturerName(view.record)
  return (
    <a
      class="internal"
      href={resolveRelative(from, notesSlug)}
      data-slug={notesSlug}
      title={who ? `${who}'s notes` : undefined}
    >
      notes
    </a>
  )
}

export const CourseUnitBar = (() => {
  const Bar: QuartzComponent = (props: QuartzComponentProps) => {
    const ctx = unitContext(props)
    if (!ctx) return null
    const { view, kind, n, unit, latex } = ctx
    const from = props.fileData.slug!
    const homeHref = resolveRelative(from, view.slug)
    const part = unit?.record.part
    const partName = part ? view.record.parts[part - 1] : undefined
    const label =
      kind === 'psets' || kind === 'pset-notes'
        ? `problem set ${pad(n)}`
        : `${pad(n)}/${view.units.length}`
    const labelName =
      kind === 'psets' || kind === 'pset-notes'
        ? label
        : `${view.record.unitKind} ${pad(n)} of ${view.units.length}`
    const due = kind === 'psets' || kind === 'pset-notes' ? dueUnit(view, n) : undefined
    const prefix =
      kind === 'lectures' || kind === 'psets'
        ? `courses/${view.dir}/${kind}/${pad(n)}/`
        : unit?.deck
          ? `courses/${view.dir}/lectures/${pad(n)}/`
          : undefined
    const isNotes = kind === 'notes' || kind === 'pset-notes'
    const taught = str(props.fileData.frontmatter?.date)
    const meta: (string | JSX.Element)[] = []
    if (unit && kind === 'lectures' && unit.record.keyDates.length > 0)
      meta.push(unit.record.keyDates.join(' · '))
    if (isNotes && taught) meta.push(`taught ${taught}`)
    if (latex?.dateSource === 'previous') meta.push('date from the previous lecture')
    if (due) meta.push(`due at ${view.record.unitKind} ${pad(due.record.n)}`)

    const endpoint = courseStudyEndpoint(props.ctx.argv)

    return (
      <nav
        class={classNames(props.displayClass, 'course-unit-bar')}
        aria-label="unit"
        data-course-prefix={prefix}
        data-course-card
      >
        <span class="course-crumb">
          <CoursesCrumb from={from} />
          <a class="internal" href={homeHref} data-slug={view.slug} data-no-popover>
            {view.record.number}
          </a>
          {isNotes ? (
            <>
              <span class="course-unit-sep">/</span>
              {notesCrumb(view, from)}
            </>
          ) : partName ? (
            <>
              <span class="course-unit-sep">/</span>
              <span>
                {roman[part! - 1] ?? part}. {partName}
              </span>
            </>
          ) : null}
          <span class="course-unit-sep">/</span>
          <span class="course-num" title={labelName}>
            {label}
          </span>
        </span>
        {meta.length > 0 ? (
          <span class="course-count course-unit-dates">
            {meta.map((item, i) => (
              <>
                {i > 0 ? ' · ' : ''}
                {item}
              </>
            ))}
          </span>
        ) : null}
        <span class="course-unit-actions">
          {unit && (kind === 'lectures' || kind === 'notes') ? (
            <StudyState view={view} unit={unit} from={from} endpoint={endpoint} />
          ) : null}
          {isNotes ? (
            <button
              type="button"
              class="course-action course-fold"
              data-course-fold-proofs
              aria-pressed="false"
            >
              fold proofs
            </button>
          ) : unit?.deck ? (
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
        </span>
      </nav>
    )
  }
  Bar.displayName = 'CourseUnitBar'
  Bar.css = concatenateResources(style, studyStyle)
  Bar.afterDOMLoaded = script
  return Bar
}) satisfies QuartzComponentConstructor

/** The desk of a problem set page, under its title. */
export const CoursePsetDesk = (() => {
  const Desk: QuartzComponent = (props: QuartzComponentProps) => {
    const ctx = unitContext(props)
    if (!ctx || (ctx.kind !== 'psets' && ctx.kind !== 'pset-notes')) return null
    return (
      <CourseDesk
        view={ctx.view}
        n={ctx.n}
        from={props.fileData.slug!}
        resources={ctx.resources}
        endpoint={courseStudyEndpoint(props.ctx.argv)}
      />
    )
  }
  Desk.displayName = 'CoursePsetDesk'
  Desk.css = concatenateResources(style, studyStyle)
  Desk.afterDOMLoaded = script
  return Desk
}) satisfies QuartzComponentConstructor

/** Byline cell on OCW notes: where the reader's own notes for the unit live, a link once they exist. */
export const CourseUnitNotes = (() => {
  const Notes: QuartzComponent = (props: QuartzComponentProps) => {
    const ctx = unitContext(props)
    if (!ctx || (ctx.kind !== 'notes' && ctx.kind !== 'pset-notes')) return null
    const { view, kind, n, unit } = ctx
    const from = props.fileData.slug!
    const path = kind === 'notes' ? `lectures/${pad(n)}.md` : `psets/${pad(n)}.md`
    const target =
      kind === 'notes'
        ? unit?.note
          ? unit.slug
          : undefined
        : (view.psets.get(n)?.slug as FullSlug | undefined)
    return (
      <div class={classNames(props.displayClass, 'course-unit-notes')}>
        <h2>{kind === 'notes' ? 'your notes' : 'solutions'}</h2>
        <p>
          {target ? (
            <a class="internal" href={resolveRelative(from, target)} data-slug={target}>
              {path}
            </a>
          ) : (
            <span>{path}</span>
          )}
        </p>
      </div>
    )
  }
  Notes.displayName = 'CourseUnitNotes'
  Notes.css = style
  return Notes
}) satisfies QuartzComponentConstructor

export const CourseUnitWorkspace = (() => {
  const Workspace: QuartzComponent = (props: QuartzComponentProps) => {
    const ctx = unitContext(props)
    const slug = props.fileData.slug
    if (!ctx || (ctx.kind !== 'lectures' && ctx.kind !== 'notes' && ctx.kind !== 'psets') || !slug)
      return null
    const { view, unit, n } = ctx
    const isPset = ctx.kind === 'psets'
    // a problem set reads its own pdf as the source
    const pdf = isPset
      ? psetRefs(ctx.resources, n).pset?.record
      : unit
        ? unitPdf(unit.record)
        : undefined
    const video = unit ? unitVideo(unit.record) : undefined
    const pdfPath = pdf?.file ? `courses/${view.dir}/${pdf.file}` : undefined
    const title = isPset
      ? `problem set ${pad(n)}`
      : `${view.record.unitKind} ${pad(n)}: ${unit?.record.title ?? props.fileData.frontmatter?.title ?? ''}`
    return (
      <aside
        class="course-workspace"
        aria-label={isPset ? 'problem set workspace' : 'lecture workspace'}
      >
        <div
          data-course-workspace-mount
          data-course-workspace-slug={slug}
          data-course-workspace-title={title}
          data-course-workspace-pdf={pdfPath}
          data-course-workspace-reader={pdfPath ? pdfReaderPath(pdfPath) : undefined}
          data-course-workspace-video={video?.youtube}
        >
          <div class="course-workspace-app">
            <div class="course-workspace-panel">
              <div class="course-workspace-loading">
                <p role="status">Loading lecture workspace…</p>
                {pdfPath && (
                  <a href={`${pdfReaderPath(pdfPath)}?view=reader`} data-router-ignore>
                    open source in reader
                  </a>
                )}
              </div>
            </div>
          </div>
        </div>
      </aside>
    )
  }
  Workspace.displayName = 'CourseUnitWorkspace'
  Workspace.css = workspaceStyle
  Workspace.afterDOMLoaded = workspaceScript
  return Workspace
}) satisfies QuartzComponentConstructor

export const CourseUnitFoot = (() => {
  const Foot: QuartzComponent = (props: QuartzComponentProps) => {
    const ctx = unitContext(props)
    if (!ctx) return null
    const { view, kind, n, unit, index, latex, resources } = ctx
    const from = props.fileData.slug!
    const record = unit?.record
    const link = (slug: FullSlug, text: string, hash = '') => (
      <a class="internal" href={resolveRelative(from, slug) + hash} data-slug={slug}>
        {text}
      </a>
    )

    const dueHere =
      record && kind === 'lectures'
        ? record.due.map(id => {
            const psetN = psetNumber(id)
            const solution = psetN === undefined ? undefined : view.psets.get(psetN)
            return (
              <li>
                {link(
                  resourceSlug(view.dir, id),
                  psetN === undefined ? id : `problem set ${psetN}`,
                )}
                {' · '}
                {solution ? (
                  link(solution.slug as FullSlug, 'our solutions')
                ) : (
                  <span>
                    solutions: <code class="course-path">psets/{pad(psetN ?? 0)}.md</code>
                  </span>
                )}
              </li>
            )
          })
        : []

    // sources: the reader, the watch page, the generated notes, the LaTeX
    const sources: JSX.Element[] = []
    if (unit) {
      const src = unitSources(unit, resources)
      const pdfUnit = unitPdf(unit.record)
      const pdfSlug = src.pdf?.slug ?? (pdfUnit ? resourceSlug(view.dir, pdfUnit.id) : undefined)
      if (pdfSlug && kind === 'lectures') sources.push(<li>{link(pdfSlug, 'open in reader')}</li>)
      if (pdfSlug && kind === 'notes') sources.push(<li>{link(pdfSlug, 'lecture pdf')}</li>)
      if (kind === 'notes' && latex?.pages) {
        const [a, b] = latex.pages
        sources.push(
          <li>
            {link(resourceSlug(view.dir, latex.pdf), `full notes pp. ${a}–${b}`, `#page=${a}`)}
          </li>,
        )
      }
      if (src.video) {
        const length = formatDuration(src.video.record.duration)
        sources.push(<li>{link(src.video.slug, length ? `watch ${length}` : 'watch')}</li>)
      }
      if (kind === 'lectures' && unit.text) {
        const who = lecturerName(view.record)
        sources.push(
          <li>{link(unit.text.slug as FullSlug, who ? `${who}'s notes` : 'latex notes')}</li>,
        )
      }
    }
    if (kind === 'pset-notes' || kind === 'psets') {
      const { pset, solution, other } = psetRefs(resources, n, solutionTerm(view.psets.get(n)))
      if (pset) sources.push(<li>{link(pset.slug, 'problem set pdf')}</li>)
      if (solution) sources.push(<li>{link(solution.slug, 'official solutions')}</li>)
      for (const older of other)
        sources.push(
          <li>
            {link(older.solution.slug, `official solutions to the ${older.term} set`)}
            {older.pset ? (
              <>
                {' · '}
                {link(older.pset.slug, `${older.term} set`)}
              </>
            ) : null}
          </li>,
        )
      if (kind === 'psets' && view.psetText.has(n))
        sources.push(<li>{link(psetTextSlug(view.dir, n), 'problem set text')}</li>)
    }
    if (latex) {
      const tex = texRef(resources, latex)
      const raw = `/${courseFileSlug(view.dir, latex.source)}`
      sources.push(
        <li>
          {tex ? link(tex.slug, 'tex') : <span>tex</span>}
          {' · '}
          <a href={raw} data-router-ignore>
            raw source
          </a>
        </li>,
      )
    }

    const neighbour = (unit: CourseUnitView | undefined, rel: 'prev' | 'next') => {
      if (!unit) return <span aria-hidden="true" />
      const target =
        kind === 'notes'
          ? (unit.text?.slug as FullSlug | undefined)
          : unit.note
            ? unit.slug
            : undefined
      const resource = unitPdf(unit.record)
      const href = target
        ? resolveRelative(from, target)
        : kind === 'notes'
          ? undefined
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

    const psetNeighbour = (m: number, rel: 'prev' | 'next') => {
      const target =
        kind === 'pset-notes'
          ? view.psetText.has(m)
            ? psetTextSlug(view.dir, m)
            : undefined
          : view.psets.get(m)
            ? unitSlug(view.dir, m, 'psets')
            : undefined
      return target ? (
        <a rel={rel} class="internal" href={resolveRelative(from, target)}>
          <span class="course-num">{pad(m)}</span>
          <span>problem set</span>
        </a>
      ) : undefined
    }
    // an empty pager is a bordered box with nothing in it
    const psetPrev = psetNeighbour(n - 1, 'prev')
    const psetNext = psetNeighbour(n + 1, 'next')

    return (
      <footer class={classNames(props.displayClass, 'course-unit-foot')}>
        {dueHere.length > 0 ? (
          <section aria-labelledby="course-due-h">
            <h2 id="course-due-h">due here</h2>
            <ul>{dueHere}</ul>
          </section>
        ) : null}
        {sources.length > 0 ? (
          <section aria-labelledby="course-sources-h">
            <h2 id="course-sources-h">sources</h2>
            <ul>{sources}</ul>
          </section>
        ) : null}
        {latex ? (
          <details
            class="course-latex"
            data-course-latex={`/${courseFileSlug(view.dir, latex.source)}`}
            data-course-latex-lines={`${latex.lines[0]}-${latex.lines[1]}`}
          >
            <summary>
              latex source · lines {latex.lines[0]}–{latex.lines[1]}
            </summary>
            <div class="course-latex-tools">
              <span class="course-count" role="status" aria-live="polite" />
              <button type="button" data-course-latex-copy>
                copy
              </button>
            </div>
            <pre class="course-latex-pre">
              <code class="language-latex" />
            </pre>
          </details>
        ) : null}
        {(kind === 'lectures' || kind === 'notes') && index >= 0 ? (
          <nav class="course-unit-pager" aria-label="units">
            {neighbour(view.units[index - 1], 'prev')}
            {neighbour(view.units[index + 1], 'next')}
          </nav>
        ) : null}
        {(kind === 'psets' || kind === 'pset-notes') && (psetPrev || psetNext) ? (
          <nav class="course-unit-pager" aria-label="problem sets">
            {psetPrev ?? <span aria-hidden="true" />}
            {psetNext ?? <span aria-hidden="true" />}
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
