import type { Element, Root } from 'hast'
import { JSX } from 'preact'
import { visit } from 'unist-util-visit'
import { QuartzPluginData } from '../plugins/vfile'
import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../types/component'
import {
  CourseResourceRef,
  ResourceView,
  compareResources,
  courseDirOf,
  courseFileSlug,
  courseOf,
  coursePageOf,
  examSolutionOf,
  formatBytes,
  formatDuration,
  latexOf,
  lecturerName,
  parseResourceRecord,
  psetTextSlug,
  resourceTitle,
  resourceView,
  unitDeckSlug,
  unitSources,
} from '../util/course'
import { courseStudyEndpoint } from '../util/course-study'
import { htmlToJsx } from '../util/jsx'
import { classNames } from '../util/lang'
import { FullSlug, resolveRelative } from '../util/path'
import { concatenateResources } from '../util/resources'
import { StudyState } from './CourseStudy'
import { CourseWatch } from './CourseWatch'

export { coursePageOf } from '../util/course'
// @ts-ignore
import watchScript from './scripts/course-watch.inline'
// @ts-ignore
import script from './scripts/course.inline'
import resourceStyle from './styles/course-resource.scss'
import studyStyle from './styles/course-study.scss'
import style from './styles/course.scss'

const pad = (n: number) => String(n).padStart(2, '0')
const str = (value: unknown): string => (typeof value === 'string' ? value : '')

/** The vendored body's `mit opencourseware` link, the one external link every resource carries. */
export function ocwLink(tree: Root): string | undefined {
  let href: string | undefined
  visit(tree, 'element', (node: Element) => {
    if (href || node.tagName !== 'a') return
    const value = node.properties.href
    if (typeof value === 'string' && value.startsWith('https://ocw.mit.edu/')) href = value
  })
  return href
}

/** The lowercased short form of a CC license name: `cc by-nc-sa`. */
export function shortLicense(name: string): string {
  const match = /Creative Commons\s+(.+?)\s+\d/i.exec(name)
  if (!match) return name.toLowerCase()
  return `cc ${match[1]
    .replace(/Attribution/i, 'by')
    .replace(/NonCommercial/i, 'nc')
    .replace(/ShareAlike/i, 'sa')
    .replace(/NoDerivatives/i, 'nd')
    .toLowerCase()}`
}

interface StripLink {
  key: string
  node: JSX.Element
  /** Owner-only cells live on this item: `course.inline.ts` fills `[data-due]`. */
  cards?: string[]
  unit?: number
}

/** The strip's link list for one resource, by kind (docs/plans/course-resources.md §2). */
function stripLinks(
  rv: ResourceView,
  from: FullSlug,
  ocw: string | undefined,
  endpoint: string | undefined,
): StripLink[] {
  const { ref, view, unit, resources } = rv
  const kind = ref.record.kind
  const links: StripLink[] = []
  const internal = (slug: FullSlug, text: string, extra: Record<string, string> = {}) => (
    <a class="internal" href={resolveRelative(from, slug)} data-slug={slug} {...extra}>
      {text}
    </a>
  )
  const resourceLink = (target: CourseResourceRef, text: string) =>
    internal(target.slug, text, { 'data-no-popover': '' })
  const push = (key: string, node: JSX.Element, extra: Partial<StripLink> = {}) =>
    links.push({ key, node, ...extra })

  // the OCW link: the body's, else the course page on OCW
  const ocwHref = ocw ?? view?.record.source
  // problem sets and exams join a unit only through `due at`; the unit's own links stay off them
  const unitLinks = !['pset', 'pset-solution', 'tex', 'exam', 'exam-solution'].includes(kind)

  if (view && unit && unitLinks) {
    const sources = unitSources(unit, resources)
    const n = unit.record.n
    if (kind !== 'notes' && kind !== 'chapter' && sources.pdf && sources.pdf.id !== ref.id) {
      push('pdf', resourceLink(sources.pdf, 'lecture pdf'))
    }
    if (unit.note) push('notes', internal(unit.slug, 'notes'))
    else {
      push('notes', <code>lectures/{pad(n)}.md</code>)
      // the lecture pdf drafts the note from its marks (course.inline.ts builds the preview)
      const file = ref.record.file
      if (kind === 'notes' && file && /\.pdf$/i.test(file))
        push(
          'notes',
          <details
            class="course-promote"
            data-promote-dir={view.dir}
            data-promote-n={String(n)}
            data-promote-kind={view.record.unitKind}
            data-promote-title={unit.record.title}
            data-promote-resource={ref.slug.replace(/\/index$/, '')}
            data-promote-pdf={courseFileSlug(ref.dir, file)}
            data-promote-unit={unit.slug}
            data-promote-endpoint={endpoint}
          >
            <summary>start notes</summary>
            <div class="course-promote-body">
              <p class="course-promote-status" role="status" aria-live="polite">
                reading marks…
              </p>
            </div>
          </details>,
        )
    }
    if (unit.deck) {
      push(
        'cards',
        <span data-cards={unit.cardIds.join(' ')} data-unit={String(n)}>
          {internal(unitDeckSlug(view.dir, n), `cards ${unit.cardIds.length}`, {
            'data-no-popover': '',
          })}
          {' · due '}
          <span data-due="">–</span>
        </span>,
        { cards: unit.cardIds, unit: n },
      )
    }
    if (kind !== 'video' && kind !== 'transcript' && kind !== 'captions' && sources.video) {
      const length = formatDuration(sources.video.record.duration)
      push('video', resourceLink(sources.video, length ? `watch ${length}` : 'watch'))
    }
    if (kind !== 'transcript' && sources.transcript) {
      push('transcript', resourceLink(sources.transcript, 'transcript'))
    }
    if (unit.text) {
      const latex = latexOf(unit.text)
      push('latex', internal(unit.text.slug as FullSlug, `latex §${latex?.section ?? n}`))
    }
    // the unit's own reading surfaces (pdf, chapter, video) carry the toggles; others show dates
    const owns = kind === 'notes' || kind === 'chapter' || kind === 'video'
    const study = <StudyState view={view} unit={unit} endpoint={owns ? endpoint : undefined} />
    if (owns ? endpoint || unit.read || unit.problems : unit.read || unit.problems)
      push('read', study)
  }

  if (kind === 'notes-full') {
    const notesIndex = `courses/${ref.dir}/notes/index` as FullSlug
    if (view?.units.some(unit => unit.text)) push('latex', internal(notesIndex, 'latex notes'))
  }

  if (kind === 'pset' || kind === 'pset-solution' || kind === 'tex') {
    const n = ref.record.n
    const dueUnit = view?.units.find(u => u.record.due.includes(ref.id))
    if (dueUnit && view) {
      push(
        'due',
        <span>
          due at {view.record.unitKind} {pad(dueUnit.record.n)}
        </span>,
      )
    }
    if (n !== undefined && view) {
      const ours = view.psets.get(n)
      if (ours) push('solutions', internal(ours.slug as FullSlug, 'our solutions'))
      else if (kind === 'pset')
        push(
          'solutions',
          <span>
            solutions: <code>psets/{pad(n)}.md</code>
          </span>,
        )
      const text = view.psetText.get(n)
      if (text && kind !== 'tex') push('rendered', internal(psetTextSlug(ref.dir, n), 'rendered'))
      // the desk sits on the rendered set, else on psets/<nn>.md; dev creates that file
      if (kind === 'pset' && text)
        push('rendered', internal(psetTextSlug(ref.dir, n), 'desk', { 'data-no-popover': '' }))
      else if (kind === 'pset' && ours)
        push('solutions', internal(ours.slug as FullSlug, 'desk', { 'data-no-popover': '' }))
      else if (kind === 'pset' && endpoint)
        push(
          'solutions',
          <button
            type="button"
            class="course-desk-toggle"
            data-desk-open={endpoint}
            data-desk-dir={view.dir}
            data-desk-n={String(n)}
          >
            open desk
          </button>,
        )
    }
    for (const other of resources.values()) {
      if (other.id === ref.id || other.record.n !== n || n === undefined) continue
      if (kind !== 'pset-solution' && other.record.kind === 'pset-solution')
        push('official', resourceLink(other, 'official solutions'))
      if (kind !== 'pset' && other.record.kind === 'pset') {
        // two psets can share a number (18.100B keeps a 2006 set beside the 2010 one)
        const year = /\b((?:19|20)\d{2})\b/.exec(resourceTitle(other))?.[1]
        push(`pset-${other.id}`, resourceLink(other, year ? `problem set ${year}` : 'problem set'))
      }
      if (kind !== 'tex' && other.record.kind === 'tex')
        push('tex', resourceLink(other, 'latex template'))
    }
  }

  if (kind === 'exam') {
    const solution = examSolutionOf(ref, resources)
    if (solution) push('solution', resourceLink(solution, 'solutions'))
  }
  if (kind === 'exam-solution') {
    for (const other of resources.values()) {
      if (other.record.kind === 'exam' && examSolutionOf(other, resources)?.id === ref.id)
        push('exam', resourceLink(other, 'exam'))
    }
  }
  if (kind === 'transcript' || kind === 'captions') {
    const video = ref.record.of ? resources.get(ref.record.of) : undefined
    if (video) {
      const length = formatDuration(video.record.duration)
      push('video', resourceLink(video, length ? `watch ${length}` : 'watch'))
    }
  }

  // Every collection that lists this resource, as the `in` line.
  for (const other of resources.values()) {
    if (other.record.kind === 'collection' && other.record.items?.includes(ref.id))
      push(`in-${other.id}`, resourceLink(other, resourceTitle(other).toLowerCase()))
  }

  if (ocwHref) {
    push(
      'ocw',
      <a href={ocwHref} target="_blank" rel="noopener noreferrer">
        mit ocw
      </a>,
    )
  }
  const license = str(ref.data.frontmatter?.license)
  const licenseUrl = str(ref.data.frontmatter?.license_url)
  if (license) {
    push(
      'license',
      licenseUrl ? (
        <a href={licenseUrl} target="_blank" rel="noopener noreferrer">
          {shortLicense(license)}
        </a>
      ) : (
        <span>{shortLicense(license)}</span>
      ),
    )
  }
  return links
}

const COURSES_SLUG = 'courses/index' as FullSlug

/** Row of the links table a key belongs to; `unit` is named after the course's unit kind. */
function rowOf(key: string): string {
  if (key.startsWith('in-')) return 'in'
  if (key.startsWith('pset-')) return 'set'
  const rows: Record<string, string> = {
    pdf: 'unit',
    video: 'unit',
    transcript: 'unit',
    cards: 'unit',
    notes: 'notes',
    latex: 'notes',
    read: 'study',
    due: 'set',
    solutions: 'set',
    rendered: 'set',
    official: 'set',
    tex: 'set',
    solution: 'exam',
    exam: 'exam',
    ocw: 'source',
    license: 'source',
  }
  if (key === 'prev' || key === 'next') return key
  return rows[key] ?? 'links'
}

const ROW_ORDER = ['prev', 'next', 'unit', 'study', 'notes', 'set', 'exam', 'in', 'source', 'links']

/**
 * The links as a key/value table in the triathlon cheat-sheet grammar: a gray row label on the
 * left, the links on the right, rows in a fixed order (the unit's own files first).
 */
export function StripTable({
  links,
  unitKind,
}: {
  links: StripLink[]
  unitKind: string
}): JSX.Element | null {
  if (links.length === 0) return null
  const rows = new Map<string, StripLink[]>()
  for (const link of links) {
    const row = rowOf(link.key)
    rows.set(row, [...(rows.get(row) ?? []), link])
  }
  const ordered = Array.from(rows).sort(([a], [b]) => ROW_ORDER.indexOf(a) - ROW_ORDER.indexOf(b))
  return (
    <table class="course-strip-table">
      <tbody>
        {ordered.map(([row, items]) => (
          <tr key={row}>
            <th scope="row">{row === 'unit' ? unitKind : row}</th>
            <td>
              {items.map((link, i) => (
                <>
                  {i > 0 ? <span class="course-sep">·</span> : null}
                  {link.node}
                </>
              ))}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/** The `courses` link that opens every course crumb. */
export function CoursesCrumb({ from }: { from: FullSlug }): JSX.Element {
  return (
    <>
      <a
        class="internal"
        href={resolveRelative(from, COURSES_SLUG)}
        data-slug={COURSES_SLUG}
        data-no-popover
      >
        courses
      </a>
      <span class="course-unit-sep">/</span>
    </>
  )
}

/** The page title without the course prefix the vendor script adds (`18.156 (s25): lecture 07: …`). */
export function resourceHeading(rv: ResourceView): string {
  const title = resourceTitle(rv.ref)
  if (rv.unit && (rv.ref.record.kind === 'notes' || rv.ref.record.kind === 'video'))
    return rv.unit.record.title
  return title.replace(/^[^:]*\(\w\d{2}\):\s*/, '').replace(/^lecture \d+:\s*/i, '')
}

/** Meta line under the title: format, size, length. */
export function resourceMeta(ref: CourseResourceRef): string[] {
  const { record } = ref
  const format = record.file ? record.file.split('.').pop()?.toLowerCase() : undefined
  return [
    record.kind === 'video' ? 'video' : format,
    formatBytes(record.bytes),
    formatDuration(record.duration),
    record.cues !== undefined ? `${record.cues} cues` : undefined,
  ].filter((part): part is string => Boolean(part))
}

/**
 * Head shared by every resource view: crumb, title, meta line, prev/next, and the links list
 * (a bottom sheet below 50rem). `[data-course-card]` is the popover contract (§6).
 */
export function ResourceHead({
  rv,
  from,
  ocw,
  pager = true,
  endpoint,
}: {
  rv: ResourceView
  from: FullSlug
  ocw?: string
  pager?: boolean
  /** Dev study endpoint; the toggles render only when it is set. */
  endpoint?: string
}): JSX.Element {
  const { ref, view, unit, siblings, index, label, crumb } = rv
  const homeSlug = `courses/${ref.dir}/index` as FullSlug
  const links = stripLinks(rv, from, ocw, endpoint)
  const prev = pager && index > 0 ? siblings[index - 1] : undefined
  const next = pager && index >= 0 ? siblings[index + 1] : undefined
  const pagerLabel = (sibling: CourseResourceRef, rel: 'prev' | 'next') => {
    const n = sibling.record.n
    const text = n === undefined ? rel : pad(n)
    return rel === 'prev' ? `‹ ${text}` : `${text} ›`
  }
  const pagerLink = (sibling: CourseResourceRef, rel: 'prev' | 'next'): StripLink => ({
    key: rel,
    node: (
      <a
        rel={rel}
        class="internal"
        href={resolveRelative(from, sibling.slug)}
        data-slug={sibling.slug}
        data-course-key={rel}
        data-no-popover
        title={resourceTitle(sibling)}
      >
        {pagerLabel(sibling, rel)}
      </a>
    ),
  })
  // prev/next open the table (aarnphm: "the next of 02 should also be an entry on the table")
  const rows = [
    ...(prev ? [pagerLink(prev, 'prev')] : []),
    ...(next ? [pagerLink(next, 'next')] : []),
    ...links,
  ]
  const hasDeck = Boolean(unit?.deck)
  const meta = resourceMeta(ref)
  return (
    <div
      class="course-strip"
      data-course-card
      data-course-prefix={
        hasDeck && view ? `courses/${view.dir}/lectures/${pad(unit!.record.n)}/` : undefined
      }
    >
      <div class="course-strip-head">
        <p class="course-crumb">
          <CoursesCrumb from={from} />
          <a
            class="internal"
            href={resolveRelative(from, homeSlug)}
            data-slug={homeSlug}
            data-course-key="up"
          >
            {crumb}
          </a>
          <span class="course-unit-sep">/</span>
          <span class="course-num">{label}</span>
        </p>
        <h1 class="course-card-title">{resourceHeading(rv)}</h1>
        {meta.length > 0 ? <p class="course-count">{meta.join(' · ')}</p> : null}
      </div>
      {rows.length > 0 ? (
        <>
          <button
            type="button"
            class="course-strip-toggle"
            popovertarget="course-strip-links"
            aria-label="links for this resource"
          >
            links
          </button>
          <div class="course-strip-links" id="course-strip-links" popover="auto">
            <StripTable links={rows} unitKind={view?.record.unitKind ?? 'unit'} />
          </div>
        </>
      ) : null}
    </div>
  )
}

/** The row above the PDF reader on a course resource page (beforeBody of the reader emitter). */
export const CourseResourceStrip = (() => {
  const Strip: QuartzComponent = (props: QuartzComponentProps) => {
    const { ctx, fileData, allFiles, tree, displayClass } = props
    const rv = resourceView(fileData, allFiles, ctx.decks)
    if (!rv) return null
    const from = fileData.slug!
    return (
      <nav class={classNames(displayClass, 'course-resource-strip')} aria-label="course resource">
        <ResourceHead
          rv={rv}
          from={from}
          ocw={ocwLink(tree as Root)}
          pager={rv.ref.record.kind !== 'notes-full'}
          endpoint={courseStudyEndpoint(ctx.argv)}
        />
      </nav>
    )
  }
  Strip.displayName = 'CourseResourceStrip'
  Strip.css = concatenateResources(style, studyStyle)
  Strip.afterDOMLoaded = script
  return Strip
}) satisfies QuartzComponentConstructor

// ---------------------------------------------------------------------------------------------
// Non-PDF resource views, dispatched by `resource.kind` from FolderContent's `Layout.resource`.

function collectionMembers(rv: ResourceView): CourseResourceRef[] {
  const items = rv.ref.record.items ?? []
  return items
    .map(id => rv.resources.get(id))
    .filter((ref): ref is CourseResourceRef => ref !== undefined)
}

function formatOf(ref: CourseResourceRef): string {
  if (ref.record.kind === 'video') return 'video'
  return ref.record.file?.split('.').pop()?.toLowerCase() ?? '–'
}

/** One table whose columns depend on the members (§3). */
function ResourceTable({
  rows,
  from,
  rv,
}: {
  rows: CourseResourceRef[]
  from: FullSlug
  rv: ResourceView
}): JSX.Element {
  const anyN = rows.some(row => row.record.n !== undefined)
  const anyVideo = rows.some(row => row.record.kind === 'video')
  const anyFile = rows.some(row => row.record.file)
  // a pset's due unit: the record's own `due`, else the unit that lists it in `due:`
  const dueOf = (row: CourseResourceRef): number | undefined =>
    row.record.due ?? rv.view?.units.find(u => u.record.due.includes(row.id))?.record.n
  const anyDue = rows.some(row => row.record.kind === 'pset' && dueOf(row) !== undefined)
  const unitKind = rv.view?.record.unitKind ?? 'lecture'
  return (
    <table class="course-resource-table">
      <thead>
        <tr>
          {anyN ? (
            <th scope="col" class="course-c-n">
              #
            </th>
          ) : null}
          <th scope="col">title</th>
          {anyVideo ? (
            <>
              <th scope="col" class="course-c-length">
                length
              </th>
              <th scope="col" class="course-c-transcript">
                transcript
              </th>
            </>
          ) : null}
          {anyFile ? (
            <>
              <th scope="col" class="course-c-format">
                format
              </th>
              <th scope="col" class="course-c-size">
                size
              </th>
            </>
          ) : null}
          {anyDue ? (
            <th scope="col" class="course-c-due">
              due
            </th>
          ) : null}
        </tr>
      </thead>
      <tbody>
        {rows.map(row => {
          const transcript = row.record.transcriptId
            ? rv.resources.get(row.record.transcriptId)
            : undefined
          return (
            <tr>
              {anyN ? (
                <td class="course-num course-c-n">
                  {row.record.n === undefined ? '–' : pad(row.record.n)}
                </td>
              ) : null}
              <td>
                <a class="internal" href={resolveRelative(from, row.slug)} data-slug={row.slug}>
                  {resourceTitle(row).toLowerCase()}
                </a>
              </td>
              {anyVideo ? (
                <>
                  <td class="course-num course-c-length">
                    {formatDuration(row.record.duration) ?? '–'}
                  </td>
                  <td class="course-c-transcript">
                    {row.record.kind === 'video' && row.record.cues ? (
                      <a class="internal" href={`${resolveRelative(from, row.slug)}#transcript`}>
                        text
                      </a>
                    ) : null}
                    {transcript ? (
                      <>
                        {row.record.cues ? ' · ' : ''}
                        <a
                          class="internal"
                          href={resolveRelative(from, transcript.slug)}
                          data-slug={transcript.slug}
                          data-no-popover
                        >
                          pdf
                        </a>
                      </>
                    ) : null}
                    {!transcript && !(row.record.kind === 'video' && row.record.cues) ? '–' : null}
                  </td>
                </>
              ) : null}
              {anyFile ? (
                <>
                  <td class="course-c-format">{formatOf(row)}</td>
                  <td class="course-num course-c-size">{formatBytes(row.record.bytes) ?? '–'}</td>
                </>
              ) : null}
              {anyDue ? (
                <td class="course-c-due">
                  {dueOf(row) !== undefined ? `${unitKind} ${pad(dueOf(row)!)}` : '–'}
                </td>
              ) : null}
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

function collectionSummary(rows: CourseResourceRef[]): string {
  const videos = rows.filter(row => row.record.kind === 'video')
  if (videos.length > 0 && videos.length === rows.length) {
    const seconds = videos.reduce((sum, row) => sum + (row.record.duration ?? 0), 0)
    const transcripts = videos.filter(row => row.record.cues).length
    return [
      `${videos.length} ${videos.length === 1 ? 'video' : 'videos'}`,
      seconds > 0 ? `${Math.round(seconds / 3600)} h` : undefined,
      transcripts > 0
        ? `${transcripts} ${transcripts === 1 ? 'transcript' : 'transcripts'}`
        : undefined,
    ]
      .filter(Boolean)
      .join(' · ')
  }
  const formats = Array.from(new Set(rows.map(formatOf))).sort()
  return `${rows.length} ${rows.length === 1 ? 'file' : 'files'} · ${formats.join(' ')}`
}

function CollectionView({ rv, props }: { rv: ResourceView; props: QuartzComponentProps }) {
  const from = props.fileData.slug!
  const rows = collectionMembers(rv)
  const description = str(props.fileData.frontmatter?.description)
  const links = (rv.ref.record.links ?? [])
    .map(slug => props.allFiles.find(file => file.slug === slug.replace(/\/$/, '/index')))
    .filter((file): file is QuartzPluginData => file !== undefined)
  return (
    <>
      <ResourceHead rv={rv} from={from} ocw={ocwLink(props.tree as Root)} pager={false} />
      <div class="course-resource-body" data-course-rows>
        <div class="course-resource-heading">
          <h2>{resourceTitle(rv.ref).toLowerCase()}</h2>
          <span class="course-count">{collectionSummary(rows)}</span>
        </div>
        {description ? <p>{description}</p> : null}
        {links.length > 0 ? (
          <ul class="course-resource-links">
            {links.map(file => (
              <li>
                <a
                  class="internal"
                  href={resolveRelative(from, file.slug as FullSlug)}
                  data-slug={file.slug}
                >
                  {str(file.frontmatter?.title).toLowerCase() || file.slug}
                </a>
              </li>
            ))}
          </ul>
        ) : null}
        {rows.length > 0 ? <ResourceTable rows={rows} from={from} rv={rv} /> : null}
      </div>
    </>
  )
}

function IndexView({ rv, props }: { rv: ResourceView; props: QuartzComponentProps }) {
  const from = props.fileData.slug!
  const collections = Array.from(rv.resources.values())
    .filter(ref => ref.record.kind === 'collection')
    .sort((a, b) => resourceTitle(a).localeCompare(resourceTitle(b)))
  const inCollection = new Set(collections.flatMap(ref => ref.record.items ?? []))
  const others = Array.from(rv.resources.values())
    .filter(
      ref =>
        !inCollection.has(ref.id) &&
        !['collection', 'index', 'captions', 'transcript', 'notes-index'].includes(ref.record.kind),
    )
    .sort(compareResources)
  return (
    <>
      <ResourceHead rv={rv} from={from} ocw={ocwLink(props.tree as Root)} pager={false} />
      <div class="course-resource-body" data-course-rows>
        <ul class="course-resource-rows">
          {collections.map(ref => {
            const rows = collectionMembers({ ...rv, ref })
            return (
              <li>
                <a class="internal" href={resolveRelative(from, ref.slug)} data-slug={ref.slug}>
                  <span class="course-row-title">{resourceTitle(ref).toLowerCase()}</span>
                  <span class="course-count">{rows.length}</span>
                  <span class="course-count">
                    {Array.from(new Set(rows.map(formatOf)))
                      .sort()
                      .join(' ')}
                  </span>
                </a>
              </li>
            )
          })}
        </ul>
        {others.length > 0 ? (
          <>
            <div class="course-resource-heading">
              <h2>other files</h2>
              <span class="course-count">{others.length}</span>
            </div>
            <ResourceTable rows={others} from={from} rv={rv} />
          </>
        ) : null}
      </div>
    </>
  )
}

function FileView({ rv, props }: { rv: ResourceView; props: QuartzComponentProps }) {
  const { fileData, tree } = props
  const from = fileData.slug!
  const { ref, view, unit } = rv
  const ocw = ocwLink(tree as Root)
  const description = str(fileData.frontmatter?.description)
  const file = ref.record.file
  const href = file ? `/${courseFileSlug(ref.dir, file)}` : undefined
  const collections = Array.from(rv.resources.values()).filter(
    other => other.record.kind === 'collection' && other.record.items?.includes(ref.id),
  )
  const content = htmlToJsx(fileData.filePath!, tree)
  return (
    <>
      <ResourceHead rv={rv} from={from} ocw={ocw} />
      <div class="course-resource-body">
        {ref.record.kind === 'image' && href ? (
          <img class="course-file-image" src={href} alt={resourceTitle(ref)} />
        ) : null}
        {description && description !== resourceTitle(ref) ? <p>{description}</p> : null}
        <dl class="course-file-meta">
          <dt>format</dt>
          <dd>{formatOf(ref)}</dd>
          <dt>size</dt>
          <dd>{formatBytes(ref.record.bytes) ?? '–'}</dd>
          {collections.length > 0 ? (
            <>
              <dt>collections</dt>
              <dd>
                {collections.map((other, i) => (
                  <>
                    {i > 0 ? ', ' : ''}
                    <a
                      class="internal"
                      href={resolveRelative(from, other.slug)}
                      data-slug={other.slug}
                    >
                      {resourceTitle(other).toLowerCase()}
                    </a>
                  </>
                ))}
              </dd>
            </>
          ) : null}
          {unit && view ? (
            <>
              <dt>unit</dt>
              <dd>
                {view.record.unitKind} {pad(unit.record.n)}: {unit.record.title}
              </dd>
            </>
          ) : null}
          {ref.record.kind === 'tex' && ref.record.notes ? (
            <>
              <dt>rendered</dt>
              <dd>
                <a
                  class="internal"
                  href={resolveRelative(
                    from,
                    ref.record.notes.replace(/\/$/, '/index') as FullSlug,
                  )}
                >
                  {ref.record.full ? 'lecture notes, one page per lecture' : 'problem set notes'}
                </a>
              </dd>
            </>
          ) : null}
          <dt>links</dt>
          <dd>
            {href && ref.record.kind !== 'archive' ? (
              <>
                <a href={href} download data-router-ignore>
                  download
                </a>
                {' · '}
              </>
            ) : null}
            {ocw ? (
              <a href={ocw} target="_blank" rel="noopener noreferrer">
                mit ocw
              </a>
            ) : null}
          </dd>
        </dl>
        {ref.record.kind === 'archive' ? content : null}
      </div>
    </>
  )
}

function CaptionsView({ rv, props }: { rv: ResourceView; props: QuartzComponentProps }) {
  const from = props.fileData.slug!
  const { ref } = rv
  const video = ref.record.of ? rv.resources.get(ref.record.of) : undefined
  const unit = rv.unit
  const label = unit && rv.view ? `${rv.view.record.unitKind} ${pad(unit.record.n)}` : 'the lecture'
  return (
    <>
      <ResourceHead rv={rv} from={from} ocw={ocwLink(props.tree as Root)} pager={false} />
      <div class="course-resource-body">
        <p>
          captions of{' '}
          {video ? (
            <a
              class="internal"
              href={`${resolveRelative(from, video.slug)}#transcript`}
              data-slug={video.slug}
            >
              {label}
            </a>
          ) : (
            label
          )}
          {' · '}
          <span class="course-count">
            {formatOf(ref)}
            {formatBytes(ref.record.bytes) ? ` · ${formatBytes(ref.record.bytes)}` : ''} · mit ocw
          </span>
        </p>
      </div>
    </>
  )
}

/** `notes/index`: the generated LaTeX notes table under a course head. */
function NotesIndexView(props: QuartzComponentProps) {
  const { ctx, fileData, allFiles, tree } = props
  const from = fileData.slug!
  const dir = courseDirOf(from)
  const { view, home } = dir ? courseOf(dir, allFiles, ctx.decks) : { view: null, home: undefined }
  const homeSlug = `courses/${dir}/index` as FullSlug
  const crumb = view?.record.number || str(home?.frontmatter?.title) || dir
  const lecturer = lecturerName(view?.record)
  const count = view?.units.filter(unit => unit.text).length ?? 0
  const content = htmlToJsx(fileData.filePath!, tree)
  return (
    <>
      <div class="course-strip" data-course-card>
        <div class="course-strip-head">
          <p class="course-crumb">
            <CoursesCrumb from={from} />
            <a
              class="internal"
              href={resolveRelative(from, homeSlug)}
              data-slug={homeSlug}
              data-course-key="up"
            >
              {crumb}
            </a>
            <span class="course-unit-sep">/</span>
            <span class="course-num">notes</span>
          </p>
          <h1 class="course-card-title">{str(fileData.frontmatter?.title)}</h1>
          <p class="course-count">
            {[
              lecturer ? `${lecturer}'s notes` : undefined,
              `${count} lectures`,
              'from the ocw latex source',
            ]
              .filter(Boolean)
              .join(' · ')}
          </p>
        </div>
      </div>
      <div class="course-resource-body" data-course-rows>
        {content}
      </div>
    </>
  )
}

/**
 * Head of a vendored `pages/<name>` page (syllabus, calendar, final project): the crumb and the
 * description, so the page carries the `[data-course-card]` popover contract (§6). The page's
 * own `h1` is the title.
 */
export function CoursePageHead(props: QuartzComponentProps): JSX.Element | null {
  const { fileData, allFiles, ctx } = props
  const page = coursePageOf(fileData.slug)
  if (!page) return null
  const from = fileData.slug!
  const { home, view } = courseOf(page.dir, allFiles, ctx.decks)
  const homeSlug = `courses/${page.dir}/index` as FullSlug
  const crumb = view?.record.number || str(home?.frontmatter?.title) || page.dir
  const description = str(fileData.frontmatter?.description)
  const license = str(fileData.frontmatter?.license)
  const licenseUrl = str(fileData.frontmatter?.license_url)
  const ocw = ocwLink(props.tree as Root)
  const pageLinks: StripLink[] = []
  if (ocw)
    pageLinks.push({
      key: 'ocw',
      node: (
        <a href={ocw} target="_blank" rel="noopener noreferrer">
          mit ocw
        </a>
      ),
    })
  if (license)
    pageLinks.push({
      key: 'license',
      node: licenseUrl ? (
        <a href={licenseUrl} target="_blank" rel="noopener noreferrer">
          {shortLicense(license)}
        </a>
      ) : (
        <span>{shortLicense(license)}</span>
      ),
    })
  return (
    <div class="course-strip course-page-head" data-course-card>
      <div class="course-strip-head">
        <p class="course-crumb">
          <CoursesCrumb from={from} />
          <a class="internal" href={resolveRelative(from, homeSlug)} data-slug={homeSlug}>
            {crumb}
          </a>
          <span class="course-unit-sep">/</span>
          <span class="course-num">pages</span>
          <span class="course-unit-sep">/</span>
          <span class="course-num">{page.name.replace(/-/g, ' ')}</span>
        </p>
        {description ? <p class="course-count">{description}</p> : null}
      </div>
      <div class="course-strip-links course-page-links">
        <StripTable links={pageLinks} unitKind="unit" />
      </div>
    </div>
  )
}

export default (() => {
  const CourseResource: QuartzComponent = (props: QuartzComponentProps) => {
    const { ctx, fileData, allFiles, displayClass } = props
    const record = parseResourceRecord(fileData.frontmatter)
    if (!record) return null
    if (record.kind === 'notes-index') {
      return (
        <div
          class={classNames(displayClass, 'course-resource', 'popover-hint')}
          data-resource-kind="notes-index"
        >
          <NotesIndexView {...props} />
        </div>
      )
    }
    const rv = resourceView(fileData, allFiles, ctx.decks)
    if (!rv) return null
    let body: JSX.Element
    switch (record.kind) {
      case 'video':
        body = <CourseWatch rv={rv} props={props} />
        break
      case 'collection':
        body = <CollectionView rv={rv} props={props} />
        break
      case 'index':
        body = <IndexView rv={rv} props={props} />
        break
      case 'captions':
        body = <CaptionsView rv={rv} props={props} />
        break
      default:
        body = <FileView rv={rv} props={props} />
    }
    return (
      <div
        class={classNames(displayClass, 'course-resource', 'popover-hint')}
        data-resource-kind={record.kind}
      >
        {body}
      </div>
    )
  }
  CourseResource.displayName = 'CourseResource'
  CourseResource.css = concatenateResources(style, resourceStyle, studyStyle)
  CourseResource.afterDOMLoaded = concatenateResources(script, watchScript)
  return CourseResource
}) satisfies QuartzComponentConstructor
