import { QuartzPluginData } from '../../plugins/vfile'
import { ReloadableWeakMap } from '../module-caches'
import { FullSlug } from '../path'
import { universityIndexOf, universityPlaceOf } from '../university'
import { isUniversityPdfPage } from '../university-pdf'
import {
  courseDirOf,
  coursePageOf,
  latexOf,
  parseCourseRecord,
  parseResourceRecord,
  str,
  unitLabel,
  unitOf,
  unitPdf,
  type CourseResourceKind,
  type CourseResourceRecord,
} from './schema'
import { courseOf, type CourseUnitView, type CourseView } from './views'

export interface CourseResourceRef {
  dir: string
  id: string
  slug: FullSlug
  data: QuartzPluginData
  record: CourseResourceRecord
}

const resourceRe = /^courses\/([^/]+)\/resources\/([^/]+)\/index$/

export const PDF_RESOURCE_KINDS = new Set<CourseResourceKind>([
  'notes',
  'notes-full',
  'chapter',
  'transcript',
  'pset',
  'pset-solution',
  'exam',
  'exam-solution',
  'reading',
  'other',
])

/**
 * Pages that draw their own course crumb (index bar, masthead, resource strip, unit bar, page head). The site
 * breadcrumbs stay off them: the crumb is the trail.
 */
export function hasCourseCrumb(data: QuartzPluginData): boolean {
  const slug = data.slug ?? ''
  if (isUniversityPdfPage(slug) || universityPlaceOf(slug) || universityIndexOf(slug)) return true
  if (!courseDirOf(slug)) return false
  return (
    parseCourseRecord(data.frontmatter) !== null ||
    parseResourceRecord(data.frontmatter)?.kind === 'notes-index' ||
    resourceOf(data) !== null ||
    unitOf(slug) !== null ||
    latexOf(data)?.pset !== undefined ||
    coursePageOf(slug) !== null ||
    data.frontmatter?.pageLayout === 'courses' ||
    data.frontmatter?.pageLayout === 'study-log' ||
    data.frontmatter?.pageLayout === 'statements'
  )
}

/** Problem set number from a resource id: `mit18_156_s25_pset2_pdf` → 2. */
export const psetNumber = (id: string): number | undefined => {
  const match = /(?:pset|ps|problem[-_ ]?set)[-_ ]?0*(\d+)/i.exec(id)
  return match ? Number(match[1]) : undefined
}

export interface PsetRefs {
  /** The term's own set: the shortest id (`pset1`, not the older-term `pset1-1`). */
  pset?: CourseResourceRef
  /** The official solution of that set. */
  solution?: CourseResourceRef
  /** Solutions OCW ships for another term's version of the set, with that set when vendored. */
  other: { term: string; pset?: CourseResourceRef; solution: CourseResourceRef }[]
}

const termOf = (ref: CourseResourceRef) => /\((\d{4})\)/.exec(resourceTitle(ref))?.[1]

/**
 * Problem set `n` and its solutions. A solution pairs with the set of the same term: 18.100B ships
 * the 2010 sets but only the 2006 solutions, and those must not open as answers to the 2010 set.
 * OCW titles most older-term files `(2006)`; `untagged` names the term of a solution whose title
 * does not (`solutionTerm:` in `psets/<nn>.md`).
 */
export function psetRefs(
  resources: Map<string, CourseResourceRef>,
  n: number,
  untagged?: string,
): PsetRefs {
  const numbered = (kind: CourseResourceKind) =>
    Array.from(resources.values())
      .filter(ref => ref.record.kind === kind && (ref.record.n ?? psetNumber(ref.id)) === n)
      .sort((a, b) => a.id.length - b.id.length || a.id.localeCompare(b.id))
  const sets = numbered('pset')
  const pset = sets[0]
  const term = pset ? termOf(pset) : undefined
  let solution: CourseResourceRef | undefined
  const other: PsetRefs['other'] = []
  for (const ref of numbered('pset-solution')) {
    const at = termOf(ref) ?? untagged
    if (at === term) solution ??= ref
    else if (at) other.push({ term: at, pset: sets.find(set => termOf(set) === at), solution: ref })
  }
  return { ...(pset ? { pset } : {}), ...(solution ? { solution } : {}), other }
}

/** A vendored resource page: `courses/<dir>/resources/<id>/index` with `layout: resource`. */
const resourceIndexRe = /^courses\/([^/]+)\/resources\/index$/

export function resourceOf(data: QuartzPluginData | undefined): CourseResourceRef | null {
  if (!data) return null
  const slug = data.slug ?? ''
  // `resources/index` is the course's own index page, id `index`
  const match = resourceRe.exec(slug) ?? resourceIndexRe.exec(slug)
  if (!match) return null
  const record = parseResourceRecord(data.frontmatter)
  if (!record) return null
  const id = match[2] ?? 'index'
  if (id === 'index' && record.kind !== 'index') return null
  return { dir: match[1], id, slug: data.slug as FullSlug, data, record }
}

/** The one predicate for resource pages the PDF reader emitter renders. */
export function pdfResourceOf(data: QuartzPluginData | undefined): CourseResourceRef | null {
  const ref = resourceOf(data)
  return ref && ref.record.file && /\.pdf$/i.test(ref.record.file) ? ref : null
}

const resourceCache = new ReloadableWeakMap<
  QuartzPluginData[],
  Map<string, Map<string, CourseResourceRef>>
>()

/** Every resource page of a course by id, memoised on the `allFiles` list identity. */
export function courseResources(
  dir: string,
  allFiles: QuartzPluginData[],
): Map<string, CourseResourceRef> {
  let byDir = resourceCache.get(allFiles)
  if (!byDir) {
    byDir = new Map()
    for (const file of allFiles) {
      const ref = resourceOf(file)
      if (!ref) continue
      const list = byDir.get(ref.dir) ?? new Map<string, CourseResourceRef>()
      list.set(ref.id, ref)
      byDir.set(ref.dir, list)
    }
    resourceCache.set(allFiles, byDir)
  }
  return byDir.get(dir) ?? new Map()
}

const numericCompare = (a: string, b: string) =>
  a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' })

/** Unit n, then pset n, then title: the collection and sibling order. */
export function compareResources(a: CourseResourceRef, b: CourseResourceRef): number {
  const an = a.record.n ?? Number.POSITIVE_INFINITY
  const bn = b.record.n ?? Number.POSITIVE_INFINITY
  if (an !== bn) return an - bn
  return numericCompare(resourceTitle(a), resourceTitle(b)) || numericCompare(a.id, b.id)
}

export function resourceTitle(ref: CourseResourceRef): string {
  return str(ref.data.frontmatter?.title) || ref.id
}

/** Exam solutions pair by id: `final_prac` with `final_prac_sol`, `midterm` with `midterm_sol`. */
export function examSolutionOf(
  ref: CourseResourceRef,
  resources: Map<string, CourseResourceRef>,
): CourseResourceRef | undefined {
  for (const candidate of resources.values()) {
    if (candidate.record.kind !== 'exam-solution' || !candidate.id.startsWith(ref.id)) continue
    const rest = candidate.id.slice(ref.id.length)
    if (/^_?sol(?:utions?|s)?$/i.test(rest)) return candidate
  }
  return undefined
}

/** The unit that lists a resource id, or carries the same number for a notes-like kind. */
export function unitForResource(
  view: CourseView | null,
  ref: CourseResourceRef,
): CourseUnitView | undefined {
  if (!view) return undefined
  const listed = view.units.find(unit => unit.record.resources.some(r => r.id === ref.id))
  if (listed) return listed
  if (ref.record.kind === 'captions' || ref.record.kind === 'transcript') {
    const of = ref.record.of
    return of ? view.units.find(unit => unit.record.resources.some(r => r.id === of)) : undefined
  }
  if (ref.record.kind === 'pset' || ref.record.kind === 'pset-solution') {
    return view.units.find(unit => unit.record.due.includes(ref.id))
  }
  if (ref.record.kind === 'notes' || ref.record.kind === 'chapter' || ref.record.kind === 'video') {
    return ref.record.n === undefined
      ? undefined
      : view.units.find(unit => unit.record.n === ref.record.n)
  }
  return undefined
}

export interface ResourceView {
  ref: CourseResourceRef
  home?: QuartzPluginData
  view: CourseView | null
  resources: Map<string, CourseResourceRef>
  unit?: CourseUnitView
  /** Pages the strip walks with prev/next; empty for kinds without a pager. */
  siblings: CourseResourceRef[]
  index: number
  /** `lecture 07 of 24`, `problem set 1`, or the kind label. */
  label: string
  /** `18.156`, or the home title when the home has no course block. */
  crumb: string
}

const kindLabels: Record<CourseResourceKind, string> = {
  video: 'lecture video',
  captions: 'captions',
  transcript: 'transcript',
  exam: 'exam',
  'exam-solution': 'exam solutions',
  pset: 'problem set',
  'pset-solution': 'problem set solutions',
  tex: 'latex source',
  notes: 'lecture notes',
  'notes-full': 'full lecture notes',
  chapter: 'chapter',
  reading: 'reading',
  archive: 'archive',
  image: 'image',
  other: 'file',
  collection: 'collection',
  index: 'resources',
  'notes-index': 'notes',
}

export function kindLabel(kind: CourseResourceKind): string {
  return kindLabels[kind]
}

/** Pages that are not unit notes: pager across the same kind, ordered by n then title. */
function kindSiblings(
  ref: CourseResourceRef,
  resources: Map<string, CourseResourceRef>,
): CourseResourceRef[] {
  return Array.from(resources.values())
    .filter(candidate => candidate.record.kind === ref.record.kind)
    .sort(compareResources)
}

/** Joins a resource page with its course, unit and pager siblings for the strip and views. */
export function resourceView(
  data: QuartzPluginData,
  allFiles: QuartzPluginData[],
  deckFiles: QuartzPluginData[] = [],
): ResourceView | null {
  const ref = resourceOf(data)
  if (!ref) return null
  const { home, view } = courseOf(ref.dir, allFiles, deckFiles)
  const resources = courseResources(ref.dir, allFiles)
  const unit = unitForResource(view, ref)
  const kind = ref.record.kind
  let siblings: CourseResourceRef[] = []
  if (view && (kind === 'notes' || kind === 'chapter' || kind === 'video')) {
    // calendar order: each unit's own resource of this kind
    siblings = view.units.flatMap(unit => {
      const own = unit.record.resources
        .map(r => resources.get(r.id))
        .filter((r): r is CourseResourceRef => r !== undefined && r.record.kind === kind)
      return own.slice(0, 1)
    })
    if (!siblings.some(s => s.id === ref.id)) siblings = kindSiblings(ref, resources)
  } else if (kind !== 'notes-full' && kind !== 'collection' && kind !== 'index') {
    siblings = kindSiblings(ref, resources)
  }
  const index = siblings.findIndex(s => s.id === ref.id)
  const n = ref.record.n
  const pad = (value: number) => String(value).padStart(2, '0')
  let label = kindLabel(kind)
  if (view && unit && (kind === 'notes' || kind === 'video' || kind === 'chapter')) {
    label = unitLabel(view.record.unitKind, unit.record.n, view.units.length)
    if (kind === 'video') label = `${label} · video`
    if (kind === 'chapter') label = `chapter ${n ?? '?'} · ${label}`
  } else if (kind === 'transcript' && unit && view) {
    label = `${unitLabel(view.record.unitKind, unit.record.n, view.units.length)} · transcript`
  } else if (kind === 'captions' && unit && view) {
    label = `${unitLabel(view.record.unitKind, unit.record.n, view.units.length)} · captions`
  } else if ((kind === 'pset' || kind === 'pset-solution' || kind === 'tex') && n !== undefined) {
    label = `problem set ${pad(n)}${kind === 'pset-solution' ? ' · solutions' : kind === 'tex' ? ' · latex' : ''}`
  } else if (kind === 'chapter' && n !== undefined) {
    label = `chapter ${n}`
  }
  const crumb = view?.record.number || str(home?.frontmatter?.title) || ref.dir
  return { ref, home, view, resources, ...(unit ? { unit } : {}), siblings, index, label, crumb }
}

export interface UnitSources {
  pdf?: CourseResourceRef
  /** Generated LaTeX note for the unit. */
  tex?: QuartzPluginData
  video?: CourseResourceRef
  videos: CourseResourceRef[]
  transcript?: CourseResourceRef
}

/** The home `src` slots for a unit: pdf, tex, video, transcript. */
export function unitSources(
  unit: CourseUnitView,
  resources: Map<string, CourseResourceRef>,
): UnitSources {
  const refs = unit.record.resources
    .map(r => resources.get(r.id))
    .filter((r): r is CourseResourceRef => r !== undefined)
  const pdfUnit = unitPdf(unit.record)
  const pdf = pdfUnit
    ? resources.get(pdfUnit.id)
    : refs.find(r => PDF_RESOURCE_KINDS.has(r.record.kind) && r.record.kind !== 'transcript')
  const videos = unit.record.resources
    .filter(r => r.youtube)
    .map(r => resources.get(r.id))
    .filter((r): r is CourseResourceRef => r !== undefined)
  const video = videos[0]
  const transcriptId = video?.record.transcriptId
  const transcript = transcriptId ? resources.get(transcriptId) : undefined
  return {
    ...(pdf ? { pdf } : {}),
    ...(unit.text ? { tex: unit.text } : {}),
    ...(video ? { video } : {}),
    videos,
    ...(transcript ? { transcript } : {}),
  }
}
