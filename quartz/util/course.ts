import { QuartzPluginData } from '../plugins/vfile'
import { FullSlug, joinSegments, slugifyFilePath, FilePath } from './path'
import { universityIndexOf, universityPlaceOf } from './university'
import { isUniversityPdfPage } from './university-pdf'

export type CourseUnitKind = 'lecture' | 'week' | 'session'

export interface CourseUnitResource {
  id: string
  file?: string
  youtube?: string
  /** Video length in seconds, from the vendored captions. */
  duration?: number
  cues?: number
}

export interface CourseUnitRecord {
  n: number
  title: string
  sessions?: string
  part?: number
  keyDates: string[]
  resources: CourseUnitResource[]
  due: string[]
}

export interface CourseRecord {
  number: string
  name: string
  term: string
  level?: string
  instructors: string[]
  source: string
  unitKind: CourseUnitKind
  parts: string[]
  trailing: string[]
  units: CourseUnitRecord[]
}

export type CourseUnitState = 'new' | 'notes' | 'deck'

/** A unit joined with the authored files that exist for it. */
export interface CourseUnitView {
  record: CourseUnitRecord
  slug: FullSlug
  note?: QuartzPluginData
  deck?: QuartzPluginData
  /** Generated `notes/<nn>` page converted from the OCW LaTeX source. */
  text?: QuartzPluginData
  state: CourseUnitState
  cardIds: string[]
  /** Dates from the study log. */
  read?: string
  problems?: string
}

export interface CourseView {
  dir: string
  slug: FullSlug
  home: QuartzPluginData
  record: CourseRecord
  units: CourseUnitView[]
  /** Authored `psets/<nn>.md` solutions, by problem set number. */
  psets: Map<number, QuartzPluginData>
  /** Generated `notes/pset-<nn>` pages, by problem set number. */
  psetText: Map<number, QuartzPluginData>
  /** The authored study log page, when the course has one. */
  log?: QuartzPluginData
}

export type CourseResourceKind =
  | 'video'
  | 'captions'
  | 'transcript'
  | 'exam'
  | 'exam-solution'
  | 'pset'
  | 'pset-solution'
  | 'tex'
  | 'notes'
  | 'notes-full'
  | 'chapter'
  | 'reading'
  | 'archive'
  | 'image'
  | 'other'
  | 'collection'
  | 'index'
  | 'notes-index'

/** The `resource:` block the vendor script writes (docs/plans/course-resources.md §1). */
export interface CourseResourceRecord {
  kind: CourseResourceKind
  file?: string
  bytes?: number
  n?: number
  /** Unit number a problem set is due at. */
  due?: number
  youtube?: string
  duration?: number
  cues?: number
  captions?: string
  captionsId?: string
  transcript?: string
  transcriptId?: string
  /** Video resource id a captions or transcript file belongs to. */
  of?: string
  full?: boolean
  /** Rendered notes slug for a `.tex` source. */
  notes?: string
  /** OCW collection label. */
  type?: string
  items?: string[]
  links?: string[]
}

export interface CourseResourceRef {
  dir: string
  id: string
  slug: FullSlug
  data: QuartzPluginData
  record: CourseResourceRecord
}

/** The `latex:` block on a generated note. */
export interface LatexRecord {
  source: string
  lines: [number, number]
  section?: number
  unit?: number
  pset?: number
  pdf: string
  pages?: [number, number]
  dateSource?: string
}

const unitRe = /^courses\/([^/]+)\/(lectures|psets|notes)\/(\d{2,3})$/
const coursePageRe = /^courses\/([^/]+)\/pages\/([^/]+)\/index$/
const resourceRe = /^courses\/([^/]+)\/resources\/([^/]+)\/index$/
const resourceKinds = new Set<string>([
  'video',
  'captions',
  'transcript',
  'exam',
  'exam-solution',
  'pset',
  'pset-solution',
  'tex',
  'notes',
  'notes-full',
  'chapter',
  'reading',
  'archive',
  'image',
  'other',
  'collection',
  'index',
  'notes-index',
])

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

function str(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function strList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((v): v is string => typeof v === 'string') : []
}

function num(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function parseUnit(value: unknown): CourseUnitRecord | null {
  if (!isRecord(value)) return null
  const n = num(value.n)
  if (n === undefined) return null
  const resources = Array.isArray(value.resources)
    ? value.resources.filter(isRecord).flatMap(entry => {
        const id = str(entry.id)
        if (!id) return []
        const file = str(entry.file)
        const youtube = str(entry.youtube)
        const duration = num(entry.duration)
        const cues = num(entry.cues)
        return [
          {
            id,
            ...(file ? { file } : {}),
            ...(youtube ? { youtube } : {}),
            ...(duration !== undefined ? { duration } : {}),
            ...(cues !== undefined ? { cues } : {}),
          },
        ]
      })
    : []
  return {
    n,
    title: str(value.title),
    ...(str(value.sessions) ? { sessions: str(value.sessions) } : {}),
    ...(num(value.part) ? { part: num(value.part) } : {}),
    keyDates: strList(value.keyDates),
    resources,
    due: strList(value.due),
  }
}

/** The vendored `course:` block on a course home, or null when the page has none. */
export function parseCourseRecord(frontmatter: unknown): CourseRecord | null {
  if (!isRecord(frontmatter)) return null
  const course = frontmatter.course
  if (!isRecord(course)) return null
  const units = Array.isArray(course.units)
    ? course.units.map(parseUnit).filter((unit): unit is CourseUnitRecord => unit !== null)
    : []
  if (units.length === 0) return null
  const kind = str(course.unitKind)
  return {
    number: str(course.number),
    name: str(course.name),
    term: str(course.term),
    ...(str(course.level) ? { level: str(course.level) } : {}),
    instructors: strList(course.instructors),
    source: str(course.source),
    unitKind: kind === 'week' || kind === 'session' ? kind : 'lecture',
    parts: strList(course.parts),
    trailing: strList(course.trailing),
    units,
  }
}

/** The `resource:` block on a page with `layout: resource`, or null. */
export function parseResourceRecord(frontmatter: unknown): CourseResourceRecord | null {
  if (!isRecord(frontmatter) || frontmatter.pageLayout !== 'resource') return null
  const resource = frontmatter.resource
  if (!isRecord(resource)) return null
  const kind = str(resource.kind)
  if (!resourceKinds.has(kind)) return null
  const out: CourseResourceRecord = { kind: kind as CourseResourceKind }
  for (const key of [
    'file',
    'youtube',
    'captions',
    'captionsId',
    'transcript',
    'transcriptId',
    'of',
    'notes',
    'type',
  ] as const) {
    const value = str(resource[key])
    if (value) out[key] = value
  }
  for (const key of ['bytes', 'n', 'due', 'duration', 'cues'] as const) {
    const value = num(resource[key])
    if (value !== undefined) out[key] = value
  }
  if (resource.full === true) out.full = true
  if (Array.isArray(resource.items)) out.items = strList(resource.items)
  if (Array.isArray(resource.links)) out.links = strList(resource.links)
  return out
}

/** The `latex:` block on a generated note, or null. */
export function latexOf(file: QuartzPluginData | undefined): LatexRecord | null {
  const latex = file?.frontmatter?.latex
  if (!isRecord(latex)) return null
  const source = str(latex.source)
  const pdf = str(latex.pdf)
  const lines = Array.isArray(latex.lines) ? latex.lines.map(num) : []
  if (!source || !pdf || lines.length !== 2 || lines.some(line => line === undefined)) return null
  const pages = Array.isArray(latex.pages) ? latex.pages.map(num) : []
  return {
    source,
    lines: lines as [number, number],
    ...(num(latex.section) !== undefined ? { section: num(latex.section) } : {}),
    ...(num(latex.unit) !== undefined ? { unit: num(latex.unit) } : {}),
    ...(num(latex.pset) !== undefined ? { pset: num(latex.pset) } : {}),
    pdf,
    ...(pages.length === 2 && !pages.some(page => page === undefined)
      ? { pages: pages as [number, number] }
      : {}),
    ...(str(latex.dateSource) ? { dateSource: str(latex.dateSource) } : {}),
  }
}

export function courseDirOf(slug: string): string | null {
  const match = /^courses\/([^/]+)(?:\/|$)/.exec(slug)
  return match ? match[1] : null
}

export type CourseUnitPath = 'lectures' | 'psets' | 'notes'

/** Unit coordinates derived from a slug such as `courses/18.905-fall-2016/lectures/07`. */
export function unitOf(slug: string): { dir: string; kind: CourseUnitPath; n: number } | null {
  const match = unitRe.exec(slug)
  if (!match) return null
  return { dir: match[1], kind: match[2] as CourseUnitPath, n: Number(match[3]) }
}

/** The course directory and name of a vendored `pages/<name>` page, else null. */
export function coursePageOf(slug: string | undefined): { dir: string; name: string } | null {
  const match = coursePageRe.exec(slug ?? '')
  return match ? { dir: match[1], name: match[2] } : null
}

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

export function unitSlug(dir: string, n: number, kind: CourseUnitPath = 'lectures'): FullSlug {
  return `courses/${dir}/${kind}/${String(n).padStart(2, '0')}` as FullSlug
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

export function psetTextSlug(dir: string, n: number): FullSlug {
  return `courses/${dir}/notes/pset-${String(n).padStart(2, '0')}` as FullSlug
}

export function unitDeckSlug(dir: string, n: number): FullSlug {
  return `${unitSlug(dir, n)}/flashcards` as FullSlug
}

export function unitLabel(kind: CourseUnitKind, n: number, total: number): string {
  return `${kind} ${String(n).padStart(2, '0')} of ${total}`
}

/** The authored study log: `courses/<dir>/lectures/index.md`. The vendor script never writes there. */
export function studyLogSlug(dir: string): FullSlug {
  return `courses/${dir}/lectures/index` as FullSlug
}

export type StudyKey = 'read' | 'problems'

export interface StudyLog {
  read: Map<number, string>
  problems: Map<number, string>
}

/** `2026-10-08`, from a YAML date or a string as it was written. */
export function studyDay(value: unknown): string | undefined {
  if (value instanceof Date) return value.toISOString().slice(0, 10)
  if (typeof value === 'string' && /^\d{4}-\d{2}-\d{2}/.test(value)) return value.slice(0, 10)
  return undefined
}

/** The `study: {read: {n: date}, problems: {n: date}}` map of a study log page. */
export function parseStudyLog(frontmatter: unknown): StudyLog {
  const log: StudyLog = { read: new Map(), problems: new Map() }
  const study = isRecord(frontmatter) ? frontmatter.study : undefined
  if (!isRecord(study)) return log
  for (const key of ['read', 'problems'] as const) {
    const map = study[key]
    if (!isRecord(map)) continue
    for (const [raw, value] of Object.entries(map)) {
      const n = Number(raw)
      const day = studyDay(value)
      if (Number.isInteger(n) && n > 0 && day) log[key].set(n, day)
    }
  }
  return log
}

const EMPTY_DECK_FILES: QuartzPluginData[] = []

interface CourseFilesIndex {
  byExactSlug: Map<string, QuartzPluginData>
  byNoteSlug: Map<string, QuartzPluginData>
  homes: QuartzPluginData[]
  homesByDir: Map<string, QuartzPluginData>
  filesByDir: Map<string, QuartzPluginData[]>
  psetsByDir: Map<string, Map<number, QuartzPluginData>>
  psetTextByDir: Map<string, Map<number, QuartzPluginData>>
  linksByDir: Map<string, { pages: CourseLink[]; resources: CourseLink[]; notes?: CourseLink }>
  cadenceByDir: Map<string, CourseCadence | undefined>
  viewsByDeckList: WeakMap<QuartzPluginData[], Map<QuartzPluginData, CourseView | null>>
  allViewsByDeckList: WeakMap<QuartzPluginData[], CourseView[]>
}

import { ReloadableWeakMap } from './module-caches'

const courseFilesIndexCache = new ReloadableWeakMap<QuartzPluginData[], CourseFilesIndex>()
const deckMapCache = new ReloadableWeakMap<QuartzPluginData[], Map<string, QuartzPluginData>>()

function getDeckMap(deckFiles: QuartzPluginData[]): Map<string, QuartzPluginData> {
  let map = deckMapCache.get(deckFiles)
  if (!map) {
    map = new Map()
    for (const file of deckFiles) {
      if (file.flashcards) {
        map.set(file.flashcards.sourceSlug as string, file)
      }
    }
    deckMapCache.set(deckFiles, map)
  }
  return map
}

function getCourseFilesIndex(allFiles: QuartzPluginData[]): CourseFilesIndex {
  let index = courseFilesIndexCache.get(allFiles)
  if (index) return index

  const byExactSlug = new Map<string, QuartzPluginData>()
  const byNoteSlug = new Map<string, QuartzPluginData>()
  const homes: QuartzPluginData[] = []
  const homesByDir = new Map<string, QuartzPluginData>()
  const filesByDir = new Map<string, QuartzPluginData[]>()
  const psetsByDir = new Map<string, Map<number, QuartzPluginData>>()
  const psetTextByDir = new Map<string, Map<number, QuartzPluginData>>()

  for (const file of allFiles) {
    const slug = file.slug ?? ''
    if (slug) byExactSlug.set(slug, file)
    if (!file.flashcards) {
      byNoteSlug.set(file.flashcardRoute?.noteSlug ?? slug, file)
    }
    if (!slug.startsWith('courses/')) continue
    const dir = courseDirOf(slug)
    if (!dir) continue

    let dirFiles = filesByDir.get(dir)
    if (!dirFiles) {
      dirFiles = []
      filesByDir.set(dir, dirFiles)
    }
    dirFiles.push(file)

    if (slug === `courses/${dir}/index`) {
      homes.push(file)
      homesByDir.set(dir, file)
      continue
    }

    const unit = unitOf(slug)
    if (unit && unit.dir === dir && unit.kind === 'psets') {
      let psets = psetsByDir.get(dir)
      if (!psets) {
        psets = new Map()
        psetsByDir.set(dir, psets)
      }
      psets.set(unit.n, file)
      continue
    }

    const pset = /^courses\/([^/]+)\/notes\/pset-(\d{2,3})$/.exec(slug)
    if (pset && pset[1] === dir) {
      let psetText = psetTextByDir.get(dir)
      if (!psetText) {
        psetText = new Map()
        psetTextByDir.set(dir, psetText)
      }
      psetText.set(Number(pset[2]), file)
    }
  }

  index = {
    byExactSlug,
    byNoteSlug,
    homes,
    homesByDir,
    filesByDir,
    psetsByDir,
    psetTextByDir,
    linksByDir: new Map(),
    cadenceByDir: new Map(),
    viewsByDeckList: new WeakMap(),
    allViewsByDeckList: new WeakMap(),
  }
  courseFilesIndexCache.set(allFiles, index)
  return index
}

/** O(1) lookup of a file in `allFiles` by its exact slug. */
export function fileBySlug(
  slug: string,
  allFiles: QuartzPluginData[],
): QuartzPluginData | undefined {
  return getCourseFilesIndex(allFiles).byExactSlug.get(slug)
}

/**
 * Joins a course home with its authored notes and decks. `allFiles` is the page list every
 * component receives; `deckFiles` is `ctx.decks`, since decks are filtered out of `allFiles`.
 */
export function courseView(
  home: QuartzPluginData,
  allFiles: QuartzPluginData[],
  deckFiles: QuartzPluginData[] = EMPTY_DECK_FILES,
): CourseView | null {
  const index = getCourseFilesIndex(allFiles)
  let byHome = index.viewsByDeckList.get(deckFiles)
  if (!byHome) {
    byHome = new Map()
    index.viewsByDeckList.set(deckFiles, byHome)
  }
  if (byHome.has(home)) return byHome.get(home)!

  const record = parseCourseRecord(home.frontmatter)
  const dir = courseDirOf(home.slug ?? '')
  if (!record || !dir) {
    byHome.set(home, null)
    return null
  }
  const bySlug = index.byNoteSlug
  const decks = getDeckMap(deckFiles)
  const logPage = bySlug.get(studyLogSlug(dir))
  const log = parseStudyLog(logPage?.frontmatter)
  const units = record.units.map((unit): CourseUnitView => {
    const slug = unitSlug(dir, unit.n)
    const note = bySlug.get(slug)
    const deck = decks.get(slug)
    const text = bySlug.get(unitSlug(dir, unit.n, 'notes'))
    const cardIds = deck?.flashcards?.cards.map(card => card.id) ?? []
    const state: CourseUnitState = deck ? 'deck' : note ? 'notes' : 'new'
    const read = log.read.get(unit.n)
    const problems = log.problems.get(unit.n)
    return {
      record: unit,
      slug,
      note,
      deck,
      ...(text ? { text } : {}),
      state,
      cardIds,
      ...(read ? { read } : {}),
      ...(problems ? { problems } : {}),
    }
  })
  const psets = new Map(index.psetsByDir.get(dir) ?? [])
  const psetText = new Map(index.psetTextByDir.get(dir) ?? [])
  const view: CourseView = {
    dir,
    slug: home.slug as FullSlug,
    home,
    record,
    units,
    psets,
    psetText,
    ...(logPage ? { log: logPage } : {}),
  }
  byHome.set(home, view)
  return view
}

/** Every course home with a study block, in course-number order. */
export function courseViews(
  allFiles: QuartzPluginData[],
  deckFiles: QuartzPluginData[] = EMPTY_DECK_FILES,
): CourseView[] {
  const index = getCourseFilesIndex(allFiles)
  const cached = index.allViewsByDeckList.get(deckFiles)
  if (cached) return cached
  const views = index.homes
    .map(file => courseView(file, allFiles, deckFiles))
    .filter((view): view is CourseView => view !== null)
    .sort((a, b) => a.record.number.localeCompare(b.record.number, undefined, { numeric: true }))
  index.allViewsByDeckList.set(deckFiles, views)
  return views
}

/** The course home for a directory, with its view when the home carries a course block. */
export function courseOf(
  dir: string,
  allFiles: QuartzPluginData[],
  deckFiles: QuartzPluginData[] = EMPTY_DECK_FILES,
): { home?: QuartzPluginData; view: CourseView | null } {
  const home = getCourseFilesIndex(allFiles).homesByDir.get(dir)
  return { home, view: home ? courseView(home, allFiles, deckFiles) : null }
}

export interface CourseCadence {
  sessions: number
  minutes: number
}

/**
 * Lecture meetings from the vendored syllabus, as OCW writes them: "Lectures: 2 sessions / week,
 * 1.5 hours / session". Undefined when the course has no syllabus or the line is missing.
 */
export function courseCadence(
  dir: string,
  allFiles: QuartzPluginData[],
): CourseCadence | undefined {
  const index = getCourseFilesIndex(allFiles)
  if (index.cadenceByDir.has(dir)) return index.cadenceByDir.get(dir)
  const syllabus = index.byExactSlug.get(`courses/${dir}/pages/syllabus/index`)
  const match =
    /lectures?:\s*(\d+)\s*sessions?\s*\/\s*week\s*[,;]\s*([\d.]+)\s*hours?\s*\/\s*session/i.exec(
      syllabus?.text ?? '',
    )
  if (!match) {
    index.cadenceByDir.set(dir, undefined)
    return undefined
  }
  const sessions = Number(match[1])
  const minutes = Math.round(Number(match[2]) * 60)
  const result = sessions > 0 && sessions <= 7 && minutes > 0 ? { sessions, minutes } : undefined
  index.cadenceByDir.set(dir, result)
  return result
}

export function unitWordCount(note: QuartzPluginData | undefined): number | undefined {
  const words = note?.readingTime?.words
  return typeof words === 'number' && words > 0 ? words : undefined
}

export function unitPdf(unit: CourseUnitRecord): CourseUnitResource | undefined {
  return unit.resources.find(
    resource => resource.file?.toLowerCase().endsWith('.pdf') && !/transcript/i.test(resource.id),
  )
}

export function unitVideo(unit: CourseUnitRecord): CourseUnitResource | undefined {
  return unit.resources.find(resource => resource.youtube)
}

export function unitVideos(unit: CourseUnitRecord): CourseUnitResource[] {
  return unit.resources.filter(resource => resource.youtube)
}

/** Resource page slug for a vendored resource id. */
export function resourceSlug(dir: string, id: string): FullSlug {
  return `courses/${dir}/resources/${id}` as FullSlug
}

/** Manifest slug of a vendored file (`static_resources/x.pdf` relative to the course dir). */
export function courseFileSlug(dir: string, file: string): string {
  return slugifyFilePath(joinSegments('courses', dir, file) as FilePath)
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

export function formatDuration(seconds: number | undefined): string | undefined {
  if (seconds === undefined || !Number.isFinite(seconds) || seconds <= 0) return undefined
  const total = Math.round(seconds)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  const mm = h > 0 ? String(m).padStart(2, '0') : String(m)
  return h > 0 ? `${h}:${mm}:${String(s).padStart(2, '0')}` : `${mm}:${String(s).padStart(2, '0')}`
}

export function formatBytes(bytes: number | undefined): string | undefined {
  if (bytes === undefined || !Number.isFinite(bytes) || bytes < 0) return undefined
  if (bytes < 1024) return `${bytes} b`
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} kb`
  return `${(bytes / (1024 * 1024)).toFixed(1)} mb`
}

/** `prof. lawrence d guth` → `guth`. */
export function lecturerName(record: CourseRecord | null | undefined): string | undefined {
  const first = record?.instructors[0]
  if (!first) return undefined
  const words = first
    .replace(/\b(?:prof|dr|mr|ms|mrs)\.?\s+/gi, '')
    .trim()
    .split(/\s+/)
  return words[words.length - 1]?.toLowerCase() || undefined
}

export interface CourseLink {
  title: string
  slug: FullSlug
  count?: number
}

/**
 * Vendored `pages/<name>` and the resource collection pages. The generated notes index leads
 * the resources list, then collections by title, each with its member count.
 */
export function courseLinks(
  dir: string,
  allFiles: QuartzPluginData[],
): { pages: CourseLink[]; resources: CourseLink[]; notes?: CourseLink } {
  const index = getCourseFilesIndex(allFiles)
  const cached = index.linksByDir.get(dir)
  if (cached) return cached

  const pages: CourseLink[] = []
  const resources: CourseLink[] = []
  let notes: CourseLink | undefined
  const base = `courses/${dir}/`
  const dirFiles = index.filesByDir.get(dir) ?? []
  for (const file of dirFiles) {
    const slug = file.slug ?? ''
    if (!slug.startsWith(base) || !slug.endsWith('/index')) continue
    const parts = slug.slice(base.length).split('/')
    const [kind, name] = parts
    const title = String(file.frontmatter?.title ?? name).toLowerCase()
    if (parts.length === 3 && kind === 'pages') pages.push({ title, slug: slug as FullSlug })
    else if (parts.length === 3 && kind === 'resources') {
      const record = parseResourceRecord(file.frontmatter)
      if (record?.kind === 'collection')
        resources.push({ title, slug: slug as FullSlug, count: record.items?.length ?? 0 })
    } else if (parts.length === 2 && kind === 'notes') {
      const record = parseResourceRecord(file.frontmatter)
      if (record?.kind === 'notes-index') notes = { title: 'notes', slug: slug as FullSlug }
    }
  }
  const byTitle = (a: CourseLink, b: CourseLink) => a.title.localeCompare(b.title)
  const result = {
    pages: pages.sort(byTitle),
    resources: resources.sort(byTitle),
    ...(notes ? { notes } : {}),
  }
  index.linksByDir.set(dir, result)
  return result
}

export function formatCount(value: number | undefined): string {
  return value === undefined ? '–' : value.toLocaleString('en-US')
}

/** JSON for the client script: deck slug, the ids it owns, and whether the unit is marked read. */
export function unitManifest(view: CourseView) {
  return view.units
    .filter(unit => unit.deck)
    .map(unit => ({
      deck: unitDeckSlug(view.dir, unit.record.n),
      course: view.dir,
      n: unit.record.n,
      read: Boolean(unit.read),
      ids: unit.cardIds,
      groups: Object.fromEntries(
        (unit.deck!.flashcards!.cards ?? [])
          .filter(card => card.groupId)
          .map(card => [card.id, card.groupId!]),
      ),
    }))
}
