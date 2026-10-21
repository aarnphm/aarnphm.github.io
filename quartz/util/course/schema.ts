import { QuartzPluginData } from '../../plugins/vfile'
import { FilePath, FullSlug, joinSegments, slugifyFilePath } from '../path'

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

export function str(value: unknown): string {
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

export function unitSlug(dir: string, n: number, kind: CourseUnitPath = 'lectures'): FullSlug {
  return `courses/${dir}/${kind}/${String(n).padStart(2, '0')}` as FullSlug
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

export function formatCount(value: number | undefined): string {
  return value === undefined ? '–' : value.toLocaleString('en-US')
}
