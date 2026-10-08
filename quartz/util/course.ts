import { QuartzPluginData } from '../plugins/vfile'
import { FullSlug } from './path'

export type CourseUnitKind = 'lecture' | 'week' | 'session'

export interface CourseUnitResource {
  id: string
  file?: string
  youtube?: string
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
  state: CourseUnitState
  cardIds: string[]
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
}

const unitRe = /^courses\/([^/]+)\/(lectures|psets)\/(\d{2,3})$/

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
        return [{ id, ...(file ? { file } : {}), ...(youtube ? { youtube } : {}) }]
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

export function courseDirOf(slug: string): string | null {
  const match = /^courses\/([^/]+)(?:\/|$)/.exec(slug)
  return match ? match[1] : null
}

/** Unit coordinates derived from a slug such as `courses/18.905-fall-2016/lectures/07`. */
export function unitOf(
  slug: string,
): { dir: string; kind: 'lectures' | 'psets'; n: number } | null {
  const match = unitRe.exec(slug)
  if (!match) return null
  return { dir: match[1], kind: match[2] as 'lectures' | 'psets', n: Number(match[3]) }
}

export function unitSlug(
  dir: string,
  n: number,
  kind: 'lectures' | 'psets' = 'lectures',
): FullSlug {
  return `courses/${dir}/${kind}/${String(n).padStart(2, '0')}` as FullSlug
}

export function unitDeckSlug(dir: string, n: number): FullSlug {
  return `${unitSlug(dir, n)}/flashcards` as FullSlug
}

export function unitLabel(kind: CourseUnitKind, n: number, total: number): string {
  return `${kind} ${String(n).padStart(2, '0')} of ${total}`
}

/** A `study.<key>` date on a unit note, formatted as it was written. */
function studyDate(note: QuartzPluginData | undefined, key: string): string | undefined {
  const study = note?.frontmatter?.study
  if (!isRecord(study)) return undefined
  const value = study[key]
  if (value instanceof Date) return value.toISOString().slice(0, 10)
  if (typeof value === 'string' && value.length > 0) return value.slice(0, 10)
  return undefined
}

/**
 * Joins a course home with its authored notes and decks. `allFiles` is the page list every
 * component receives; `deckFiles` is `ctx.decks`, since decks are filtered out of `allFiles`.
 */
export function courseView(
  home: QuartzPluginData,
  allFiles: QuartzPluginData[],
  deckFiles: QuartzPluginData[] = [],
): CourseView | null {
  const record = parseCourseRecord(home.frontmatter)
  const dir = courseDirOf(home.slug ?? '')
  if (!record || !dir) return null
  const bySlug = new Map(allFiles.map(file => [file.slug as string, file]))
  const decks = new Map(
    deckFiles
      .filter(file => file.flashcards)
      .map(file => [file.flashcards!.sourceSlug as string, file]),
  )
  const units = record.units.map((unit): CourseUnitView => {
    const slug = unitSlug(dir, unit.n)
    const note = bySlug.get(slug)
    const deck = decks.get(slug)
    const cardIds = deck?.flashcards?.cards.map(card => card.id) ?? []
    const state: CourseUnitState = deck ? 'deck' : note ? 'notes' : 'new'
    const read = studyDate(note, 'read')
    const problems = studyDate(note, 'problems')
    return {
      record: unit,
      slug,
      note,
      deck,
      state,
      cardIds,
      ...(read ? { read } : {}),
      ...(problems ? { problems } : {}),
    }
  })
  const psets = new Map<number, QuartzPluginData>()
  for (const file of allFiles) {
    const unit = unitOf(file.slug ?? '')
    if (unit && unit.dir === dir && unit.kind === 'psets') psets.set(unit.n, file)
  }
  return { dir, slug: home.slug as FullSlug, home, record, units, psets }
}

/** Every course home with a study block, in course-number order. */
export function courseViews(
  allFiles: QuartzPluginData[],
  deckFiles: QuartzPluginData[] = [],
): CourseView[] {
  return allFiles
    .filter(file => /^courses\/[^/]+\/index$/.test(file.slug ?? ''))
    .map(file => courseView(file, allFiles, deckFiles))
    .filter((view): view is CourseView => view !== null)
    .sort((a, b) => a.record.number.localeCompare(b.record.number, undefined, { numeric: true }))
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

/** Resource page slug for a vendored resource id. */
export function resourceSlug(dir: string, id: string): FullSlug {
  return `courses/${dir}/resources/${id}` as FullSlug
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
