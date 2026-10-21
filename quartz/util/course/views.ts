import { QuartzPluginData } from '../../plugins/vfile'
import { ReloadableWeakMap } from '../module-caches'
import { FullSlug } from '../path'
import {
  courseDirOf,
  parseCourseRecord,
  parseResourceRecord,
  parseStudyLog,
  studyLogSlug,
  unitDeckSlug,
  unitOf,
  unitSlug,
  type CourseRecord,
  type CourseUnitRecord,
} from './schema'

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
