import path from 'node:path'
import { convertLatexCourse, LatexGateError, type LatexResult } from '../vendor-ocw-latex'
import {
  contentRoot,
  normalizeText,
  normalizeTitle,
  recordList,
  slugTitle,
  stringValue,
  type CourseContext,
  type CoursePage,
  type CourseResource,
  type CourseStudy,
  type CourseUnit,
  type CourseUnitKind,
  type JsonRecord,
} from './context'
import { lectureNumber, problemSetNumberRe, resourceId } from './resources'

function cellText(html: string): string[] {
  return html
    .replace(/<\/(?:p|br|li)\s*>|<br\s*\/?>/gi, '\n')
    .replace(/<[^>]+>/g, ' ')
    .split('\n')
    .map(line =>
      normalizeText(
        line
          .replace(/&nbsp;/g, ' ')
          .replace(/&[lr]squo;/g, "'")
          .replace(/&[lr]dquo;/g, '"'),
      ),
    )
    .filter(line => line.length > 0)
}

function unitKindFromHeader(header: string): CourseUnitKind {
  const value = header.toLowerCase()
  if (value.includes('week')) return 'week'
  if (/\bses/.test(value)) return 'session'
  return 'lecture'
}

// The OCW calendar table: a header row names the unit kind, a `<th colspan>` row opens a part,
// a row whose first cell is an integer is a unit, and a numberless row with text is trailing.
// Columns past the topic are key dates only on a calendar; a readings table lists sections there.
function parseCalendarUnits(html: string, withKeyDates = true): CourseStudy | null {
  const rows = Array.from(html.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/gi), match => match[1])
  if (rows.length === 0) return null
  const study: CourseStudy = { unitKind: 'lecture', parts: [], trailing: [], units: [] }
  let sawHeader = false
  for (const row of rows) {
    const cells = Array.from(row.matchAll(/<(t[hd])\b([^>]*)>([\s\S]*?)<\/t[hd]>/gi), match => ({
      tag: match[1].toLowerCase(),
      attrs: match[2],
      lines: cellText(match[3]),
    }))
    if (cells.length === 0) continue
    const first = cells[0]
    if (!sawHeader && first.tag === 'th' && cells.length > 1) {
      sawHeader = true
      study.unitKind = unitKindFromHeader(first.lines.join(' '))
      continue
    }
    // OCW writes part rows as `<td colspan><strong>II. …</strong>`; a colspan row without a
    // roman numeral (the oral exam) stays trailing.
    const spanLabel = first.lines.join(' ')
    if (/colspan/i.test(first.attrs) && (first.tag === 'th' || /^[IVXLC]+\.\s/.test(spanLabel))) {
      const label = spanLabel.replace(/^[IVXLC]+\.\s*/i, '')
      if (label.length > 0) study.parts.push(label.toLowerCase())
      continue
    }
    const number = first.lines.join(' ')
    const topicLines = cells[1]?.lines ?? []
    const keyDates = withKeyDates ? cells.slice(2).flatMap(cell => cell.lines) : []
    if (/^\d+$/.test(number)) {
      const topic = topicLines.join(' ')
      const session = topic.match(/^ses(?:sion)?s?\.?\s*([\d\s,and-]+?):\s*(.*)$/i)
      study.units.push({
        n: Number(number),
        title: normalizeTitle(session ? session[2] : topic),
        sessions: session ? normalizeText(session[1]).replace(/\s+/g, '') : '',
        part: study.parts.length,
        keyDates: keyDates.map(value => value.toLowerCase()),
        resources: [],
        due: [],
      })
      continue
    }
    const text = [...first.lines, ...topicLines, ...keyDates].join(' ').toLowerCase()
    if (text.length > 0) study.trailing.push(text)
  }
  if (study.units.length === 0) return null
  const seen = new Set<number>()
  study.units = study.units.filter(unit => !seen.has(unit.n) && seen.add(unit.n))
  return study
}

// Courses without a calendar page take their units from numbered lecture-note resources.
function unitsFromResources(resources: CourseResource[]): CourseStudy | null {
  const byNumber = new Map<number, CourseUnit>()
  for (const resource of resources) {
    if (!resource.resourceTypes.includes('Lecture Notes')) continue
    const n = lectureNumber(resource)
    if (n === null || byNumber.has(n)) continue
    const title = resource.title.replace(/^.*?\blecture\s*\d+\s*[:.-]?\s*/i, '')
    byNumber.set(n, {
      n,
      title: normalizeTitle(title.length > 0 ? title : resource.title),
      sessions: '',
      part: 0,
      keyDates: [],
      resources: [],
      due: [],
    })
  }
  if (byNumber.size === 0) return null
  const units = Array.from(byNumber.values()).sort((left, right) => left.n - right.n)
  return { unitKind: 'lecture', parts: [], trailing: [], units }
}

function attachUnitResources(study: CourseStudy, resources: CourseResource[]): void {
  const units = new Map(study.units.map(unit => [unit.n, unit]))
  const psets = new Map<number, string>()
  for (const resource of resources) {
    const id = resourceId(resource)
    const lecture =
      resource.kind === 'notes' || resource.kind === 'video' ? lectureNumber(resource) : null
    if (lecture !== null && units.has(lecture)) {
      const unit = units.get(lecture)!
      if (!unit.resources.some(entry => entry.id === id)) {
        unit.resources.push(
          resource.kind === 'video'
            ? { id, youtube: resource.youtubeId, duration: resource.duration, cues: resource.cues }
            : { id, file: path.posix.join('static_resources', resource.localFile) },
        )
      }
    }
    if (resource.kind === 'pset' && resource.n !== null && !psets.has(resource.n)) {
      psets.set(resource.n, id)
    }
  }
  const byId = new Map(resources.map(resource => [resourceId(resource), resource]))
  for (const unit of study.units) {
    for (const date of unit.keyDates) {
      const match = date.match(problemSetNumberRe)
      if (!match) continue
      const id = psets.get(Number(match[1]))
      if (id && !unit.due.includes(id)) unit.due.push(id)
    }
    for (const id of unit.due) {
      const resource = byId.get(id)
      if (resource && resource.due === null) resource.due = unit.n
    }
  }
}

export function courseStudy(pages: CoursePage[], resources: CourseResource[]): CourseStudy | null {
  const calendar = pages.find(page => slugTitle(page.title) === 'calendar')
  // last, the session table of a readings page (18.100b has neither calendar nor numbered notes)
  const readings = pages.find(page => /^readings|^lecture-notes/.test(slugTitle(page.title)))
  const study =
    (calendar && parseCalendarUnits(calendar.content)) ||
    unitsFromResources(resources) ||
    (readings && parseCalendarUnits(readings.content, false))
  if (!study) return null
  attachUnitResources(study, resources)
  return study
}

// LaTeX sources become `notes/`; the converter's gates fail the whole course rather than ship a wrong page.
export async function latexNotes(
  context: CourseContext,
  data: JsonRecord,
  study: CourseStudy | null,
  resources: CourseResource[],
): Promise<LatexResult | null> {
  if (!study || !resources.some(resource => resource.kind === 'tex')) return null
  const instructors = recordList(data, 'instructors').map(instructor =>
    stringValue(instructor, 'title'),
  )
  try {
    const result = await convertLatexCourse({
      root: context.root,
      slug: context.slug,
      number: context.number,
      term: `${stringValue(data, 'term')} ${stringValue(data, 'year')}`.trim(),
      instructors,
      licenseName: context.licenseName,
      licenseUrl: context.licenseUrl,
      units: study.units.map(unit => ({ n: unit.n, title: unit.title })),
      resources: resources.map(resource => ({
        relDir: resource.relDir,
        title: resource.title,
        localFile: resource.localFile,
        resourceTypes: resource.resourceTypes,
      })),
      contentRoot,
    })
    if (result) {
      console.log(
        `${context.slug}: latex: ${result.notes.length} lectures, ${result.psets.length} problem sets, ${result.figures} figures`,
      )
      for (const line of result.report) console.log(`  ${line}`)
    }
    return result
  } catch (error) {
    if (error instanceof LatexGateError) throw new Error(`${context.slug}: ${error.message}`)
    throw error
  }
}
