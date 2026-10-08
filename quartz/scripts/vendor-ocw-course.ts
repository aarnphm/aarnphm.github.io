import { globby } from 'globby'
import { fromHtml } from 'hast-util-from-html'
import { toMdast } from 'hast-util-to-mdast'
import { gfmToMarkdown } from 'mdast-util-gfm'
import { toMarkdown } from 'mdast-util-to-markdown'
import fs from 'node:fs/promises'
import path from 'node:path'
import { isRecord } from '../util/type-guards'

type JsonRecord = Record<string, unknown>

type CourseContext = {
  root: string
  slug: string
  sourcePath: string
  sourceUrl: string
  licenseName: string
  licenseUrl: string
  idPrefix: string
  label: string
  number: string
  title: string
  tags: string[]
  aliases: string[]
  linkableDirs: Set<string>
}

type CoursePage = {
  relDir: string
  title: string
  description: string
  content: string
  resourceTypes: string[]
}

type CourseResource = {
  relDir: string
  title: string
  description: string
  content: string
  resourceType: string
  resourceTypes: string[]
  youtubeId: string
  localFile: string
  originalFile: string
}

type CourseCollection = {
  relDir: string
  title: string
  description: string
  resources: CourseResource[]
  extraLinks: string[]
}

type CourseUnitKind = 'lecture' | 'week' | 'session'

type CourseUnit = {
  n: number
  title: string
  sessions: string
  part: number
  keyDates: string[]
  resources: { id: string; file: string; youtube: string }[]
  due: string[]
}

type CourseStudy = {
  unitKind: CourseUnitKind
  parts: string[]
  trailing: string[]
  units: CourseUnit[]
}

const defaultCourseRoot = path.join('content', 'courses')
const contentRoot = path.resolve('content')
const mitBaseUrl = 'https://ocw.mit.edu/'
const ocwHost = 'ocw.mit.edu'
const fetchConcurrency = 8
const ocwLayout = 'A|L'
const defaultLicenseName = 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International'
const defaultLicenseUrl = 'https://creativecommons.org/licenses/by-nc-sa/4.0/'

function stringValue(record: JsonRecord, key: string): string {
  const value = record[key]
  return typeof value === 'string' ? value : ''
}

function stringList(record: JsonRecord, key: string): string[] {
  const value = record[key]
  if (!Array.isArray(value)) return []
  return value.filter(item => typeof item === 'string').map(normalizeText)
}

function recordList(record: JsonRecord, key: string): JsonRecord[] {
  const value = record[key]
  if (!Array.isArray(value)) return []
  return value.filter(isRecord)
}

function nestedStringLists(record: JsonRecord, key: string): string[][] {
  const value = record[key]
  if (!Array.isArray(value)) return []
  return value
    .filter(item => Array.isArray(item))
    .map(item => item.filter(child => typeof child === 'string').map(normalizeText))
    .filter(item => item.length > 0)
}

function nestedRecord(record: JsonRecord, key: string): JsonRecord | undefined {
  const value = record[key]
  return isRecord(value) ? value : undefined
}

function nestedString(record: JsonRecord, key: string, childKey: string): string {
  const child = nestedRecord(record, key)
  return child ? stringValue(child, childKey) : ''
}

function firstString(values: string[]): string {
  return values.find(value => value.length > 0) ?? ''
}

function normalizeText(value: string): string {
  return value
    .replace(/\u00a0/g, ' ')
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/[–—]/g, '-')
    .replace(/&amp;/g, '&')
    .replace(/\s+\n/g, '\n')
    .trim()
}

function titleWithoutExtension(value: string): string {
  return value.replace(/\.(pdf|jpg|jpeg|png|gif|mp3|mp4|zip|html?)$/i, '')
}

function normalizeTitle(value: string): string {
  const spaced = titleWithoutExtension(value)
    .replace(/commentsonstyle/i, 'comments on style')
    .replace(/erratafortop/i, 'errata for topology')
    .replace(/problem[\s_-]*set/i, 'problem set')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/[_-]+/g, ' ')
  return normalizeText(spaced).toLowerCase()
}

function compareTitle(left: { title: string }, right: { title: string }): number {
  return left.title.localeCompare(right.title, undefined, { numeric: true })
}

function descriptionTitle(description: string): string {
  const [candidate] = normalizeText(description).split(':')
  if (candidate.length === 0 || candidate.length > 48) return ''
  if (!/^[a-zA-Z0-9 ._-]+$/.test(candidate)) return ''
  return normalizeTitle(candidate)
}

function resourceTitle(context: CourseContext, data: JsonRecord, relDir: string): string {
  const rawTitle = stringValue(data, 'title')
  const description = stringValue(data, 'description')
  const titleBase = titleWithoutExtension(rawTitle)
  const fromDescription = descriptionTitle(description)
  if (/^\d[\d._-]*$/.test(titleBase) && fromDescription.length > 0) {
    return fromDescription
  }
  if (rawTitle.length === 0 && description.length === 0) {
    const inferred = inferredResourceTitle(context, data, relDir)
    if (inferred.length > 0) return inferred
  }
  return normalizeTitle(rawTitle || description || relDir)
}

function inferredResourceTitle(context: CourseContext, data: JsonRecord, relDir: string): string {
  const source = resourceInferenceSource(data, relDir)
  const lecture = source.match(/\blec(?:ture)?\s*(\d+)\b/)
  if (lecture) return `${context.title}: lecture ${lecture[1]}`
  const problemSet = source.match(/\b(?:pset|problem set)\s*(\d+)\b/)
  if (problemSet) return `${context.title}: problem set ${problemSet[1]}`
  if (source.includes('lecture notes')) return `${context.title}: lecture notes`
  return ''
}

function resourceInferenceSource(data: JsonRecord, relDir: string): string {
  return normalizeTitle(
    [
      stringValue(data, 'title'),
      stringValue(data, 'description'),
      stringValue(data, 'file'),
      path.posix.basename(relDir),
    ].join(' '),
  )
}

function resourceTypes(data: JsonRecord, relDir: string): string[] {
  const explicit = stringList(data, 'learning_resource_types')
  if (explicit.length > 0) return explicit
  const source = resourceInferenceSource(data, relDir)
  if (/\blec(?:ture)?\s*\d+\b/.test(source) || source.includes('lecture notes')) {
    return ['Lecture Notes']
  }
  if (/\b(?:pset|problem set|assignments?)\s*\d*\b/.test(source)) return ['Assignments']
  return []
}

function slugTitle(value: string): string {
  return normalizeTitle(value)
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
}

function yamlString(value: string): string {
  return `"${normalizeText(value).replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n')}"`
}

function yamlList(key: string, values: string[]): string[] {
  const items = uniqueValues(values.map(normalizeText))
  if (items.length === 0) return []
  return [key + ':', ...items.map(value => `  - ${yamlString(value)}`)]
}

function frontmatter(fields: Record<string, string | string[]>): string {
  const lines: string[] = ['---']
  for (const [key, value] of Object.entries(fields)) {
    if (Array.isArray(value)) {
      lines.push(...yamlList(key, value))
    } else if (value.length > 0) {
      lines.push(`${key}: ${yamlString(value)}`)
    }
  }
  lines.push('---')
  return lines.join('\n')
}

function courseFrontmatter(
  context: CourseContext,
  fields: Record<string, string | string[]>,
  layout: string = ocwLayout,
): string {
  return frontmatter({
    ...fields,
    layout,
    license: context.licenseName,
    license_url: context.licenseUrl,
  })
}

async function readJson(filePath: string): Promise<JsonRecord> {
  const raw: unknown = JSON.parse(await fs.readFile(filePath, 'utf8'))
  if (!isRecord(raw)) {
    throw new Error(`${filePath} does not contain a JSON object`)
  }
  return raw
}

async function htmlToMarkdown(context: CourseContext, html: string): Promise<string> {
  if (html.trim().length === 0) return ''
  const hast = fromHtml(normalizeMalformedHtml(html), { fragment: true })
  const mdast = toMdast(hast)
  return normalizeMarkdown(
    context,
    toMarkdown(mdast, { bullet: '-', fences: true, extensions: [gfmToMarkdown()] }),
  )
}

function normalizeMalformedHtml(value: string): string {
  return value.replace(
    /<(td|th)>\s*([\s\S]*?)\s*<\/\1>/gi,
    (_match, tag: string, content: string) => {
      const cell = content
        .replace(/<\/p>\s*<p>/gi, '<br>')
        .replace(/<p>\s*/gi, '')
        .replace(/\s*<\/p>/gi, '')
      return `<${tag}>${cell}</${tag}>`
    },
  )
}

function normalizeMarkdown(context: CourseContext, value: string): string {
  return normalizeText(value)
    .replace(/^(#{1,6})\s+(.+)$/gm, (_match, hashes: string, title: string) => {
      return `${hashes} ${normalizeText(title).toLowerCase()}`
    })
    .replace(/\[PDF\]/g, '[pdf]')
    .replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, (match: string, _label: string, url: string) => {
      const fileName = staticResourceFile(context, url)
      return fileName ? assetEmbed(context, fileName) : match
    })
    .replace(
      /(^|[^!])\[([^\]]+)\]\(([^)\s]+)\)/gm,
      (match: string, prefix: string, label: string, url: string) => {
        const relDir = courseRelativeDir(context, url)
        if (relDir) return `${prefix}${resourceLink(context, relDir, label)}`
        const fileName = staticResourceFile(context, url)
        if (fileName) {
          return `${prefix}[[${context.slug}/static_resources/${fileName}|${normalizeText(label).toLowerCase()}]]`
        }
        return match
      },
    )
    .replace(/^(\|.+)$/gm, (line: string) => {
      return line.replace(/\[\[([^\]\n|]+)\|([^\]\n]+)\]\]/g, '[[$1\\|$2]]')
    })
    .replace(/\)\)(?=[A-Z])/g, ')); ')
    .replace(
      /([a-z0-9.)=])\s+(?=(?:Sec\.|Problem Set|Midterm Exam|Final Exam|Notes [A-K]))/g,
      '$1<br>',
    )
    .replace(/(Principle)\s+(?=Ses \d+:|Midterm Exam)/g, '$1<br>')
    .replace(/([^\n])\n(#{1,6} )/g, '$1\n\n$2')
    .replace(/^(#{1,6} .+)\n(?!\n)/gm, '$1\n\n')
    .replace(/(^[^|\n].*\S)\n(\| [^\n]+\|)/gm, '$1\n\n$2')
    .replace(/(\|[^\n]+\|)\n(#{1,6} )/g, '$1\n\n$2')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

function strippedUrlPath(value: string): string {
  const [withoutHash] = value.split('#')
  const [withoutQuery] = withoutHash.split('?')
  return withoutQuery
}

function ocwLocalPath(context: CourseContext, value: string): string {
  const stripped = strippedUrlPath(value)
  let local = stripped
  if (/^https?:\/\//.test(stripped)) {
    try {
      const parsed = new URL(stripped)
      if (parsed.hostname !== 'ocw.mit.edu') return ''
      local = parsed.pathname.replace(/^\/+/, '')
    } catch {
      return ''
    }
  } else if (stripped.startsWith('/')) {
    local = stripped.replace(/^\/+/, '')
  } else {
    local = stripped.replace(/^(\.\/|\.\.\/)+/, '')
  }

  const sourcePath = context.sourcePath.replace(/^\/+|\/+$/g, '')
  if (sourcePath.length > 0 && local.startsWith(`${sourcePath}/`)) {
    return local.slice(sourcePath.length + 1)
  }
  if (!local.startsWith('courses/')) {
    return local
  }
  return ''
}

function courseRelativeDir(context: CourseContext, value: string): string {
  const local = ocwLocalPath(context, value)
  if (local.length === 0) return ''
  const candidate = local
    .replace(/index\.html?$/i, '')
    .replace(/\.html?$/i, '/')
    .replace(/\/+$/, '')
  if (context.linkableDirs.has(candidate)) return candidate
  if (/^(pages|resources)\//.test(candidate)) return candidate
  return ''
}

function staticResourceFile(context: CourseContext, value: string): string {
  const local = ocwLocalPath(context, value)
  if (local.length === 0) return ''
  if (local.startsWith('static_resources/')) return path.posix.basename(local)
  if (/\.(pdf|jpg|jpeg|png|gif|mp3|mp4|zip|html?)$/i.test(local)) return path.posix.basename(local)
  return ''
}

function resourceLink(context: CourseContext, relDir: string, label: string): string {
  const target = relDir.length > 0 ? `${context.slug}/${relDir}/` : `${context.slug}/`
  return `[[${target}|${normalizeText(label).toLowerCase()}]]`
}

function assetEmbed(context: CourseContext, localFile: string): string {
  if (localFile.length === 0) return ''
  return `![[${context.slug}/static_resources/${localFile}]]`
}

function originalAssetUrl(context: CourseContext, originalFile: string, localFile: string): string {
  if (/^https?:\/\//.test(originalFile)) return originalFile
  if (originalFile.startsWith('/')) return new URL(originalFile, mitBaseUrl).toString()
  if (originalFile.length > 0)
    return new URL(path.posix.basename(originalFile), context.sourceUrl).toString()
  if (localFile.length > 0) return new URL(localFile, context.sourceUrl).toString()
  return context.sourceUrl
}

async function writeMarkdown(
  context: CourseContext,
  relDir: string,
  content: string,
): Promise<void> {
  const outPath = path.join(context.root, relDir, 'index.md')
  await fs.mkdir(path.dirname(outPath), { recursive: true })
  await fs.writeFile(outPath, `${content.trim()}\n`, 'utf8')
}

function uniqueValues(values: string[]): string[] {
  return Array.from(new Set(values.map(normalizeText).filter(value => value.length > 0)))
}

function markdownDocument(parts: string[]): string {
  return parts
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

function metadataId(context: CourseContext, scope: string, value: string): string {
  return `${context.idPrefix}-${scope}-${slugTitle(value)}`
}

function learningTypesForFrontmatter(types: string[]): string[] {
  return uniqueValues(types.map(value => value.toLowerCase()))
}

function parseResource(context: CourseContext, relDir: string, data: JsonRecord): CourseResource {
  const originalFile = stringValue(data, 'file')
  const localFile = path.posix.basename(originalFile)
  const description = stringValue(data, 'description')
  return {
    relDir,
    title: resourceTitle(context, data, relDir),
    description: normalizeText(description),
    content: stringValue(data, 'content'),
    resourceType: normalizeText(
      firstString([stringValue(data, 'resourcetype'), stringValue(data, 'resource_type')]),
    ).toLowerCase(),
    resourceTypes: resourceTypes(data, relDir),
    youtubeId: nestedString(data, 'video_metadata', 'youtube_id'),
    localFile,
    originalFile,
  }
}

function parsePage(relDir: string, data: JsonRecord): CoursePage {
  const title = normalizeTitle(stringValue(data, 'title') || path.posix.basename(relDir))
  return {
    relDir,
    title,
    description: normalizeText(stringValue(data, 'description')),
    content: stringValue(data, 'content'),
    resourceTypes: stringList(data, 'learning_resource_types'),
  }
}

function topicLabels(data: JsonRecord): string[] {
  return uniqueValues(nestedStringLists(data, 'topics').flatMap(topic => topic))
}

function topicTags(data: JsonRecord): string[] {
  const labels = topicLabels(data).flatMap(label => {
    const normalized = label.toLowerCase()
    const parts = normalized.split(/\s+and\s+/).map(normalizeText)
    const aliases = normalized === 'mathematics' ? ['math'] : []
    return [normalized, ...parts, ...aliases]
  })
  return uniqueValues(labels)
}

function topicSummary(data: JsonRecord): string {
  return nestedStringLists(data, 'topics')
    .map(topic => topic.join(' / '))
    .filter(value => value.length > 0)
    .join(', ')
    .toLowerCase()
}

function extraCourseNumbers(data: JsonRecord): string[] {
  return stringValue(data, 'extra_course_numbers')
    .split(',')
    .map(normalizeText)
    .filter(value => value.length > 0)
}

function toPosixPath(value: string): string {
  return value.split(path.sep).join(path.posix.sep)
}

function contentSlug(courseRoot: string): string {
  const relative = toPosixPath(path.relative(contentRoot, path.resolve(courseRoot)))
  if (relative.length > 0 && !relative.startsWith('..')) return relative
  return path.posix.join('courses', path.basename(courseRoot))
}

function sourcePath(data: JsonRecord): string {
  const explicit = stringValue(data, 'site_url_path')
  if (explicit.length > 0) return explicit.replace(/^\/+|\/+$/g, '')
  const imageFile = nestedString(data, 'course_image_metadata', 'file')
  if (imageFile.startsWith('/courses/')) {
    return path.posix.dirname(imageFile).replace(/^\/+|\/+$/g, '')
  }
  return ''
}

function sourceUrl(data: JsonRecord): string {
  const value = sourcePath(data)
  return value.length > 0 ? new URL(`${value}/`, mitBaseUrl).toString() : mitBaseUrl
}

function licenseUrl(data: JsonRecord): string {
  return nestedString(data, 'course_image_metadata', 'license') || defaultLicenseUrl
}

function siteShortId(courseRoot: string, data: JsonRecord): string {
  const explicit = stringValue(data, 'site_short_id')
  if (explicit.length > 0) return explicit
  const number = stringValue(data, 'primary_course_number')
  const term = stringValue(data, 'term')
  const year = stringValue(data, 'year')
  const fromFields = [number, term, year]
    .map(normalizeText)
    .filter(value => value.length > 0)
    .join('-')
  return fromFields.length > 0 ? fromFields : path.basename(courseRoot)
}

function courseContext(courseRoot: string, data: JsonRecord): CourseContext {
  const number = normalizeText(stringValue(data, 'primary_course_number'))
  const title = normalizeTitle(stringValue(data, 'course_title') || path.basename(courseRoot))
  const shortId = siteShortId(courseRoot, data)
  const aliases = uniqueValues([number, ...extraCourseNumbers(data), title, shortId])
  const tags = uniqueValues(['course', 'mit', 'ocw', ...topicTags(data)])
  return {
    root: courseRoot,
    slug: contentSlug(courseRoot),
    sourcePath: sourcePath(data),
    sourceUrl: sourceUrl(data),
    licenseName: defaultLicenseName,
    licenseUrl: licenseUrl(data),
    idPrefix: `mit-${slugTitle(shortId)}`,
    label: number.length > 0 ? `mit ${number}` : 'mit ocw',
    number,
    title,
    tags,
    aliases,
    linkableDirs: new Set(),
  }
}

function resourceTypesInOrder(data: JsonRecord, resources: CourseResource[]): string[] {
  return uniqueValues([
    ...stringList(data, 'learning_resource_types'),
    ...resources.flatMap(resource => resource.resourceTypes),
  ])
}

function resourceCollectionRelDir(resourceType: string): string {
  return path.posix.join('resources', slugTitle(resourceType))
}

function cellText(html: string): string[] {
  return html
    .replace(/<\/(?:p|br|li)\s*>|<br\s*\/?>/gi, '\n')
    .replace(/<[^>]+>/g, ' ')
    .split('\n')
    .map(line => normalizeText(line.replace(/&nbsp;/g, ' ')))
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
function parseCalendarUnits(html: string): CourseStudy | null {
  const rows = Array.from(html.matchAll(/<tr[^>]*>([\s\S]*?)<\/tr>/gi), match => match[1])
  if (rows.length === 0) return null
  const study: CourseStudy = { unitKind: 'lecture', parts: [], trailing: [], units: [] }
  let sawHeader = false
  for (const row of rows) {
    const cells = Array.from(
      row.matchAll(/<(t[hd])\b([^>]*)>([\s\S]*?)<\/t[hd]>/gi),
      match => ({ tag: match[1].toLowerCase(), attrs: match[2], lines: cellText(match[3]) }),
    )
    if (cells.length === 0) continue
    const first = cells[0]
    if (!sawHeader && first.tag === 'th' && cells.length > 1) {
      sawHeader = true
      study.unitKind = unitKindFromHeader(first.lines.join(' '))
      continue
    }
    if (first.tag === 'th' && /colspan/i.test(first.attrs)) {
      const label = first.lines.join(' ').replace(/^[IVXLC]+\.\s*/i, '')
      if (label.length > 0) study.parts.push(label.toLowerCase())
      continue
    }
    const number = first.lines.join(' ')
    const topicLines = cells[1]?.lines ?? []
    const keyDates = cells.slice(2).flatMap(cell => cell.lines)
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

const lectureNumberRe = /\blec(?:ture)?[\s_#]*0*(\d+)\b/i
const problemSetNumberRe = /\b(?:pset|problem set|ps)[\s_#]*0*(\d+)\b/i

function lectureNumber(resource: CourseResource): number | null {
  const match =
    path.posix.basename(resource.relDir).match(lectureNumberRe) ??
    resource.title.match(lectureNumberRe)
  return match ? Number(match[1]) : null
}

function problemSetNumber(resource: CourseResource): number | null {
  const match =
    path.posix.basename(resource.relDir).match(problemSetNumberRe) ??
    resource.title.match(problemSetNumberRe)
  return match ? Number(match[1]) : null
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
    const id = path.posix.basename(resource.relDir)
    const isNotes =
      resource.resourceTypes.includes('Lecture Notes') ||
      resource.resourceTypes.includes('Lecture Videos') ||
      resource.youtubeId.length > 0
    const lecture = isNotes ? lectureNumber(resource) : null
    if (lecture !== null && units.has(lecture)) {
      const unit = units.get(lecture)!
      if (!unit.resources.some(entry => entry.id === id)) {
        unit.resources.push({
          id,
          file: resource.localFile ? path.posix.join('static_resources', resource.localFile) : '',
          youtube: resource.youtubeId,
        })
      }
    }
    const pset = problemSetNumber(resource)
    if (pset !== null && !resource.resourceTypes.some(type => /solution/i.test(type))) {
      if (!psets.has(pset)) psets.set(pset, id)
    }
  }
  for (const unit of study.units) {
    for (const date of unit.keyDates) {
      const match = date.match(problemSetNumberRe)
      if (!match) continue
      const id = psets.get(Number(match[1]))
      if (id && !unit.due.includes(id)) unit.due.push(id)
    }
  }
}

function courseStudy(
  pages: CoursePage[],
  resources: CourseResource[],
): CourseStudy | null {
  const calendar = pages.find(page => slugTitle(page.title) === 'calendar')
  const study = (calendar && parseCalendarUnits(calendar.content)) || unitsFromResources(resources)
  if (!study) return null
  attachUnitResources(study, resources)
  return study
}

function yamlFlow(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(yamlFlow).join(', ')}]`
  if (typeof value === 'number') return String(value)
  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>).filter(([, entry]) =>
      Array.isArray(entry) ? entry.length > 0 : entry !== '' && entry !== null && entry !== 0
    )
    return `{${entries.map(([key, entry]) => `${key}: ${yamlFlow(entry)}`).join(', ')}}`
  }
  return yamlString(String(value))
}

// The study block is one flow-style line per unit, so a 38-lecture course adds ~45 lines.
function courseStudyYaml(context: CourseContext, data: JsonRecord, study: CourseStudy): string[] {
  const instructors = recordList(data, 'instructors')
    .map(instructor => normalizeText(stringValue(instructor, 'title')).toLowerCase())
    .filter(value => value.length > 0)
  const course: Record<string, unknown> = {
    number: context.number || stringValue(data, 'site_short_id'),
    name: context.title,
    term: `${normalizeText(stringValue(data, 'term')).toLowerCase()} ${stringValue(data, 'year')}`.trim(),
    level: stringList(data, 'level').map(value => value.toLowerCase()).join(', '),
    instructors,
    source: context.sourceUrl,
    unitKind: study.unitKind,
    parts: study.parts,
    trailing: study.trailing,
  }
  const lines = ['course:']
  for (const [key, value] of Object.entries(course)) {
    if (Array.isArray(value) ? value.length === 0 : value === '') continue
    lines.push(`  ${key}: ${yamlFlow(value)}`)
  }
  lines.push('  units:')
  for (const unit of study.units) lines.push(`    - ${yamlFlow(unit)}`)
  return lines
}

function courseCollections(
  context: CourseContext,
  data: JsonRecord,
  pages: CoursePage[],
  resources: CourseResource[],
): CourseCollection[] {
  const collections: CourseCollection[] = resourceTypesInOrder(data, resources).map(
    resourceType => {
      const title = normalizeTitle(resourceType)
      return {
        relDir: resourceCollectionRelDir(resourceType),
        title,
        description: `${context.label} ${title}`,
        resources: resources.filter(resource => resource.resourceTypes.includes(resourceType)),
        extraLinks: [],
      }
    },
  )

  const assignmentPage = pages.find(page => slugTitle(page.title) === 'assignments')
  if (assignmentPage) {
    const relDir = resourceCollectionRelDir('Assignments')
    const pageLink = `- ${resourceLink(context, assignmentPage.relDir, assignmentPage.title)}`
    const problemSetLink = collections.some(
      collection => collection.relDir === resourceCollectionRelDir('Problem Sets'),
    )
      ? `- ${resourceLink(context, resourceCollectionRelDir('Problem Sets'), 'problem sets')}`
      : ''
    const existing = collections.find(collection => collection.relDir === relDir)
    if (existing) {
      existing.extraLinks = uniqueValues([pageLink, problemSetLink, ...existing.extraLinks])
    } else {
      collections.push({
        relDir,
        title: 'assignments',
        description: `${context.label} assignments`,
        resources: [],
        extraLinks: uniqueValues([pageLink, problemSetLink]),
      })
    }
  }

  return collections.filter(
    collection => collection.resources.length > 0 || collection.extraLinks.length > 0,
  )
}

function indexLinkableDirs(
  context: CourseContext,
  pages: CoursePage[],
  resources: CourseResource[],
  collections: CourseCollection[],
): void {
  context.linkableDirs.clear()
  context.linkableDirs.add('')
  context.linkableDirs.add('pages')
  for (const page of pages) context.linkableDirs.add(page.relDir)
  for (const resource of resources) context.linkableDirs.add(resource.relDir)
  for (const collection of collections) context.linkableDirs.add(collection.relDir)
}

async function courseHome(
  context: CourseContext,
  data: JsonRecord,
  pages: CoursePage[],
  collections: CourseCollection[],
  study: CourseStudy | null,
): Promise<string> {
  const description = normalizeText(
    stringValue(data, 'course_description') || stringValue(data, 'course_description_html'),
  )
  const image = path.posix.basename(stringValue(data, 'image_src'))
  const instructors = recordList(data, 'instructors')
    .map(instructor => stringValue(instructor, 'title'))
    .filter(value => value.length > 0)
  const pageLinks = pages
    .sort(compareTitle)
    .map(page => `- ${resourceLink(context, page.relDir, page.title)}`)
  const collectionLinks = collections.map(
    collection => `- ${resourceLink(context, collection.relDir, collection.title)}`,
  )
  const head = courseFrontmatter(
    context,
    {
      title: `${context.label}: ${context.title}`,
      description,
      id: context.idPrefix,
      tags: context.tags,
      aliases: context.aliases,
    },
    study ? 'course' : ocwLayout,
  )
  // A course with units renders the study spine; the frontmatter closes after the study block.
  const front = study
    ? [...head.split('\n').slice(0, -1), ...courseStudyYaml(context, data, study), '---'].join('\n')
    : head
  const body = [
    front,
    '',
    assetEmbed(context, image),
    '',
    description,
    '',
    '## course metadata',
    '',
    `- course: \`${context.number || stringValue(data, 'site_short_id')}\``,
    `- term: ${normalizeText(stringValue(data, 'term')).toLowerCase()} ${stringValue(data, 'year')}`,
    `- level: ${stringList(data, 'level')
      .map(value => value.toLowerCase())
      .join(', ')}`,
    `- instructor: ${instructors.join(', ')}`,
    `- topics: ${topicSummary(data)}`,
    `- license: [${context.licenseName.toLowerCase()}](${context.licenseUrl})`,
    `- source: [mit opencourseware](${context.sourceUrl})`,
    '',
    '## pages',
    '',
    ...pageLinks,
    '',
    '## resources',
    '',
    ...collectionLinks,
  ]
  return markdownDocument(body)
}

async function pageMarkdown(context: CourseContext, page: CoursePage): Promise<string> {
  const content = await htmlToMarkdown(context, page.content)
  return markdownDocument([
    courseFrontmatter(context, {
      title: page.title,
      description: page.description || `${context.label} ${page.title}`,
      id: metadataId(context, 'page', page.relDir),
      tags: uniqueValues([...context.tags, ...learningTypesForFrontmatter(page.resourceTypes)]),
    }),
    '',
    `up: ${resourceLink(context, '', context.label)}`,
    '',
    content,
  ])
}

async function resourceMarkdown(context: CourseContext, resource: CourseResource): Promise<string> {
  const content = await htmlToMarkdown(context, resource.content)
  const video = resource.youtubeId.length > 0
  const watchUrl = `https://www.youtube.com/watch?v=${resource.youtubeId}`
  const source = video
    ? new URL(`${resource.relDir}/`, context.sourceUrl).toString()
    : originalAssetUrl(context, resource.originalFile, resource.localFile)
  const embed = video
    ? `![${resource.title}](${watchUrl})`
    : assetEmbed(context, resource.localFile)
  const assetLine = video
    ? `- youtube: [${resource.youtubeId}](${watchUrl})`
    : `- file: [[${context.slug}/static_resources/${resource.localFile}|${resource.localFile}]]`
  return markdownDocument([
    courseFrontmatter(context, {
      title: resource.title,
      description: resource.description || `${context.label} ${resource.title}`,
      id: metadataId(context, 'resource', resource.relDir),
      tags: uniqueValues([...context.tags, ...learningTypesForFrontmatter(resource.resourceTypes)]),
    }),
    '',
    `up: ${resourceLink(context, '', context.label)}`,
    '',
    embed,
    '',
    content || resource.description,
    '',
    '## metadata',
    '',
    `- type: ${resource.resourceType || 'resource'}`,
    assetLine,
    `- source: [mit opencourseware](${source})`,
  ])
}

function collectionMarkdown(context: CourseContext, collection: CourseCollection): string {
  const links = [
    ...collection.extraLinks,
    ...collection.resources
      .sort(compareTitle)
      .map(resource => `- ${resourceLink(context, resource.relDir, resource.title)}`),
  ]
  return markdownDocument([
    courseFrontmatter(context, {
      title: collection.title,
      description: collection.description,
      id: metadataId(context, 'collection', collection.relDir),
      tags: context.tags,
    }),
    '',
    `up: ${resourceLink(context, '', context.label)}`,
    '',
    ...links,
  ])
}

async function pathExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath)
    return true
  } catch {
    return false
  }
}

async function removeGeneratedCollectionMarkdown(context: CourseContext): Promise<void> {
  const files = await globby(['pages/index.md', 'resources/*/index.md'], {
    cwd: context.root,
    onlyFiles: true,
  })
  for (const file of files) {
    const dataPath = path.join(context.root, path.dirname(file), 'data.json')
    if (!(await pathExists(dataPath))) {
      await fs.rm(path.join(context.root, file), { force: true })
    }
  }
}

async function removeDeadOcwShell(context: CourseContext): Promise<void> {
  const files = await globby(
    [
      'index.html',
      'download/index.html',
      'download/index.xml',
      'pages/**/index.html',
      'resources/**/index.html',
      'resources/**/index.xml',
      'external-resources/**/*.html',
      'external-resources/**/*.xml',
      'static_resources/index.html',
      'sitemap.xml',
      'favicon.ico',
    ],
    { cwd: context.root, dot: true, onlyFiles: true },
  )
  for (const file of files) {
    await fs.rm(path.join(context.root, file), { force: true })
  }
  await fs.rm(path.join(context.root, 'static_shared'), { recursive: true, force: true })
}

function isCoursePage(relDir: string, contentType: string): boolean {
  if (contentType === 'page') return relDir !== 'pages' && relDir !== 'resources'
  return relDir.startsWith('pages/') && relDir !== 'pages'
}

function isCourseResource(relDir: string, data: JsonRecord, contentType: string): boolean {
  if (contentType === 'resource') return true
  return relDir.startsWith('resources/') && stringValue(data, 'file').length > 0
}

async function fetchText(url: string): Promise<string> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`fetch ${url} -> ${response.status}`)
  return response.text()
}

function isCourseUrl(input: string): boolean {
  const candidate = /^https?:\/\//i.test(input)
    ? input
    : input.startsWith(`${ocwHost}/`)
      ? `https://${input}`
      : ''
  if (candidate.length === 0) return false
  try {
    return new URL(candidate).hostname === ocwHost
  } catch {
    return false
  }
}

function courseSlug(input: string): string {
  const normalized = /^https?:\/\//i.test(input) ? input : `https://${input}`
  const match = new URL(normalized).pathname.match(/\/courses\/([^/]+)/)
  if (!match) throw new Error(`no OCW course slug found in: ${input}`)
  return match[1]
}

function isVideoAsset(data: JsonRecord, file: string): boolean {
  if (stringValue(data, 'file_type').toLowerCase().startsWith('video/')) return true
  if (stringValue(data, 'resourcetype').toLowerCase() === 'video') return true
  return /\.(mp4|m4v|mov|webm)$/i.test(strippedUrlPath(file))
}

async function downloadAsset(courseRoot: string, file: string): Promise<void> {
  const target = path.join(
    courseRoot,
    'static_resources',
    path.posix.basename(strippedUrlPath(file)),
  )
  if (await pathExists(target)) return
  const response = await fetch(new URL(file.replace(/^\/+/, ''), mitBaseUrl).toString())
  if (!response.ok) throw new Error(`asset ${file} -> ${response.status}`)
  await fs.mkdir(path.dirname(target), { recursive: true })
  await fs.writeFile(target, Buffer.from(await response.arrayBuffer()))
}

async function downloadNodeAsset(courseRoot: string, data: JsonRecord): Promise<void> {
  const file = stringValue(data, 'file')
  if (file.length === 0 || isVideoAsset(data, file)) return
  await downloadAsset(courseRoot, file)
}

async function writeFetchedJson(
  courseRoot: string,
  relPath: string,
  body: string,
): Promise<JsonRecord> {
  const target = path.join(courseRoot, relPath)
  await fs.mkdir(path.dirname(target), { recursive: true })
  await fs.writeFile(target, body, 'utf8')
  const parsed: unknown = JSON.parse(body)
  return isRecord(parsed) ? parsed : {}
}

async function mapWithConcurrency<T>(
  items: T[],
  limit: number,
  run: (item: T) => Promise<void>,
): Promise<void> {
  let cursor = 0
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (cursor < items.length) {
      await run(items[cursor++])
    }
  })
  await Promise.all(workers)
}

// Renamed course directories are frozen: their names sit inside every stored deck slug.
async function existingRootForSite(siteUid: string): Promise<string | null> {
  if (siteUid.length === 0) return null
  const files = await globby(['*/data.json'], { cwd: defaultCourseRoot, onlyFiles: true })
  for (const file of files) {
    const data = await readJson(path.join(defaultCourseRoot, file))
    if (stringValue(data, 'site_uid') === siteUid) {
      return path.join(defaultCourseRoot, path.dirname(file))
    }
  }
  return null
}

async function fetchCourse(input: string): Promise<string> {
  const slug = courseSlug(input)
  const base = new URL(`courses/${slug}/`, mitBaseUrl).toString()
  const rootText = await fetchText(`${base}data.json`)
  const parsedRoot: unknown = JSON.parse(rootText)
  const siteUid = isRecord(parsedRoot) ? stringValue(parsedRoot, 'site_uid') : ''
  const courseRoot =
    (await existingRootForSite(siteUid)) ?? path.join(defaultCourseRoot, slug)
  await fs.mkdir(courseRoot, { recursive: true })

  const rootData = await writeFetchedJson(courseRoot, 'data.json', rootText)
  const contentMapText = await fetchText(`${base}content_map.json`)
  await fs.writeFile(path.join(courseRoot, 'content_map.json'), contentMapText, 'utf8')
  const imageMeta = nestedRecord(rootData, 'course_image_metadata')
  if (imageMeta) await downloadNodeAsset(courseRoot, imageMeta)

  const contentMap: unknown = JSON.parse(contentMapText)
  const prefix = `courses/${slug}/`
  const nodePaths = isRecord(contentMap)
    ? Array.from(
        new Set(
          Object.values(contentMap)
            .filter((value): value is string => typeof value === 'string')
            .map(value => value.replace(/^\/+/, ''))
            .filter(value => value.startsWith(prefix))
            .map(value => value.slice(prefix.length)),
        ),
      )
    : []

  await mapWithConcurrency(nodePaths, fetchConcurrency, async relPath => {
    const data = await writeFetchedJson(courseRoot, relPath, await fetchText(`${base}${relPath}`))
    await downloadNodeAsset(courseRoot, data)
  })

  return courseRoot
}

async function vendorCourse(courseRoot: string): Promise<void> {
  const rootData = await readJson(path.join(courseRoot, 'data.json'))
  const context = courseContext(courseRoot, rootData)
  const dataFiles = await globby(['**/data.json'], {
    cwd: context.root,
    ignore: ['data.json', 'static_resources/data.json'],
  })
  const pages: CoursePage[] = []
  const resources: CourseResource[] = []

  for (const dataFile of dataFiles.sort()) {
    const relDir = path.posix.dirname(dataFile)
    const data = await readJson(path.join(context.root, dataFile))
    const contentType = stringValue(data, 'content_type')
    if (isCourseResource(relDir, data, contentType)) {
      resources.push(parseResource(context, relDir, data))
    } else if (isCoursePage(relDir, contentType)) {
      pages.push(parsePage(relDir, data))
    }
  }

  const collections = courseCollections(context, rootData, pages, resources)
  indexLinkableDirs(context, pages, resources, collections)

  await removeGeneratedCollectionMarkdown(context)
  const study = courseStudy(pages, resources)
  await writeMarkdown(
    context,
    '',
    await courseHome(context, rootData, pages, collections, study),
  )

  for (const page of pages) {
    await writeMarkdown(context, page.relDir, await pageMarkdown(context, page))
  }

  for (const resource of resources) {
    await writeMarkdown(context, resource.relDir, await resourceMarkdown(context, resource))
  }

  await writeMarkdown(
    context,
    'pages',
    collectionMarkdown(context, {
      relDir: 'pages',
      title: 'course pages',
      description: `${context.label} course pages`,
      resources: [],
      extraLinks: pages
        .sort(compareTitle)
        .map(page => `- ${resourceLink(context, page.relDir, page.title)}`),
    }),
  )

  for (const collection of collections) {
    await writeMarkdown(context, collection.relDir, collectionMarkdown(context, collection))
  }

  await removeDeadOcwShell(context)
}

async function courseRootsFromPath(inputPath: string): Promise<string[]> {
  if (await pathExists(path.join(inputPath, 'data.json'))) {
    return [inputPath]
  }
  const files = await globby(['*/data.json'], { cwd: inputPath, onlyFiles: true })
  return files.map(file => path.join(inputPath, path.dirname(file))).sort()
}

async function resolveCourseInput(input: string): Promise<string[]> {
  if (isCourseUrl(input)) return [await fetchCourse(input)]
  return courseRootsFromPath(input)
}

async function courseRoots(inputs: string[]): Promise<string[]> {
  const roots = await Promise.all(
    (inputs.length > 0 ? inputs : [defaultCourseRoot]).map(resolveCourseInput),
  )
  return uniqueValues(roots.flat().map(root => path.normalize(root)))
}

async function main(): Promise<void> {
  const roots = await courseRoots(process.argv.slice(2))
  if (roots.length === 0) {
    throw new Error('No MIT OCW course roots found')
  }
  for (const root of roots) {
    await vendorCourse(root)
  }
}

main().catch(error => {
  console.error(error)
  process.exit(1)
})
