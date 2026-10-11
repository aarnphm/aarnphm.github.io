import { globby } from 'globby'
import { fromHtml } from 'hast-util-from-html'
import { toMdast } from 'hast-util-to-mdast'
import { gfmToMarkdown } from 'mdast-util-gfm'
import { toMarkdown } from 'mdast-util-to-markdown'
import { execFile } from 'node:child_process'
import fs from 'node:fs/promises'
import path from 'node:path'
import { promisify } from 'node:util'
import { isRecord } from '../util/type-guards'
import { parseVtt, transcriptMarkdown } from './ocw-vtt'
import { convertLatexCourse, LatexGateError, type LatexResult } from './vendor-ocw-latex'

const run = promisify(execFile)

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

// First match in `resourceKind` wins; the order is the one in docs/plans/course-resources.md §1.
type ResourceKind =
  | 'video'
  | 'captions'
  | 'transcript'
  | 'exam-solution'
  | 'exam'
  | 'pset-solution'
  | 'pset'
  | 'tex'
  | 'archive'
  | 'image'
  | 'notes'
  | 'notes-full'
  | 'chapter'
  | 'reading'
  | 'other'

type Sidecar = { video: string; role: 'captions' | 'transcript' }

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
  fileType: string
  bytes: number
  captionsFile: string
  transcriptFile: string
  kind: ResourceKind
  sidecar: Sidecar | null
  n: number | null
  due: number | null
  duration: number
  cues: number
  transcript: string
}

type CourseCollection = {
  relDir: string
  title: string
  type: string
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
  resources: CourseUnitResource[]
  due: string[]
}

type CourseUnitResource = {
  id: string
  file?: string
  youtube?: string
  duration?: number
  cues?: number
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
  return value.replace(/\.(pdf|jpg|jpeg|png|gif|mp3|mp4|zip|tex|vtt|html?)$/i, '')
}

function normalizeTitle(value: string): string {
  const spaced = titleWithoutExtension(value)
    .replace(/LaTeX/g, 'latex')
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
  // captions are never lecture notes, whatever their file name says
  if (stringValue(data, 'file_type').toLowerCase() === 'text/vtt') return []
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

// `{flow}` values are written as one YAML flow line (the `resource:` contract).
type FrontmatterValue = string | string[] | boolean | { flow: unknown }

function frontmatter(fields: Record<string, FrontmatterValue>): string {
  const lines: string[] = ['---']
  for (const [key, value] of Object.entries(fields)) {
    if (Array.isArray(value)) {
      lines.push(...yamlList(key, value))
    } else if (typeof value === 'boolean') {
      lines.push(`${key}: ${value}`)
    } else if (typeof value === 'object') {
      lines.push(`${key}: ${yamlFlow(value.flow)}`)
    } else if (value.length > 0) {
      lines.push(`${key}: ${yamlString(value)}`)
    }
  }
  lines.push('---')
  return lines.join('\n')
}

function courseFrontmatter(
  context: CourseContext,
  fields: Record<string, FrontmatterValue>,
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
  const size = data.file_size
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
    fileType: stringValue(data, 'file_type').toLowerCase(),
    bytes: typeof size === 'number' && Number.isFinite(size) ? size : 0,
    captionsFile: path.posix.basename(nestedString(data, 'video_files', 'video_captions_file')),
    transcriptFile: path.posix.basename(nestedString(data, 'video_files', 'video_transcript_file')),
    kind: 'other',
    sidecar: null,
    n: null,
    due: null,
    duration: 0,
    cues: 0,
    transcript: '',
  }
}

function resourceId(resource: CourseResource): string {
  return path.posix.basename(resource.relDir)
}

/** Captions and transcript files named by a video's `video_files` belong to that video. */
function sidecarIndex(resources: CourseResource[]): Map<string, Sidecar> {
  const sidecars = new Map<string, Sidecar>()
  for (const resource of resources) {
    if (resource.youtubeId.length === 0) continue
    const video = resourceId(resource)
    if (resource.captionsFile.length > 0) {
      sidecars.set(resource.captionsFile, { video, role: 'captions' })
    }
    if (resource.transcriptFile.length > 0) {
      sidecars.set(resource.transcriptFile, { video, role: 'transcript' })
    }
  }
  return sidecars
}

const examIdRe = /exam|midterm|final|quiz/i
const chapterNumberRe = /\bchapter[\s_-]*0*(\d+)\b/i

function resourceKind(resource: CourseResource): ResourceKind {
  const id = resourceId(resource)
  const types = resource.resourceTypes
  const file = resource.localFile
  if (resource.youtubeId.length > 0) return 'video'
  if (
    resource.sidecar?.role === 'captions' ||
    resource.fileType === 'text/vtt' ||
    /\.vtt$/i.test(file)
  )
    return 'captions'
  if (resource.sidecar?.role === 'transcript' || /transcript/i.test(id)) return 'transcript'
  if (types.includes('Exam Solutions') || (examIdRe.test(id) && /sol(?:utions?)?$/i.test(id)))
    return 'exam-solution'
  if (types.includes('Exams') || examIdRe.test(id)) return 'exam'
  if (types.includes('Problem Set Solutions') || /pset\d+_?sol/i.test(id)) return 'pset-solution'
  if (/\.tex$/i.test(file)) return 'tex'
  if (/\.(zip|tar|gz|tgz)$/i.test(file)) return 'archive'
  if (resource.fileType.startsWith('image/') || /\.(jpe?g|png|gif|svg|webp)$/i.test(file))
    return 'image'
  if (
    types.includes('Problem Sets') ||
    types.includes('Assignments') ||
    problemSetNumber(resource) !== null
  )
    return 'pset'
  if (lectureNumber(resource) !== null) return 'notes'
  if (types.includes('Lecture Notes') && /full|lecture_notes$/i.test(id)) return 'notes-full'
  if (types.includes('Lecture Notes')) return 'chapter'
  if (types.includes('Readings')) return 'reading'
  return 'other'
}

function resourceNumber(resource: CourseResource): number | null {
  switch (resource.kind) {
    case 'video':
    case 'transcript':
    case 'notes':
      return lectureNumber(resource)
    case 'pset':
    case 'pset-solution':
    case 'tex':
      return problemSetNumber(resource) ?? lectureNumber(resource)
    case 'chapter': {
      const match =
        resourceId(resource).match(chapterNumberRe) ?? resource.title.match(chapterNumberRe)
      return match ? Number(match[1]) : null
    }
    default:
      return null
  }
}

async function classifyResources(
  context: CourseContext,
  resources: CourseResource[],
): Promise<void> {
  const sidecars = sidecarIndex(resources)
  for (const resource of resources) {
    resource.sidecar = sidecars.get(resource.localFile) ?? null
    if (resource.sidecar || resource.fileType === 'text/vtt') resource.resourceTypes = []
    resource.kind = resourceKind(resource)
    resource.n = resourceNumber(resource)
  }
  const summary = { written: 0, unreadable: 0, missing: 0 }
  for (const resource of resources) {
    if (resource.kind !== 'video') continue
    const captionsPath = resource.captionsFile
      ? path.join(context.root, 'static_resources', resource.captionsFile)
      : ''
    if (captionsPath.length === 0 || !(await pathExists(captionsPath))) {
      summary.missing++
      continue
    }
    const { cues, skipped } = parseVtt(await fs.readFile(captionsPath, 'utf8'))
    if (cues.length === 0) {
      console.warn(
        `${context.slug}: captions ${resource.captionsFile} unreadable (${skipped} blocks skipped)`,
      )
      summary.unreadable++
      continue
    }
    resource.cues = cues.length
    resource.duration = Math.ceil(cues[cues.length - 1].end)
    resource.transcript = transcriptMarkdown(cues, resource.youtubeId, resource.duration)
    summary.written++
  }
  if (resources.some(resource => resource.kind === 'video')) {
    console.log(
      `${context.slug}: transcripts: ${summary.written} written, ${summary.unreadable} unreadable, ${summary.missing} missing`,
    )
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

function courseStudy(pages: CoursePage[], resources: CourseResource[]): CourseStudy | null {
  const calendar = pages.find(page => slugTitle(page.title) === 'calendar')
  const study = (calendar && parseCalendarUnits(calendar.content)) || unitsFromResources(resources)
  if (!study) return null
  attachUnitResources(study, resources)
  return study
}

function yamlFlow(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(yamlFlow).join(', ')}]`
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>).filter(([, entry]) =>
      Array.isArray(entry)
        ? entry.length > 0
        : entry !== '' && entry !== null && entry !== undefined && entry !== 0 && entry !== false,
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
    level: stringList(data, 'level')
      .map(value => value.toLowerCase())
      .join(', '),
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

function compareResources(left: CourseResource, right: CourseResource): number {
  const rank = (resource: CourseResource) =>
    resource.kind === 'pset' || resource.kind === 'pset-solution' ? 1 : 0
  if (left.n !== null && right.n !== null && rank(left) === rank(right) && left.n !== right.n)
    return left.n - right.n
  if (left.n !== null && right.n === null) return -1
  if (left.n === null && right.n !== null) return 1
  if (rank(left) !== rank(right)) return rank(left) - rank(right)
  return compareTitle(left, right)
}

async function listDescription(context: CourseContext, resourceType: string): Promise<string> {
  const file = path.join(context.root, 'lists', slugTitle(resourceType), 'data.json')
  if (!(await pathExists(file))) return ''
  return normalizeText(stringValue(await readJson(file), 'description'))
}

async function courseCollections(
  context: CourseContext,
  data: JsonRecord,
  pages: CoursePage[],
  resources: CourseResource[],
  latex: LatexResult | null,
): Promise<CourseCollection[]> {
  const collections: CourseCollection[] = []
  for (const resourceType of resourceTypesInOrder(data, resources)) {
    const title = normalizeTitle(resourceType)
    collections.push({
      relDir: resourceCollectionRelDir(resourceType),
      title,
      type: resourceType,
      description: (await listDescription(context, resourceType)) || `${context.label} ${title}`,
      resources: resources
        .filter(resource => resource.resourceTypes.includes(resourceType))
        .sort(compareResources),
      extraLinks: [],
    })
  }
  if (latex) {
    const editable = collections.find(
      collection => collection.relDir === resourceCollectionRelDir('Editable Files'),
    )
    if (editable) {
      editable.extraLinks = [
        `- [[${context.slug}/notes/|rendered lecture notes]]`,
        ...latex.psets.map(note => `- [[${note.slug}|rendered ${note.title}]]`),
      ]
    }
  }

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
        type: 'Assignments',
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

function formatBytes(bytes: number): string {
  if (bytes <= 0) return ''
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} kb`
  return `${(bytes / (1024 * 1024)).toFixed(1)} mb`
}

function fileLine(context: CourseContext, resource: CourseResource): string {
  return `- file: [[${context.slug}/static_resources/${resource.localFile}|${resource.localFile}]]`
}

async function archiveFiles(context: CourseContext, resource: CourseResource): Promise<string[]> {
  const file = path.join(context.root, 'static_resources', resource.localFile)
  if (!/\.zip$/i.test(resource.localFile) || !(await pathExists(file))) return []
  try {
    const { stdout } = await run('unzip', ['-Z1', file], { maxBuffer: 8 * 1024 * 1024 })
    return stdout
      .split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0 && !line.startsWith('__MACOSX/') && !line.endsWith('/'))
      .filter(line => !/(^|\/)\.DS_Store$/.test(line))
      .sort()
  } catch {
    return []
  }
}

function resourceFlow(
  resource: CourseResource,
  byId: Map<string, CourseResource>,
  latex: LatexResult | null,
): Record<string, unknown> {
  const file = resource.localFile ? path.posix.join('static_resources', resource.localFile) : ''
  const base = { kind: resource.kind, file, bytes: resource.bytes }
  switch (resource.kind) {
    case 'video': {
      const captions = resource.captionsFile
      const transcript = resource.transcriptFile
      const captionsId = Array.from(byId.values()).find(
        other => other.kind === 'captions' && other.localFile === captions,
      )
      const transcriptId = Array.from(byId.values()).find(
        other => other.kind === 'transcript' && other.localFile === transcript,
      )
      return {
        kind: 'video',
        n: resource.n,
        youtube: resource.youtubeId,
        duration: resource.duration,
        cues: resource.cues,
        captions: captions ? path.posix.join('static_resources', captions) : '',
        captionsId: captionsId ? resourceId(captionsId) : '',
        transcript: transcript ? path.posix.join('static_resources', transcript) : '',
        transcriptId: transcriptId ? resourceId(transcriptId) : '',
      }
    }
    case 'captions':
      return { ...base, of: resource.sidecar?.video ?? '' }
    case 'transcript':
      return { ...base, of: resource.sidecar?.video ?? '', n: resource.n }
    case 'pset':
      return { ...base, n: resource.n, due: resource.due }
    case 'pset-solution':
    case 'notes':
    case 'chapter':
      return { ...base, n: resource.n }
    case 'tex': {
      const notes = latex?.pointers.get(resource.relDir) ?? ''
      const full =
        resource.resourceTypes.includes('Lecture Notes') && problemSetNumber(resource) === null
      return full
        ? { ...base, full: true, notes }
        : { ...base, n: problemSetNumber(resource), notes }
    }
    default:
      return base
  }
}

async function resourceMarkdown(
  context: CourseContext,
  resource: CourseResource,
  byId: Map<string, CourseResource>,
  latex: LatexResult | null,
): Promise<string> {
  const content = await htmlToMarkdown(context, resource.content)
  const video = resource.kind === 'video'
  const source = video
    ? new URL(`${resource.relDir}/`, context.sourceUrl).toString()
    : originalAssetUrl(context, resource.originalFile, resource.localFile)
  const flow = resourceFlow(resource, byId, latex)
  const description =
    resource.description ||
    (video ? content.split('\n\n')[0] : '') ||
    `${context.label} ${resource.title}`
  const fields: Record<string, FrontmatterValue> = {
    title: resource.title,
    description,
    id: metadataId(context, 'resource', resource.relDir),
    tags: uniqueValues([...context.tags, ...learningTypesForFrontmatter(resource.resourceTypes)]),
  }
  if (resource.kind === 'captions' || resource.kind === 'transcript') fields.noindex = true
  if (video) fields.semantic = false
  fields.resource = { flow }
  const head = courseFrontmatter(context, fields, 'resource')
  const up = `up: ${resourceLink(context, '', context.label)}`
  const metadata = (lines: string[]) => [
    '## metadata',
    '',
    `- type: ${resource.resourceType || 'resource'}`,
    ...lines,
    `- source: [mit opencourseware](${source})`,
  ]

  if (video) {
    const transcript =
      resource.transcript.length > 0
        ? resource.transcript
        : `_no transcript: captions for this lecture are not vendored._`
    return markdownDocument([head, '', transcript])
  }
  if (resource.kind === 'captions') {
    const of = resource.sidecar?.video ?? ''
    const watch = of ? byId.get(of) : undefined
    const label =
      watch && watch.n !== null
        ? `lecture ${String(watch.n).padStart(2, '0')}`
        : 'the lecture video'
    return markdownDocument([
      head,
      '',
      up,
      '',
      of
        ? `captions of [[${context.slug}/resources/${of}/#transcript|${label}]], ${formatBytes(resource.bytes) || 'webvtt'}.`
        : `captions file, ${formatBytes(resource.bytes) || 'webvtt'}.`,
      '',
      ...metadata([fileLine(context, resource)]),
    ])
  }
  if (resource.kind === 'tex') {
    const notes = typeof flow.notes === 'string' ? flow.notes : ''
    const rendered = notes
      ? flow.full === true
        ? `rendered: [[${notes}|lecture notes, one page per lecture]]`
        : `rendered: [[${notes}|${latex?.psets.find(note => note.slug === notes)?.title ?? 'problem set'}]]`
      : ''
    return markdownDocument([
      head,
      '',
      up,
      '',
      rendered,
      '',
      content || resource.description,
      '',
      ...metadata([fileLine(context, resource)]),
    ])
  }
  if (resource.kind === 'archive') {
    const files = await archiveFiles(context, resource)
    return markdownDocument([
      head,
      '',
      up,
      '',
      content || resource.description,
      '',
      ...(files.length > 0 ? ['## files', '', ...files.map(file => `- \`${file}\``), ''] : []),
      ...metadata([
        `- file: ${resource.localFile} (${formatBytes(resource.bytes) || 'zip'}, [download](${source}))`,
      ]),
    ])
  }
  return markdownDocument([
    head,
    '',
    up,
    '',
    resource.kind === 'image' ? assetEmbed(context, resource.localFile) : '',
    '',
    content || resource.description,
    '',
    ...metadata([fileLine(context, resource)]),
  ])
}

function collectionMarkdown(context: CourseContext, collection: CourseCollection): string {
  const links = [
    ...collection.extraLinks,
    ...collection.resources.map(
      resource => `- ${resourceLink(context, resource.relDir, resource.title)}`,
    ),
  ]
  const pageLinks = collection.extraLinks
    .map(link => /\[\[([^\]|]+)\|/.exec(link)?.[1] ?? '')
    .filter(target => target.length > 0)
  return markdownDocument([
    courseFrontmatter(
      context,
      {
        title: collection.title,
        description: collection.description,
        id: metadataId(context, 'collection', collection.relDir),
        tags: context.tags,
        resource: {
          flow: {
            kind: 'collection',
            type: collection.type,
            items: collection.resources.map(resourceId),
            links: pageLinks,
          },
        },
      },
      'resource',
    ),
    '',
    `up: ${resourceLink(context, '', context.label)}`,
    '',
    ...links,
  ])
}

function resourcesIndexMarkdown(context: CourseContext, collections: CourseCollection[]): string {
  return markdownDocument([
    courseFrontmatter(
      context,
      {
        title: 'resources',
        description: `${context.label} resources by type`,
        id: metadataId(context, 'collection', 'resources'),
        tags: context.tags,
        resource: { flow: { kind: 'index' } },
      },
      'resource',
    ),
    '',
    `up: ${resourceLink(context, '', context.label)}`,
    '',
    ...collections.map(
      collection => `- ${resourceLink(context, collection.relDir, collection.title)}`,
    ),
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
  const files = await globby(['pages/index.md', 'resources/index.md', 'resources/*/index.md'], {
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
  const captions = nestedString(data, 'video_files', 'video_captions_file')
  if (captions.length > 0) {
    if (/^https?:\/\//.test(captions) || /\.webvtt$/i.test(captions)) {
      console.warn(`captions not vendored: ${captions}`)
    } else {
      await downloadAsset(courseRoot, captions)
    }
  }
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
  const courseRoot = (await existingRootForSite(siteUid)) ?? path.join(defaultCourseRoot, slug)
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

  await classifyResources(context, resources)
  const study = courseStudy(pages, resources)
  const latex = await latexNotes(context, rootData, study, resources)
  const collections = await courseCollections(context, rootData, pages, resources, latex)
  indexLinkableDirs(context, pages, resources, collections)

  await removeGeneratedCollectionMarkdown(context)
  await writeMarkdown(context, '', await courseHome(context, rootData, pages, collections, study))

  for (const page of pages) {
    await writeMarkdown(context, page.relDir, await pageMarkdown(context, page))
  }

  const byId = new Map(resources.map(resource => [resourceId(resource), resource]))
  for (const resource of resources) {
    await writeMarkdown(
      context,
      resource.relDir,
      await resourceMarkdown(context, resource, byId, latex),
    )
  }
  await writeMarkdown(context, 'resources', resourcesIndexMarkdown(context, collections))

  await writeMarkdown(
    context,
    'pages',
    collectionMarkdown(context, {
      relDir: 'pages',
      title: 'course pages',
      type: 'Pages',
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

// LaTeX sources become `notes/`; the converter's gates fail the whole course rather than ship a wrong page.
async function latexNotes(
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
