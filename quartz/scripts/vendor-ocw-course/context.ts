import { globby } from 'globby'
import fs from 'node:fs/promises'
import path from 'node:path'
import { isRecord } from '../../util/type-guards'

export type JsonRecord = Record<string, unknown>

export type CourseContext = {
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
  // OCW site path -> content slug of every vendored course, for links between courses
  vendoredCourses: Map<string, string>
}

export type CoursePage = {
  relDir: string
  title: string
  description: string
  content: string
  resourceTypes: string[]
}

// First match in `resourceKind` wins; the order is the one in docs/plans/course-resources.md §1.
export type ResourceKind =
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

export type Sidecar = { video: string; role: 'captions' | 'transcript' }

export type CourseResource = {
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

export type CourseCollection = {
  relDir: string
  title: string
  type: string
  description: string
  resources: CourseResource[]
  extraLinks: string[]
}

export type CourseUnitKind = 'lecture' | 'week' | 'session'

export type CourseUnit = {
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

export type CourseStudy = {
  unitKind: CourseUnitKind
  parts: string[]
  trailing: string[]
  units: CourseUnit[]
}

export const defaultCourseRoot = path.join('content', 'courses')
export const contentRoot = path.resolve('content')
export const mitBaseUrl = 'https://ocw.mit.edu/'

export const ocwLayout = 'A|L'
const defaultLicenseName = 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International'
const defaultLicenseUrl = 'https://creativecommons.org/licenses/by-nc-sa/4.0/'

export function stringValue(record: JsonRecord, key: string): string {
  const value = record[key]
  return typeof value === 'string' ? value : ''
}

export function stringList(record: JsonRecord, key: string): string[] {
  const value = record[key]
  if (!Array.isArray(value)) return []
  return value.filter(item => typeof item === 'string').map(normalizeText)
}

export function recordList(record: JsonRecord, key: string): JsonRecord[] {
  const value = record[key]
  if (!Array.isArray(value)) return []
  return value.filter(isRecord)
}

export function nestedStringLists(record: JsonRecord, key: string): string[][] {
  const value = record[key]
  if (!Array.isArray(value)) return []
  return value
    .filter(item => Array.isArray(item))
    .map(item => item.filter(child => typeof child === 'string').map(normalizeText))
    .filter(item => item.length > 0)
}

export function nestedRecord(record: JsonRecord, key: string): JsonRecord | undefined {
  const value = record[key]
  return isRecord(value) ? value : undefined
}

export function nestedString(record: JsonRecord, key: string, childKey: string): string {
  const child = nestedRecord(record, key)
  return child ? stringValue(child, childKey) : ''
}

export function firstString(values: string[]): string {
  return values.find(value => value.length > 0) ?? ''
}

export function normalizeText(value: string): string {
  return (
    value
      .replace(/\u00a0/g, ' ')
      .replace(/[“”]/g, '"')
      .replace(/[‘’]/g, "'")
      .replace(/[–—]/g, '-')
      .replace(/&amp;/g, '&')
      // trailing spaces only: `\s+\n` also ate the blank line between paragraphs
      .replace(/[^\S\n]+\n/g, '\n')
      .trim()
  )
}

export function titleWithoutExtension(value: string): string {
  return value.replace(/\.(pdf|jpg|jpeg|png|gif|mp3|mp4|zip|tex|vtt|html?)$/i, '')
}

export function normalizeTitle(value: string): string {
  const spaced = titleWithoutExtension(value)
    .replace(/LaTeX/g, 'latex')
    .replace(/commentsonstyle/i, 'comments on style')
    .replace(/erratafortop/i, 'errata for topology')
    .replace(/problem[\s_-]*set/i, 'problem set')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/[_-]+/g, ' ')
  return normalizeText(spaced).toLowerCase()
}

export function compareTitle(left: { title: string }, right: { title: string }): number {
  return left.title.localeCompare(right.title, undefined, { numeric: true })
}

export function slugTitle(value: string): string {
  return normalizeTitle(value)
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
}

export async function readJson(filePath: string): Promise<JsonRecord> {
  const raw: unknown = JSON.parse(await fs.readFile(filePath, 'utf8'))
  if (!isRecord(raw)) {
    throw new Error(`${filePath} does not contain a JSON object`)
  }
  return raw
}

export function strippedUrlPath(value: string): string {
  const [withoutHash] = value.split('#')
  const [withoutQuery] = withoutHash.split('?')
  return withoutQuery
}

export function uniqueValues(values: string[]): string[] {
  return Array.from(new Set(values.map(normalizeText).filter(value => value.length > 0)))
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

export function courseContext(courseRoot: string, data: JsonRecord): CourseContext {
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
    vendoredCourses: new Map(),
  }
}

export async function pathExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath)
    return true
  } catch {
    return false
  }
}

export async function vendoredCourses(): Promise<Map<string, string>> {
  const files = await globby(['*/data.json'], { cwd: defaultCourseRoot, onlyFiles: true })
  const courses = new Map<string, string>()
  for (const file of files) {
    const root = path.join(defaultCourseRoot, path.dirname(file))
    const site = sourcePath(await readJson(path.join(defaultCourseRoot, file)))
    if (site.length > 0) courses.set(site, contentSlug(root))
  }
  return courses
}
