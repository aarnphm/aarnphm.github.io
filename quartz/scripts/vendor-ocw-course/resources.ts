import fs from 'node:fs/promises'
import path from 'node:path'
import type { LatexResult } from '../vendor-ocw-latex'
import { parseVtt, transcriptMarkdown } from '../ocw-vtt'
import {
  compareTitle,
  firstString,
  nestedString,
  normalizeText,
  normalizeTitle,
  pathExists,
  readJson,
  slugTitle,
  stringList,
  stringValue,
  titleWithoutExtension,
  uniqueValues,
  type CourseCollection,
  type CourseContext,
  type CoursePage,
  type CourseResource,
  type JsonRecord,
  type ResourceKind,
  type Sidecar,
} from './context'
import { resourceLink } from './markdown'

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

export function parseResource(
  context: CourseContext,
  relDir: string,
  data: JsonRecord,
): CourseResource {
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

export function resourceId(resource: CourseResource): string {
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

export async function classifyResources(
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

export function parsePage(relDir: string, data: JsonRecord): CoursePage {
  const title = normalizeTitle(stringValue(data, 'title') || path.posix.basename(relDir))
  return {
    relDir,
    title,
    description: normalizeText(stringValue(data, 'description')),
    content: stringValue(data, 'content'),
    resourceTypes: stringList(data, 'learning_resource_types'),
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

// `_` is a word character, so `\b` misses `mit18_905f16_lec28`; the file name beats OCW titles
// like "Lecture 27 Products in Cohomology" on lec28.
const lectureNumberRe = /(?<![a-z])lec(?:ture)?[\s_#]*0*(\d+)\b/i
export const problemSetNumberRe = /\b(?:pset|problem set|ps)[\s_#]*0*(\d+)\b/i

export function lectureNumber(resource: CourseResource): number | null {
  const match =
    path.posix.basename(resource.relDir).match(lectureNumberRe) ??
    resource.title.match(lectureNumberRe)
  return match ? Number(match[1]) : null
}

export function problemSetNumber(resource: CourseResource): number | null {
  const match =
    path.posix.basename(resource.relDir).match(problemSetNumberRe) ??
    resource.title.match(problemSetNumberRe)
  return match ? Number(match[1]) : null
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

export async function courseCollections(
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

export function indexLinkableDirs(
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

export function isCoursePage(relDir: string, contentType: string): boolean {
  if (contentType === 'page') return relDir !== 'pages' && relDir !== 'resources'
  return relDir.startsWith('pages/') && relDir !== 'pages'
}

export function isCourseResource(relDir: string, data: JsonRecord, contentType: string): boolean {
  if (contentType === 'resource') return true
  return relDir.startsWith('resources/') && stringValue(data, 'file').length > 0
}
