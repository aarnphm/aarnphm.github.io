import { execFile } from 'node:child_process'
import path from 'node:path'
import { promisify } from 'node:util'
import type { LatexResult } from '../vendor-ocw-latex'
import {
  compareTitle,
  mitBaseUrl,
  nestedStringLists,
  normalizeText,
  ocwLayout,
  pathExists,
  recordList,
  slugTitle,
  stringList,
  stringValue,
  uniqueValues,
  type CourseCollection,
  type CourseContext,
  type CoursePage,
  type CourseResource,
  type CourseStudy,
  type JsonRecord,
} from './context'
import {
  assetEmbed,
  courseFrontmatter,
  courseStudyYaml,
  htmlToMarkdown,
  resourceLink,
  type FrontmatterValue,
} from './markdown'
import { problemSetNumber, resourceId } from './resources'

const run = promisify(execFile)

function originalAssetUrl(context: CourseContext, originalFile: string, localFile: string): string {
  if (/^https?:\/\//.test(originalFile)) return originalFile
  if (originalFile.startsWith('/')) return new URL(originalFile, mitBaseUrl).toString()
  if (originalFile.length > 0)
    return new URL(path.posix.basename(originalFile), context.sourceUrl).toString()
  if (localFile.length > 0) return new URL(localFile, context.sourceUrl).toString()
  return context.sourceUrl
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

function topicSummary(data: JsonRecord): string {
  return nestedStringLists(data, 'topics')
    .map(topic => topic.join(' / '))
    .filter(value => value.length > 0)
    .join(', ')
    .toLowerCase()
}

export async function courseHome(
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

export async function pageMarkdown(context: CourseContext, page: CoursePage): Promise<string> {
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

export async function resourceMarkdown(
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

export function collectionMarkdown(context: CourseContext, collection: CourseCollection): string {
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

export function resourcesIndexMarkdown(
  context: CourseContext,
  collections: CourseCollection[],
): string {
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
