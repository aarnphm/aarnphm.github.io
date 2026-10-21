import { globby } from 'globby'
import fs from 'node:fs/promises'
import path from 'node:path'
import { isRecord } from '../util/type-guards'
import {
  compareTitle,
  courseContext,
  defaultCourseRoot,
  mitBaseUrl,
  nestedRecord,
  nestedString,
  pathExists,
  readJson,
  stringValue,
  strippedUrlPath,
  uniqueValues,
  vendoredCourses,
  type CourseContext,
  type CoursePage,
  type CourseResource,
  type JsonRecord,
} from './vendor-ocw-course/context'
import { resourceLink, writeMarkdown } from './vendor-ocw-course/markdown'
import {
  collectionMarkdown,
  courseHome,
  pageMarkdown,
  resourceMarkdown,
  resourcesIndexMarkdown,
} from './vendor-ocw-course/render'
import {
  classifyResources,
  courseCollections,
  indexLinkableDirs,
  isCoursePage,
  isCourseResource,
  parsePage,
  parseResource,
  resourceId,
} from './vendor-ocw-course/resources'
import { courseStudy, latexNotes } from './vendor-ocw-course/study'

const ocwHost = 'ocw.mit.edu'
const fetchConcurrency = 8

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
  context.vendoredCourses = await vendoredCourses()
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
