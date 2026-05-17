import { globby } from 'globby'
import { fromHtml } from 'hast-util-from-html'
import { toMdast } from 'hast-util-to-mdast'
import { gfmToMarkdown } from 'mdast-util-gfm'
import { toMarkdown } from 'mdast-util-to-markdown'
import fs from 'node:fs/promises'
import path from 'node:path'

type JsonRecord = Record<string, unknown>

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
  localFile: string
  originalFile: string
}

const courseRoot = process.argv[2] ?? path.join('content', 'courses', '18.901-fall-2004')
const courseSlug = path.posix.join('courses', path.basename(courseRoot))
const ocwUrl = 'https://ocw.mit.edu/courses/18-901-introduction-to-topology-fall-2004/'

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function stringValue(record: JsonRecord, key: string): string {
  const value = record[key]
  return typeof value === 'string' ? value : ''
}

function stringList(record: JsonRecord, key: string): string[] {
  const value = record[key]
  if (!Array.isArray(value)) return []
  return value.filter(item => typeof item === 'string')
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
    .map(item => item.filter(child => typeof child === 'string'))
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

function normalizeTitle(value: string): string {
  const withoutExtension = value.replace(/\.(pdf|jpg|jpeg|png)$/i, '')
  const spaced = withoutExtension
    .replace(/commentsonstyle/i, 'comments on style')
    .replace(/erratafortop/i, 'errata for topology')
    .replace(/problemset/i, 'problem set')
    .replace(/^18901$/i, 'notes a')
    .replace(/[_-]+/g, ' ')
  return normalizeText(spaced).toLowerCase()
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
  if (values.length === 0) return []
  return [key + ':', ...values.map(value => `  - ${yamlString(value)}`)]
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

async function readJson(filePath: string): Promise<JsonRecord> {
  const raw: unknown = JSON.parse(await fs.readFile(filePath, 'utf8'))
  if (!isRecord(raw)) {
    throw new Error(`${filePath} does not contain a JSON object`)
  }
  return raw
}

async function htmlToMarkdown(html: string): Promise<string> {
  if (html.trim().length === 0) return ''
  const hast = fromHtml(normalizeMalformedHtml(html), { fragment: true })
  const mdast = toMdast(hast)
  return normalizeMarkdown(
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

function normalizeMarkdown(value: string): string {
  return normalizeText(value)
    .replace(/^(#{1,6})\s+(.+)$/gm, (_match, hashes: string, title: string) => {
      return `${hashes} ${normalizeText(title).toLowerCase()}`
    })
    .replace(/\[PDF\]/g, '[pdf]')
    .replace(/\]\(([^)\s]+)index\.html\)/g, (_match, url: string) => {
      return `](${url.replace(/index\.html$/, '')})`
    })
    .replace(/\]\(([^)\s]+)\.html\)/g, (_match, url: string) => {
      return `](${url.replace(/\.html$/, '/')})`
    })
    .replace(/!\[\]\((\.\.\/\.\.\/static_resources\/[^)\s]+)\)/g, (_match, url: string) => {
      const fileName = path.posix.basename(url)
      return `![[${courseSlug}/static_resources/${fileName}]]`
    })
    .replace(
      /(^|[^!])\[([^\]]+)\]\((\.\.\/\.\.\/(?:pages|resources)\/[^)\s]+)\)/gm,
      (_match, prefix: string, label: string, url: string) => {
        return `${prefix}${resourceLink(courseRelativeDir(url), label)}`
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

function courseRelativeDir(url: string): string {
  const hashStart = url.indexOf('#')
  const urlWithoutHash = hashStart >= 0 ? url.slice(0, hashStart) : url
  return urlWithoutHash
    .replace(/^\.\.\/\.\.\//, '')
    .replace(/index\.html$/, '')
    .replace(/\.html$/, '/')
    .replace(/\/+$/, '')
}

function resourceLink(relDir: string, label: string): string {
  const target = relDir.length > 0 ? `${courseSlug}/${relDir}/` : `${courseSlug}/`
  return `[[${target}|${normalizeText(label).toLowerCase()}]]`
}

function assetEmbed(localFile: string): string {
  if (localFile.length === 0) return ''
  return `![[${courseSlug}/static_resources/${localFile}]]`
}

function originalAssetUrl(originalFile: string, localFile: string): string {
  if (originalFile.length > 0) {
    return new URL(path.posix.basename(originalFile), ocwUrl).toString()
  }
  if (localFile.length > 0) {
    return new URL(localFile, ocwUrl).toString()
  }
  return ocwUrl
}

async function writeMarkdown(relDir: string, content: string): Promise<void> {
  const outPath = path.join(courseRoot, relDir, 'index.md')
  await fs.mkdir(path.dirname(outPath), { recursive: true })
  await fs.writeFile(outPath, `${content.trim()}\n`, 'utf8')
}

function uniqueValues(values: string[]): string[] {
  return Array.from(new Set(values.filter(value => value.length > 0)))
}

function markdownDocument(parts: string[]): string {
  return parts
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

function metadataId(scope: string, value: string): string {
  return `mit-18-901-${scope}-${slugTitle(value)}`
}

function learningTypesForFrontmatter(types: string[]): string[] {
  return uniqueValues(types.map(value => value.toLowerCase()))
}

function parseResource(relDir: string, data: JsonRecord): CourseResource {
  const originalFile = stringValue(data, 'file')
  const localFile = path.posix.basename(originalFile)
  const description = stringValue(data, 'description')
  const title = normalizeTitle(stringValue(data, 'title') || description || relDir)
  return {
    relDir,
    title,
    description: normalizeText(description),
    content: stringValue(data, 'content'),
    resourceType: normalizeText(stringValue(data, 'resourcetype')).toLowerCase(),
    resourceTypes: stringList(data, 'learning_resource_types'),
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

async function courseHome(data: JsonRecord, pages: CoursePage[]): Promise<string> {
  const title = normalizeText(stringValue(data, 'course_title')).toLowerCase()
  const description = normalizeText(
    stringValue(data, 'course_description') || stringValue(data, 'course_description_html'),
  )
  const image = path.posix.basename(stringValue(data, 'image_src'))
  const instructors = recordList(data, 'instructors')
    .map(instructor => stringValue(instructor, 'title'))
    .filter(value => value.length > 0)
  const topics = nestedStringLists(data, 'topics')
    .map(topic => topic.join(' / '))
    .filter(value => value.length > 0)
  const pageLinks = pages
    .filter(page => page.relDir.startsWith('pages/') && page.title !== 'pages')
    .sort((left, right) => left.title.localeCompare(right.title))
    .map(page => `- ${resourceLink(page.relDir, page.title)}`)
  const body = [
    frontmatter({
      title: `mit 18.901: ${title}`,
      description,
      id: 'mit-18-901-fall-2004',
      tags: ['math', 'topology', 'course', 'mit'],
      aliases: ['18.901', 'introduction to topology'],
    }),
    '',
    assetEmbed(image),
    '',
    description,
    '',
    '## course metadata',
    '',
    `- course: \`${stringValue(data, 'primary_course_number')}\``,
    `- term: ${normalizeText(stringValue(data, 'term')).toLowerCase()} ${stringValue(data, 'year')}`,
    `- level: ${stringList(data, 'level')
      .map(value => value.toLowerCase())
      .join(', ')}`,
    `- instructor: ${instructors.join(', ')}`,
    `- topics: ${topics.join(', ').toLowerCase()}`,
    `- license: [cc by-nc-sa 4.0](${nestedString(data, 'course_image_metadata', 'license') || 'https://creativecommons.org/licenses/by-nc-sa/4.0/'})`,
    `- source: [mit opencourseware](${ocwUrl})`,
    '',
    '## pages',
    '',
    ...pageLinks,
    '',
    '## resources',
    '',
    `- ${resourceLink('resources/lecture-notes', 'lecture notes')}`,
    `- ${resourceLink('resources/problem-sets', 'problem sets')}`,
    `- ${resourceLink('resources/readings', 'readings')}`,
    `- ${resourceLink('resources/assignments', 'assignments')}`,
  ]
  return body.join('\n')
}

async function pageMarkdown(page: CoursePage): Promise<string> {
  const content = await htmlToMarkdown(page.content)
  const tags = ['math', 'topology', 'course', ...learningTypesForFrontmatter(page.resourceTypes)]
  return markdownDocument([
    frontmatter({
      title: page.title,
      description: page.description || `mit 18.901 ${page.title}`,
      id: metadataId('page', page.relDir.replace(/^pages\//, '')),
      tags: uniqueValues(tags),
    }),
    '',
    `up: ${resourceLink('', 'mit 18.901')}`,
    '',
    content,
  ])
}

async function resourceMarkdown(resource: CourseResource): Promise<string> {
  const content = await htmlToMarkdown(resource.content)
  const sourceUrl = originalAssetUrl(resource.originalFile, resource.localFile)
  const tags = [
    'math',
    'topology',
    'course',
    ...learningTypesForFrontmatter(resource.resourceTypes),
  ]
  return markdownDocument([
    frontmatter({
      title: resource.title,
      description: resource.description || `mit 18.901 ${resource.title}`,
      id: metadataId('resource', resource.relDir.replace(/^resources\//, '')),
      tags: uniqueValues(tags),
      aliases: [path.posix.basename(resource.originalFile)],
    }),
    '',
    `up: ${resourceLink('', 'mit 18.901')}`,
    '',
    assetEmbed(resource.localFile),
    '',
    content || resource.description,
    '',
    '## metadata',
    '',
    `- type: ${resource.resourceType || 'resource'}`,
    `- file: [[${courseSlug}/static_resources/${resource.localFile}|${resource.localFile}]]`,
    `- source: [mit opencourseware](${sourceUrl})`,
  ])
}

function collectionMarkdown(
  title: string,
  description: string,
  resources: CourseResource[],
  extraLinks: string[],
): string {
  const links = [
    ...extraLinks,
    ...resources
      .sort((left, right) => left.title.localeCompare(right.title))
      .map(resource => `- ${resourceLink(resource.relDir, resource.title)}`),
  ]
  return markdownDocument([
    frontmatter({
      title,
      description,
      id: metadataId('collection', title),
      tags: ['math', 'topology', 'course'],
    }),
    '',
    `up: ${resourceLink('', 'mit 18.901')}`,
    '',
    ...links,
  ])
}

async function removeDeadHtml(): Promise<void> {
  const files = await globby(['**/*.html', '**/*.xml', 'sitemap.xml', 'favicon.ico'], {
    cwd: courseRoot,
    dot: true,
    onlyFiles: true,
  })
  for (const file of files) {
    await fs.rm(path.join(courseRoot, file), { force: true })
  }
  await fs.rm(path.join(courseRoot, 'static_shared'), { recursive: true, force: true })
}

async function main(): Promise<void> {
  const rootData = await readJson(path.join(courseRoot, 'data.json'))
  const dataFiles = await globby(['**/data.json'], {
    cwd: courseRoot,
    ignore: ['data.json', 'static_resources/data.json'],
  })
  const pages: CoursePage[] = []
  const resources: CourseResource[] = []

  for (const dataFile of dataFiles.sort()) {
    const relDir = path.posix.dirname(dataFile)
    const data = await readJson(path.join(courseRoot, dataFile))
    const contentType = stringValue(data, 'content_type')
    if (contentType === 'resource') {
      resources.push(parseResource(relDir, data))
    } else if (relDir.startsWith('pages/')) {
      pages.push(parsePage(relDir, data))
    }
  }

  await writeMarkdown('', await courseHome(rootData, pages))

  for (const page of pages) {
    await writeMarkdown(page.relDir, await pageMarkdown(page))
  }

  for (const resource of resources) {
    await writeMarkdown(resource.relDir, await resourceMarkdown(resource))
  }

  const byType = (type: string) =>
    resources.filter(resource => resource.resourceTypes.includes(type))

  await writeMarkdown(
    'pages',
    collectionMarkdown(
      'course pages',
      'mit 18.901 course pages',
      [],
      pages
        .filter(page => page.title !== 'pages')
        .sort((left, right) => left.title.localeCompare(right.title))
        .map(page => `- ${resourceLink(page.relDir, page.title)}`),
    ),
  )
  await writeMarkdown(
    'resources/lecture-notes',
    collectionMarkdown(
      'lecture notes',
      'mit 18.901 supplementary lecture notes',
      byType('Lecture Notes'),
      [],
    ),
  )
  await writeMarkdown(
    'resources/problem-sets',
    collectionMarkdown(
      'problem sets',
      'mit 18.901 problem-set resources',
      byType('Problem Sets'),
      [],
    ),
  )
  await writeMarkdown(
    'resources/readings',
    collectionMarkdown('readings', 'mit 18.901 reading resources', byType('Readings'), []),
  )
  await writeMarkdown(
    'resources/assignments',
    collectionMarkdown(
      'assignments',
      'mit 18.901 assignments',
      [],
      [
        `- ${resourceLink('pages/assignments', 'assignments')}`,
        `- ${resourceLink('resources/problem-sets', 'problem sets')}`,
      ],
    ),
  )

  await removeDeadHtml()
}

main().catch(error => {
  console.error(error)
  process.exit(1)
})
