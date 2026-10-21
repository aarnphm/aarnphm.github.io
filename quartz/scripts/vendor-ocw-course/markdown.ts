import { fromHtml } from 'hast-util-from-html'
import { toMdast } from 'hast-util-to-mdast'
import { gfmToMarkdown } from 'mdast-util-gfm'
import { toMarkdown } from 'mdast-util-to-markdown'
import fs from 'node:fs/promises'
import path from 'node:path'
import {
  mitBaseUrl,
  normalizeText,
  ocwLayout,
  recordList,
  stringList,
  stringValue,
  strippedUrlPath,
  uniqueValues,
  type CourseContext,
  type CourseStudy,
  type JsonRecord,
} from './context'

function yamlString(value: string): string {
  return `"${normalizeText(value).replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n')}"`
}

function yamlList(key: string, values: string[]): string[] {
  const items = uniqueValues(values.map(normalizeText))
  if (items.length === 0) return []
  return [key + ':', ...items.map(value => `  - ${yamlString(value)}`)]
}

// `{flow}` values are written as one YAML flow line (the `resource:` contract).
export type FrontmatterValue = string | string[] | boolean | { flow: unknown }

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

export function courseFrontmatter(
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

export async function htmlToMarkdown(context: CourseContext, html: string): Promise<string> {
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
        const otherCourse = otherCourseLink(context, url, label)
        if (otherCourse) return `${prefix}${otherCourse}`
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

// A root-relative `/courses/…` link points at another OCW course: link the vendored copy when
// there is one, else the OCW page, since the bare path 404s on this site.
function otherCourseLink(context: CourseContext, value: string, label: string): string {
  const stripped = strippedUrlPath(value)
  let local = ''
  if (stripped.startsWith('/')) local = stripped
  else if (/^https?:\/\/ocw\.mit\.edu\//.test(stripped)) local = new URL(stripped).pathname
  const site = local.match(/^\/(courses\/[^/]+)/)?.[1]
  if (!site) return ''
  const slug = context.vendoredCourses.get(site)
  if (slug) {
    const [, mark = '', inner = label] = label.match(/^([_*]?)(.+)\1$/) ?? []
    return `${mark}[[${slug}/|${inner}]]${mark}`
  }
  return /^https?:/.test(value) ? '' : `[${label}](${new URL(value, mitBaseUrl).toString()})`
}

function staticResourceFile(context: CourseContext, value: string): string {
  const local = ocwLocalPath(context, value)
  if (local.length === 0) return ''
  if (local.startsWith('static_resources/')) return path.posix.basename(local)
  if (/\.(pdf|jpg|jpeg|png|gif|mp3|mp4|zip|html?)$/i.test(local)) return path.posix.basename(local)
  return ''
}

export function resourceLink(context: CourseContext, relDir: string, label: string): string {
  const target = relDir.length > 0 ? `${context.slug}/${relDir}/` : `${context.slug}/`
  return `[[${target}|${normalizeText(label).toLowerCase()}]]`
}

export function assetEmbed(context: CourseContext, localFile: string): string {
  if (localFile.length === 0) return ''
  return `![[${context.slug}/static_resources/${localFile}]]`
}

export async function writeMarkdown(
  context: CourseContext,
  relDir: string,
  content: string,
): Promise<void> {
  const outPath = path.join(context.root, relDir, 'index.md')
  await fs.mkdir(path.dirname(outPath), { recursive: true })
  await fs.writeFile(outPath, `${content.trim()}\n`, 'utf8')
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
export function courseStudyYaml(
  context: CourseContext,
  data: JsonRecord,
  study: CourseStudy,
): string[] {
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
