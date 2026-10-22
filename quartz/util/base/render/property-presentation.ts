import { h } from 'hastscript'
import { QuartzPluginData } from '../../../plugins/vfile'
import { ReloadableWeakMap } from '../../module-caches'
import {
  FilePath,
  FullSlug,
  isAbsoluteURL,
  resolveRelative,
  simplifySlug,
  slugifyFilePath,
} from '../../path'
import { extractWikilinksWithPositions, resolveWikilinkTarget } from '../../wikilinks'
import { PropertyConfig } from '../types'

export type RenderElement = ReturnType<typeof h>
export type RenderNode = RenderElement | string
type IconToken = { kind: 'icon'; value: string }

export const isIconToken = (value: unknown): value is IconToken =>
  typeof value === 'object' &&
  value !== null &&
  (value as IconToken).kind === 'icon' &&
  typeof (value as IconToken).value === 'string'

const splitIconClasses = (raw: string): string[] =>
  raw
    .trim()
    .split(/\s+/)
    .filter(part => part.length > 0)

const normalizeIconName = (raw: string): string => {
  const trimmed = raw.trim()
  if (!trimmed) return ''
  const colonIndex = trimmed.lastIndexOf(':')
  return colonIndex >= 0 ? trimmed.slice(colonIndex + 1).trim() : trimmed
}

const buildIconClassList = (raw: string): string[] => {
  const parts = splitIconClasses(raw)
  if (parts.length === 0) return []
  if (parts.length > 1) return parts
  const normalized = normalizeIconName(parts[0])
  if (!normalized) return []
  if (normalized.startsWith('icon-')) return [normalized]
  return [`icon-${normalized}`]
}

export const renderIconNode = (raw: string): RenderElement => {
  const classList = buildIconClassList(raw)
  return h('i', { class: classList.join(' '), ariaHidden: true })
}

export const renderIconHtml = (raw: string): string => {
  const classList = buildIconClassList(raw)
  if (classList.length === 0) return ''
  return `<i class="${classList.join(' ')}" aria-hidden="true"></i>`
}

export function getFileBaseName(filePath?: string, slug?: string): string | undefined {
  const source = filePath ?? slug
  if (!source) return undefined
  const fragment = source.split('/').pop() || source
  return fragment.replace(/\.[^/.]+$/, '')
}

export function getFileDisplayName(file?: QuartzPluginData): string | undefined {
  if (!file) return undefined
  const title = file.frontmatter?.title
  if (typeof title === 'string' && title.length > 0) return title
  const baseFromPath = getFileBaseName(file.filePath as string | undefined)
  if (baseFromPath) return baseFromPath
  const baseFromSlug = getFileBaseName(file.slug)
  if (baseFromSlug) return baseFromSlug.replace(/-/g, ' ')
  return undefined
}

export function fallbackNameFromSlug(slug: FullSlug): string {
  const base = getFileBaseName(slug) ?? slug
  return base.replace(/-/g, ' ')
}

const fileLookupCache = new ReloadableWeakMap<
  QuartzPluginData[],
  { bySimple: Map<string, QuartzPluginData>; byExact: Map<string, QuartzPluginData> }
>()

export function getFileLookup(allFiles: QuartzPluginData[]): {
  bySimple: Map<string, QuartzPluginData>
  byExact: Map<string, QuartzPluginData>
} {
  let index = fileLookupCache.get(allFiles)
  if (!index) {
    const bySimple = new Map<string, QuartzPluginData>()
    const byExact = new Map<string, QuartzPluginData>()
    for (const entry of allFiles) {
      if (!entry.slug) continue
      const exact = String(entry.slug)
      if (!byExact.has(exact)) {
        byExact.set(exact, entry)
      }
      const simple = simplifySlug(entry.slug as FullSlug)
      if (!bySimple.has(simple)) {
        bySimple.set(simple, entry)
      }
    }
    index = { bySimple, byExact }
    fileLookupCache.set(allFiles, index)
  }
  return index
}

function findFileBySlug(
  allFiles: QuartzPluginData[],
  targetSlug: FullSlug,
): QuartzPluginData | undefined {
  const index = getFileLookup(allFiles)
  const targetSimple = simplifySlug(targetSlug)
  return index.bySimple.get(targetSimple) ?? index.byExact.get(targetSlug)
}

function renderInternalLinkNode(
  targetSlug: FullSlug,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
  alias?: string,
  anchor?: string,
): RenderElement {
  const targetFile = findFileBySlug(allFiles, targetSlug)
  const displayText =
    alias && alias.trim().length > 0
      ? alias.trim()
      : (getFileDisplayName(targetFile) ?? fallbackNameFromSlug(targetSlug))

  const hrefBase = resolveRelative(currentSlug, targetSlug)
  const href = anchor && anchor.length > 0 ? `${hrefBase}${anchor}` : hrefBase
  const dataSlug = anchor && anchor.length > 0 ? `${targetSlug}${anchor}` : targetSlug

  return h('a.internal', { href, 'data-slug': dataSlug }, displayText)
}

export function buildFileLinkNode(
  slug: FullSlug,
  currentSlug: FullSlug,
  label: string,
): RenderElement {
  const href = resolveRelative(currentSlug, slug)
  return h('a.internal', { href, 'data-slug': slug }, label)
}

function splitTargetAndAlias(raw: string): { target: string; alias?: string } {
  let buffer = ''
  let alias: string | undefined
  let escaped = false
  for (let i = 0; i < raw.length; i++) {
    const ch = raw[i]
    if (escaped) {
      buffer += ch
      escaped = false
      continue
    }
    if (ch === '\\') {
      escaped = true
      continue
    }
    if (ch === '|' && alias === undefined) {
      alias = raw.slice(i + 1)
      break
    }
    buffer += ch
  }

  const target = buffer.replace(/\\\|/g, '|').trim()
  const cleanedAlias = alias?.replace(/\\\|/g, '|').trim()
  return { target, alias: cleanedAlias?.length ? cleanedAlias : undefined }
}

function normalizeTargetSlug(
  target: string,
  currentSlug: FullSlug,
  anchor?: string,
): { slug: FullSlug; anchor?: string } {
  const trimmed = target.trim()
  if (!trimmed) return { slug: currentSlug, anchor }
  const slug = slugifyFilePath(trimmed as FilePath)
  return { slug, anchor }
}

export function renderInlineString(
  value: string,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
): RenderNode[] {
  if (!value.includes('[[')) {
    return [value]
  }

  const nodes: RenderNode[] = []
  const ranges = extractWikilinksWithPositions(value)
  let lastIndex = 0
  for (const range of ranges) {
    const start = range.start
    if (start > lastIndex) nodes.push(value.slice(lastIndex, start))

    const parsed = range.wikilink
    const raw = value.slice(range.start, range.end)
    if (parsed.embed) {
      nodes.push(raw)
      lastIndex = range.end
      continue
    }

    const resolved = resolveWikilinkTarget(parsed, currentSlug)
    if (!resolved) {
      nodes.push(parsed.alias ?? parsed.target ?? raw)
      lastIndex = range.end
      continue
    }

    nodes.push(
      renderInternalLinkNode(resolved.slug, currentSlug, allFiles, parsed.alias, resolved.anchor),
    )
    lastIndex = range.end
  }

  if (lastIndex < value.length) {
    nodes.push(value.slice(lastIndex))
  }

  return nodes
}

export function renderBacklinkNodes(
  backlinks: string[],
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
): RenderNode[] {
  const nodes: RenderNode[] = []
  for (const entry of backlinks) {
    if (!entry) continue
    let raw = entry.trim()
    if (!raw) continue
    let alias: string | undefined
    if (raw.startsWith('!')) {
      raw = raw.slice(1)
    }
    if (raw.startsWith('[[') && raw.endsWith(']]')) {
      const inner = raw.slice(2, -2)
      const parsed = splitTargetAndAlias(inner)
      raw = parsed.target
      alias = parsed.alias
    }
    const { slug: targetSlug, anchor } = normalizeTargetSlug(raw, currentSlug)
    if (nodes.length > 0) {
      nodes.push(', ')
    }
    nodes.push(
      renderInternalLinkNode(
        targetSlug,
        currentSlug,
        allFiles,
        alias,
        anchor && anchor.length > 0 ? anchor : undefined,
      ),
    )
  }
  return nodes
}

export function getPropertyDisplayName(
  property: string,
  properties?: Record<string, PropertyConfig>,
): string {
  const candidates: string[] = []

  const addCandidate = (candidate: string | undefined) => {
    if (!candidate) return
    if (!candidates.includes(candidate)) {
      candidates.push(candidate)
    }
  }

  addCandidate(property)

  const withoutPrefix = property.replace(/^(?:note|file)\./, '')
  addCandidate(withoutPrefix)

  if (!property.startsWith('note.')) {
    addCandidate(`note.${property}`)
  }
  if (!property.startsWith('file.')) {
    addCandidate(`file.${property}`)
  }

  addCandidate(withoutPrefix.split('.').pop())

  for (const candidate of candidates) {
    const displayName = properties?.[candidate]?.displayName
    if (displayName && displayName.length > 0) {
      return displayName
    }
  }

  const base = withoutPrefix.length > 0 ? withoutPrefix : property
  return base
    .split('.')
    .pop()!
    .replace(/-/g, ' ')
    .replace(/_/g, ' ')
    .replace(/([A-Z])/g, ' $1')
    .trim()
}

export function listValueToPlainText(value: unknown): string | undefined {
  if (value === undefined || value === null) {
    return undefined
  }
  if (isIconToken(value)) {
    const cleaned = value.value.trim()
    return cleaned.length > 0 ? cleaned : undefined
  }
  if (Array.isArray(value)) {
    const parts = value
      .map(item => listValueToPlainText(item))
      .filter((part): part is string => Boolean(part && part.length > 0))
    if (parts.length === 0) return undefined
    return parts.join(', ')
  }
  if (value instanceof Date) {
    return value.toISOString().split('T')[0]
  }
  if (typeof value === 'string') {
    const cleaned = value
      .replace(/\[\[([^\]|]+)\|([^\]]+)\]\]/g, '$2')
      .replace(/\[\[([^\]]+)\]\]/g, '$1')
      .trim()
    return cleaned.length > 0 ? cleaned : undefined
  }
  const stringified = String(value).trim()
  return stringified.length > 0 ? stringified : undefined
}

export function hasRenderableValue(value: unknown): boolean {
  if (value === undefined || value === null) return false
  if (isIconToken(value)) {
    return value.value.trim().length > 0
  }
  if (Array.isArray(value)) {
    return value.some(item => hasRenderableValue(item))
  }
  if (value instanceof Date) return true
  if (typeof value === 'string') return value.trim().length > 0
  return true
}

export function renderPropertyValueNodes(
  value: unknown,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
): RenderNode[] {
  if (value === undefined || value === null) return []
  if (isIconToken(value)) {
    const cleaned = value.value.trim()
    if (cleaned.length === 0) return []
    return [renderIconNode(cleaned)]
  }
  if (Array.isArray(value)) {
    const nodes: RenderNode[] = []
    value.forEach((item, idx) => {
      nodes.push(...renderPropertyValueNodes(item, currentSlug, allFiles))
      if (idx < value.length - 1) {
        nodes.push(', ')
      }
    })
    return nodes
  }
  if (value instanceof Date) {
    return [value.toISOString().split('T')[0]]
  }
  if (typeof value === 'string') {
    return renderInlineString(value, currentSlug, allFiles)
  }
  return [String(value)]
}

export function resolveCardImage(
  imageValue: unknown,
  currentSlug: FullSlug,
): { url: string; slug?: FullSlug } | undefined {
  const source =
    typeof imageValue === 'string'
      ? imageValue
      : Array.isArray(imageValue) && typeof imageValue[0] === 'string'
        ? imageValue[0]
        : undefined

  if (!source) return undefined

  const trimmed = source.trim()
  if (!trimmed) return undefined

  const wl = trimmed.match(/^\[\[(.+?)\]\]$/)
  if (wl) {
    const inner = wl[1]
    const { target } = splitTargetAndAlias(inner)
    const { slug } = normalizeTargetSlug(target, currentSlug)
    return { url: resolveRelative(currentSlug, slug), slug }
  }

  if (isAbsoluteURL(trimmed)) return { url: trimmed }
  const slug = slugifyFilePath(trimmed as FilePath)
  return { url: resolveRelative(currentSlug, slug), slug }
}

const mapLatexCommandText: Record<string, string> = {
  cdot: '*',
  ge: '>=',
  geq: '>=',
  infty: 'infinity',
  le: '<=',
  leq: '<=',
  left: '',
  right: '',
  mathcal: '',
  mathbb: '',
  mathbf: '',
  mathit: '',
  mathrm: '',
  operatorname: '',
  text: '',
  times: 'x',
  top: '^T',
}

function latexToPlainText(math: string): string {
  let text = math.trim()
  let previous = ''
  while (text !== previous) {
    previous = text
    text = text.replace(
      /\\(mathcal|mathbb|mathbf|mathit|mathrm|operatorname|text)\s*\{([^{}]*)\}/g,
      '$2',
    )
    text = text.replace(/\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, '($1)/($2)')
    text = text.replace(/\\sqrt\s*\{([^{}]*)\}/g, 'sqrt($1)')
  }
  return text
    .replace(/\\([A-Za-z]+)/g, (_match, command: string) => mapLatexCommandText[command] ?? command)
    .replace(/\\([{}])/g, '$1')
    .replace(/[{}]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

export function plainTextDescription(
  description: string | undefined,
  currentSlug: FullSlug,
): string {
  if (!description) return ''
  let text = description
  if (text.includes('[[')) {
    const ranges = extractWikilinksWithPositions(text)
    for (let i = ranges.length - 1; i >= 0; i--) {
      const { wikilink: link, start, end } = ranges[i]
      const resolved = resolveWikilinkTarget(link, currentSlug)
      const display = link.alias || link.anchorText || link.target || resolved?.slug || ''
      text = text.slice(0, start) + display + text.slice(end)
    }
  }
  if (text.includes('[')) {
    text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
  }
  if (text.includes('$')) {
    text = text
      .replace(/\$\$([\s\S]*?)\$\$/g, (_m, math: string) => latexToPlainText(math))
      .replace(/(?<!\$)\$([^$\n]+?)\$(?!\$)/g, (_m, math: string) => latexToPlainText(math))
  }
  if (text.includes('<')) {
    text = text.replace(/<((?:https?:|mailto:)[^>]+)>/gi, '$1').replace(/<[^>]+>/g, '')
  }
  return text.replace(/\s+/g, ' ').trim()
}
