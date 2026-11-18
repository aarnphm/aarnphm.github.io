import { toString } from 'hast-util-to-string'
import { h } from 'hastscript'
import { QuartzPluginData } from '../../../plugins/vfile'
import { ReloadableWeakMap } from '../../module-caches'
import { FullSlug, isAbsoluteURL, resolveRelative, simplifySlug } from '../../path'
import { referenceCatalogFromFiles, referenceSourcePath } from '../../references/context'
import { parseLiteralReference, resolveLiteralReference } from '../../references/literal'
import { PropertyConfig } from '../types'
import { inlineReferenceNodes } from './inline-reference'

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

export function buildFileLinkNode(
  slug: FullSlug,
  currentSlug: FullSlug,
  label: string,
): RenderElement {
  const href = resolveRelative(currentSlug, slug)
  return h('a.internal', { href, 'data-slug': slug }, label)
}

export function renderInlineString(
  value: string,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
  sourceFile?: QuartzPluginData,
): RenderNode[] {
  if (!/[[\]<>*&_`\\]/.test(value)) return [value]
  const originFile = sourceFile ?? findFileBySlug(allFiles, currentSlug)
  const nodes = inlineReferenceNodes(value, currentSlug, {
    catalog: referenceCatalogFromFiles(allFiles),
    origin: originFile ? referenceSourcePath(originFile) : currentSlug,
  })
  return nodes.map(node => (node.type === 'text' ? node.value : node))
}

export function renderBacklinkNodes(
  backlinks: string[],
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
  sourceFile?: QuartzPluginData,
): RenderNode[] {
  const nodes: RenderNode[] = []
  for (const entry of backlinks) {
    const raw = entry.trim().replace(/^!/, '')
    if (!raw) continue
    if (nodes.length > 0) nodes.push(', ')
    nodes.push(
      ...renderInlineString(
        parseLiteralReference(raw) ? raw : `[[${raw}]]`,
        currentSlug,
        allFiles,
        sourceFile,
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
    const cleaned = plainTextDescription(value)
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
  sourceFile?: QuartzPluginData,
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
      nodes.push(...renderPropertyValueNodes(item, currentSlug, allFiles, sourceFile))
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
    return renderInlineString(value, currentSlug, allFiles, sourceFile)
  }
  return [String(value)]
}

export function resolveCardImage(
  imageValue: unknown,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
  sourceFile?: QuartzPluginData,
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

  if (isAbsoluteURL(trimmed)) return { url: trimmed }
  const link = parseLiteralReference(trimmed) ?? parseLiteralReference(`[[${trimmed}]]`)
  if (!link) return undefined
  const originFile = sourceFile ?? findFileBySlug(allFiles, currentSlug)
  const resolved = resolveLiteralReference(link, currentSlug, {
    catalog: referenceCatalogFromFiles(allFiles),
    origin: originFile ? referenceSourcePath(originFile) : currentSlug,
  })
  if (!resolved) return undefined
  return {
    url: `${resolveRelative(currentSlug, resolved.slug)}${resolved.anchor ?? ''}`,
    slug: resolved.slug,
  }
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

export function plainTextDescription(description: string | undefined): string {
  if (!description) return ''
  if (!/[[\]$<>&*_`\\]/.test(description)) return description.replace(/\s+/g, ' ').trim()
  let text = description
  if (text.includes('$')) {
    text = text
      .replace(/\$\$([\s\S]*?)\$\$/g, (_m, math: string) => latexToPlainText(math))
      .replace(/(?<!\$)\$([^$\n]+?)\$(?!\$)/g, (_m, math: string) => latexToPlainText(math))
  }
  if (text.includes('<')) {
    text = text.replace(/<((?:https?:|mailto:)[^>]+)>/gi, '$1').replace(/<[^>]+>/g, '')
  }
  return inlineReferenceNodes(text, undefined, undefined, true)
    .map(node => toString(node))
    .join('')
    .replace(/\s+/g, ' ')
    .trim()
}
