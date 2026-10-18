import { Root } from 'hast'
import { h, s } from 'hastscript'
import type { MarkerData } from '../../components/scripts/base-map-data'
import { QuartzPluginData } from '../../plugins/vfile'
import {
  resolveRelative,
  FullSlug,
  joinSegments,
  FilePath,
  slugifyFilePath,
  simplifySlug,
  isAbsoluteURL,
} from '../../util/path'
import { extractWikilinksWithPositions, resolveWikilinkTarget } from '../../util/wikilinks'
import {
  BOOKCASE_DEFAULTS,
  BOOKCASE_LIMITS,
  clampOption,
  fallbackSpine,
  packShelves,
  spineInk,
} from './bookcase/geometry'
import { bookcaseScene, type SceneBook } from './bookcase/scene'
import { isPublicBaseFile } from './browser-data'
import {
  BaseExpressionDiagnostic,
  ProgramIR,
  buildPropertyExpressionSource,
  evaluateExpression,
  evaluateFilterExpression,
  valueToUnknown,
  EvalContext,
  Value,
} from './compiler'
import { computeViewSummaries } from './query'
import {
  BaseView,
  BaseGroupBy,
  BaseGroupValue,
  BaseFile,
  BaseSortConfig,
  PropertyConfig,
  ViewSummaryConfig,
  parseViewSummaries,
  BOOKCASE_PROPERTY_DEFAULTS,
} from './types'

type RenderElement = ReturnType<typeof h>
type RenderNode = RenderElement | string
type IconToken = { kind: 'icon'; value: string }

const isIconToken = (value: unknown): value is IconToken =>
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

const renderIconNode = (raw: string): RenderElement => {
  const classList = buildIconClassList(raw)
  return h('i', { class: classList.join(' '), ariaHidden: true })
}

const renderIconHtml = (raw: string): string => {
  const classList = buildIconClassList(raw)
  if (classList.length === 0) return ''
  return `<i class="${classList.join(' ')}" aria-hidden="true"></i>`
}

function getFileBaseName(filePath?: string, slug?: string): string | undefined {
  const source = filePath ?? slug
  if (!source) return undefined
  const fragment = source.split('/').pop() || source
  return fragment.replace(/\.[^/.]+$/, '')
}

function getFileDisplayName(file?: QuartzPluginData): string | undefined {
  if (!file) return undefined
  const title = file.frontmatter?.title
  if (typeof title === 'string' && title.length > 0) return title
  const baseFromPath = getFileBaseName(file.filePath as string | undefined)
  if (baseFromPath) return baseFromPath
  const baseFromSlug = getFileBaseName(file.slug)
  if (baseFromSlug) return baseFromSlug.replace(/-/g, ' ')
  return undefined
}

function fallbackNameFromSlug(slug: FullSlug): string {
  const base = getFileBaseName(slug) ?? slug
  return base.replace(/-/g, ' ')
}

const fileLookupCache = new WeakMap<
  QuartzPluginData[],
  { bySimple: Map<string, QuartzPluginData>; byExact: Map<string, QuartzPluginData> }
>()

function getFileLookup(allFiles: QuartzPluginData[]): {
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

function buildFileLinkNode(slug: FullSlug, currentSlug: FullSlug, label: string): RenderElement {
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

function renderInlineString(
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

function renderBacklinkNodes(
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

function getPropertyDisplayName(
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

function renderBooleanCheckbox(value: boolean): RenderElement {
  return h('input', {
    type: 'checkbox',
    checked: value ? true : undefined,
    disabled: true,
    class: 'base-checkbox',
  })
}

function wrapTableCellContent(content: RenderNode | RenderNode[]): RenderElement {
  const nodes = Array.isArray(content) ? content : [content]
  return h('td', {}, [h('div.base-table-cell-content', nodes)])
}

function buildTableHead(
  columns: string[],
  properties?: Record<string, PropertyConfig>,
): RenderElement {
  return h(
    'tr',
    columns.map(col => h('th', {}, getPropertyDisplayName(col, properties))),
  )
}

type EvalContextFactory = (file: QuartzPluginData) => EvalContext

type PropertyExprGetter = (property: string) => ProgramIR | null

const unknownPropertyCache = new WeakMap<Map<string, Value>, Map<string, unknown>>()
const diagnosticSourceCache = new Map<string, string>()
const sortCollator = new Intl.Collator()

function getDiagnosticSource(property: string): string {
  let src = diagnosticSourceCache.get(property)
  if (src === undefined) {
    src = buildPropertyExpressionSource(property) ?? property
    diagnosticSourceCache.set(property, src)
  }
  return src
}

function resolveValueWithFormulas(
  file: QuartzPluginData,
  property: string,
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
): unknown {
  const expr = getPropertyExpr(property)
  if (!expr) return undefined
  const ctx = getContext(file)
  const cacheKey = property.trim()
  if (cacheKey.length > 0 && ctx.propertyCache) {
    let unknownMap = unknownPropertyCache.get(ctx.propertyCache)
    if (!unknownMap) {
      unknownMap = new Map()
      unknownPropertyCache.set(ctx.propertyCache, unknownMap)
    }
    if (unknownMap.has(cacheKey)) {
      return unknownMap.get(cacheKey)
    }
    if (ctx.propertyCache.has(cacheKey)) {
      const converted = valueToUnknown(ctx.propertyCache.get(cacheKey)!)
      unknownMap.set(cacheKey, converted)
      return converted
    }
    ctx.diagnosticContext = `property.${property}`
    ctx.diagnosticSource = getDiagnosticSource(property)
    const value = evaluateExpression(expr, ctx)
    ctx.propertyCache.set(cacheKey, value)
    const converted = valueToUnknown(value)
    unknownMap.set(cacheKey, converted)
    return converted
  }
  ctx.diagnosticContext = `property.${property}`
  ctx.diagnosticSource = getDiagnosticSource(property)
  const value = evaluateExpression(expr, ctx)
  return valueToUnknown(value)
}

function buildTableCell(
  file: QuartzPluginData,
  column: string,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
): RenderElement {
  const slug = (file.slug || '') as FullSlug

  const linkProperty =
    column === 'file.name'
      ? 'file.name'
      : column === 'title' || column === 'file.title' || column === 'note.title'
        ? 'file.title'
        : undefined

  if (linkProperty) {
    const rawValue = resolveValueWithFormulas(file, linkProperty, getContext, getPropertyExpr)
    let resolvedValue: string
    if (typeof rawValue === 'string' && rawValue.length > 0) {
      resolvedValue = rawValue
    } else {
      const fallbackSlugSegment = file.slug?.split('/').pop() || ''
      resolvedValue =
        getFileBaseName(file.filePath as string | undefined) ||
        fallbackSlugSegment.replace(/\.[^/.]+$/, '').replace(/-/g, ' ')
    }
    return wrapTableCellContent(buildFileLinkNode(slug, currentSlug, resolvedValue))
  }

  if (column === 'file.links') {
    const links = resolveValueWithFormulas(file, 'file.links', getContext, getPropertyExpr)
    const count = Array.isArray(links) ? links.length : 0
    return wrapTableCellContent(String(count))
  }

  if (column === 'file.backlinks' || column === 'file.inlinks') {
    const backlinks = resolveValueWithFormulas(file, column, getContext, getPropertyExpr)
    if (!Array.isArray(backlinks) || backlinks.length === 0) {
      return wrapTableCellContent('')
    }
    const entries = backlinks.filter((entry): entry is string => typeof entry === 'string')
    if (entries.length === 0) {
      return wrapTableCellContent('')
    }
    const nodes = renderBacklinkNodes(entries, currentSlug, allFiles)
    return wrapTableCellContent(nodes)
  }

  const canEvalExpr = Boolean(getPropertyExpr(column))

  if (!canEvalExpr && column.startsWith('note.')) {
    const actualColumn = column.replace('note.', '')
    return buildTableCell(file, actualColumn, currentSlug, allFiles, getContext, getPropertyExpr)
  }

  const value = resolveValueWithFormulas(file, column, getContext, getPropertyExpr)

  if (value === undefined || value === null) {
    return wrapTableCellContent('')
  }

  if (isIconToken(value)) {
    const cleaned = value.value.trim()
    return cleaned.length > 0
      ? wrapTableCellContent(renderIconNode(cleaned))
      : wrapTableCellContent('')
  }

  if (Array.isArray(value)) {
    const parts: RenderNode[] = []
    value.forEach((item, idx) => {
      if (isIconToken(item)) {
        const cleaned = item.value.trim()
        if (cleaned.length > 0) {
          parts.push(renderIconNode(cleaned))
        }
      } else if (typeof item === 'string') {
        parts.push(...renderInlineString(item, currentSlug, allFiles))
      } else {
        parts.push(String(item))
      }
      if (idx < value.length - 1) {
        parts.push(', ')
      }
    })
    return wrapTableCellContent(parts)
  }

  if (value instanceof Date) {
    return wrapTableCellContent(value.toISOString().split('T')[0])
  }

  if (typeof value === 'string') {
    const rendered = renderInlineString(value, currentSlug, allFiles)
    return wrapTableCellContent(rendered)
  }

  if (typeof value === 'boolean') {
    return wrapTableCellContent(renderBooleanCheckbox(value))
  }

  return wrapTableCellContent(String(value))
}

function applySorting(
  files: QuartzPluginData[],
  sortConfig: BaseSortConfig[] = [],
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
): QuartzPluginData[] {
  if (sortConfig.length === 0 || files.length <= 1) return files

  const normalizeSortValue = (val: unknown): string | number | null | undefined => {
    if (isIconToken(val)) {
      return val.value
    }
    if (val instanceof Date) {
      return val.getTime()
    }
    if (Array.isArray(val)) {
      return val.join(', ')
    }
    if (typeof val === 'string' || typeof val === 'number') {
      return val
    }
    if (typeof val === 'boolean') {
      return val ? 1 : 0
    }
    if (val === null || val === undefined) {
      return val
    }
    return String(val)
  }

  type SortKey = string | number | null | undefined
  const numRules = sortConfig.length
  const firstProp = sortConfig[0].property

  const decorated = files.map((file, index) => {
    const keys: SortKey[] = [
      normalizeSortValue(resolveValueWithFormulas(file, firstProp, getContext, getPropertyExpr)),
    ]
    const computed: boolean[] = [true]
    return { file, index, keys, computed }
  })

  const getKey = (entry: (typeof decorated)[number], ruleIdx: number): SortKey => {
    if (!entry.computed[ruleIdx]) {
      entry.keys[ruleIdx] = normalizeSortValue(
        resolveValueWithFormulas(
          entry.file,
          sortConfig[ruleIdx].property,
          getContext,
          getPropertyExpr,
        ),
      )
      entry.computed[ruleIdx] = true
    }
    return entry.keys[ruleIdx]
  }

  decorated.sort((a, b) => {
    for (let i = 0; i < numRules; i++) {
      const direction = sortConfig[i].direction
      const aVal = getKey(a, i)
      const bVal = getKey(b, i)

      let comparison = 0
      if (aVal === undefined || aVal === null || aVal === '') {
        if (bVal === undefined || bVal === null || bVal === '') {
          comparison = 0
        } else {
          comparison = 1
        }
      } else if (bVal === undefined || bVal === null || bVal === '') {
        comparison = -1
      } else if (typeof aVal === 'string' && typeof bVal === 'string') {
        comparison = sortCollator.compare(aVal, bVal)
      } else {
        const aNumber = typeof aVal === 'number' ? aVal : Number(aVal)
        const bNumber = typeof bVal === 'number' ? bVal : Number(bVal)
        if (Number.isFinite(aNumber) && Number.isFinite(bNumber)) {
          comparison = aNumber > bNumber ? 1 : aNumber < bNumber ? -1 : 0
        } else {
          comparison = sortCollator.compare(String(aVal), String(bVal))
        }
      }

      if (comparison !== 0) {
        return direction === 'ASC' ? comparison : -comparison
      }
    }
    return a.index - b.index
  })

  return decorated.map(entry => entry.file)
}

function groupValue(value: unknown): BaseGroupValue {
  if (value === undefined || value === null || value === '') return null
  if (Array.isArray(value)) return value.length > 0 ? String(value) : null
  if (isIconToken(value)) return value.value
  if (value instanceof Date) return value.toISOString().split('T')[0]
  if (typeof value === 'number' || typeof value === 'boolean') return value
  return String(value)
}

const groupLabel = (value: BaseGroupValue): string => (value === null ? '(empty)' : String(value))

function fileGroupValues(
  file: QuartzPluginData,
  groupBy: string | BaseGroupBy,
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
): BaseGroupValue[] {
  const property = typeof groupBy === 'string' ? groupBy : groupBy.property
  const value = resolveValueWithFormulas(file, property, getContext, getPropertyExpr)
  if (Array.isArray(value)) {
    return value.length > 0 ? [...new Set(value.map(groupValue))] : [null]
  }
  return [groupValue(value)]
}

function groupFiles(
  files: QuartzPluginData[],
  groupBy: string | BaseGroupBy,
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
  groupOrder?: BaseGroupValue[],
): Map<BaseGroupValue, QuartzPluginData[]> {
  const groups = new Map<BaseGroupValue, QuartzPluginData[]>()

  const direction = typeof groupBy === 'string' ? 'ASC' : groupBy.direction

  for (const file of files) {
    for (const key of fileGroupValues(file, groupBy, getContext, getPropertyExpr)) {
      const members = groups.get(key) ?? []
      members.push(file)
      groups.set(key, members)
    }
  }

  if (groupOrder !== undefined) {
    return new Map(groupOrder.map(value => [value, groups.get(value) ?? []]))
  }

  const sortedGroups = new Map(
    [...groups.entries()].sort(([a], [b]) => {
      if (a === null) return b === null ? 0 : 1
      if (b === null) return -1
      const comparison =
        typeof a === 'number' && typeof b === 'number' ? a - b : String(a).localeCompare(String(b))
      return direction === 'ASC' ? comparison : -comparison
    }),
  )

  return sortedGroups
}

function buildTableSummaryRow(
  columns: string[],
  files: QuartzPluginData[],
  summaryConfig: ViewSummaryConfig | undefined,
  allFiles: QuartzPluginData[],
  summaryExpressions: Record<string, ProgramIR> | undefined,
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
): RenderElement | undefined {
  if (!summaryConfig?.columns || Object.keys(summaryConfig.columns).length === 0) {
    return undefined
  }

  const summaryValues = computeViewSummaries(
    columns,
    files,
    summaryConfig,
    allFiles,
    getContext,
    (file, column) => resolveValueWithFormulas(file, column, getContext, getPropertyExpr),
    summaryExpressions,
  )

  const hasValues = Object.values(summaryValues).some(v => v !== undefined)
  if (!hasValues) {
    return undefined
  }

  const cells: RenderElement[] = columns.map(col => {
    const value = summaryValues[col]
    if (value === undefined) {
      return h('td.base-summary-cell', {}, '')
    }
    if (isIconToken(value)) {
      const cleaned = value.value.trim()
      return cleaned.length > 0
        ? h('td.base-summary-cell', {}, [renderIconNode(cleaned)])
        : h('td.base-summary-cell', {}, '')
    }
    return h('td.base-summary-cell', {}, String(value))
  })

  return h('tfoot', [h('tr.base-summary-row', cells)])
}

function buildTable(
  files: QuartzPluginData[],
  view: BaseView,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
  properties?: Record<string, PropertyConfig>,
  topLevelSummaries?: Record<string, string>,
  summaryExpressions?: Record<string, ProgramIR>,
): RenderElement {
  const columns = view.order || []

  const summaryConfig = parseViewSummaries(view.summaries, topLevelSummaries)

  if (view.groupBy) {
    const groups = groupFiles(files, view.groupBy, getContext, getPropertyExpr, view.groupOrder)
    const allRows: RenderElement[] = []
    const groupByKey = typeof view.groupBy === 'string' ? view.groupBy : view.groupBy.property
    const groupByLabel = getPropertyDisplayName(groupByKey, properties)

    for (const [groupName, groupFiles] of groups) {
      const groupHeader = h('tr.base-group-header', [
        h('td', { colspan: columns.length }, [
          h('span.base-group-header-label', groupByLabel.toLowerCase()),
          ' ',
          groupLabel(groupName),
        ]),
      ])
      allRows.push(groupHeader)

      for (const file of groupFiles) {
        const cells = columns.map(col =>
          buildTableCell(file, col, currentSlug, allFiles, getContext, getPropertyExpr),
        )
        allRows.push(h('tr', { 'data-base-slug': file.slug }, cells))
      }
    }

    const tbody = h('tbody', allRows)
    const thead = h('thead', buildTableHead(columns, properties))
    const table = h('table.base-table', [thead, tbody])
    return h('div.base-table-wrapper', [table])
  }

  const rows = files.map(f => {
    const cells = columns.map(col =>
      buildTableCell(f, col, currentSlug, allFiles, getContext, getPropertyExpr),
    )
    return h('tr', { 'data-base-slug': f.slug }, cells)
  })

  const tbody = h('tbody', rows)
  const thead = h('thead', buildTableHead(columns, properties))
  const tfoot = buildTableSummaryRow(
    columns,
    files,
    summaryConfig,
    allFiles,
    summaryExpressions,
    getContext,
    getPropertyExpr,
  )
  const tableChildren = tfoot ? [thead, tbody, tfoot] : [thead, tbody]
  const table = h('table.base-table', tableChildren)
  return h('div.base-table-wrapper', [table])
}

function listValueToPlainText(value: unknown): string | undefined {
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

function hasRenderableValue(value: unknown): boolean {
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

function renderPropertyValueNodes(
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

function createListItemRenderer(
  view: BaseView,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
  properties?: Record<string, PropertyConfig>,
): (file: QuartzPluginData) => RenderElement {
  const nestedProperties = view.nestedProperties === true || view.indentProperties === true
  const order = Array.isArray(view.order) && view.order.length > 0 ? view.order : ['title']
  const [primaryProp, ...secondaryProps] = order
  const rawSeparator = typeof view.separator === 'string' ? view.separator : ','
  const separator = rawSeparator.endsWith(' ') ? rawSeparator : `${rawSeparator} `

  return file => {
    const slug = (file.slug || '') as FullSlug
    const fallbackTitle = getFileDisplayName(file) ?? fallbackNameFromSlug(slug)

    const primaryValue = primaryProp
      ? resolveValueWithFormulas(file, primaryProp, getContext, getPropertyExpr)
      : resolveValueWithFormulas(file, 'title', getContext, getPropertyExpr)
    const primaryText = listValueToPlainText(primaryValue) ?? fallbackTitle
    const anchor = buildFileLinkNode(slug, currentSlug, primaryText)

    const seen = new Set<string>()
    if (primaryProp) {
      seen.add(primaryProp)
    }

    if (!nestedProperties) {
      const inlineNodes: RenderNode[] = []

      for (const propertyKey of secondaryProps) {
        if (!propertyKey || seen.has(propertyKey)) continue
        const value = resolveValueWithFormulas(file, propertyKey, getContext, getPropertyExpr)
        if (!hasRenderableValue(value)) continue

        const renderedValue = renderPropertyValueNodes(value, currentSlug, allFiles)
        if (renderedValue.length === 0) continue

        inlineNodes.push(separator)
        inlineNodes.push(...renderedValue)
        seen.add(propertyKey)
      }

      return h('li', { 'data-base-slug': slug }, [anchor, ...inlineNodes])
    }

    const metadataItems: RenderElement[] = []

    for (const propertyKey of secondaryProps) {
      if (!propertyKey || seen.has(propertyKey)) continue
      const value = resolveValueWithFormulas(file, propertyKey, getContext, getPropertyExpr)
      if (!hasRenderableValue(value)) continue

      const renderedValue = renderPropertyValueNodes(value, currentSlug, allFiles)
      if (renderedValue.length === 0) continue

      const label = getPropertyDisplayName(propertyKey, properties)
      metadataItems.push(h('li', [h('span.base-list-meta-label', `${label}: `), ...renderedValue]))
      seen.add(propertyKey)
    }

    if (metadataItems.length === 0) {
      return h('li', { 'data-base-slug': slug }, [anchor])
    }

    return h('li', { 'data-base-slug': slug }, [anchor, h('ul.base-list-nested', metadataItems)])
  }
}

function buildList(
  files: QuartzPluginData[],
  view: BaseView,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
  properties?: Record<string, PropertyConfig>,
): RenderElement {
  const renderListItem = createListItemRenderer(
    view,
    currentSlug,
    allFiles,
    getContext,
    getPropertyExpr,
    properties,
  )

  if (view.groupBy) {
    const groups = groupFiles(files, view.groupBy, getContext, getPropertyExpr, view.groupOrder)
    const groupElements: RenderElement[] = []

    for (const [groupName, groupedFiles] of groups) {
      const items = groupedFiles.map(file => renderListItem(file))
      groupElements.push(
        h('div.base-list-group', [
          h('h3.base-list-group-header', groupLabel(groupName)),
          h('ul.base-list', items),
        ]),
      )
    }

    return h('div.base-list-container', groupElements)
  }

  const items = files.map(file => renderListItem(file))
  return h('ul.base-list', items)
}

function normalizeCalendarDate(value: unknown): string | undefined {
  if (value instanceof Date) {
    return value.toISOString().split('T')[0]
  }
  if (typeof value === 'string') {
    const match = value.match(/^\d{4}-\d{2}-\d{2}/)
    if (match) return match[0]
    const parsed = new Date(value)
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toISOString().split('T')[0]
    }
    return undefined
  }
  if (typeof value === 'number' && Number.isFinite(value)) {
    const parsed = new Date(value)
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toISOString().split('T')[0]
    }
  }
  return undefined
}

function buildCalendar(
  files: QuartzPluginData[],
  view: BaseView,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
  properties?: Record<string, PropertyConfig>,
): RenderElement {
  const dateField =
    typeof view.date === 'string'
      ? view.date
      : typeof view.dateField === 'string'
        ? view.dateField
        : typeof view.dateProperty === 'string'
          ? view.dateProperty
          : 'date'

  const renderListItem = createListItemRenderer(
    view,
    currentSlug,
    allFiles,
    getContext,
    getPropertyExpr,
    properties,
  )

  const groups = new Map<string, QuartzPluginData[]>()
  for (const file of files) {
    const dateValue = resolveValueWithFormulas(file, dateField, getContext, getPropertyExpr)
    const dateKey = normalizeCalendarDate(dateValue) ?? '(no date)'
    if (!groups.has(dateKey)) {
      groups.set(dateKey, [])
    }
    groups.get(dateKey)!.push(file)
  }

  const groupElements: RenderElement[] = []
  const sorted = [...groups.entries()].sort(([a], [b]) => a.localeCompare(b))
  for (const [dateKey, groupedFiles] of sorted) {
    const items = groupedFiles.map(file => renderListItem(file))
    groupElements.push(
      h('div.base-calendar-group', [
        h('h3.base-calendar-group-header', dateKey),
        h('ul.base-list', items),
      ]),
    )
  }

  return h('div.base-calendar-container', groupElements)
}

function resolveCardImage(
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

function resolveCardImageUrl(imageValue: unknown, currentSlug: FullSlug): string | undefined {
  return resolveCardImage(imageValue, currentSlug)?.url
}

function buildCards(
  files: QuartzPluginData[],
  view: BaseView,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
  properties?: Record<string, PropertyConfig>,
): RenderElement {
  const isKanban = view.type === 'kanban'
  if (isKanban && !view.groupBy) {
    return h('p.base-kanban-empty', 'Choose a groupBy property to organize this view into columns.')
  }
  const imageField = view.image || (isKanban ? undefined : 'image')
  const order = view.order ?? []
  const titleField = isKanban ? (order[0] ?? 'file.name') : undefined

  const renderCard = (file: QuartzPluginData): RenderElement => {
    const slug = (file.slug || '') as FullSlug
    const titleValue = titleField
      ? resolveValueWithFormulas(file, titleField, getContext, getPropertyExpr)
      : undefined
    const title =
      listValueToPlainText(titleValue) ?? getFileDisplayName(file) ?? fallbackNameFromSlug(slug)
    const href = resolveRelative(currentSlug, slug)

    const imageValue = imageField
      ? resolveValueWithFormulas(file, imageField, getContext, getPropertyExpr)
      : undefined
    const imageUrl = resolveCardImageUrl(imageValue, currentSlug)

    const metadataItems: RenderElement[] = []
    const metadataFields = (isKanban ? order.slice(1) : order).filter(
      (field): field is string =>
        typeof field === 'string' &&
        (isKanban || (field !== 'title' && field !== 'file.title' && field !== 'note.title')) &&
        field !== imageField,
    )

    for (const field of metadataFields) {
      const value = resolveValueWithFormulas(file, field, getContext, getPropertyExpr)
      if (!hasRenderableValue(value)) continue
      const renderedValue = renderPropertyValueNodes(value, currentSlug, allFiles)
      if (renderedValue.length === 0) continue
      const label = getPropertyDisplayName(field, properties)
      metadataItems.push(
        h('div.base-card-meta-item', [
          h('span.base-card-meta-label', label),
          h('span.base-card-meta-value', renderedValue),
        ]),
      )
    }

    const cardChildren: RenderElement[] = []
    if (imageUrl) {
      cardChildren.push(
        h('a.base-card-image-link', { href, 'data-slug': slug, 'aria-label': title }, [
          h('img.base-card-image', { src: imageUrl, alt: '', loading: 'lazy', decoding: 'async' }),
        ]),
      )
    }

    const contentChildren: RenderElement[] = [
      h('a.base-card-title-link', { href, 'data-slug': slug }, [h('h3.base-card-title', title)]),
    ]
    if (metadataItems.length > 0) {
      contentChildren.push(h('div.base-card-meta', metadataItems))
    }

    cardChildren.push(h('div.base-card-content', contentChildren))

    return h(isKanban ? 'li.base-card' : 'div.base-card', { 'data-base-slug': slug }, cardChildren)
  }

  const styleParts: string[] = []
  if (typeof view.cardSize === 'number' && view.cardSize > 0) {
    styleParts.push(`--base-card-min: ${view.cardSize}px;`)
  }
  // Obsidian stores height / width; CSS aspect-ratio uses width / height.
  if (view.imageAspectRatio !== undefined) {
    styleParts.push(`--base-card-aspect: 1 / ${view.imageAspectRatio};`)
  } else if (typeof view.cardAspect === 'number' && view.cardAspect > 0) {
    styleParts.push(`--base-card-aspect: ${view.cardAspect};`)
  }
  if (view.imageFit) styleParts.push(`--base-card-image-fit: ${view.imageFit};`)
  if (isKanban) styleParts.push(`--base-kanban-width: ${view.columnWidth ?? 280}px;`)
  const varStyle = styleParts.length > 0 ? styleParts.join(' ') : undefined

  if (view.groupBy) {
    const groups = groupFiles(files, view.groupBy, getContext, getPropertyExpr, view.groupOrder)
    const groupElements: RenderElement[] = []

    const groupSizes = view.groupSizes
    const groupAspects = view.groupAspects

    for (const [groupName, groupFiles] of groups) {
      const cards = groupFiles.map(file => renderCard(file))
      const label = groupLabel(groupName)
      if (isKanban) {
        groupElements.push(
          h('section.base-kanban-column', { 'aria-label': groupName === null ? 'None' : label }, [
            h('div.base-kanban-column-header', [
              h('h3.base-kanban-column-title', groupName === null ? 'None' : label),
              h(
                'span.base-kanban-count',
                { 'aria-label': `${cards.length} results` },
                String(cards.length),
              ),
            ]),
            h('ul.base-kanban-cards', cards),
            h('p.base-kanban-column-empty', { hidden: cards.length > 0 }, 'No results'),
          ]),
        )
        continue
      }
      const parts: string[] = []
      const size = groupSizes?.[label]
      if (typeof size === 'number' && size > 0) {
        parts.push(`--base-card-min: ${size}px;`)
      }
      const aspect = groupAspects?.[label]
      if (typeof aspect === 'number' && aspect > 0) {
        parts.push(`--base-card-aspect: ${aspect};`)
      }
      const gridStyle = parts.length > 0 ? parts.join(' ') : undefined

      groupElements.push(
        h('div.base-card-group', [
          h('h3.base-card-group-header', label),
          h('div.base-card-grid', gridStyle ? { style: gridStyle } : {}, cards),
        ]),
      )
    }

    if (isKanban) {
      return groupElements.length > 0
        ? h(
            'div.base-kanban',
            { style: varStyle, tabindex: 0, role: 'region', 'aria-label': `${view.name} board` },
            groupElements,
          )
        : h('p.base-kanban-empty', 'No results in the visible groups.')
    }

    return h('div.base-card-container', varStyle ? { style: varStyle } : {}, groupElements)
  }

  const cards = files.map(file => renderCard(file))
  return h('div.base-card-grid', varStyle ? { style: varStyle } : {}, cards)
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

function plainTextDescription(description: string | undefined, currentSlug: FullSlug): string {
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

const SPINE_HEX = /^#(?:[0-9a-f]{3}|[0-9a-f]{6})$/i

function expandHex(hex: string): string {
  return hex.length === 4 ? `#${Array.from(hex.slice(1), c => c + c).join('')}` : hex
}

// Spines carry the surname, as a shelf does; the index keeps the full credit.
function spineSurname(author: string): string {
  const first = author.split(/,|&| and /)[0].trim()
  return first.split(/\s+/).pop() ?? first
}

function buildBookcase(
  files: QuartzPluginData[],
  view: BaseView,
  currentSlug: FullSlug,
  allFiles: QuartzPluginData[],
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
  properties?: Record<string, PropertyConfig>,
): RenderElement {
  const bookHeight = clampOption(
    view.bookHeight,
    BOOKCASE_DEFAULTS.bookHeight,
    BOOKCASE_LIMITS.bookHeight,
  )
  const shelfWidth = clampOption(
    view.shelfWidth,
    BOOKCASE_DEFAULTS.shelfWidth,
    BOOKCASE_LIMITS.shelfWidth,
  )
  const scrollStep = clampOption(
    view.scrollStep,
    BOOKCASE_DEFAULTS.scrollStep,
    BOOKCASE_LIMITS.scrollStep,
  )
  const [imageField, titleField, authorField, colorField] = BOOKCASE_PROPERTY_DEFAULTS.map(
    ([key, fallback]) => {
      const value = view[key]
      return typeof value === 'string' ? value : fallback
    },
  )
  const pageContent = view.pageContent !== false
  // `author` in `order` and `note.author` in `spineAuthor` name the same property.
  const bare = (field: string | undefined) => field?.replace(/^note\./, '')
  const shown = new Set([titleField, authorField, imageField].map(bare))
  const order = (view.order ?? []).filter(field => !shown.has(bare(field)))
  const resolve = (file: QuartzPluginData, field: string | undefined) =>
    field ? resolveValueWithFormulas(file, field, getContext, getPropertyExpr) : undefined

  const grouped: [string | null, QuartzPluginData[]][] = view.groupBy
    ? [...groupFiles(files, view.groupBy, getContext, getPropertyExpr, view.groupOrder)].map(
        ([value, members]) => [value === null ? 'none' : groupLabel(value), members],
      )
    : [[null, files]]
  const groups = grouped.filter(([, members]) => members.length > 0)

  type Entry = SceneBook & { file: QuartzPluginData; fullTitle: string; byline: string }
  const toEntry = (file: QuartzPluginData): Entry => {
    const slug = (file.slug || '') as FullSlug
    const fullTitle =
      listValueToPlainText(resolve(file, titleField)) ??
      getFileDisplayName(file) ??
      fallbackNameFromSlug(slug)
    const author = listValueToPlainText(resolve(file, authorField))
    const cover = resolveCardImage(resolve(file, imageField), currentSlug)
    const explicit = listValueToPlainText(resolve(file, colorField))
    const path = String(file.relativePath ?? file.filePath ?? slug)
    const spine = explicit && SPINE_HEX.test(explicit) ? expandHex(explicit) : fallbackSpine(path)
    const lead = order[0] ? listValueToPlainText(resolve(file, order[0])) : undefined
    return {
      file,
      slug,
      fullTitle,
      title: fullTitle.split(':')[0].trim(),
      author: author ? spineSurname(author) : undefined,
      byline: [author, lead].filter(Boolean).join(' · '),
      spine,
      ink: spineInk(spine),
      cover: cover?.url,
      coverSlug: cover?.slug,
      sampleSpine: !(explicit && SPINE_HEX.test(explicit)),
    }
  }

  const entryGroups = groups.map(([label, members]) => ({ label, entries: members.map(toEntry) }))
  const layout = packShelves(
    entryGroups.map(({ label, entries }) => ({
      label,
      books: entries.map(entry => ({
        path: String(entry.file.relativePath ?? entry.file.filePath ?? entry.slug),
        item: entry,
      })),
    })),
    { bookHeight, shelfWidth },
  )
  const total = layout.books.length

  const detailOf = (entry: Entry): RenderElement[] => {
    const rows: RenderElement[] = []
    for (const field of order) {
      const value = resolve(entry.file, field)
      if (!hasRenderableValue(value)) continue
      const rendered = renderPropertyValueNodes(value, currentSlug, allFiles)
      if (rendered.length === 0) continue
      rows.push(
        h('div.bookcase-prop', [
          h('dt', getPropertyDisplayName(field, properties)),
          h('dd', rendered),
        ]),
      )
    }
    const nodes: RenderElement[] = []
    if (rows.length > 0) nodes.push(h('dl.bookcase-props', rows))
    const description = entry.file.frontmatter?.description
    if (pageContent && typeof description === 'string' && description.trim()) {
      nodes.push(
        h('p.bookcase-description', plainTextDescription(description, entry.slug as FullSlug)),
      )
    }
    return nodes
  }

  let index = 0
  const indexGroups = entryGroups.map(({ label, entries }) => {
    const rows = entries.map(entry => {
      const href = resolveRelative(currentSlug, entry.slug as FullSlug)
      return h(
        'li.bookcase-entry',
        {
          'data-base-slug': entry.slug,
          'data-book-index': index++,
          'data-cover': entry.cover,
          'data-bookcase-cover': entry.coverSlug,
        },
        [
          h('a.bookcase-entry-link', { href, 'data-slug': entry.slug }, [
            h('span.bookcase-entry-title', entry.fullTitle),
            entry.byline ? h('span.bookcase-entry-byline', entry.byline) : '',
          ]),
          h('div.bookcase-entry-detail', detailOf(entry)),
        ],
      )
    })
    const list = h('ul.base-list.bookcase-index-list', rows)
    return label === null
      ? list
      : h('details.base-list-group.bookcase-index-group', { open: true }, [
          h('summary.base-list-group-header', [
            h('span.bookcase-index-label', label),
            h('span.bookcase-index-count', String(entries.length)),
          ]),
          list,
        ])
  })

  const first = entryGroups[0]?.entries[0]
  const page = first
    ? [
        h('div.bookcase-page-cover', [
          h(
            'button.bookcase-preview-open',
            { type: 'button', 'aria-label': 'Preview selected book' },
            [
              h('img', {
                src: first.cover,
                alt: '',
                hidden: first.cover ? undefined : true,
                decoding: 'async',
                'data-no-popover': '',
                'data-bookcase-cover': first.coverSlug,
              }),
            ],
          ),
        ]),
        h('div.bookcase-page-body', [
          h(
            'a.bookcase-page-title',
            { href: resolveRelative(currentSlug, first.slug as FullSlug), 'data-slug': first.slug },
            first.fullTitle,
          ),
          h('p.bookcase-page-byline', first.byline),
          h('div.bookcase-page-detail', detailOf(first)),
        ]),
      ]
    : [h('p.bookcase-empty', 'No books in this view.')]

  const pad = (n: number) => String(n).padStart(String(total).length, '0')
  const controlIcon = (paths: string[]) =>
    s(
      'svg',
      { viewBox: '0 0 24 24', 'aria-hidden': 'true' },
      paths.map(d => s('path', { d })),
    )
  return h(
    'div.bookcase',
    { style: `--bookcase-step: ${scrollStep}px;`, 'data-scroll-step': scrollStep },
    [
      h('div.bookcase-stage', [
        h('div.bookcase-plate', [
          h(
            'button.bookcase-sidebar-toggle',
            {
              type: 'button',
              'aria-expanded': 'true',
              'aria-label': 'Hide list',
              title: 'Hide list',
            },
            controlIcon(['M9 5h12M9 12h12M9 19h12M3 5h1M3 12h1M3 19h1']),
          ),
          h(
            'span.bookcase-corner',
            { 'data-corner': 'title', 'data-bookcase-shelf': '' },
            entryGroups[0]?.label ?? '',
          ),
          h(
            'div.bookcase-viewport',
            {
              tabindex: 0,
              role: 'group',
              'aria-label':
                'Bookcase drawing. Arrow keys pan, plus and minus zoom, Home fits all books, Enter previews the selected book.',
            },
            [
              h('div.bookcase-world', [
                bookcaseScene(
                  layout,
                  `${view.name}: ${total} books on ${layout.shelves.length} shelves`,
                ),
              ]),
            ],
          ),
          h('div.bookcase-camera-controls', { role: 'group', 'aria-label': 'Bookcase controls' }, [
            h(
              'button',
              { type: 'button', 'data-camera': 'in', 'aria-label': 'Zoom in', title: 'Zoom in' },
              controlIcon(['M12 5v14M5 12h14']),
            ),
            h(
              'button',
              { type: 'button', 'data-camera': 'out', 'aria-label': 'Zoom out', title: 'Zoom out' },
              controlIcon(['M5 12h14']),
            ),
            h(
              'button',
              {
                type: 'button',
                'data-camera': 'fit',
                'aria-label': 'Fit bookcase',
                title: 'Fit bookcase',
              },
              controlIcon(['M8 3H3v5M16 3h5v5M21 16v5h-5M3 16v5h5']),
            ),
          ]),
          h('p.bookcase-no-results', 'no books match this search'),
          h(
            'span.bookcase-corner',
            { 'data-corner': 'hint' },
            'drag to explore · click to preview',
          ),
          h(
            'span.bookcase-corner',
            { 'data-corner': 'readout', 'data-bookcase-readout': '' },
            total > 0 ? `${pad(1)} / ${total}` : '–',
          ),
        ]),
        h('div.bookcase-page', page),
      ]),
      h('div.bookcase-index', indexGroups),
      h('dialog.bookcase-preview', { 'aria-label': 'Book preview' }, [
        h('div.bookcase-preview-toolbar', [
          h('span.bookcase-preview-position'),
          h(
            'button',
            {
              type: 'button',
              'data-preview': 'close',
              'aria-label': 'Close preview',
              title: 'Close preview',
              autofocus: true,
            },
            controlIcon(['M6 6l12 12M18 6 6 18']),
          ),
        ]),
        h('div.bookcase-preview-content', [
          h('div.bookcase-preview-art', { 'aria-hidden': 'true' }),
          h('div.bookcase-preview-body', [
            h('h2', [h('a.bookcase-preview-title.bookcase-preview-link')]),
            h('p.bookcase-preview-byline'),
            h('div.bookcase-preview-detail'),
          ]),
        ]),
        h('div.bookcase-preview-navigation', [
          h(
            'button',
            {
              type: 'button',
              'data-preview': 'previous',
              'aria-label': 'Previous book',
              title: 'Previous book',
            },
            controlIcon(['M15 5l-7 7 7 7']),
          ),
          h(
            'button',
            {
              type: 'button',
              'data-preview': 'next',
              'aria-label': 'Next book',
              title: 'Next book',
            },
            controlIcon(['M9 5l7 7-7 7']),
          ),
        ]),
      ]),
    ],
  )
}

function buildMap(
  files: QuartzPluginData[],
  view: BaseView,
  currentSlug: FullSlug,
  getContext: EvalContextFactory,
  getPropertyExpr: PropertyExprGetter,
  properties?: Record<string, PropertyConfig>,
): RenderElement {
  const resolveMapProperty = (file: QuartzPluginData, prop: string | undefined): unknown => {
    if (!prop) return undefined
    const key = prop.trim()
    if (!key) return undefined
    if (getPropertyExpr(key)) {
      return resolveValueWithFormulas(file, key, getContext, getPropertyExpr)
    }
    if (key.startsWith('note.')) {
      const stripped = key.replace(/^note\./, '')
      if (getPropertyExpr(stripped)) {
        return resolveValueWithFormulas(file, stripped, getContext, getPropertyExpr)
      }
    } else {
      const prefixed = `note.${key}`
      if (getPropertyExpr(prefixed)) {
        return resolveValueWithFormulas(file, prefixed, getContext, getPropertyExpr)
      }
    }
    return undefined
  }

  const coordinatesProp = view.coordinates || 'coordinates'

  const markers: MarkerData[] = []

  for (const file of files) {
    const coordsValue = resolveMapProperty(file, coordinatesProp)
    if (!coordsValue || !Array.isArray(coordsValue) || coordsValue.length < 2) {
      continue
    }

    const lat = parseFloat(String(coordsValue[0]))
    const lon = parseFloat(String(coordsValue[1]))

    if (isNaN(lat) || isNaN(lon)) {
      continue
    }

    const title = getFileDisplayName(file) ?? fallbackNameFromSlug((file.slug || '') as FullSlug)
    const slug = (file.slug || '') as FullSlug

    const popupFields: Record<string, unknown> = {}
    const order = Array.isArray(view.order) ? view.order : []
    for (const field of order) {
      if (typeof field !== 'string') continue
      if (field === 'title' || field === 'file.title' || field === 'note.title') continue
      const value = resolveValueWithFormulas(file, field, getContext, getPropertyExpr)
      if (hasRenderableValue(value)) {
        popupFields[field] = value
      }
    }

    const icon = resolveMapProperty(file, view.markerIcon)
    const color = resolveMapProperty(file, view.markerColor)
    const iconHtml = isIconToken(icon)
      ? renderIconHtml(icon.value)
      : icon
        ? String(icon)
        : undefined

    markers.push({
      lat,
      lon,
      title,
      slug,
      icon: iconHtml,
      color: color ? String(color) : undefined,
      description:
        typeof file.frontmatter?.description === 'string'
          ? plainTextDescription(file.frontmatter.description, slug)
          : typeof file.description === 'string'
            ? file.description
            : undefined,
      address: typeof file.frontmatter?.address === 'string' ? file.frontmatter.address : undefined,
      categories: Array.isArray(file.frontmatter?.type)
        ? file.frontmatter.type.filter((value): value is string => typeof value === 'string')
        : typeof file.frontmatter?.type === 'string'
          ? [file.frontmatter.type]
          : [],
      rating: typeof file.frontmatter?.rating === 'number' ? file.frontmatter.rating : undefined,
      url: typeof file.frontmatter?.url === 'string' ? file.frontmatter.url : undefined,
      popupFields,
    })
  }

  const config = {
    defaultZoom: view.defaultZoom ?? 12,
    defaultCenter: view.defaultCenter,
    clustering: view.clustering !== false,
  }

  const attrs: Record<string, string> = {
    'data-markers': JSON.stringify(markers),
    'data-config': JSON.stringify(config),
    'data-current-slug': currentSlug,
  }

  if (properties) {
    attrs['data-properties'] = JSON.stringify(properties)
  }

  return h('div.base-map', attrs)
}

function resolveViewSlug(baseSlug: FullSlug, viewName: string, viewIndex: number): FullSlug {
  if (viewIndex === 0) return baseSlug
  const slugifiedName = slugifyFilePath((viewName + '.tmp') as FilePath, true)
  return joinSegments(baseSlug, slugifiedName) as FullSlug
}

function renderDiagnostics(
  diagnostics: BaseExpressionDiagnostic[] | undefined,
  currentSlug: FullSlug,
): RenderElement | undefined {
  if (!diagnostics || diagnostics.length === 0) {
    return undefined
  }
  const items: RenderElement[] = diagnostics.map((diag, index) => {
    const line = diag.span.start.line
    const column = diag.span.start.column
    const location = Number.isFinite(line) && Number.isFinite(column) ? `${line}:${column}` : ''
    const label = location.length > 0 ? `${diag.context} (${location})` : diag.context
    return h('li.base-diagnostics-item', { key: String(index) }, [
      h('div.base-diagnostics-label', label),
      h('div.base-diagnostics-message', diag.message),
      h('code.base-diagnostics-source', diag.source),
    ])
  })
  return h('div.base-diagnostics', [
    h('div.base-diagnostics-title', 'bases diagnostics'),
    h('div.base-diagnostics-meta', [
      h('span', 'page'),
      h('span.base-diagnostics-page', currentSlug),
    ]),
    h('ul.base-diagnostics-list', items),
  ])
}

export type BaseViewMeta = { name: string; type: BaseView['type']; slug: FullSlug }

export type RenderedBaseView = {
  view: BaseView
  slug: FullSlug
  tree: Root
  resultCount: number
  totalCount: number
  matchedSlugs: string[]
  diagnostics: BaseExpressionDiagnostic[]
}

export type BaseMetadata = { baseSlug: FullSlug; currentView: string; allViews: BaseViewMeta[] }

const publicFilesCache = new WeakMap<QuartzPluginData[], QuartzPluginData[]>()

function getPublicBaseFiles(allFiles: QuartzPluginData[]): QuartzPluginData[] {
  let filtered = publicFilesCache.get(allFiles)
  if (!filtered) {
    filtered = allFiles.filter(isPublicBaseFile)
    publicFilesCache.set(allFiles, filtered)
  }
  return filtered
}

export function renderBaseViewsForFile(
  baseFileData: QuartzPluginData,
  allFiles: QuartzPluginData[],
  thisFile?: QuartzPluginData,
): { views: RenderedBaseView[]; allViews: BaseViewMeta[] } {
  const config = baseFileData.basesConfig as BaseFile | undefined
  if (!config || !baseFileData.slug) {
    return { views: [], allViews: [] }
  }
  allFiles = getPublicBaseFiles(allFiles)

  const baseSlug = baseFileData.slug as FullSlug
  const expressions = baseFileData.basesExpressions
  const formulaExpressions = expressions?.formulas
  const summaryExpressionsByView = expressions?.viewSummaries
  const viewFilterExpressions = expressions?.viewFilters
  const formulaCaches = new Map<string, Map<string, Value>>()
  const propertyCaches = new Map<string, Map<string, Value>>()
  const runtimeDiagnostics: BaseExpressionDiagnostic[] = []
  const runtimeDiagnosticSet = new Set<string>()
  const propertyExpressions = expressions?.propertyExpressions ?? {}

  const fileIndex = getFileLookup(allFiles).bySimple

  const getPropertyExpr: PropertyExprGetter = property => {
    const key = property.trim()
    if (!key) return null
    return propertyExpressions[key] ?? null
  }

  const baseThisFile = thisFile ?? baseFileData
  const evalContexts = new Map<QuartzPluginData, EvalContext>()

  const getEvalContext: EvalContextFactory = file => {
    let ctx = evalContexts.get(file)
    if (!ctx) {
      const slug = file.slug ? String(file.slug) : file.filePath ? String(file.filePath) : ''
      let cache = formulaCaches.get(slug)
      if (!cache) {
        cache = new Map()
        formulaCaches.set(slug, cache)
      }
      let propertyCache = propertyCaches.get(slug)
      if (!propertyCache) {
        propertyCache = new Map()
        propertyCaches.set(slug, propertyCache)
      }
      ctx = {
        file,
        thisFile: baseThisFile,
        allFiles,
        fileIndex,
        formulas: formulaExpressions,
        formulaSources: config.formulas,
        formulaCache: cache,
        formulaStack: new Set(),
        propertyCache,
        diagnostics: runtimeDiagnostics,
        diagnosticSet: runtimeDiagnosticSet,
      }
      evalContexts.set(file, ctx)
    }
    return ctx
  }

  const allViews = config.views.map((view, idx) => ({
    name: view.name,
    type: view.type,
    slug: resolveViewSlug(baseSlug, view.name, idx),
  }))

  const baseFilterSource = typeof config.filters === 'string' ? config.filters : ''
  const baseMatchedFiles = expressions?.filters
    ? allFiles.filter(file => {
        const ctx = getEvalContext(file)
        ctx.diagnosticContext = 'filters'
        ctx.diagnosticSource = baseFilterSource
        return evaluateFilterExpression(expressions.filters!, ctx)
      })
    : allFiles

  const views: RenderedBaseView[] = []

  for (const [viewIndex, view] of config.views.entries()) {
    const slug = resolveViewSlug(baseSlug, view.name, viewIndex)

    let matchedFiles = baseMatchedFiles
    const viewFilter = viewFilterExpressions ? viewFilterExpressions[String(viewIndex)] : undefined
    if (viewFilter) {
      const viewFilterSource = typeof view.filters === 'string' ? view.filters : ''
      matchedFiles = matchedFiles.filter(file => {
        const ctx = getEvalContext(file)
        ctx.diagnosticContext = `views[${viewIndex}].filters`
        ctx.diagnosticSource = viewFilterSource
        return evaluateFilterExpression(viewFilter, ctx)
      })
    }

    const groupBy = view.groupBy
    const visibleGroups = view.groupOrder !== undefined ? new Set(view.groupOrder) : undefined
    const visibleFiles =
      groupBy && visibleGroups
        ? matchedFiles.filter(file =>
            fileGroupValues(file, groupBy, getEvalContext, getPropertyExpr).some(value =>
              visibleGroups.has(value),
            ),
          )
        : view.type === 'kanban' && !groupBy
          ? []
          : matchedFiles
    const sortedFiles = applySorting(visibleFiles, view.sort, getEvalContext, getPropertyExpr)
    const limitedFiles = view.limit !== undefined ? sortedFiles.slice(0, view.limit) : sortedFiles
    const totalCount = visibleFiles.length
    const resultCount = limitedFiles.length

    const wrapView = (node: RenderElement) =>
      h('div.base-view', { 'data-base-view-type': view.type, 'data-base-view-name': view.name }, [
        node,
      ])

    let viewNode: RenderElement | undefined
    switch (view.type) {
      case 'table':
        viewNode = buildTable(
          limitedFiles,
          view,
          slug,
          allFiles,
          getEvalContext,
          getPropertyExpr,
          config.properties,
          config.summaries,
          summaryExpressionsByView ? summaryExpressionsByView[String(viewIndex)] : undefined,
        )
        break
      case 'list':
        viewNode = buildList(
          limitedFiles,
          view,
          slug,
          allFiles,
          getEvalContext,
          getPropertyExpr,
          config.properties,
        )
        break
      case 'card':
      case 'cards':
      case 'gallery':
      case 'board':
      case 'kanban':
        viewNode = buildCards(
          limitedFiles,
          view,
          slug,
          allFiles,
          getEvalContext,
          getPropertyExpr,
          config.properties,
        )
        break
      case 'bookcase':
        viewNode = buildBookcase(
          limitedFiles,
          view,
          slug,
          allFiles,
          getEvalContext,
          getPropertyExpr,
          config.properties,
        )
        break
      case 'calendar':
        viewNode = buildCalendar(
          limitedFiles,
          view,
          slug,
          allFiles,
          getEvalContext,
          getPropertyExpr,
          config.properties,
        )
        break
      case 'map':
        viewNode = buildMap(
          limitedFiles,
          view,
          slug,
          getEvalContext,
          getPropertyExpr,
          config.properties,
        )
        break
      default:
        viewNode = undefined
    }

    if (!viewNode) continue

    const diagnostics = [...(baseFileData.basesDiagnostics ?? []), ...runtimeDiagnostics]
    const diagnosticsNode = renderDiagnostics(
      diagnostics.length > 0 ? diagnostics : undefined,
      slug,
    )
    const wrapped = wrapView(viewNode)
    const tree: Root = {
      type: 'root',
      children: diagnosticsNode ? [diagnosticsNode, wrapped] : [wrapped],
    }

    const matchedSlugs = matchedFiles.flatMap(file =>
      file.slug ? [simplifySlug(file.slug as FullSlug)] : [],
    )

    views.push({ view, slug, tree, resultCount, totalCount, matchedSlugs, diagnostics })
  }

  return { views, allViews }
}
