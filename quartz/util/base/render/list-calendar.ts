import { h } from 'hastscript'
import { QuartzPluginData } from '../../../plugins/vfile'
import { FullSlug } from '../../path'
import { BaseView, PropertyConfig } from '../types'
import {
  groupFiles,
  groupLabel,
  resolveValueWithFormulas,
  type EvalContextFactory,
  type PropertyExprGetter,
} from './property-evaluation'
import {
  buildFileLinkNode,
  fallbackNameFromSlug,
  getFileDisplayName,
  getPropertyDisplayName,
  hasRenderableValue,
  listValueToPlainText,
  renderPropertyValueNodes,
  type RenderElement,
  type RenderNode,
} from './property-presentation'

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

    // Missing values keep their slot as "–" so the property columns line up across rows.
    for (const propertyKey of secondaryProps) {
      if (!propertyKey || seen.has(propertyKey)) continue
      const value = resolveValueWithFormulas(file, propertyKey, getContext, getPropertyExpr)
      const renderedValue = hasRenderableValue(value)
        ? renderPropertyValueNodes(value, currentSlug, allFiles)
        : []

      const label = getPropertyDisplayName(propertyKey, properties)
      metadataItems.push(
        h('li', [
          h('span.base-list-meta-label', `${label}:`),
          ...(renderedValue.length > 0 ? renderedValue : ['–']),
        ]),
      )
      seen.add(propertyKey)
    }

    if (metadataItems.length === 0) {
      return h('li', { 'data-base-slug': slug }, [anchor])
    }

    return h('li', { 'data-base-slug': slug }, [anchor, h('ul.base-list-nested', metadataItems)])
  }
}

export function buildList(
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

export function buildCalendar(
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
