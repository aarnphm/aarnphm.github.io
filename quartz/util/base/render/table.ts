import { h } from 'hastscript'
import { QuartzPluginData } from '../../../plugins/vfile'
import { FullSlug } from '../../path'
import { ProgramIR } from '../compiler'
import { computeViewSummaries } from '../query'
import { BaseView, parseViewSummaries, PropertyConfig, ViewSummaryConfig } from '../types'
import {
  groupFiles,
  groupLabel,
  resolveValueWithFormulas,
  type EvalContextFactory,
  type PropertyExprGetter,
} from './property-evaluation'
import {
  buildFileLinkNode,
  getFileBaseName,
  getPropertyDisplayName,
  isIconToken,
  renderBacklinkNodes,
  renderIconNode,
  renderInlineString,
  type RenderElement,
  type RenderNode,
} from './property-presentation'

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
  if (nodes.length === 0 || (nodes.length === 1 && nodes[0] === '')) return h('td', {})
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
      return h('td', {})
    }
    const entries = backlinks.filter((entry): entry is string => typeof entry === 'string')
    if (entries.length === 0) {
      return h('td', {})
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
    return h('td', {})
  }

  if (isIconToken(value)) {
    const cleaned = value.value.trim()
    return cleaned.length > 0 ? h('td', {}, [renderIconNode(cleaned)]) : h('td', {})
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
    return h('td', {}, [renderBooleanCheckbox(value)])
  }

  return wrapTableCellContent(String(value))
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

export function buildTable(
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
  const wrapTable = (table: RenderElement) =>
    h(
      'div.base-table-wrapper',
      {
        tabindex: 0,
        role: 'region',
        'aria-label': `${view.name} table`,
        'data-base-table-group-by':
          typeof view.groupBy === 'string' ? view.groupBy : view.groupBy?.property,
      },
      [table],
    )

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
    return wrapTable(table)
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
  return wrapTable(table)
}
