import { toHtml } from 'hast-util-to-html'
import yaml from 'js-yaml'
import type { QuartzPluginData } from '../../plugins/vfile'
import type { BaseExpressionDiagnostic } from './compiler'
import { isFullSlug } from '../path'
import { isRecord } from '../type-guards'
import {
  BASE_QUERY_DATA_PATH,
  hydrateBaseQueryRecord,
  readBaseQueryData,
  type BrowserBaseRecord,
} from './browser-data'
import { compileBaseConfig } from './compile'
import { renderBaseViewsForFile } from './render'
import { BUILTIN_SUMMARY_TYPES, type BaseFile, type BaseView } from './types'

export { baseQuerySource, parseBaseQueryDocument } from './browser-data'

export interface BaseQueryFailure {
  ok: false
  diagnostics: string[]
}

export type ParsedBaseQuery = { ok: true; config: BaseFile } | BaseQueryFailure

export type BaseQueryResult =
  | {
      ok: true
      html: string
      resultCount: number
      totalCount: number
      config: BaseFile
      view: BaseView
    }
  | BaseQueryFailure

export interface BaseQueryRequest {
  source: string
  baseFile: BrowserBaseRecord
  thisFile?: BrowserBaseRecord
  files: QuartzPluginData[]
  viewName: string
  currentSlug?: string
}

const nonEmptyString = (value: unknown): value is string =>
  typeof value === 'string' && value.trim().length > 0

const sortDirection = (value: unknown): boolean => value === 'ASC' || value === 'DESC'

const isSort = (value: unknown): boolean =>
  isRecord(value) && nonEmptyString(value.property) && sortDirection(value.direction)

function validFilter(value: unknown): boolean {
  if (nonEmptyString(value)) return true
  if (!isRecord(value) || Object.keys(value).length !== 1) return false
  const key = Object.keys(value)[0]
  if (key !== 'and' && key !== 'or' && key !== 'not') return false
  const children = value[key]
  return Array.isArray(children) && children.every(validFilter)
}

function stringExpressions(value: unknown): boolean {
  return isRecord(value) && Object.values(value).every(nonEmptyString)
}

function validSummary(value: unknown): boolean {
  if (nonEmptyString(value)) return true
  if (!isRecord(value)) return false
  if (value.type === 'builtin')
    return BUILTIN_SUMMARY_TYPES.some(type => type === value.builtinType)
  return value.type === 'formula' && nonEmptyString(value.expression)
}

function validateSource(source: string): string[] {
  const raw: unknown = yaml.load(source)
  if (!isRecord(raw)) return ['A base must be a YAML object with a views list.']
  const diagnostics: string[] = []
  const rootKeys = new Set(['filters', 'views', 'properties', 'summaries', 'formulas'])
  for (const key of Object.keys(raw))
    if (!rootKeys.has(key)) diagnostics.push(`Unknown base option: ${key}.`)
  if (raw.filters !== undefined && !validFilter(raw.filters))
    diagnostics.push('filters must be an expression or one and/or/not group.')
  for (const key of ['formulas', 'summaries'])
    if (raw[key] !== undefined && !stringExpressions(raw[key]))
      diagnostics.push(`${key} must map names to non-empty expressions.`)
  if (
    raw.properties !== undefined &&
    (!isRecord(raw.properties) ||
      !Object.values(raw.properties).every(
        value =>
          isRecord(value) &&
          (value.displayName === undefined || typeof value.displayName === 'string'),
      ))
  )
    diagnostics.push('properties must map property names to display settings.')
  if (!Array.isArray(raw.views) || raw.views.length === 0)
    return [...diagnostics, 'views must contain at least one named view.']
  const names = new Set<string>()
  raw.views.forEach((view: unknown, index: number) => {
    const context = `views[${index}]`
    if (!isRecord(view)) {
      diagnostics.push(`${context} must be an object.`)
      return
    }
    if (typeof view.name === 'string') {
      if (names.has(view.name)) diagnostics.push(`${context}.name must be unique.`)
      names.add(view.name)
    }
    if (view.filters !== undefined && !validFilter(view.filters))
      diagnostics.push(`${context}.filters must be an expression or one and/or/not group.`)
    if (
      view.order !== undefined &&
      (!Array.isArray(view.order) || !view.order.every(nonEmptyString))
    )
      diagnostics.push(`${context}.order must be a list of property names.`)
    if (view.sort !== undefined && (!Array.isArray(view.sort) || !view.sort.every(isSort)))
      diagnostics.push(`${context}.sort must contain property and ASC/DESC direction pairs.`)
    if (view.groupBy !== undefined && !nonEmptyString(view.groupBy) && !isSort(view.groupBy))
      diagnostics.push(`${context}.groupBy must be a property or property/direction pair.`)
    if (view.type === 'kanban' && view.groupBy === undefined)
      diagnostics.push(`${context}.groupBy is required for a kanban view.`)
    if (
      view.limit !== undefined &&
      (typeof view.limit !== 'number' || !Number.isInteger(view.limit) || view.limit < 0)
    )
      diagnostics.push(`${context}.limit must be a non-negative integer.`)
    for (const property of [
      'image',
      'separator',
      'date',
      'dateField',
      'dateProperty',
      'coordinates',
      'markerIcon',
      'markerColor',
    ])
      if (view[property] !== undefined && typeof view[property] !== 'string')
        diagnostics.push(`${context}.${property} must be a string.`)
    for (const property of ['nestedProperties', 'indentProperties', 'clustering'])
      if (view[property] !== undefined && typeof view[property] !== 'boolean')
        diagnostics.push(`${context}.${property} must be true or false.`)
    for (const property of ['cardSize', 'cardAspect', 'columnWidth', 'imageAspectRatio'])
      if (
        view[property] !== undefined &&
        (typeof view[property] !== 'number' ||
          !Number.isFinite(view[property]) ||
          view[property] <= 0)
      )
        diagnostics.push(`${context}.${property} must be a positive number.`)
    if (
      view.defaultZoom !== undefined &&
      (typeof view.defaultZoom !== 'number' ||
        !Number.isFinite(view.defaultZoom) ||
        view.defaultZoom < 0 ||
        view.defaultZoom > 24)
    )
      diagnostics.push(`${context}.defaultZoom must be between 0 and 24.`)
    if (
      view.defaultCenter !== undefined &&
      (!Array.isArray(view.defaultCenter) ||
        view.defaultCenter.length !== 2 ||
        !view.defaultCenter.every(value => typeof value === 'number' && Number.isFinite(value)))
    )
      diagnostics.push(`${context}.defaultCenter must contain two coordinates.`)
    if (view.summaries !== undefined) {
      const summaries = view.summaries
      const columns = isRecord(summaries) && 'columns' in summaries ? summaries.columns : summaries
      if (!isRecord(columns) || !Object.values(columns).every(validSummary))
        diagnostics.push(`${context}.summaries must map properties to summary expressions.`)
    }
  })
  return diagnostics
}

function diagnosticMessage(diagnostic: BaseExpressionDiagnostic): string {
  const { line, column } = diagnostic.span.start
  return `${diagnostic.context} (${line}:${column}): ${diagnostic.message}`
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function compileQuery(source: string) {
  const diagnostics = validateSource(source)
  if (diagnostics.length > 0) return { ok: false, diagnostics } satisfies BaseQueryFailure
  const compiled = compileBaseConfig(source)
  if (compiled.diagnostics.length > 0)
    return {
      ok: false,
      diagnostics: compiled.diagnostics.map(diagnosticMessage),
    } satisfies BaseQueryFailure
  return { ok: true, compiled } satisfies { ok: true; compiled: typeof compiled }
}

export function parseBaseQuery(source: string): ParsedBaseQuery {
  try {
    const parsed = compileQuery(source)
    return parsed.ok ? { ok: true, config: parsed.compiled.config } : parsed
  } catch (error) {
    return { ok: false, diagnostics: [errorMessage(error)] }
  }
}

export function runBaseQuery(request: BaseQueryRequest): BaseQueryResult {
  try {
    const parsed = compileQuery(request.source)
    if (!parsed.ok) return parsed
    const { config, expressions } = parsed.compiled
    const viewIndex = Math.max(
      0,
      config.views.findIndex(view => view.name === request.viewName),
    )
    const view = config.views[viewIndex]
    if (!view) return { ok: false, diagnostics: ['No view is available.'] }
    const baseFile = hydrateBaseQueryRecord(request.baseFile)
    const thisFile = request.thisFile ? hydrateBaseQueryRecord(request.thisFile) : baseFile
    const currentSlug = request.currentSlug ?? request.baseFile.slug
    if (!isFullSlug(currentSlug)) return { ok: false, diagnostics: ['Invalid current page path.'] }
    // Rendering one view keeps Apply bounded while retaining the complete source for export.
    const renderingData: QuartzPluginData = {
      ...baseFile,
      slug: currentSlug,
      basesConfig: { ...config, views: [view] },
      basesExpressions: {
        ...expressions,
        viewFilters: expressions.viewFilters[String(viewIndex)]
          ? { '0': expressions.viewFilters[String(viewIndex)] }
          : {},
        viewSummaries: expressions.viewSummaries[String(viewIndex)]
          ? { '0': expressions.viewSummaries[String(viewIndex)] }
          : {},
      },
      basesDiagnostics: [],
    }
    const rendered = renderBaseViewsForFile(renderingData, request.files, thisFile).views[0]
    if (!rendered) return { ok: false, diagnostics: ['The selected view could not be rendered.'] }
    if (rendered.diagnostics.length > 0)
      return { ok: false, diagnostics: rendered.diagnostics.map(diagnosticMessage) }
    return {
      ok: true,
      html: toHtml(rendered.tree),
      resultCount: rendered.resultCount,
      totalCount: rendered.totalCount,
      config,
      view,
    }
  } catch (error) {
    return { ok: false, diagnostics: [errorMessage(error)] }
  }
}

const dataCache = new Map<string, Promise<QuartzPluginData[]>>()

if (typeof document !== 'undefined') {
  document.addEventListener('nav', () => {
    if (!document.querySelector('[data-base-query]')) {
      dataCache.clear()
    }
  })
}

export function loadBaseQueryData(url = BASE_QUERY_DATA_PATH): Promise<QuartzPluginData[]> {
  const cached = dataCache.get(url)
  if (cached) return cached
  const pending = (async () => {
    const response = await fetch(url)
    if (!response.ok) throw new Error(`Base data could not be loaded (${response.status}).`)
    const value: unknown = await response.json()
    return readBaseQueryData(value)
  })().catch(cause => {
    dataCache.delete(url)
    throw cause
  })
  dataCache.set(url, pending)
  return pending
}
