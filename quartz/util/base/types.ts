import { isRecord } from '../type-guards'
import {
  SummaryDefinition,
  ViewSummaryConfig,
  PropertyConfig,
  BuiltinSummaryType,
  BUILTIN_SUMMARY_TYPES,
} from './compiler/schema'

export type { SummaryDefinition, ViewSummaryConfig, PropertyConfig, BuiltinSummaryType }
export { BUILTIN_SUMMARY_TYPES }

const isNonEmptyString = (value: unknown): value is string =>
  typeof value === 'string' && value.trim().length > 0

const BASE_VIEW_TYPES = [
  'table',
  'list',
  'gallery',
  'board',
  'calendar',
  'card',
  'cards',
  'map',
] as const

type BaseViewType = (typeof BASE_VIEW_TYPES)[number]

const isBaseViewType = (value: unknown): value is BaseViewType =>
  BASE_VIEW_TYPES.some(type => type === value)

export type BaseFileFilter =
  | string
  | { and: BaseFileFilter[] }
  | { or: BaseFileFilter[] }
  | { not: BaseFileFilter[] }

export interface BaseFile {
  filters?: BaseFileFilter
  views: BaseView[]
  properties?: Record<string, PropertyConfig>
  summaries?: Record<string, string>
  formulas?: Record<string, string>
}

export interface BaseView {
  type: BaseViewType
  name: string
  order?: string[]
  sort?: BaseSortConfig[]
  columnSize?: Record<string, number>
  groupBy?: string | BaseGroupBy
  limit?: number
  filters?: BaseFileFilter
  summaries?: Record<string, string> | ViewSummaryConfig
  image?: string
  cardSize?: number
  cardAspect?: number
  nestedProperties?: boolean
  indentProperties?: boolean
  separator?: string
  date?: string
  dateField?: string
  dateProperty?: string
  coordinates?: string
  markerIcon?: string
  markerColor?: string
  defaultZoom?: number
  defaultCenter?: [number, number]
  clustering?: boolean
  groupSizes?: Record<string, number>
  groupAspects?: Record<string, number>
}

export interface BaseSortConfig {
  property: string
  direction: 'ASC' | 'DESC'
}

export interface BaseGroupBy {
  property: string
  direction: 'ASC' | 'DESC'
}

export function parseViews(raw: unknown[]): BaseView[] {
  return raw.map((entry, index) => {
    if (!isRecord(entry)) throw new Error(`View at index ${index} must be an object`)
    const { type, name } = entry
    if (!isNonEmptyString(type) || !isNonEmptyString(name)) {
      throw new Error(`View at index ${index} must have non-empty 'type' and 'name' fields`)
    }
    if (!isBaseViewType(type)) {
      throw new Error(`View at index ${index} has unsupported type '${type}'`)
    }
    return { ...entry, type, name }
  })
}

export function parseViewSummaries(
  viewSummaries: Record<string, string> | ViewSummaryConfig | undefined,
  topLevelSummaries?: Record<string, string>,
): ViewSummaryConfig | undefined {
  if (!viewSummaries || typeof viewSummaries !== 'object') return undefined

  if ('columns' in viewSummaries && typeof viewSummaries.columns === 'object') {
    return viewSummaries as ViewSummaryConfig
  }

  const columns: Record<string, SummaryDefinition> = {}

  for (const [column, summaryValue] of Object.entries(viewSummaries)) {
    if (typeof summaryValue !== 'string') continue

    const normalized = summaryValue.toLowerCase().trim()

    if (BUILTIN_SUMMARY_TYPES.includes(normalized as BuiltinSummaryType)) {
      columns[column] = { type: 'builtin', builtinType: normalized as BuiltinSummaryType }
      continue
    }

    if (topLevelSummaries && summaryValue in topLevelSummaries) {
      columns[column] = {
        type: 'formula',
        formulaRef: summaryValue,
        expression: topLevelSummaries[summaryValue],
      }
      continue
    }

    if (summaryValue.includes('(') || summaryValue.includes('.')) {
      columns[column] = { type: 'formula', expression: summaryValue }
    }
  }

  return Object.keys(columns).length > 0 ? { columns } : undefined
}
