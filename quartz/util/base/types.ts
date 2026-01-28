import { QuartzPluginData } from "../../plugins/vfile"
import {
  SummaryDefinition,
  ViewSummaryConfig,
  PropertyConfig,
  BuiltinSummaryType,
  BUILTIN_SUMMARY_TYPES,
} from "./compiler/schema"

export type { SummaryDefinition, ViewSummaryConfig, PropertyConfig, BuiltinSummaryType }
export { BUILTIN_SUMMARY_TYPES }

export type BasesConfigFileFilter =
  | string
  | { and: BasesConfigFileFilter[] }
  | { or: BasesConfigFileFilter[] }
  | { not: BasesConfigFileFilter[] }

export interface BasesConfigFile {
  filters?: BasesConfigFileFilter
  views: BasesConfigFileView[]
  properties?: Record<string, PropertyConfig>
  summaries?: Record<string, string>
  formulas?: Record<string, string>
}

export type BaseFile = BasesConfigFile

export interface BasesConfigFileView {
  type: "table" | "list" | "gallery" | "board" | "calendar" | "card" | "cards" | "map"
  name: string
  order?: string[]
  sort?: BasesSortConfig[]
  columnSize?: Record<string, number>
  groupBy?: string | BasesGroupByConfig
  limit?: number
  filters?: BasesConfigFileFilter
  summaries?: Record<string, string> | ViewSummaryConfig
  image?: string
  cardSize?: number
  nestedProperties?: boolean
  coordinates?: string
  markerIcon?: string
  markerColor?: string
  defaultZoom?: number
  defaultCenter?: [number, number]
  clustering?: boolean
  [key: string]: any
}

export type BaseView = BasesConfigFileView

export interface BasesSortConfig {
  property: string
  direction: "ASC" | "DESC"
}

export type BaseSortConfig = BasesSortConfig

export interface BasesGroupByConfig {
  property: string
  direction: "ASC" | "DESC"
}

export type BaseGroupBy = BasesGroupByConfig

export interface BaseTableData {
  view: BaseView
  rows: QuartzPluginData[]
  columns: string[]
}

export function parseDuration(durationStr: string): number {
  const str = durationStr.trim()

  const asNumber = Number(str)
  if (!isNaN(asNumber)) {
    return asNumber
  }

  let totalMs = 0
  const regex = /(\d+(?:\.\d+)?)\s*([a-zA-Z]+)/g
  let match
  while ((match = regex.exec(str)) !== null) {
    const value = parseFloat(match[1])
    const unitRaw = match[2]
    const unit = unitRaw.toLowerCase()
    if (unitRaw === "M" || unit === "mo" || unit === "month" || unit === "months") {
      totalMs += value * 30 * 24 * 60 * 60 * 1000
      continue
    }
    if (unit === "ms" || unit === "millisecond" || unit === "milliseconds") {
      totalMs += value
      continue
    }
    if (
      unit === "s" ||
      unit === "sec" ||
      unit === "secs" ||
      unit === "second" ||
      unit === "seconds"
    ) {
      totalMs += value * 1000
      continue
    }
    if (
      unit === "m" ||
      unit === "min" ||
      unit === "mins" ||
      unit === "minute" ||
      unit === "minutes"
    ) {
      totalMs += value * 60 * 1000
      continue
    }
    if (unit === "h" || unit === "hr" || unit === "hrs" || unit === "hour" || unit === "hours") {
      totalMs += value * 60 * 60 * 1000
      continue
    }
    if (unit === "d" || unit === "day" || unit === "days") {
      totalMs += value * 24 * 60 * 60 * 1000
      continue
    }
    if (unit === "w" || unit === "week" || unit === "weeks") {
      totalMs += value * 7 * 24 * 60 * 60 * 1000
      continue
    }
    if (unit === "y" || unit === "yr" || unit === "yrs" || unit === "year" || unit === "years") {
      totalMs += value * 365 * 24 * 60 * 60 * 1000
      continue
    }
  }

  return totalMs
}

export function parseViews(raw: any): BaseView[] {
  if (!Array.isArray(raw)) throw new Error("Views must be an array")

  return raw.map((v) => {
    if (!v || typeof v !== "object") throw new Error("Each view must be an object")
    if (!v.type || !v.name) throw new Error("Each view must have 'type' and 'name' fields")

    return {
      ...v,
      type: v.type,
      name: v.name,
      order: v.order,
      sort: v.sort,
      columnSize: v.columnSize,
      groupBy: v.groupBy,
      limit: v.limit,
      filters: v.filters,
      summaries: v.summaries,
    } as BaseView
  })
}

export function parseViewSummaries(
  viewSummaries: Record<string, string> | ViewSummaryConfig | undefined,
  topLevelSummaries?: Record<string, string>,
): ViewSummaryConfig | undefined {
  if (!viewSummaries || typeof viewSummaries !== "object") return undefined

  if ("columns" in viewSummaries && typeof viewSummaries.columns === "object") {
    return viewSummaries as ViewSummaryConfig
  }

  const columns: Record<string, SummaryDefinition> = {}

  for (const [column, summaryValue] of Object.entries(viewSummaries)) {
    if (typeof summaryValue !== "string") continue

    const normalized = summaryValue.toLowerCase().trim()

    if (BUILTIN_SUMMARY_TYPES.includes(normalized as BuiltinSummaryType)) {
      columns[column] = { type: "builtin", builtinType: normalized as BuiltinSummaryType }
      continue
    }

    if (topLevelSummaries && summaryValue in topLevelSummaries) {
      columns[column] = {
        type: "formula",
        formulaRef: summaryValue,
        expression: topLevelSummaries[summaryValue],
      }
      continue
    }

    if (summaryValue.includes("(") || summaryValue.includes(".")) {
      columns[column] = { type: "formula", expression: summaryValue }
    }
  }

  return Object.keys(columns).length > 0 ? { columns } : undefined
}
