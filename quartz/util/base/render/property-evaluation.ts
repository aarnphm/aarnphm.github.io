import { QuartzPluginData } from '../../../plugins/vfile'
import {
  buildPropertyExpressionSource,
  EvalContext,
  evaluateExpression,
  ProgramIR,
  Value,
  valueToUnknown,
} from '../compiler'
import { BaseGroupBy, BaseGroupValue, BaseSortConfig } from '../types'
import { isIconToken } from './property-presentation'

export type EvalContextFactory = (file: QuartzPluginData) => EvalContext

export type PropertyExprGetter = (property: string) => ProgramIR | null

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

export function resolveValueWithFormulas(
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

export function applySorting(
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

export const groupLabel = (value: BaseGroupValue): string =>
  value === null ? '(empty)' : String(value)

export function fileGroupValues(
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

export function groupFiles(
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
