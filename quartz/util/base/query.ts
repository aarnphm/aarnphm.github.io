import { QuartzPluginData } from '../../plugins/vfile'
import { evaluateSummaryExpression, valueToUnknown, EvalContext, ProgramIR } from './compiler'
import { SummaryDefinition, ViewSummaryConfig, BuiltinSummaryType } from './types'

type SummaryValueResolver = (
  file: QuartzPluginData,
  column: string,
  allFiles: QuartzPluginData[],
) => unknown

type SummaryContextFactory = (file: QuartzPluginData) => EvalContext

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)

const isValidDate = (value: unknown): value is Date =>
  value instanceof Date && Number.isFinite(value.getTime())

const formatDate = (timestamp: number): string | undefined => {
  const date = new Date(timestamp)
  return Number.isFinite(date.getTime()) ? date.toISOString().split('T')[0] : undefined
}

const dateTimestamp = (value: unknown): number | undefined => {
  if (isValidDate(value)) return value.getTime()
  if (typeof value === 'number') {
    return Number.isFinite(new Date(value).getTime()) ? value : undefined
  }
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}/.test(value)) return undefined
  const timestamp = new Date(value).getTime()
  return Number.isFinite(timestamp) ? timestamp : undefined
}

const computeExtremum = (values: unknown[], type: 'min' | 'max'): string | number | undefined => {
  const dates = values.filter(isValidDate)
  const numbers = values.filter(isFiniteNumber)
  const numericValues = [...numbers, ...dates.map(date => date.getTime())]

  if (numericValues.length > 0) {
    const extremum = type === 'min' ? Math.min(...numericValues) : Math.max(...numericValues)
    return dates.length > 0 ? formatDate(extremum) : extremum
  }

  const strings = values.filter((value): value is string => typeof value === 'string')
  if (strings.length === 0) return undefined
  return strings.reduce((extremum, value) => {
    if (type === 'min') return value < extremum ? value : extremum
    return value > extremum ? value : extremum
  })
}

export function computeColumnSummary(
  column: string,
  files: QuartzPluginData[],
  summary: SummaryDefinition,
  allFiles: QuartzPluginData[] = [],
  valueResolver: SummaryValueResolver,
  getContext: SummaryContextFactory,
  summaryExpression?: ProgramIR,
): string | number | undefined {
  if (files.length === 0) {
    return undefined
  }

  const values = files.map(file => valueResolver(file, column, allFiles))

  if (summary.type === 'builtin' && summary.builtinType) {
    return computeBuiltinSummary(values, summary.builtinType)
  }

  if (summary.type === 'formula' && summary.expression) {
    if (summaryExpression) {
      const summaryCtx = getContext(files[0])
      summaryCtx.diagnosticContext = `summaries.${column}`
      summaryCtx.diagnosticSource = summary.expression
      summaryCtx.rows = files
      const value = evaluateSummaryExpression(summaryExpression, values, summaryCtx)
      const unknownValue = valueToUnknown(value)
      if (typeof unknownValue === 'number' || typeof unknownValue === 'string') {
        return unknownValue
      }
      return undefined
    }
  }

  return undefined
}

function computeBuiltinSummary(
  values: unknown[],
  type: BuiltinSummaryType,
): string | number | undefined {
  switch (type) {
    case 'count':
      return values.length

    case 'sum': {
      const nums = values.filter(isFiniteNumber)
      if (nums.length === 0) return undefined
      return nums.reduce((acc, v) => acc + v, 0)
    }

    case 'average':
    case 'avg': {
      const nums = values.filter(isFiniteNumber)
      if (nums.length === 0) return undefined
      const sum = nums.reduce((acc, v) => acc + v, 0)
      return Math.round((sum / nums.length) * 100) / 100
    }

    case 'min':
      return computeExtremum(values, 'min')

    case 'max':
      return computeExtremum(values, 'max')

    case 'range': {
      const dates = values.filter(isValidDate)
      const nums = [...values.filter(isFiniteNumber), ...dates.map(date => date.getTime())]
      if (nums.length === 0) return undefined
      const min = Math.min(...nums)
      const max = Math.max(...nums)
      if (dates.length > 0) {
        const start = formatDate(min)
        const end = formatDate(max)
        return start && end ? `${start} - ${end}` : undefined
      }
      return `${min} - ${max}`
    }

    case 'unique': {
      const nonNull = values.filter(v => v !== undefined && v !== null && v !== '')
      const normalized = nonNull.flatMap(value => {
        if (!(value instanceof Date)) return [String(value)]
        return isValidDate(value) ? [value.toISOString()] : []
      })
      const unique = new Set(normalized)
      return unique.size
    }

    case 'filled': {
      const filled = values.filter(v => v !== undefined && v !== null && v !== '')
      return filled.length
    }

    case 'missing': {
      const missing = values.filter(v => v === undefined || v === null || v === '')
      return missing.length
    }

    case 'median': {
      const nums = values.filter(isFiniteNumber)
      if (nums.length === 0) return undefined
      const sorted = [...nums].sort((a, b) => a - b)
      const mid = Math.floor(sorted.length / 2)
      if (sorted.length % 2 === 0) {
        return (sorted[mid - 1] + sorted[mid]) / 2
      }
      return sorted[mid]
    }

    case 'stddev': {
      const nums = values.filter(isFiniteNumber)
      if (nums.length === 0) return undefined
      const mean = nums.reduce((acc, v) => acc + v, 0) / nums.length
      const variance = nums.reduce((acc, v) => acc + (v - mean) * (v - mean), 0) / nums.length
      return Math.round(Math.sqrt(variance) * 100) / 100
    }

    case 'checked':
      return values.filter(v => v === true).length

    case 'unchecked':
      return values.filter(v => v === false).length

    case 'empty': {
      const count = values.filter(
        v =>
          v === undefined ||
          v === null ||
          v === '' ||
          (Array.isArray(v) && v.length === 0) ||
          (typeof v === 'object' && v !== null && !Array.isArray(v) && Object.keys(v).length === 0),
      ).length
      return count
    }

    case 'earliest': {
      const timestamps = values.flatMap(value => {
        const timestamp = dateTimestamp(value)
        return timestamp === undefined ? [] : [timestamp]
      })
      if (timestamps.length === 0) return undefined
      const earliest = Math.min(...timestamps)
      return formatDate(earliest)
    }

    case 'latest': {
      const timestamps = values.flatMap(value => {
        const timestamp = dateTimestamp(value)
        return timestamp === undefined ? [] : [timestamp]
      })
      if (timestamps.length === 0) return undefined
      const latest = Math.max(...timestamps)
      return formatDate(latest)
    }

    default:
      return undefined
  }
}

export function computeViewSummaries(
  columns: string[],
  files: QuartzPluginData[],
  summaryConfig: ViewSummaryConfig | undefined,
  allFiles: QuartzPluginData[] = [],
  getContext: SummaryContextFactory,
  valueResolver: SummaryValueResolver,
  summaryExpressions?: Record<string, ProgramIR>,
): Record<string, string | number | undefined> {
  const results: Record<string, string | number | undefined> = {}

  if (!summaryConfig?.columns) {
    return results
  }

  for (const column of columns) {
    const summary = summaryConfig.columns[column]
    if (summary) {
      const expression = summaryExpressions ? summaryExpressions[column] : undefined
      results[column] = computeColumnSummary(
        column,
        files,
        summary,
        allFiles,
        valueResolver,
        getContext,
        expression,
      )
    }
  }

  return results
}
