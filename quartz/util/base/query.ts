import { QuartzPluginData } from "../../plugins/vfile"
import { simplifySlug, FullSlug } from "../path"
import {
  ComparisonOp,
  compileExpression,
  parseDuration,
  FormulaDefinition,
  SummaryDefinition,
  ViewSummaryConfig,
  BuiltinSummaryType,
} from "./types"

export type BaseFilter =
  | { type: "and"; conditions: BaseFilter[] }
  | { type: "or"; conditions: BaseFilter[] }
  | { type: "not"; conditions: BaseFilter[] }
  | { type: "function"; name: string; args: string[] }
  | {
      type: "comparison"
      property: string
      operator: ComparisonOp
      value: string | number | boolean | Date
      isExpression?: boolean
    }
  | { type: "method"; property: string; method: string; args: string[]; negated: boolean }

export type FilePredicate = (file: QuartzPluginData, allFiles: QuartzPluginData[]) => boolean

export function resolvePropertyValue(
  file: QuartzPluginData,
  property: string,
  allFiles: QuartzPluginData[] = [],
): any {
  if (property.startsWith("file.")) {
    switch (property) {
      case "file.name": {
        const filePath = file.filePath as string | undefined
        if (filePath) {
          const segment = filePath.split("/").pop() || ""
          return segment.replace(/\.[^/.]+$/, "")
        }
        const slug = file.slug || ""
        const segment = slug.split("/").pop() || ""
        return segment.replace(/\.[^/.]+$/, "")
      }
      case "file.title": {
        const title = file.frontmatter?.title
        if (typeof title === "string" && title.length > 0) {
          return title
        }
        const filePath = file.filePath as string | undefined
        if (filePath) {
          const segment = filePath.split("/").pop() || ""
          return segment.replace(/\.[^/.]+$/, "")
        }
        const slug = file.slug || ""
        const segment = slug.split("/").pop() || ""
        return segment.replace(/\.[^/.]+$/, "")
      }
      case "file.path":
        return file.filePath || file.slug || ""
      case "file.link":
        return file.slug || ""
      case "file.folder": {
        const parts = file.slug?.split("/") || []
        return parts.length > 1 ? parts.slice(0, -1).join("/") : ""
      }
      case "file.ext": {
        const name = file.filePath?.split("/").pop() || file.slug?.split("/").pop() || ""
        const match = name.match(/\.([^.]+)$/)
        return match ? match[1] : "md"
      }
      case "file.tags":
        return file.frontmatter?.tags || []
      case "file.outlinks":
      case "file.links":
        return file.links || []
      case "file.inlinks":
      case "file.backlinks": {
        const slug = file.slug
        if (!slug) {
          return []
        }
        const simpleSlug = simplifySlug(slug)
        return allFiles
          .filter((f) => f.links?.includes(simpleSlug))
          .map((f) => f.slug)
          .filter(Boolean)
      }
      case "file.aliases":
        return file.frontmatter?.aliases || []
      case "file.ctime":
        return file.dates?.created ? new Date(file.dates.created) : undefined
      case "file.mtime":
        return file.dates?.modified ? new Date(file.dates.modified) : undefined
      default:
        return undefined
    }
  }

  let key = property
  if (key.startsWith("note.")) {
    key = key.slice("note.".length)
  }

  if (key.includes(".")) {
    const segments = key.split(".")
    let current: any = file.frontmatter
    for (const segment of segments) {
      if (current && typeof current === "object" && segment in current) {
        current = current[segment as keyof typeof current]
      } else {
        return undefined
      }
    }
    return current
  }

  return file.frontmatter?.[key]
}

export function evaluateFormula(
  formula: FormulaDefinition,
  file: QuartzPluginData,
  allFiles: QuartzPluginData[] = [],
): any {
  if (formula.property && formula.operator !== undefined && formula.value !== undefined) {
    const propValue = resolvePropertyValue(file, formula.property, allFiles)

    const normalizeValue = (val: any): any => {
      if (val instanceof Date) return val.getTime()
      if (typeof val === "string" && /^\d{4}-\d{2}-\d{2}/.test(val)) {
        const parsed = new Date(val)
        if (!isNaN(parsed.getTime())) return parsed.getTime()
      }
      return val
    }

    const normalizedProp = normalizeValue(propValue)
    const normalizedTarget = normalizeValue(formula.value)

    switch (formula.operator) {
      case "==":
        return normalizedProp === normalizedTarget
      case "!=":
        return normalizedProp !== normalizedTarget
      case ">":
        return (
          typeof normalizedProp === "number" &&
          typeof normalizedTarget === "number" &&
          normalizedProp > normalizedTarget
        )
      case "<":
        return (
          typeof normalizedProp === "number" &&
          typeof normalizedTarget === "number" &&
          normalizedProp < normalizedTarget
        )
      case ">=":
        return (
          typeof normalizedProp === "number" &&
          typeof normalizedTarget === "number" &&
          normalizedProp >= normalizedTarget
        )
      case "<=":
        return (
          typeof normalizedProp === "number" &&
          typeof normalizedTarget === "number" &&
          normalizedProp <= normalizedTarget
        )
      default:
        return undefined
    }
  }

  return undefined
}

function normalizeLinkTarget(raw: unknown): string | undefined {
  if (typeof raw !== "string") {
    return undefined
  }

  let slug = raw.trim()
  if (!slug) {
    return undefined
  }

  if (slug.startsWith("!")) {
    slug = slug.slice(1)
  }

  if (slug.startsWith("[[") && slug.endsWith("]]")) {
    slug = slug.slice(2, -2)
  }

  slug = slug.replace(/\.md$/i, "")
  slug = slug.replace(/\\/g, "/")
  slug = slug.replace(/^\/+/, "")

  try {
    return simplifySlug(slug as FullSlug)
  } catch {
    return slug.toLowerCase()
  }
}

function findFileByNormalizedSlug(
  normalized: string | undefined,
  allFiles: QuartzPluginData[],
): QuartzPluginData | undefined {
  if (!normalized) {
    return undefined
  }

  return allFiles.find((candidate) => {
    if (!candidate.slug) {
      return false
    }
    const candidateSlug = normalizeLinkTarget(candidate.slug)
    return candidateSlug === normalized
  })
}

function parseRegexInput(pattern: string): RegExp | null {
  if (!pattern) {
    return null
  }

  let source = pattern
  let flags = ""

  const literalMatch = pattern.match(/^\/(.*)\/([gimsuy]*)$/)
  if (literalMatch) {
    source = literalMatch[1]
    flags = literalMatch[2]
  }

  try {
    return new RegExp(source, flags)
  } catch {
    return null
  }
}

export function evaluateFilter(
  filter: BaseFilter,
  allFiles: QuartzPluginData[],
): QuartzPluginData[] {
  const predicate = buildPredicate(filter)
  return allFiles.filter((file) => predicate(file, allFiles))
}

function buildPredicate(filter: BaseFilter): FilePredicate {
  switch (filter.type) {
    case "and":
      return and(filter.conditions.map(buildPredicate))
    case "or":
      return or(filter.conditions.map(buildPredicate))
    case "not":
      return not(buildPredicate(filter.conditions[0]))
    case "function":
      return parseFunction(filter.name, filter.args)
    case "comparison":
      return parseComparison(filter.property, filter.operator, filter.value, filter.isExpression)
    case "method":
      return buildMethodCall(filter.property, filter.method, filter.args, filter.negated)
  }
}

function and(predicates: FilePredicate[]): FilePredicate {
  return (file, allFiles) => predicates.every((p) => p(file, allFiles))
}

function or(predicates: FilePredicate[]): FilePredicate {
  return (file, allFiles) => predicates.some((p) => p(file, allFiles))
}

function not(predicate: FilePredicate): FilePredicate {
  return (file, allFiles) => !predicate(file, allFiles)
}

function parseComparison(
  property: string,
  operator: ComparisonOp,
  value: string | number | boolean | Date,
  isExpression?: boolean,
): FilePredicate {
  if (isExpression) {
    const evaluator = compileExpression(property)
    return (file, _allFiles) => {
      try {
        const computedValue = evaluator(file.frontmatter || {})

        switch (operator) {
          case "==":
            return computedValue === value
          case "!=":
            return computedValue !== value
          case ">":
            if (typeof value === "number") {
              return computedValue > value
            }
            return false
          case "<":
            if (typeof value === "number") {
              return computedValue < value
            }
            return false
          case ">=":
            if (typeof value === "number") {
              return computedValue >= value
            }
            return false
          case "<=":
            if (typeof value === "number") {
              return computedValue <= value
            }
            return false
          default:
            return false
        }
      } catch {
        return false
      }
    }
  }

  return (file, allFiles) => {
    const fileValue = resolvePropertyValue(file, property, allFiles)

    const normalizeForComparison = (val: any) => {
      if (val instanceof Date) {
        return val.getTime()
      }
      if (typeof val === "string" && /^\d{4}-\d{2}-\d{2}/.test(val)) {
        const parsed = new Date(val)
        if (!isNaN(parsed.getTime())) {
          return parsed.getTime()
        }
      }
      return val
    }

    const normalizedFile = normalizeForComparison(fileValue)
    const normalizedValue = normalizeForComparison(value)

    switch (operator) {
      case "==":
        return normalizedFile === normalizedValue
      case "!=":
        return normalizedFile !== normalizedValue
      case ">":
        if (typeof normalizedFile === "number" && typeof normalizedValue === "number") {
          return normalizedFile > normalizedValue
        }
        if (typeof normalizedFile === "string" && typeof normalizedValue === "string") {
          return normalizedFile > normalizedValue
        }
        return false
      case "<":
        if (typeof normalizedFile === "number" && typeof normalizedValue === "number") {
          return normalizedFile < normalizedValue
        }
        if (typeof normalizedFile === "string" && typeof normalizedValue === "string") {
          return normalizedFile < normalizedValue
        }
        return false
      case ">=":
        if (typeof normalizedFile === "number" && typeof normalizedValue === "number") {
          return normalizedFile >= normalizedValue
        }
        if (typeof normalizedFile === "string" && typeof normalizedValue === "string") {
          return normalizedFile >= normalizedValue
        }
        return false
      case "<=":
        if (typeof normalizedFile === "number" && typeof normalizedValue === "number") {
          return normalizedFile <= normalizedValue
        }
        if (typeof normalizedFile === "string" && typeof normalizedValue === "string") {
          return normalizedFile <= normalizedValue
        }
        return false
      case "contains":
        if (Array.isArray(fileValue)) {
          return fileValue.includes(value)
        }
        if (typeof fileValue === "string" && typeof value === "string") {
          return fileValue.includes(value)
        }
        return false
      case "!contains":
        if (Array.isArray(fileValue)) {
          return !fileValue.includes(value)
        }
        if (typeof fileValue === "string" && typeof value === "string") {
          return !fileValue.includes(value)
        }
        return false
      default:
        return false
    }
  }
}

function evaluateArgument(arg: string): string {
  const trimmed = arg.trim()

  const linkMatch = trimmed.match(/^link\(["']?(.+?)["']?\)$/)
  if (linkMatch) {
    const target = linkMatch[1]
    return `[[${target}]]`
  }

  return arg
}

function buildMethodCall(
  property: string,
  method: string,
  args: string[],
  negated: boolean,
): FilePredicate {
  return (file, allFiles) => {
    const propValue = resolvePropertyValue(file, property, allFiles)
    const evaluatedArgs = args.map(evaluateArgument)
    const applyNegation = (result: boolean) => (negated ? !result : result)

    switch (method) {
      case "toString": {
        return applyNegation(propValue !== undefined && propValue !== null)
      }

      case "contains": {
        if (evaluatedArgs.length === 0) return applyNegation(false)
        const needle = evaluatedArgs[0]
        if (Array.isArray(propValue)) {
          return applyNegation(propValue.includes(needle))
        }
        if (typeof propValue === "string") {
          return applyNegation(propValue.includes(String(needle)))
        }
        return applyNegation(false)
      }

      case "containsAny": {
        if (Array.isArray(propValue)) {
          return applyNegation(evaluatedArgs.some((arg) => propValue.includes(arg)))
        }
        if (typeof propValue === "string") {
          return applyNegation(evaluatedArgs.some((arg) => propValue.includes(String(arg))))
        }
        return applyNegation(false)
      }

      case "containsAll": {
        if (Array.isArray(propValue)) {
          return applyNegation(evaluatedArgs.every((arg) => propValue.includes(arg)))
        }
        if (typeof propValue === "string") {
          return applyNegation(evaluatedArgs.every((arg) => propValue.includes(String(arg))))
        }
        return applyNegation(false)
      }

      case "startsWith": {
        if (typeof propValue === "string" && evaluatedArgs.length > 0) {
          return applyNegation(propValue.startsWith(String(evaluatedArgs[0])))
        }
        return applyNegation(false)
      }

      case "endsWith": {
        if (typeof propValue === "string" && evaluatedArgs.length > 0) {
          return applyNegation(propValue.endsWith(String(evaluatedArgs[0])))
        }
        return applyNegation(false)
      }

      case "isEmpty": {
        return applyNegation(
          propValue === undefined ||
            propValue === null ||
            propValue === "" ||
            (Array.isArray(propValue) && propValue.length === 0) ||
            (typeof propValue === "object" &&
              propValue !== null &&
              !Array.isArray(propValue) &&
              Object.keys(propValue).length === 0),
        )
      }

      case "isTruthy": {
        return applyNegation(Boolean(propValue))
      }

      case "isType": {
        if (evaluatedArgs.length === 0) return applyNegation(false)
        const typeArg = evaluatedArgs[0].toLowerCase()
        let result = false

        switch (typeArg) {
          case "null":
            result = propValue === null || propValue === undefined
            break
          case "string":
            result = typeof propValue === "string"
            break
          case "number":
            result = typeof propValue === "number"
            break
          case "boolean":
            result = typeof propValue === "boolean"
            break
          case "array":
            result = Array.isArray(propValue)
            break
          case "object":
            result =
              typeof propValue === "object" && propValue !== null && !Array.isArray(propValue)
            break
          default:
            result = false
        }
        return applyNegation(result)
      }

      case "replace": {
        return applyNegation(typeof propValue === "string" && evaluatedArgs.length >= 2)
      }

      case "lower": {
        return applyNegation(typeof propValue === "string")
      }

      case "upper": {
        return applyNegation(typeof propValue === "string")
      }

      case "slice": {
        return applyNegation(
          (typeof propValue === "string" || Array.isArray(propValue)) && evaluatedArgs.length >= 1,
        )
      }

      case "split": {
        return applyNegation(typeof propValue === "string" && evaluatedArgs.length >= 1)
      }

      case "trim": {
        return applyNegation(typeof propValue === "string")
      }

      case "abs": {
        return applyNegation(typeof propValue === "number")
      }

      case "ceil": {
        return applyNegation(typeof propValue === "number")
      }

      case "floor": {
        return applyNegation(typeof propValue === "number")
      }

      case "round": {
        return applyNegation(typeof propValue === "number")
      }

      case "toFixed": {
        return applyNegation(typeof propValue === "number" && evaluatedArgs.length >= 1)
      }

      case "join": {
        return applyNegation(Array.isArray(propValue))
      }

      case "reverse": {
        return applyNegation(typeof propValue === "string" || Array.isArray(propValue))
      }

      case "sort": {
        return applyNegation(Array.isArray(propValue))
      }

      case "flat": {
        return applyNegation(Array.isArray(propValue))
      }

      case "map":
      case "filter": {
        return applyNegation(Array.isArray(propValue))
      }

      case "unique": {
        return applyNegation(Array.isArray(propValue))
      }

      case "length": {
        return applyNegation(
          typeof propValue === "string" ||
            Array.isArray(propValue) ||
            (typeof propValue === "object" && propValue !== null),
        )
      }

      case "title": {
        return applyNegation(typeof propValue === "string")
      }

      case "keys": {
        return applyNegation(
          typeof propValue === "object" && propValue !== null && !Array.isArray(propValue),
        )
      }

      case "values": {
        return applyNegation(
          typeof propValue === "object" && propValue !== null && !Array.isArray(propValue),
        )
      }

      case "matches": {
        if (typeof propValue !== "string" || evaluatedArgs.length === 0) return applyNegation(false)
        const regex = parseRegexInput(evaluatedArgs[0])
        return applyNegation(regex ? regex.test(propValue) : false)
      }

      case "linksTo": {
        if (evaluatedArgs.length === 0) return applyNegation(false)
        const normalizedTarget = normalizeLinkTarget(evaluatedArgs[0])
        if (!normalizedTarget) return applyNegation(false)

        const values = Array.isArray(propValue) ? propValue : [propValue]
        const result = values.some((val) => {
          if (typeof val !== "string") {
            return false
          }
          const normalizedValue = normalizeLinkTarget(val)
          return normalizedValue === normalizedTarget
        })
        return applyNegation(result)
      }

      case "asFile": {
        if (typeof propValue !== "string") return applyNegation(false)
        const normalized = normalizeLinkTarget(propValue)
        return applyNegation(Boolean(findFileByNormalizedSlug(normalized, allFiles)))
      }

      default:
        throw new Error(`Unknown method: ${method}`)
    }
  }
}

function parseValueExpression(expr: string, file: QuartzPluginData): any {
  const trimmed = expr.trim()

  if (
    (trimmed.startsWith('"') && trimmed.endsWith('"')) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'"))
  ) {
    return trimmed.slice(1, -1)
  }

  const num = Number(trimmed)
  if (!isNaN(num)) return num

  if (trimmed === "true") return true
  if (trimmed === "false") return false

  return file.frontmatter?.[trimmed]
}

function parseFunction(name: string, args: string[]): FilePredicate {
  const registry: Record<string, (...args: string[]) => FilePredicate> = {
    icon:
      (...params: string[]) =>
      () =>
        params.length > 0,
    image:
      (...params: string[]) =>
      () =>
        params.length > 0,
    file: (target?: string) => (_file, allFiles) => {
      const normalized = normalizeLinkTarget(target)
      return Boolean(findFileByNormalizedSlug(normalized, allFiles))
    },

    "file.asLink": () => (file) => Boolean(file.slug),
    "file.hasTag":
      (...tags: string[]) =>
      (file, _allFiles) => {
        const fileTags = file.frontmatter?.tags
        if (!Array.isArray(fileTags)) return false
        return tags.some((tag) => fileTags.includes(tag))
      },

    "file.inFolder": (folder: string) => (file, _allFiles) => {
      if (!file.slug) return false
      const normalizedFolder = folder.endsWith("/") ? folder : folder + "/"
      return file.slug.startsWith(normalizedFolder)
    },

    "file.hasProperty": (prop: string) => (file, _allFiles) =>
      file.frontmatter?.[prop] !== undefined,

    "file.hasLink": (target: string) => (file, _allFiles) =>
      file.links?.some((link) => link === target) ?? false,

    now: () => () => true,

    today: () => () => true,

    date: (value: string) => (_file, _allFiles) => !isNaN(new Date(value).getTime()),

    if: (conditionStr: string, valueIfTrue: string, valueIfFalse: string) => (file, _allFiles) => {
      const conditionValue = parseValueExpression(conditionStr, file)
      const result = conditionValue
        ? parseValueExpression(valueIfTrue, file)
        : parseValueExpression(valueIfFalse, file)
      return !!result
    },

    number: (_value: string) => () => true,

    max:
      (...values: string[]) =>
      (file, _allFiles) => {
        const nums = values.map((v) => {
          const parsed = parseValueExpression(v, file)
          return typeof parsed === "number" ? parsed : Number(parsed)
        })
        const result = Math.max(...nums)
        return !isNaN(result)
      },

    min:
      (...values: string[]) =>
      (file, _allFiles) => {
        const nums = values.map((v) => {
          const parsed = parseValueExpression(v, file)
          return typeof parsed === "number" ? parsed : Number(parsed)
        })
        const result = Math.min(...nums)
        return !isNaN(result)
      },

    duration: (durationStr: string) => () => {
      const ms = parseDuration(durationStr)
      return !isNaN(ms) && ms >= 0
    },

    link: (target: string) => (file, _allFiles) => {
      const simpleTarget = simplifySlug(target as FullSlug)
      return file.links?.includes(simpleTarget) ?? false
    },

    list: () => () => true,
  }

  const factory = registry[name]
  if (!factory) {
    throw new Error(`Unknown filter function: ${name}`)
  }

  return factory(...args)
}

export function computeColumnSummary(
  column: string,
  files: QuartzPluginData[],
  summary: SummaryDefinition,
  allFiles: QuartzPluginData[] = [],
): string | number | undefined {
  if (files.length === 0) {
    return undefined
  }

  const values = files.map((file) => resolvePropertyValue(file, column, allFiles))

  if (summary.type === "builtin" && summary.builtinType) {
    return computeBuiltinSummary(values, summary.builtinType)
  }

  if (summary.type === "formula" && summary.expression) {
    return computeFormulaSummary(values, summary.expression)
  }

  return undefined
}

function computeBuiltinSummary(
  values: any[],
  type: BuiltinSummaryType,
): string | number | undefined {
  switch (type) {
    case "count":
      return values.length

    case "sum": {
      const nums = values.filter((v) => typeof v === "number")
      if (nums.length === 0) return undefined
      return nums.reduce((acc, v) => acc + v, 0)
    }

    case "average":
    case "avg": {
      const nums = values.filter((v) => typeof v === "number")
      if (nums.length === 0) return undefined
      const sum = nums.reduce((acc, v) => acc + v, 0)
      return Math.round((sum / nums.length) * 100) / 100
    }

    case "min": {
      const comparable = values.filter(
        (v) => typeof v === "number" || v instanceof Date || typeof v === "string",
      )
      if (comparable.length === 0) return undefined
      const normalized = comparable.map((v) => (v instanceof Date ? v.getTime() : v))
      const min = Math.min(...normalized.filter((v) => typeof v === "number"))
      if (isNaN(min)) {
        const strings = comparable.filter((v) => typeof v === "string") as string[]
        if (strings.length === 0) return undefined
        return strings.sort()[0]
      }
      if (comparable.some((v) => v instanceof Date)) {
        return new Date(min).toISOString().split("T")[0]
      }
      return min
    }

    case "max": {
      const comparable = values.filter(
        (v) => typeof v === "number" || v instanceof Date || typeof v === "string",
      )
      if (comparable.length === 0) return undefined
      const normalized = comparable.map((v) => (v instanceof Date ? v.getTime() : v))
      const max = Math.max(...normalized.filter((v) => typeof v === "number"))
      if (isNaN(max)) {
        const strings = comparable.filter((v) => typeof v === "string") as string[]
        if (strings.length === 0) return undefined
        return strings.sort().reverse()[0]
      }
      if (comparable.some((v) => v instanceof Date)) {
        return new Date(max).toISOString().split("T")[0]
      }
      return max
    }

    case "range": {
      const comparable = values.filter(
        (v) => typeof v === "number" || v instanceof Date || typeof v === "string",
      )
      if (comparable.length === 0) return undefined
      const normalized = comparable.map((v) => (v instanceof Date ? v.getTime() : v))
      const nums = normalized.filter((v) => typeof v === "number")
      if (nums.length === 0) return undefined
      const min = Math.min(...nums)
      const max = Math.max(...nums)
      if (comparable.some((v) => v instanceof Date)) {
        return `${new Date(min).toISOString().split("T")[0]} - ${new Date(max).toISOString().split("T")[0]}`
      }
      return `${min} - ${max}`
    }

    case "unique": {
      const nonNull = values.filter((v) => v !== undefined && v !== null && v !== "")
      const unique = new Set(nonNull.map((v) => (v instanceof Date ? v.toISOString() : String(v))))
      return unique.size
    }

    case "filled": {
      const filled = values.filter((v) => v !== undefined && v !== null && v !== "")
      return filled.length
    }

    case "missing": {
      const missing = values.filter((v) => v === undefined || v === null || v === "")
      return missing.length
    }

    case "earliest": {
      const dates = values.filter(
        (v) =>
          v instanceof Date ||
          (typeof v === "string" && /^\d{4}-\d{2}-\d{2}/.test(v)) ||
          typeof v === "number",
      )
      if (dates.length === 0) return undefined
      const timestamps = dates.map((v) => {
        if (v instanceof Date) return v.getTime()
        if (typeof v === "string") return new Date(v).getTime()
        return v
      })
      const earliest = Math.min(...timestamps)
      return new Date(earliest).toISOString().split("T")[0]
    }

    case "latest": {
      const dates = values.filter(
        (v) =>
          v instanceof Date ||
          (typeof v === "string" && /^\d{4}-\d{2}-\d{2}/.test(v)) ||
          typeof v === "number",
      )
      if (dates.length === 0) return undefined
      const timestamps = dates.map((v) => {
        if (v instanceof Date) return v.getTime()
        if (typeof v === "string") return new Date(v).getTime()
        return v
      })
      const latest = Math.max(...timestamps)
      return new Date(latest).toISOString().split("T")[0]
    }

    default:
      return undefined
  }
}

function computeFormulaSummary(values: any[], expression: string): string | number | undefined {
  const isTypeNullMatch = expression.match(
    /values\.filter\(value\.isType\(['"](null|undefined)['"]\)\)\.length/,
  )
  if (isTypeNullMatch) {
    return values.filter((v) => v === null || v === undefined).length
  }

  const filterTrueMatch = expression.match(
    /values\.filter\(value\s*==\s*true\)\.reduce\(acc\s*\+\s*1,\s*0\)/,
  )
  if (filterTrueMatch) {
    return values.filter((v) => v === true).length
  }

  const filterFalseMatch = expression.match(
    /values\.filter\(value\s*==\s*false\)\.reduce\(acc\s*\+\s*1,\s*0\)/,
  )
  if (filterFalseMatch) {
    return values.filter((v) => v === false).length
  }

  const sumMatch = expression.match(/values\.reduce\(acc\s*\+\s*value,\s*0\)/)
  if (sumMatch) {
    const nums = values.filter((v) => typeof v === "number")
    return nums.reduce((acc, v) => acc + v, 0)
  }

  if (expression.trim() === "values.length") {
    return values.length
  }

  const filterLengthMatch = expression.match(/values\.filter\(.+\)\.length/)
  if (filterLengthMatch) {
    return values.filter((v) => v !== null && v !== undefined).length
  }

  return undefined
}

export function computeViewSummaries(
  columns: string[],
  files: QuartzPluginData[],
  summaryConfig: ViewSummaryConfig | undefined,
  allFiles: QuartzPluginData[] = [],
): Record<string, string | number | undefined> {
  const results: Record<string, string | number | undefined> = {}

  if (!summaryConfig?.columns) {
    return results
  }

  for (const column of columns) {
    const summary = summaryConfig.columns[column]
    if (summary) {
      results[column] = computeColumnSummary(column, files, summary, allFiles)
    }
  }

  return results
}
