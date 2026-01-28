import { Expr, Literal, Span } from "./ast"
import { parseDuration } from "../types"
import { QuartzPluginData } from "../../../plugins/vfile"

export type Value =
  | { kind: "null" }
  | { kind: "boolean"; value: boolean }
  | { kind: "number"; value: number }
  | { kind: "string"; value: string }
  | { kind: "date"; value: Date }
  | { kind: "duration"; value: number }
  | { kind: "list"; value: Value[] }
  | { kind: "object"; value: Record<string, Value> }
  | { kind: "file"; value: QuartzPluginData }
  | { kind: "link"; value: string; display?: string }
  | { kind: "regex"; value: RegExp }
  | { kind: "html"; value: string }
  | { kind: "icon"; value: string }
  | { kind: "image"; value: string }

export type EvalContext = {
  file: QuartzPluginData
  allFiles: QuartzPluginData[]
  formulas?: Record<string, Expr>
  formulaCache?: Map<string, Value>
  formulaStack?: Set<string>
  locals?: Record<string, Value>
  values?: Value[]
}

const nullValue: Value = { kind: "null" }

const makeNull = (): Value => nullValue
const makeBoolean = (value: boolean): Value => ({ kind: "boolean", value })
const makeNumber = (value: number): Value => ({ kind: "number", value })
const makeString = (value: string): Value => ({ kind: "string", value })
const makeDate = (value: Date): Value => ({ kind: "date", value })
const makeDuration = (value: number): Value => ({ kind: "duration", value })
const makeList = (value: Value[]): Value => ({ kind: "list", value })
const makeObject = (value: Record<string, Value>): Value => ({ kind: "object", value })
const makeFile = (value: QuartzPluginData): Value => ({ kind: "file", value })
const makeLink = (value: string, display?: string): Value => ({ kind: "link", value, display })
const makeRegex = (value: RegExp): Value => ({ kind: "regex", value })
const makeHtml = (value: string): Value => ({ kind: "html", value })
const makeIcon = (value: string): Value => ({ kind: "icon", value })
const makeImage = (value: string): Value => ({ kind: "image", value })

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const isValue = (value: unknown): value is Value =>
  typeof value === "object" && value !== null && "kind" in value

const isNumberValue = (value: Value): value is { kind: "number"; value: number } =>
  value.kind === "number"

const isStringValue = (value: Value): value is { kind: "string"; value: string } =>
  value.kind === "string"

const isBooleanValue = (value: Value): value is { kind: "boolean"; value: boolean } =>
  value.kind === "boolean"

const isListValue = (value: Value): value is { kind: "list"; value: Value[] } =>
  value.kind === "list"

const isObjectValue = (value: Value): value is { kind: "object"; value: Record<string, Value> } =>
  value.kind === "object"

const isDateValue = (value: Value): value is { kind: "date"; value: Date } =>
  value.kind === "date"

const isDurationValue = (value: Value): value is { kind: "duration"; value: number } =>
  value.kind === "duration"

const isFileValue = (value: Value): value is { kind: "file"; value: QuartzPluginData } =>
  value.kind === "file"

const isLinkValue = (value: Value): value is { kind: "link"; value: string; display?: string } =>
  value.kind === "link"

const isRegexValue = (value: Value): value is { kind: "regex"; value: RegExp } =>
  value.kind === "regex"

const valueToString = (value: Value): string => {
  switch (value.kind) {
    case "null":
      return ""
    case "boolean":
      return value.value ? "true" : "false"
    case "number":
      return Number.isFinite(value.value) ? String(value.value) : ""
    case "string":
      return value.value
    case "date":
      return formatDate(value.value)
    case "duration":
      return String(value.value)
    case "list":
      return value.value.map(valueToString).join(", ")
    case "object":
      return Object.keys(value.value).length > 0 ? "[object]" : ""
    case "file":
      return value.value.slug ? String(value.value.slug) : ""
    case "link":
      return value.value
    case "regex":
      return value.value.source
    case "html":
      return value.value
    case "icon":
      return value.value
    case "image":
      return value.value
  }
}

const valueToNumber = (value: Value): number => {
  switch (value.kind) {
    case "number":
      return value.value
    case "duration":
      return value.value
    case "boolean":
      return value.value ? 1 : 0
    case "string": {
      const num = Number(value.value)
      return Number.isFinite(num) ? num : Number.NaN
    }
    case "date":
      return value.value.getTime()
    default:
      return Number.NaN
  }
}

const valueToBoolean = (value: Value): boolean => {
  switch (value.kind) {
    case "null":
      return false
    case "boolean":
      return value.value
    case "number":
      return Number.isFinite(value.value) && value.value !== 0
    case "string":
      return value.value.length > 0
    case "date":
      return true
    case "duration":
      return value.value !== 0
    case "list":
      return value.value.length > 0
    case "object":
      return Object.keys(value.value).length > 0
    case "file":
      return true
    case "link":
      return value.value.length > 0
    case "regex":
      return true
    case "html":
      return value.value.length > 0
    case "icon":
      return value.value.length > 0
    case "image":
      return value.value.length > 0
  }
}

const valueEquals = (left: Value, right: Value): boolean => {
  if (left.kind !== right.kind) return false
  if (left.kind === "null") return true
  if (isBooleanValue(left) && isBooleanValue(right)) return left.value === right.value
  if (isNumberValue(left) && isNumberValue(right)) return left.value === right.value
  if (isStringValue(left) && isStringValue(right)) return left.value === right.value
  if (isDateValue(left) && isDateValue(right)) return left.value.getTime() === right.value.getTime()
  if (isDurationValue(left) && isDurationValue(right)) return left.value === right.value
  if (isLinkValue(left) && isLinkValue(right)) return left.value === right.value
  if (isRegexValue(left) && isRegexValue(right)) return left.value.source === right.value.source
  if (isListValue(left) && isListValue(right)) {
    if (left.value.length !== right.value.length) return false
    for (let i = 0; i < left.value.length; i += 1) {
      if (!valueEquals(left.value[i], right.value[i])) return false
    }
    return true
  }
  if (isObjectValue(left) && isObjectValue(right)) {
    const leftKeys = Object.keys(left.value)
    const rightKeys = Object.keys(right.value)
    if (leftKeys.length !== rightKeys.length) return false
    for (const key of leftKeys) {
      const l = left.value[key]
      const r = right.value[key]
      if (!r || !valueEquals(l, r)) return false
    }
    return true
  }
  if (isFileValue(left) && isFileValue(right)) {
    return left.value.slug === right.value.slug
  }
  return false
}

const formatDate = (date: Date): string => {
  const year = String(date.getUTCFullYear()).padStart(4, "0")
  const month = String(date.getUTCMonth() + 1).padStart(2, "0")
  const day = String(date.getUTCDate()).padStart(2, "0")
  return `${year}-${month}-${day}`
}

const formatTime = (date: Date): string => {
  const hour = String(date.getUTCHours()).padStart(2, "0")
  const minute = String(date.getUTCMinutes()).padStart(2, "0")
  const second = String(date.getUTCSeconds()).padStart(2, "0")
  return `${hour}:${minute}:${second}`
}

const formatDatePattern = (date: Date, pattern: string): string => {
  const replacements: Record<string, string> = {
    YYYY: String(date.getUTCFullYear()).padStart(4, "0"),
    YY: String(date.getUTCFullYear() % 100).padStart(2, "0"),
    MM: String(date.getUTCMonth() + 1).padStart(2, "0"),
    DD: String(date.getUTCDate()).padStart(2, "0"),
    HH: String(date.getUTCHours()).padStart(2, "0"),
    mm: String(date.getUTCMinutes()).padStart(2, "0"),
    ss: String(date.getUTCSeconds()).padStart(2, "0"),
    SSS: String(date.getUTCMilliseconds()).padStart(3, "0"),
  }
  let result = pattern
  for (const [token, replacement] of Object.entries(replacements)) {
    result = result.split(token).join(replacement)
  }
  return result
}

const formatRelative = (date: Date): string => {
  const now = Date.now()
  const diff = date.getTime() - now
  const abs = Math.abs(diff)
  const seconds = Math.round(abs / 1000)
  const minutes = Math.round(abs / 60000)
  const hours = Math.round(abs / 3600000)
  const days = Math.round(abs / 86400000)
  const weeks = Math.round(abs / 604800000)
  const direction = diff < 0 ? "ago" : "from now"
  if (seconds < 60) return `${seconds}s ${direction}`
  if (minutes < 60) return `${minutes}m ${direction}`
  if (hours < 24) return `${hours}h ${direction}`
  if (days < 7) return `${days}d ${direction}`
  return `${weeks}w ${direction}`
}

const parseDurationValue = (raw: Value): number | null => {
  if (isDurationValue(raw)) return raw.value
  if (isNumberValue(raw)) return raw.value
  if (isStringValue(raw)) {
    const value = parseDuration(raw.value)
    return Number.isFinite(value) ? value : null
  }
  return null
}

const toValue = (input: unknown): Value => {
  if (input === null || input === undefined) return makeNull()
  if (typeof input === "boolean") return makeBoolean(input)
  if (typeof input === "number") return makeNumber(input)
  if (typeof input === "string") {
    const trimmed = input.trim()
    if (/^\d{4}-\d{2}-\d{2}/.test(trimmed)) {
      const parsed = new Date(trimmed)
      if (!Number.isNaN(parsed.getTime())) {
        return makeDate(parsed)
      }
    }
    return makeString(input)
  }
  if (input instanceof Date) return makeDate(input)
  if (input instanceof RegExp) return makeRegex(input)
  if (Array.isArray(input)) return makeList(input.map(toValue))
  if (isRecord(input)) {
    const obj: Record<string, Value> = {}
    for (const [key, value] of Object.entries(input)) {
      obj[key] = toValue(value)
    }
    return makeObject(obj)
  }
  return makeNull()
}

export const valueToUnknown = (value: Value): unknown => {
  switch (value.kind) {
    case "null":
      return undefined
    case "boolean":
      return value.value
    case "number":
      return value.value
    case "string":
      return value.value
    case "date":
      return value.value
    case "duration":
      return value.value
    case "list":
      return value.value.map(valueToUnknown)
    case "object": {
      const obj: Record<string, unknown> = {}
      for (const [key, entry] of Object.entries(value.value)) {
        obj[key] = valueToUnknown(entry)
      }
      return obj
    }
    case "file":
      return value.value
    case "link":
      return value.value
    case "regex":
      return value.value
    case "html":
      return value.value
    case "icon":
      return value.value
    case "image":
      return value.value
  }
}

export const evaluateExpression = (expr: Expr, ctx: EvalContext): Value => {
  switch (expr.type) {
    case "Literal":
      return evalLiteral(expr)
    case "Identifier":
      return resolveIdentifier(expr.name, ctx)
    case "UnaryExpr":
      return evalUnary(expr, ctx)
    case "BinaryExpr":
      return evalBinary(expr, ctx)
    case "LogicalExpr":
      return evalLogical(expr, ctx)
    case "MemberExpr":
      return evalMember(expr, ctx)
    case "IndexExpr":
      return evalIndex(expr, ctx)
    case "CallExpr":
      return evalCall(expr, ctx)
    case "ListExpr":
      return makeList(expr.elements.map((item) => evaluateExpression(item, ctx)))
    case "ErrorExpr":
      return makeNull()
  }
}

export const evaluateFilterExpression = (expr: Expr, ctx: EvalContext): boolean =>
  valueToBoolean(evaluateExpression(expr, ctx))

export const evaluateSummaryExpression = (
  expr: Expr,
  values: unknown[],
  ctx: EvalContext,
): Value => {
  const valueList = values.map(toValue)
  const summaryCtx: EvalContext = { ...ctx, values: valueList }
  return evaluateExpression(expr, summaryCtx)
}

const evalLiteral = (expr: Literal): Value => {
  if (expr.kind === "number") return makeNumber(expr.value)
  if (expr.kind === "string") return makeString(expr.value)
  if (expr.kind === "boolean") return makeBoolean(expr.value)
  if (expr.kind === "null") return makeNull()
  if (expr.kind === "date") return makeDate(new Date(expr.value))
  if (expr.kind === "duration") {
    const duration = parseDuration(expr.value)
    return makeDuration(Number.isFinite(duration) ? duration : 0)
  }
  if (expr.kind === "regex") {
    const regex = new RegExp(expr.value, expr.flags)
    return makeRegex(regex)
  }
  return makeNull()
}

const resolveIdentifier = (name: string, ctx: EvalContext): Value => {
  if (ctx.locals && name in ctx.locals) {
    const local = ctx.locals[name]
    if (isValue(local)) return local
  }
  if (name === "this") return makeFile(ctx.file)
  if (name === "file") return makeFile(ctx.file)
  if (name === "note") {
    const fm = ctx.file.frontmatter
    return toValue(fm)
  }
  if (name === "values" && ctx.values) {
    return makeList(ctx.values)
  }
  if (name === "formula") {
    return makeObject({})
  }
  const raw: unknown = ctx.file.frontmatter ? ctx.file.frontmatter[name] : undefined
  return toValue(raw)
}

const evalUnary = (expr: { operator: "!" | "-"; argument: Expr }, ctx: EvalContext): Value => {
  const value = evaluateExpression(expr.argument, ctx)
  if (expr.operator === "!") {
    return makeBoolean(!valueToBoolean(value))
  }
  const num = valueToNumber(value)
  return Number.isFinite(num) ? makeNumber(-num) : makeNull()
}

const evalLogical = (
  expr: { operator: "&&" | "||"; left: Expr; right: Expr },
  ctx: EvalContext,
): Value => {
  if (expr.operator === "&&") {
    const left = evaluateExpression(expr.left, ctx)
    if (!valueToBoolean(left)) return makeBoolean(false)
    return makeBoolean(valueToBoolean(evaluateExpression(expr.right, ctx)))
  }
  const left = evaluateExpression(expr.left, ctx)
  if (valueToBoolean(left)) return makeBoolean(true)
  return makeBoolean(valueToBoolean(evaluateExpression(expr.right, ctx)))
}

const evalBinary = (
  expr: { operator: BinaryOperator; left: Expr; right: Expr },
  ctx: EvalContext,
): Value => {
  const left = evaluateExpression(expr.left, ctx)
  const right = evaluateExpression(expr.right, ctx)
  if (expr.operator === "==") return makeBoolean(valueEquals(left, right))
  if (expr.operator === "!=") return makeBoolean(!valueEquals(left, right))

  if (expr.operator === "+" || expr.operator === "-") {
    return evalAdditive(expr.operator, left, right)
  }
  if (expr.operator === "*" || expr.operator === "/" || expr.operator === "%") {
    const leftNum = valueToNumber(left)
    const rightNum = valueToNumber(right)
    if (!Number.isFinite(leftNum) || !Number.isFinite(rightNum)) return makeNull()
    if (expr.operator === "*") return makeNumber(leftNum * rightNum)
    if (expr.operator === "/") return makeNumber(rightNum === 0 ? Number.NaN : leftNum / rightNum)
    return makeNumber(rightNum === 0 ? Number.NaN : leftNum % rightNum)
  }

  const compare = compareValues(left, right)
  if (compare === null) return makeNull()
  if (expr.operator === ">") return makeBoolean(compare > 0)
  if (expr.operator === ">=") return makeBoolean(compare >= 0)
  if (expr.operator === "<") return makeBoolean(compare < 0)
  if (expr.operator === "<=") return makeBoolean(compare <= 0)
  return makeNull()
}

const evalAdditive = (operator: "+" | "-", left: Value, right: Value): Value => {
  if (operator === "+" && (isStringValue(left) || isStringValue(right))) {
    return makeString(`${valueToString(left)}${valueToString(right)}`)
  }
  if (isDateValue(left)) {
    const duration = parseDurationValue(right)
    if (duration === null) return makeNull()
    const date = new Date(left.value.getTime() + (operator === "+" ? duration : -duration))
    return makeDate(date)
  }
  if (isDateValue(right) && operator === "+") {
    const duration = parseDurationValue(left)
    if (duration === null) return makeNull()
    const date = new Date(right.value.getTime() + duration)
    return makeDate(date)
  }
  if (isDateValue(left) && isDateValue(right) && operator === "-") {
    return makeDuration(left.value.getTime() - right.value.getTime())
  }
  if (isDurationValue(left) && isDurationValue(right)) {
    return makeDuration(operator === "+" ? left.value + right.value : left.value - right.value)
  }
  const leftNum = valueToNumber(left)
  const rightNum = valueToNumber(right)
  if (!Number.isFinite(leftNum) || !Number.isFinite(rightNum)) return makeNull()
  return makeNumber(operator === "+" ? leftNum + rightNum : leftNum - rightNum)
}

const compareValues = (left: Value, right: Value): number | null => {
  if (isNumberValue(left) && isNumberValue(right)) return left.value - right.value
  if (isDurationValue(left) && isDurationValue(right)) return left.value - right.value
  if (isDateValue(left) && isDateValue(right)) return left.value.getTime() - right.value.getTime()
  if (isStringValue(left) && isStringValue(right)) {
    if (left.value === right.value) return 0
    return left.value > right.value ? 1 : -1
  }
  return null
}

const evalMember = (
  expr: { object: Expr; property: string },
  ctx: EvalContext,
): Value => {
  if (expr.object.type === "Identifier") {
    if (expr.object.name === "file") {
      return resolveFileProperty(ctx.file, expr.property, ctx.allFiles)
    }
    if (expr.object.name === "note") {
      const raw: unknown = ctx.file.frontmatter ? ctx.file.frontmatter[expr.property] : undefined
      return toValue(raw)
    }
    if (expr.object.name === "formula") {
      return resolveFormulaProperty(expr.property, ctx)
    }
    if (expr.object.name === "this") {
      if (expr.property === "file") return makeFile(ctx.file)
      return resolveFileProperty(ctx.file, expr.property, ctx.allFiles)
    }
  }
  const objectValue = evaluateExpression(expr.object, ctx)
  return accessProperty(objectValue, expr.property, ctx)
}

const evalIndex = (
  expr: { object: Expr; index: Expr },
  ctx: EvalContext,
): Value => {
  const objectValue = evaluateExpression(expr.object, ctx)
  const indexValue = evaluateExpression(expr.index, ctx)
  if (isListValue(objectValue)) {
    const index = Math.trunc(valueToNumber(indexValue))
    if (!Number.isFinite(index)) return makeNull()
    const item = objectValue.value[index]
    return item ?? makeNull()
  }
  if (isObjectValue(objectValue) && isStringValue(indexValue)) {
    const item = objectValue.value[indexValue.value]
    return item ?? makeNull()
  }
  return makeNull()
}

const evalCall = (expr: { callee: Expr; args: Expr[] }, ctx: EvalContext): Value => {
  if (expr.callee.type === "Identifier") {
    return evalGlobalCall(expr.callee.name, expr.args, ctx)
  }
  if (expr.callee.type === "MemberExpr") {
    const receiver = evaluateExpression(expr.callee.object, ctx)
    return evalMethodCall(receiver, expr.callee.property, expr.args, ctx)
  }
  const calleeValue = evaluateExpression(expr.callee, ctx)
  if (calleeValue.kind === "html") {
    return makeHtml(calleeValue.value)
  }
  return makeNull()
}

const evalGlobalCall = (name: string, args: Expr[], ctx: EvalContext): Value => {
  if (name === "if") {
    const condition = args[0] ? evaluateExpression(args[0], ctx) : makeNull()
    if (valueToBoolean(condition)) {
      return args[1] ? evaluateExpression(args[1], ctx) : makeNull()
    }
    return args[2] ? evaluateExpression(args[2], ctx) : makeNull()
  }
  if (name === "now") return makeDate(new Date())
  if (name === "today") {
    const d = new Date()
    d.setUTCHours(0, 0, 0, 0)
    return makeDate(d)
  }
  if (name === "date") {
    const arg = args[0] ? evaluateExpression(args[0], ctx) : makeNull()
    const str = valueToString(arg)
    const parsed = new Date(str)
    if (Number.isNaN(parsed.getTime())) return makeNull()
    return makeDate(parsed)
  }
  if (name === "duration") {
    const arg = args[0] ? evaluateExpression(args[0], ctx) : makeNull()
    const parsed = parseDuration(valueToString(arg))
    if (!Number.isFinite(parsed)) return makeNull()
    return makeDuration(parsed)
  }
  if (name === "min" || name === "max") {
    const values = args.map((arg) => valueToNumber(evaluateExpression(arg, ctx)))
    const nums = values.filter((value) => Number.isFinite(value))
    if (nums.length === 0) return makeNull()
    return makeNumber(name === "min" ? Math.min(...nums) : Math.max(...nums))
  }
  if (name === "number") {
    const value = args[0] ? evaluateExpression(args[0], ctx) : makeNull()
    const num = valueToNumber(value)
    return Number.isFinite(num) ? makeNumber(num) : makeNull()
  }
  if (name === "link") {
    const target = args[0] ? evaluateExpression(args[0], ctx) : makeNull()
    const display = args[1] ? evaluateExpression(args[1], ctx) : makeNull()
    const targetStr = valueToString(target)
    const displayStr = valueToString(display)
    return makeLink(targetStr, displayStr.length > 0 ? displayStr : undefined)
  }
  if (name === "list") {
    const value = args[0] ? evaluateExpression(args[0], ctx) : makeNull()
    if (isListValue(value)) return value
    return makeList([value])
  }
  if (name === "file") {
    const arg = args[0] ? evaluateExpression(args[0], ctx) : makeNull()
    const target = valueToString(arg)
    const file = findFileByTarget(target, ctx.allFiles)
    return file ? makeFile(file) : makeNull()
  }
  if (name === "image") {
    const arg = args[0] ? evaluateExpression(args[0], ctx) : makeNull()
    const target = valueToString(arg)
    return makeImage(target)
  }
  if (name === "icon") {
    const arg = args[0] ? evaluateExpression(args[0], ctx) : makeNull()
    const target = valueToString(arg)
    return makeIcon(target)
  }
  if (name === "html") {
    const arg = args[0] ? evaluateExpression(args[0], ctx) : makeNull()
    const target = valueToString(arg)
    return makeHtml(target)
  }
  if (name === "escapeHTML") {
    const arg = args[0] ? evaluateExpression(args[0], ctx) : makeNull()
    const target = valueToString(arg)
    const escaped = target
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;")
    return makeString(escaped)
  }
  return makeNull()
}

const evalMethodCall = (receiver: Value, method: string, args: Expr[], ctx: EvalContext): Value => {
  if (method === "isTruthy") {
    return makeBoolean(valueToBoolean(receiver))
  }
  if (method === "isType") {
    const arg = args[0] ? evaluateExpression(args[0], ctx) : makeNull()
    const typeName = valueToString(arg).toLowerCase()
    return makeBoolean(isValueType(receiver, typeName))
  }
  if (method === "toString") {
    return makeString(valueToString(receiver))
  }

  if (isStringValue(receiver)) {
    return evalStringMethod(receiver, method, args, ctx)
  }
  if (isNumberValue(receiver)) {
    return evalNumberMethod(receiver, method, args, ctx)
  }
  if (isListValue(receiver)) {
    return evalListMethod(receiver, method, args, ctx)
  }
  if (isDateValue(receiver)) {
    return evalDateMethod(receiver, method, args, ctx)
  }
  if (isFileValue(receiver)) {
    return evalFileMethod(receiver, method, args, ctx)
  }
  if (isLinkValue(receiver)) {
    return evalLinkMethod(receiver, method, args, ctx)
  }
  if (isObjectValue(receiver)) {
    return evalObjectMethod(receiver, method)
  }
  if (isRegexValue(receiver)) {
    if (method === "matches") {
      const value = args[0] ? evaluateExpression(args[0], ctx) : makeNull()
      return makeBoolean(receiver.value.test(valueToString(value)))
    }
  }
  return makeNull()
}

const evalStringMethod = (
  receiver: { kind: "string"; value: string },
  method: string,
  args: Expr[],
  ctx: EvalContext,
): Value => {
  const value = receiver.value
  if (method === "contains") {
    const arg = args[0] ? evaluateExpression(args[0], ctx) : makeNull()
    return makeBoolean(value.includes(valueToString(arg)))
  }
  if (method === "containsAny") {
    const values = args.map((arg) => valueToString(evaluateExpression(arg, ctx)))
    return makeBoolean(values.some((entry) => value.includes(entry)))
  }
  if (method === "containsAll") {
    const values = args.map((arg) => valueToString(evaluateExpression(arg, ctx)))
    return makeBoolean(values.every((entry) => value.includes(entry)))
  }
  if (method === "startsWith") {
    const arg = args[0] ? evaluateExpression(args[0], ctx) : makeNull()
    return makeBoolean(value.startsWith(valueToString(arg)))
  }
  if (method === "endsWith") {
    const arg = args[0] ? evaluateExpression(args[0], ctx) : makeNull()
    return makeBoolean(value.endsWith(valueToString(arg)))
  }
  if (method === "isEmpty") {
    return makeBoolean(value.length === 0)
  }
  if (method === "lower") {
    return makeString(value.toLowerCase())
  }
  if (method === "upper") {
    return makeString(value.toUpperCase())
  }
  if (method === "title") {
    const parts = value.split(/\s+/).map((part) => {
      const lower = part.toLowerCase()
      return lower.length > 0 ? `${lower[0].toUpperCase()}${lower.slice(1)}` : lower
    })
    return makeString(parts.join(" "))
  }
  if (method === "trim") {
    return makeString(value.trim())
  }
  if (method === "replace") {
    const patternVal = args[0] ? evaluateExpression(args[0], ctx) : makeNull()
    const replacementVal = args[1] ? evaluateExpression(args[1], ctx) : makeNull()
    const replacement = valueToString(replacementVal)
    if (isRegexValue(patternVal)) {
      return makeString(value.replace(patternVal.value, replacement))
    }
    return makeString(value.replace(valueToString(patternVal), replacement))
  }
  if (method === "repeat") {
    const count = args[0] ? valueToNumber(evaluateExpression(args[0], ctx)) : 0
    return makeString(value.repeat(Number.isFinite(count) ? Math.max(0, count) : 0))
  }
  if (method === "reverse") {
    return makeString(value.split("").reverse().join(""))
  }
  if (method === "slice") {
    const start = args[0] ? valueToNumber(evaluateExpression(args[0], ctx)) : 0
    const end = args[1] ? valueToNumber(evaluateExpression(args[1], ctx)) : undefined
    const startIndex = Number.isFinite(start) ? Math.trunc(start) : 0
    const endIndex = end !== undefined && Number.isFinite(end) ? Math.trunc(end) : undefined
    return makeString(value.slice(startIndex, endIndex))
  }
  if (method === "split") {
    const separator = args[0] ? valueToString(evaluateExpression(args[0], ctx)) : ""
    const limit = args[1] ? valueToNumber(evaluateExpression(args[1], ctx)) : undefined
    const parts = limit && Number.isFinite(limit) ? value.split(separator, limit) : value.split(separator)
    return makeList(parts.map((entry) => makeString(entry)))
  }
  if (method === "length") {
    return makeNumber(value.length)
  }
  return makeNull()
}

const evalNumberMethod = (
  receiver: { kind: "number"; value: number },
  method: string,
  args: Expr[],
  ctx: EvalContext,
): Value => {
  const value = receiver.value
  if (method === "abs") return makeNumber(Math.abs(value))
  if (method === "ceil") return makeNumber(Math.ceil(value))
  if (method === "floor") return makeNumber(Math.floor(value))
  if (method === "round") {
    const digits = args[0] ? valueToNumber(evaluateExpression(args[0], ctx)) : 0
    if (!Number.isFinite(digits)) return makeNumber(Math.round(value))
    const factor = 10 ** Math.trunc(digits)
    return makeNumber(Math.round(value * factor) / factor)
  }
  if (method === "toFixed") {
    const digits = args[0] ? valueToNumber(evaluateExpression(args[0], ctx)) : 0
    const precision = Number.isFinite(digits) ? Math.trunc(digits) : 0
    return makeString(value.toFixed(Math.max(0, precision)))
  }
  if (method === "isEmpty") {
    return makeBoolean(!Number.isFinite(value))
  }
  return makeNull()
}

const evalListMethod = (
  receiver: { kind: "list"; value: Value[] },
  method: string,
  args: Expr[],
  ctx: EvalContext,
): Value => {
  const list = receiver.value
  if (method === "contains") {
    const arg = args[0] ? evaluateExpression(args[0], ctx) : makeNull()
    return makeBoolean(list.some((entry) => valueEquals(entry, arg)))
  }
  if (method === "containsAny") {
    const values = args.map((arg) => evaluateExpression(arg, ctx))
    return makeBoolean(values.some((entry) => list.some((item) => valueEquals(item, entry))))
  }
  if (method === "containsAll") {
    const values = args.map((arg) => evaluateExpression(arg, ctx))
    return makeBoolean(values.every((entry) => list.some((item) => valueEquals(item, entry))))
  }
  if (method === "filter") {
    const expr = args[0]
    if (!expr) return makeList(list)
    const filtered = list.filter((value, index) => {
      const locals: Record<string, Value> = { value, index: makeNumber(index) }
      const baseLocals = ctx.locals ? ctx.locals : {}
      const nextCtx: EvalContext = { ...ctx, locals: { ...baseLocals, ...locals } }
      return valueToBoolean(evaluateExpression(expr, nextCtx))
    })
    return makeList(filtered)
  }
  if (method === "map") {
    const expr = args[0]
    if (!expr) return makeList(list)
    const mapped = list.map((value, index) => {
      const locals: Record<string, Value> = { value, index: makeNumber(index) }
      const baseLocals = ctx.locals ? ctx.locals : {}
      const nextCtx: EvalContext = { ...ctx, locals: { ...baseLocals, ...locals } }
      return evaluateExpression(expr, nextCtx)
    })
    return makeList(mapped)
  }
  if (method === "reduce") {
    const expr = args[0]
    const initial = args[1] ? evaluateExpression(args[1], ctx) : makeNull()
    if (!expr) return initial
    let acc = initial
    for (let index = 0; index < list.length; index += 1) {
      const value = list[index]
      const locals: Record<string, Value> = {
        value,
        index: makeNumber(index),
        acc,
      }
      const baseLocals = ctx.locals ? ctx.locals : {}
      const nextCtx: EvalContext = { ...ctx, locals: { ...baseLocals, ...locals } }
      acc = evaluateExpression(expr, nextCtx)
    }
    return acc
  }
  if (method === "flat") {
    const flattened: Value[] = []
    for (const item of list) {
      if (isListValue(item)) {
        flattened.push(...item.value)
      } else {
        flattened.push(item)
      }
    }
    return makeList(flattened)
  }
  if (method === "join") {
    const separator = args[0] ? valueToString(evaluateExpression(args[0], ctx)) : ","
    return makeString(list.map(valueToString).join(separator))
  }
  if (method === "reverse") {
    return makeList([...list].reverse())
  }
  if (method === "slice") {
    const start = args[0] ? valueToNumber(evaluateExpression(args[0], ctx)) : 0
    const end = args[1] ? valueToNumber(evaluateExpression(args[1], ctx)) : undefined
    const startIndex = Number.isFinite(start) ? Math.trunc(start) : 0
    const endIndex = end !== undefined && Number.isFinite(end) ? Math.trunc(end) : undefined
    return makeList(list.slice(startIndex, endIndex))
  }
  if (method === "sort") {
    const sorted = [...list].sort((a, b) => {
      const aNum = valueToNumber(a)
      const bNum = valueToNumber(b)
      if (Number.isFinite(aNum) && Number.isFinite(bNum)) return aNum - bNum
      const aStr = valueToString(a)
      const bStr = valueToString(b)
      if (aStr === bStr) return 0
      return aStr > bStr ? 1 : -1
    })
    return makeList(sorted)
  }
  if (method === "unique") {
    const unique: Value[] = []
    for (const item of list) {
      if (!unique.some((entry) => valueEquals(entry, item))) {
        unique.push(item)
      }
    }
    return makeList(unique)
  }
  if (method === "isEmpty") {
    return makeBoolean(list.length === 0)
  }
  if (method === "length") {
    return makeNumber(list.length)
  }
  return makeNull()
}

const evalDateMethod = (
  receiver: { kind: "date"; value: Date },
  method: string,
  args: Expr[],
  ctx: EvalContext,
): Value => {
  const value = receiver.value
  if (method === "date") {
    const date = new Date(value.getTime())
    date.setUTCHours(0, 0, 0, 0)
    return makeDate(date)
  }
  if (method === "format") {
    const pattern = args[0] ? valueToString(evaluateExpression(args[0], ctx)) : "YYYY-MM-DD"
    return makeString(formatDatePattern(value, pattern))
  }
  if (method === "time") {
    return makeString(formatTime(value))
  }
  if (method === "relative") {
    return makeString(formatRelative(value))
  }
  if (method === "isEmpty") {
    return makeBoolean(false)
  }
  if (method === "year") return makeNumber(value.getUTCFullYear())
  if (method === "month") return makeNumber(value.getUTCMonth() + 1)
  if (method === "day") return makeNumber(value.getUTCDate())
  if (method === "hour") return makeNumber(value.getUTCHours())
  if (method === "minute") return makeNumber(value.getUTCMinutes())
  if (method === "second") return makeNumber(value.getUTCSeconds())
  if (method === "millisecond") return makeNumber(value.getUTCMilliseconds())
  return makeNull()
}

const evalObjectMethod = (receiver: { kind: "object"; value: Record<string, Value> }, method: string): Value => {
  const entries = Object.entries(receiver.value)
  if (method === "isEmpty") return makeBoolean(entries.length === 0)
  if (method === "keys") return makeList(entries.map(([key]) => makeString(key)))
  if (method === "values") return makeList(entries.map(([, value]) => value))
  return makeNull()
}

const evalFileMethod = (
  receiver: { kind: "file"; value: QuartzPluginData },
  method: string,
  args: Expr[],
  ctx: EvalContext,
): Value => {
  const file = receiver.value
  if (method === "asLink") {
    const display = args[0] ? valueToString(evaluateExpression(args[0], ctx)) : undefined
    const slug = file.slug ? String(file.slug) : ""
    return makeLink(slug, display)
  }
  if (method === "hasTag") {
    const tags = args.map((arg) => valueToString(evaluateExpression(arg, ctx)))
    const fileTags = Array.isArray(file.frontmatter?.tags) ? file.frontmatter?.tags : []
    if (!fileTags) return makeBoolean(false)
    return makeBoolean(tags.some((tag) => fileTags.includes(tag)))
  }
  if (method === "inFolder") {
    const folder = args[0] ? valueToString(evaluateExpression(args[0], ctx)) : ""
    const slug = file.slug ? String(file.slug) : ""
    const normalized = folder.endsWith("/") ? folder : `${folder}/`
    return makeBoolean(slug.startsWith(normalized))
  }
  if (method === "hasProperty") {
    const prop = args[0] ? valueToString(evaluateExpression(args[0], ctx)) : ""
    const fm = file.frontmatter
    return makeBoolean(Boolean(fm && prop in fm))
  }
  if (method === "hasLink") {
    const arg = args[0] ? evaluateExpression(args[0], ctx) : makeNull()
    const target = extractLinkTarget(arg, ctx.allFiles)
    const links = Array.isArray(file.links) ? file.links : []
    return makeBoolean(target.length > 0 && links.includes(target))
  }
  return makeNull()
}

const evalLinkMethod = (
  receiver: { kind: "link"; value: string },
  method: string,
  args: Expr[],
  ctx: EvalContext,
): Value => {
  if (method === "asFile") {
    const file = findFileByTarget(receiver.value, ctx.allFiles)
    return file ? makeFile(file) : makeNull()
  }
  if (method === "linksTo") {
    const arg = args[0] ? evaluateExpression(args[0], ctx) : makeNull()
    const target = extractLinkTarget(arg, ctx.allFiles)
    return makeBoolean(target.length > 0 && normalizeLinkTarget(receiver.value) === target)
  }
  return makeNull()
}

const resolveFormulaProperty = (name: string, ctx: EvalContext): Value => {
  if (!ctx.formulas || !ctx.formulas[name]) return makeNull()
  if (!ctx.formulaCache) ctx.formulaCache = new Map()
  if (!ctx.formulaStack) ctx.formulaStack = new Set()
  const cached = ctx.formulaCache.get(name)
  if (cached) return cached
  if (ctx.formulaStack.has(name)) return makeNull()
  ctx.formulaStack.add(name)
  const expr = ctx.formulas[name]
  const value = evaluateExpression(expr, ctx)
  ctx.formulaCache.set(name, value)
  ctx.formulaStack.delete(name)
  return value
}

const resolveFileProperty = (
  file: QuartzPluginData,
  property: string,
  allFiles: QuartzPluginData[],
): Value => {
  if (property === "file") return makeFile(file)
  if (property === "name" || property === "basename") {
    const filePath = typeof file.filePath === "string" ? file.filePath : ""
    const source = filePath.length > 0 ? filePath : (file.slug ? String(file.slug) : "")
    const segment = source.split("/").pop() || ""
    const basename = segment.replace(/\.[^/.]+$/, "")
    return makeString(basename)
  }
  if (property === "title") {
    const title = typeof file.frontmatter?.title === "string" ? file.frontmatter.title : ""
    if (title.length > 0) return makeString(title)
    return resolveFileProperty(file, "name", allFiles)
  }
  if (property === "path") {
    const path = file.filePath || file.slug || ""
    return makeString(String(path))
  }
  if (property === "folder") {
    const slug = file.slug ? String(file.slug) : ""
    const parts = slug.split("/")
    const folder = parts.length > 1 ? parts.slice(0, -1).join("/") : ""
    return makeString(folder)
  }
  if (property === "ext") {
    const filePath = typeof file.filePath === "string" ? file.filePath : ""
    const slug = file.slug ? String(file.slug) : ""
    const source = filePath.length > 0 ? filePath : slug
    const match = source.match(/\.([^.]+)$/)
    return makeString(match ? match[1] : "md")
  }
  if (property === "size") {
    return makeNull()
  }
  if (property === "ctime") {
    const created = file.dates?.created
    if (!created) return makeNull()
    return makeDate(new Date(created))
  }
  if (property === "mtime") {
    const modified = file.dates?.modified
    if (!modified) return makeNull()
    return makeDate(new Date(modified))
  }
  if (property === "tags") {
    const tags = Array.isArray(file.frontmatter?.tags) ? file.frontmatter?.tags : []
    if (!tags) return makeList([])
    return makeList(tags.map((tag) => makeString(String(tag))))
  }
  if (property === "links" || property === "outlinks") {
    const links = Array.isArray(file.links) ? file.links : []
    return makeList(links.map((link) => makeString(String(link))))
  }
  if (property === "backlinks" || property === "inlinks") {
    const slug = file.slug ? String(file.slug) : ""
    const target = normalizeLinkTarget(slug)
    const backlinks = allFiles
      .filter((entry) => {
        const links = Array.isArray(entry.links) ? entry.links : []
        return target.length > 0 && links.includes(target)
      })
      .map((entry) => (entry.slug ? String(entry.slug) : ""))
      .filter((entry) => entry.length > 0)
    return makeList(backlinks.map((link) => makeString(link)))
  }
  if (property === "embeds") {
    return makeList([])
  }
  if (property === "properties") {
    return toValue(file.frontmatter)
  }
  if (property === "link") {
    const slug = file.slug ? String(file.slug) : ""
    return makeLink(slug)
  }
  const raw: unknown = file.frontmatter ? file.frontmatter[property] : undefined
  return toValue(raw)
}

const accessProperty = (value: Value, property: string, ctx: EvalContext): Value => {
  if (isStringValue(value) && property === "length") return makeNumber(value.value.length)
  if (isListValue(value) && property === "length") return makeNumber(value.value.length)
  if (isDateValue(value)) {
    if (property === "year") return makeNumber(value.value.getUTCFullYear())
    if (property === "month") return makeNumber(value.value.getUTCMonth() + 1)
    if (property === "day") return makeNumber(value.value.getUTCDate())
    if (property === "hour") return makeNumber(value.value.getUTCHours())
    if (property === "minute") return makeNumber(value.value.getUTCMinutes())
    if (property === "second") return makeNumber(value.value.getUTCSeconds())
    if (property === "millisecond") return makeNumber(value.value.getUTCMilliseconds())
  }
  if (isObjectValue(value)) {
    return value.value[property] ?? makeNull()
  }
  if (isFileValue(value)) {
    return resolveFileProperty(value.value, property, ctx.allFiles)
  }
  if (isLinkValue(value)) {
    if (property === "value") return makeString(value.value)
  }
  return makeNull()
}

const isValueType = (value: Value, typeName: string): boolean => {
  if (typeName === "null" || typeName === "undefined") return value.kind === "null"
  if (typeName === "string") return value.kind === "string"
  if (typeName === "number") return value.kind === "number"
  if (typeName === "boolean") return value.kind === "boolean"
  if (typeName === "array" || typeName === "list") return value.kind === "list"
  if (typeName === "object") return value.kind === "object"
  if (typeName === "date") return value.kind === "date"
  if (typeName === "duration") return value.kind === "duration"
  if (typeName === "file") return value.kind === "file"
  if (typeName === "link") return value.kind === "link"
  return false
}

const extractLinkTarget = (value: Value, allFiles: QuartzPluginData[]): string => {
  if (isLinkValue(value)) return normalizeLinkTarget(value.value)
  if (isFileValue(value)) {
    return normalizeLinkTarget(value.value.slug ? String(value.value.slug) : "")
  }
  const asString = valueToString(value)
  const normalized = normalizeLinkTarget(asString)
  if (normalized.length > 0) return normalized
  const file = findFileByTarget(asString, allFiles)
  return file && file.slug ? normalizeLinkTarget(String(file.slug)) : ""
}

const findFileByTarget = (target: string, allFiles: QuartzPluginData[]): QuartzPluginData | undefined => {
  const normalized = normalizeLinkTarget(target)
  if (!normalized) return undefined
  return allFiles.find((entry) => normalizeLinkTarget(entry.slug ? String(entry.slug) : "") === normalized)
}

const normalizeLinkTarget = (raw: string): string => {
  let slug = raw.trim()
  if (!slug) return ""
  if (slug.startsWith("!")) slug = slug.slice(1)
  if (slug.startsWith("[[") && slug.endsWith("]]")) {
    slug = slug.slice(2, -2)
  }
  slug = slug.replace(/\.md$/i, "")
  slug = slug.replace(/\\/g, "/")
  slug = slug.replace(/^\/+/, "")
  return slug.toLowerCase()
}

type BinaryOperator = BinaryExpr['operator']

export const buildSyntheticSpan = (): Span => ({
  start: { offset: 0, line: 0, column: 0 },
  end: { offset: 0, line: 0, column: 0 },
})
