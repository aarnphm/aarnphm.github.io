import { Span } from '../ast'
import { findFileByTarget } from './file-context'
import {
  parseDurationParts,
  valueToBoolean,
  valueToNumber,
  valueToString,
} from './value-conversion'
import {
  evalDateMethod,
  evalFileMethod,
  evalLinkMethod,
  evalListMethod,
  evalNumberMethod,
  evalObjectMethod,
  evalStringMethod,
} from './value-methods'
import {
  isDateValue,
  isFileValue,
  isLinkValue,
  isListValue,
  isNumberValue,
  isObjectValue,
  isRegexValue,
  isStringValue,
  makeBoolean,
  makeDate,
  makeDuration,
  makeFile,
  makeHtml,
  makeIcon,
  makeImage,
  makeLink,
  makeList,
  makeNull,
  makeNumber,
  makeString,
  type EvalContext,
  type Value,
} from './value-model'

const stringMethods = new Set([
  'contains',
  'containsAny',
  'containsAll',
  'startsWith',
  'endsWith',
  'isEmpty',
  'lower',
  'title',
  'trim',
  'replace',
  'repeat',
  'reverse',
  'slice',
  'split',
  'length',
])

const numberMethods = new Set(['abs', 'ceil', 'floor', 'round', 'toFixed', 'isEmpty'])

const listMethods = new Set([
  'contains',
  'containsAny',
  'containsAll',
  'flat',
  'join',
  'reverse',
  'slice',
  'sort',
  'unique',
  'isEmpty',
  'length',
  'sum',
  'mean',
  'average',
  'median',
  'stddev',
  'min',
  'max',
])

const dateMethods = new Set([
  'date',
  'format',
  'time',
  'relative',
  'isEmpty',
  'year',
  'month',
  'day',
  'hour',
  'minute',
  'second',
  'millisecond',
])

const fileMethods = new Set(['asLink', 'hasTag', 'inFolder', 'hasProperty', 'hasLink'])

const linkMethods = new Set(['asFile', 'linksTo'])

const objectMethods = new Set(['isEmpty', 'keys', 'values'])

const pushRuntimeDiagnostic = (ctx: EvalContext, message: string, span: Span) => {
  if (!ctx.diagnostics || !ctx.diagnosticContext) return
  const source = ctx.diagnosticSource ?? ''
  const key = `${ctx.diagnosticContext}|${message}|${span.start.offset}|${span.end.offset}|${source}`
  if (ctx.diagnosticSet) {
    if (ctx.diagnosticSet.has(key)) return
    ctx.diagnosticSet.add(key)
  }
  ctx.diagnostics.push({ kind: 'runtime', message, span, context: ctx.diagnosticContext, source })
}

export const evalGlobalCallValues = (
  name: string,
  args: Value[],
  ctx: EvalContext,
  span: Span,
): Value => {
  if (name === 'if') {
    if (args.length < 2) {
      pushRuntimeDiagnostic(ctx, 'if() expects at least 2 arguments', span)
    }
    const condition = args[0] ?? makeNull()
    if (valueToBoolean(condition)) {
      return args[1] ?? makeNull()
    }
    return args[2] ?? makeNull()
  }
  if (name === 'now') return makeDate(new Date())
  if (name === 'today') {
    const d = new Date()
    d.setUTCHours(0, 0, 0, 0)
    return makeDate(d)
  }
  if (name === 'date') {
    if (args.length < 1) {
      pushRuntimeDiagnostic(ctx, 'date() expects 1 argument', span)
    }
    const arg = args[0] ?? makeNull()
    const str = valueToString(arg)
    const parsed = new Date(str)
    if (Number.isNaN(parsed.getTime())) return makeNull()
    return makeDate(parsed)
  }
  if (name === 'duration') {
    if (args.length < 1) {
      pushRuntimeDiagnostic(ctx, 'duration() expects 1 argument', span)
    }
    const arg = args[0] ?? makeNull()
    const parsed = parseDurationParts(valueToString(arg))
    return makeDuration(parsed.ms, parsed.months)
  }
  if (name === 'min' || name === 'max') {
    if (args.length < 1) {
      pushRuntimeDiagnostic(ctx, `${name}() expects at least 1 argument`, span)
    }
    const values = args.map(arg => valueToNumber(arg))
    const nums = values.filter(value => Number.isFinite(value))
    if (nums.length === 0) return makeNull()
    return makeNumber(name === 'min' ? Math.min(...nums) : Math.max(...nums))
  }
  if (name === 'number') {
    if (args.length < 1) {
      pushRuntimeDiagnostic(ctx, 'number() expects 1 argument', span)
    }
    const value = args[0] ?? makeNull()
    const num = valueToNumber(value)
    return Number.isFinite(num) ? makeNumber(num) : makeNull()
  }
  if (name === 'link') {
    if (args.length < 1) {
      pushRuntimeDiagnostic(ctx, 'link() expects at least 1 argument', span)
    }
    const target = args[0] ?? makeNull()
    const display = args[1] ?? makeNull()
    const targetStr = valueToString(target)
    const displayStr = valueToString(display)
    return makeLink(targetStr, displayStr.length > 0 ? displayStr : undefined)
  }
  if (name === 'list') {
    if (args.length < 1) {
      pushRuntimeDiagnostic(ctx, 'list() expects 1 argument', span)
    }
    const value = args[0] ?? makeNull()
    if (isListValue(value)) return value
    return makeList([value])
  }
  if (name === 'file') {
    if (args.length < 1) {
      pushRuntimeDiagnostic(ctx, 'file() expects 1 argument', span)
    }
    const arg = args[0] ?? makeNull()
    const target = valueToString(arg)
    const file = findFileByTarget(target, ctx)
    return file ? makeFile(file) : makeNull()
  }
  if (name === 'image') {
    if (args.length < 1) {
      pushRuntimeDiagnostic(ctx, 'image() expects 1 argument', span)
    }
    const arg = args[0] ?? makeNull()
    const target = valueToString(arg)
    return makeImage(target)
  }
  if (name === 'icon') {
    if (args.length < 1) {
      pushRuntimeDiagnostic(ctx, 'icon() expects 1 argument', span)
    }
    const arg = args[0] ?? makeNull()
    const target = valueToString(arg)
    return makeIcon(target)
  }
  if (name === 'html') {
    if (args.length < 1) {
      pushRuntimeDiagnostic(ctx, 'html() expects 1 argument', span)
    }
    const arg = args[0] ?? makeNull()
    const target = valueToString(arg)
    return makeHtml(target)
  }
  if (name === 'escapeHTML') {
    if (args.length < 1) {
      pushRuntimeDiagnostic(ctx, 'escapeHTML() expects 1 argument', span)
    }
    const arg = args[0] ?? makeNull()
    const target = valueToString(arg)
    const escaped = target
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;')
    return makeString(escaped)
  }
  pushRuntimeDiagnostic(ctx, `unknown function: ${name}`, span)
  return makeNull()
}

export const evalMethodCallValues = (
  receiver: Value,
  method: string,
  args: Value[],
  ctx: EvalContext,
  span: Span,
): Value => {
  if (method === 'isTruthy') {
    return makeBoolean(valueToBoolean(receiver))
  }
  if (method === 'isType') {
    const arg = args[0] ?? makeNull()
    const typeName = valueToString(arg).toLowerCase()
    return makeBoolean(isValueType(receiver, typeName))
  }
  if (method === 'toString') {
    return makeString(valueToString(receiver))
  }
  if (receiver.kind === 'null') {
    if (method === 'isEmpty') return makeBoolean(true)
    if (method === 'length') return makeNumber(0)
    if (
      method === 'contains' ||
      method === 'containsAny' ||
      method === 'containsAll' ||
      method === 'startsWith' ||
      method === 'endsWith' ||
      method === 'matches'
    ) {
      return makeBoolean(false)
    }
    if (method === 'asFile' || method === 'asLink') {
      return makeNull()
    }
    return makeNull()
  }

  if (isStringValue(receiver)) {
    if (method === 'asFile') {
      const file = findFileByTarget(receiver.value, ctx)
      return file ? makeFile(file) : makeNull()
    }
    if (!stringMethods.has(method)) {
      pushRuntimeDiagnostic(ctx, `unknown string method: ${method}`, span)
      return makeNull()
    }
    return evalStringMethod(receiver, method, args)
  }
  if (isNumberValue(receiver)) {
    if (!numberMethods.has(method)) {
      pushRuntimeDiagnostic(ctx, `unknown number method: ${method}`, span)
      return makeNull()
    }
    return evalNumberMethod(receiver, method, args)
  }
  if (isListValue(receiver)) {
    if (!listMethods.has(method)) {
      pushRuntimeDiagnostic(ctx, `unknown list method: ${method}`, span)
      return makeNull()
    }
    return evalListMethod(receiver, method, args, ctx)
  }
  if (isDateValue(receiver)) {
    if (!dateMethods.has(method)) {
      pushRuntimeDiagnostic(ctx, `unknown date method: ${method}`, span)
      return makeNull()
    }
    return evalDateMethod(receiver, method, args)
  }
  if (isFileValue(receiver)) {
    if (!fileMethods.has(method)) {
      pushRuntimeDiagnostic(ctx, `unknown file method: ${method}`, span)
      return makeNull()
    }
    return evalFileMethod(receiver, method, args, ctx)
  }
  if (isLinkValue(receiver)) {
    if (!linkMethods.has(method)) {
      pushRuntimeDiagnostic(ctx, `unknown link method: ${method}`, span)
      return makeNull()
    }
    return evalLinkMethod(receiver, method, args, ctx)
  }
  if (isObjectValue(receiver)) {
    if (!objectMethods.has(method)) {
      pushRuntimeDiagnostic(ctx, `unknown object method: ${method}`, span)
      return makeNull()
    }
    return evalObjectMethod(receiver, method)
  }
  if (isRegexValue(receiver)) {
    if (method === 'matches') {
      const value = args[0] ?? makeNull()
      return makeBoolean(receiver.value.test(valueToString(value)))
    }
    pushRuntimeDiagnostic(ctx, `unknown regex method: ${method}`, span)
    return makeNull()
  }
  pushRuntimeDiagnostic(ctx, `unknown ${receiver.kind} method: ${method}`, span)
  return makeNull()
}

const isValueType = (value: Value, typeName: string): boolean => {
  if (typeName === 'null' || typeName === 'undefined') return value.kind === 'null'
  if (typeName === 'string') return value.kind === 'string'
  if (typeName === 'number') return value.kind === 'number'
  if (typeName === 'boolean') return value.kind === 'boolean'
  if (typeName === 'array' || typeName === 'list') return value.kind === 'list'
  if (typeName === 'object') return value.kind === 'object'
  if (typeName === 'date') return value.kind === 'date'
  if (typeName === 'duration') return value.kind === 'duration'
  if (typeName === 'file') return value.kind === 'file'
  if (typeName === 'link') return value.kind === 'link'
  return false
}
