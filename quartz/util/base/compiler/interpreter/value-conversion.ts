import { isRecord } from '../../../type-guards'
import { Literal } from '../ast'
import {
  isDateValue,
  isDurationValue,
  isNumberValue,
  isStringValue,
  makeBoolean,
  makeDate,
  makeDuration,
  makeList,
  makeNull,
  makeNumber,
  makeObject,
  makeRegex,
  makeString,
  type DurationValue,
  type LinkValue,
  type Value,
} from './value-model'

const formatLinkValue = (value: LinkValue): string => {
  const target = value.value
  const display = value.display?.trim()
  if (display && display.length > 0) {
    return `[[${target}|${display}]]`
  }
  return `[[${target}]]`
}

export const valueToString = (value: Value): string => {
  switch (value.kind) {
    case 'null':
      return ''
    case 'boolean':
      return value.value ? 'true' : 'false'
    case 'number':
      return Number.isFinite(value.value) ? String(value.value) : ''
    case 'string':
      return value.value
    case 'date':
      return formatDate(value.value)
    case 'duration':
      return String(durationToMs(value))
    case 'list':
      return value.value.map(valueToString).join(', ')
    case 'object':
      return Object.keys(value.value).length > 0 ? '[object]' : ''
    case 'file':
      return value.value.slug ? String(value.value.slug) : ''
    case 'link':
      return value.value
    case 'regex':
      return value.value.source
    case 'html':
      return value.value
    case 'icon':
      return value.value
    case 'image':
      return value.value
  }
}

export const valueToNumber = (value: Value): number => {
  switch (value.kind) {
    case 'number':
      return value.value
    case 'duration':
      return durationToMs(value)
    case 'boolean':
      return value.value ? 1 : 0
    case 'string': {
      const num = Number(value.value)
      return Number.isFinite(num) ? num : Number.NaN
    }
    case 'date':
      return value.value.getTime()
    default:
      return Number.NaN
  }
}

export const coerceToNumber = (value: Value): number | null => {
  const num = valueToNumber(value)
  return Number.isFinite(num) ? num : null
}

export const coerceToDate = (value: Value): Date | null => {
  if (isDateValue(value)) return value.value
  if (isNumberValue(value)) {
    const date = new Date(value.value)
    return Number.isNaN(date.getTime()) ? null : date
  }
  if (isStringValue(value)) {
    const parsed = new Date(value.value)
    return Number.isNaN(parsed.getTime()) ? null : parsed
  }
  return null
}

export const isScalarValue = (value: Value): boolean =>
  !(
    value.kind === 'list' ||
    value.kind === 'object' ||
    value.kind === 'file' ||
    value.kind === 'link' ||
    value.kind === 'regex' ||
    value.kind === 'html' ||
    value.kind === 'icon' ||
    value.kind === 'image'
  )

export const valueToBoolean = (value: Value): boolean => {
  switch (value.kind) {
    case 'null':
      return false
    case 'boolean':
      return value.value
    case 'number':
      return Number.isFinite(value.value) && value.value !== 0
    case 'string':
      return value.value.length > 0
    case 'date':
      return true
    case 'duration':
      return durationToMs(value) !== 0
    case 'list':
      return value.value.length > 0
    case 'object':
      return Object.keys(value.value).length > 0
    case 'file':
      return true
    case 'link':
      return value.value.length > 0
    case 'regex':
      return true
    case 'html':
      return value.value.length > 0
    case 'icon':
      return value.value.length > 0
    case 'image':
      return value.value.length > 0
  }
}

const formatDate = (date: Date): string => {
  const year = String(date.getUTCFullYear()).padStart(4, '0')
  const month = String(date.getUTCMonth() + 1).padStart(2, '0')
  const day = String(date.getUTCDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

export const formatTime = (date: Date): string => {
  const hour = String(date.getUTCHours()).padStart(2, '0')
  const minute = String(date.getUTCMinutes()).padStart(2, '0')
  const second = String(date.getUTCSeconds()).padStart(2, '0')
  return `${hour}:${minute}:${second}`
}

export const formatDatePattern = (date: Date, pattern: string): string => {
  const replacements: Record<string, string> = {
    YYYY: String(date.getUTCFullYear()).padStart(4, '0'),
    YY: String(date.getUTCFullYear() % 100).padStart(2, '0'),
    MM: String(date.getUTCMonth() + 1).padStart(2, '0'),
    DD: String(date.getUTCDate()).padStart(2, '0'),
    HH: String(date.getUTCHours()).padStart(2, '0'),
    mm: String(date.getUTCMinutes()).padStart(2, '0'),
    ss: String(date.getUTCSeconds()).padStart(2, '0'),
    SSS: String(date.getUTCMilliseconds()).padStart(3, '0'),
  }
  let result = pattern
  for (const [token, replacement] of Object.entries(replacements)) {
    result = result.split(token).join(replacement)
  }
  return result
}

export const formatRelative = (date: Date): string => {
  const now = Date.now()
  const diff = date.getTime() - now
  const abs = Math.abs(diff)
  const seconds = Math.round(abs / 1000)
  const minutes = Math.round(abs / 60000)
  const hours = Math.round(abs / 3600000)
  const days = Math.round(abs / 86400000)
  const weeks = Math.round(abs / 604800000)
  const direction = diff < 0 ? 'ago' : 'from now'
  if (seconds < 60) return `${seconds}s ${direction}`
  if (minutes < 60) return `${minutes}m ${direction}`
  if (hours < 24) return `${hours}h ${direction}`
  if (days < 7) return `${days}d ${direction}`
  return `${weeks}w ${direction}`
}

const durationToMs = (duration: DurationValue): number =>
  duration.value + duration.months * 30 * 24 * 60 * 60 * 1000

export const parseDurationParts = (input: string): { months: number; ms: number } => {
  const trimmed = input.trim()
  const asNumber = Number(trimmed)
  if (!isNaN(asNumber)) {
    return { months: 0, ms: asNumber }
  }

  let months = 0
  let ms = 0
  const regex = /(\d+(?:\.\d+)?)\s*([a-zA-Z]+)/g
  let match
  while ((match = regex.exec(trimmed)) !== null) {
    const value = parseFloat(match[1])
    const unitRaw = match[2]
    const unit = unitRaw.toLowerCase()
    if (unitRaw === 'M' || unit === 'mo' || unit === 'month' || unit === 'months') {
      months += value
      continue
    }
    if (unit === 'y' || unit === 'yr' || unit === 'yrs' || unit === 'year' || unit === 'years') {
      months += value * 12
      continue
    }
    if (unit === 'ms' || unit === 'millisecond' || unit === 'milliseconds') {
      ms += value
      continue
    }
    if (
      unit === 's' ||
      unit === 'sec' ||
      unit === 'secs' ||
      unit === 'second' ||
      unit === 'seconds'
    ) {
      ms += value * 1000
      continue
    }
    if (
      unit === 'm' ||
      unit === 'min' ||
      unit === 'mins' ||
      unit === 'minute' ||
      unit === 'minutes'
    ) {
      ms += value * 60 * 1000
      continue
    }
    if (unit === 'h' || unit === 'hr' || unit === 'hrs' || unit === 'hour' || unit === 'hours') {
      ms += value * 60 * 60 * 1000
      continue
    }
    if (unit === 'd' || unit === 'day' || unit === 'days') {
      ms += value * 24 * 60 * 60 * 1000
      continue
    }
    if (unit === 'w' || unit === 'week' || unit === 'weeks') {
      ms += value * 7 * 24 * 60 * 60 * 1000
      continue
    }
  }

  return { months, ms }
}

export const addDurationToDate = (date: Date, duration: DurationValue, direction: 1 | -1): Date => {
  const result = new Date(date.getTime())
  if (duration.months !== 0) {
    const totalMonths = duration.months * direction
    const wholeMonths = Math.trunc(totalMonths)
    const fractional = totalMonths - wholeMonths
    if (wholeMonths !== 0) {
      result.setUTCMonth(result.getUTCMonth() + wholeMonths)
    }
    if (fractional !== 0) {
      result.setTime(result.getTime() + fractional * 30 * 24 * 60 * 60 * 1000)
    }
  }
  if (duration.value !== 0) {
    result.setTime(result.getTime() + direction * duration.value)
  }
  return result
}

export const parseDurationValue = (raw: Value): DurationValue | null => {
  if (isDurationValue(raw)) return raw
  if (isNumberValue(raw)) return makeDuration(raw.value)
  if (isStringValue(raw)) {
    const parsed = parseDurationParts(raw.value)
    return makeDuration(parsed.ms, parsed.months)
  }
  return null
}

export const toValue = (input: unknown): Value => {
  if (input === null || input === undefined) return makeNull()
  if (typeof input === 'boolean') return makeBoolean(input)
  if (typeof input === 'number') return makeNumber(input)
  if (typeof input === 'string') {
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
    case 'null':
      return undefined
    case 'boolean':
      return value.value
    case 'number':
      return value.value
    case 'string':
      return value.value
    case 'date':
      return value.value
    case 'duration':
      return durationToMs(value)
    case 'list':
      return value.value.map(valueToUnknown)
    case 'object': {
      const obj: Record<string, unknown> = {}
      for (const [key, entry] of Object.entries(value.value)) {
        obj[key] = valueToUnknown(entry)
      }
      return obj
    }
    case 'file':
      return value.value
    case 'link':
      return formatLinkValue(value)
    case 'regex':
      return value.value
    case 'html':
      return value.value
    case 'icon':
      return { kind: 'icon', value: value.value }
    case 'image':
      return value.value
  }
}

export const literalToValue = (expr: Literal): Value => {
  if (expr.kind === 'number') return makeNumber(expr.value)
  if (expr.kind === 'string') return makeString(expr.value)
  if (expr.kind === 'boolean') return makeBoolean(expr.value)
  if (expr.kind === 'null') return makeNull()
  if (expr.kind === 'date') return makeDate(new Date(expr.value))
  if (expr.kind === 'duration') {
    const parsed = parseDurationParts(expr.value)
    return makeDuration(parsed.ms, parsed.months)
  }
  if (expr.kind === 'regex') {
    const regex = new RegExp(expr.value, expr.flags)
    return makeRegex(regex)
  }
  return makeNull()
}
