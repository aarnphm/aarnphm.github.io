import {
  findFileByTarget,
  resolveLinkSlugFromText,
  resolveLinkSlugFromValue,
  vaultPath,
} from './file-context'
import {
  formatDatePattern,
  formatRelative,
  formatTime,
  valueToNumber,
  valueToString,
} from './value-conversion'
import {
  isListValue,
  isRegexValue,
  makeBoolean,
  makeDate,
  makeFile,
  makeLink,
  makeList,
  makeNull,
  makeNumber,
  makeString,
  type DateValue,
  type EvalContext,
  type FileValue,
  type LinkValue,
  type ListValue,
  type NumberValue,
  type ObjectValue,
  type StringValue,
  type Value,
} from './value-model'
import { valueEquals } from './value-operators'

export const evalStringMethod = (receiver: StringValue, method: string, args: Value[]): Value => {
  const value = receiver.value
  if (method === 'contains') {
    const arg = args[0] ?? makeNull()
    return makeBoolean(value.includes(valueToString(arg)))
  }
  if (method === 'containsAny') {
    const values = args.map(arg => valueToString(arg))
    return makeBoolean(values.some(entry => value.includes(entry)))
  }
  if (method === 'containsAll') {
    const values = args.map(arg => valueToString(arg))
    return makeBoolean(values.every(entry => value.includes(entry)))
  }
  if (method === 'startsWith') {
    const arg = args[0] ?? makeNull()
    return makeBoolean(value.startsWith(valueToString(arg)))
  }
  if (method === 'endsWith') {
    const arg = args[0] ?? makeNull()
    return makeBoolean(value.endsWith(valueToString(arg)))
  }
  if (method === 'isEmpty') {
    return makeBoolean(value.length === 0)
  }
  if (method === 'lower') {
    return makeString(value.toLowerCase())
  }
  if (method === 'title') {
    const parts = value.split(/\s+/).map(part => {
      const lower = part.toLowerCase()
      return lower.length > 0 ? `${lower[0].toUpperCase()}${lower.slice(1)}` : lower
    })
    return makeString(parts.join(' '))
  }
  if (method === 'trim') {
    return makeString(value.trim())
  }
  if (method === 'replace') {
    const patternVal = args[0] ?? makeNull()
    const replacementVal = args[1] ?? makeNull()
    const replacement = valueToString(replacementVal)
    if (isRegexValue(patternVal)) {
      return makeString(value.replace(patternVal.value, replacement))
    }
    return makeString(value.replace(valueToString(patternVal), replacement))
  }
  if (method === 'repeat') {
    const count = args[0] ? valueToNumber(args[0]) : 0
    return makeString(value.repeat(Number.isFinite(count) ? Math.max(0, count) : 0))
  }
  if (method === 'reverse') {
    return makeString(value.split('').reverse().join(''))
  }
  if (method === 'slice') {
    const start = args[0] ? valueToNumber(args[0]) : 0
    const end = args[1] ? valueToNumber(args[1]) : undefined
    const startIndex = Number.isFinite(start) ? Math.trunc(start) : 0
    const endIndex = end !== undefined && Number.isFinite(end) ? Math.trunc(end) : undefined
    return makeString(value.slice(startIndex, endIndex))
  }
  if (method === 'split') {
    const separatorValue = args[0] ?? makeString('')
    const separator = isRegexValue(separatorValue)
      ? separatorValue.value
      : valueToString(separatorValue)
    const limitValue = args[1] ? valueToNumber(args[1]) : undefined
    const limit =
      limitValue !== undefined && Number.isFinite(limitValue) ? Math.trunc(limitValue) : undefined
    const parts = limit !== undefined ? value.split(separator, limit) : value.split(separator)
    return makeList(parts.map(entry => makeString(entry)))
  }
  if (method === 'length') {
    return makeNumber(value.length)
  }
  return makeNull()
}

export const evalNumberMethod = (receiver: NumberValue, method: string, args: Value[]): Value => {
  const value = receiver.value
  if (method === 'abs') return makeNumber(Math.abs(value))
  if (method === 'ceil') return makeNumber(Math.ceil(value))
  if (method === 'floor') return makeNumber(Math.floor(value))
  if (method === 'round') {
    const digits = args[0] ? valueToNumber(args[0]) : 0
    if (!Number.isFinite(digits)) return makeNumber(Math.round(value))
    const factor = 10 ** Math.trunc(digits)
    return makeNumber(Math.round(value * factor) / factor)
  }
  if (method === 'toFixed') {
    const digits = args[0] ? valueToNumber(args[0]) : 0
    const precision = Number.isFinite(digits) ? Math.trunc(digits) : 0
    return makeString(value.toFixed(Math.max(0, precision)))
  }
  if (method === 'isEmpty') {
    return makeBoolean(!Number.isFinite(value))
  }
  return makeNull()
}

export const evalListMethod = (
  receiver: ListValue,
  method: string,
  args: Value[],
  ctx: EvalContext,
): Value => {
  const list = receiver.value
  if (method === 'contains') {
    const arg = args[0] ?? makeNull()
    return makeBoolean(list.some(entry => valueEquals(entry, arg, ctx)))
  }
  if (method === 'containsAny') {
    const values = args
    return makeBoolean(values.some(entry => list.some(item => valueEquals(item, entry, ctx))))
  }
  if (method === 'containsAll') {
    const values = args
    return makeBoolean(values.every(entry => list.some(item => valueEquals(item, entry, ctx))))
  }
  if (method === 'flat') {
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
  if (method === 'join') {
    const separator = args[0] ? valueToString(args[0]) : ','
    return makeString(list.map(valueToString).join(separator))
  }
  if (method === 'reverse') {
    return makeList([...list].reverse())
  }
  if (method === 'slice') {
    const start = args[0] ? valueToNumber(args[0]) : 0
    const end = args[1] ? valueToNumber(args[1]) : undefined
    const startIndex = Number.isFinite(start) ? Math.trunc(start) : 0
    const endIndex = end !== undefined && Number.isFinite(end) ? Math.trunc(end) : undefined
    return makeList(list.slice(startIndex, endIndex))
  }
  if (method === 'sort') {
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
  if (method === 'unique') {
    const unique: Value[] = []
    for (const item of list) {
      if (!unique.some(entry => valueEquals(entry, item, ctx))) {
        unique.push(item)
      }
    }
    return makeList(unique)
  }
  if (method === 'sum') {
    const nums = list.map(valueToNumber).filter(value => Number.isFinite(value))
    if (nums.length === 0) return makeNull()
    return makeNumber(nums.reduce((acc, value) => acc + value, 0))
  }
  if (method === 'mean' || method === 'average') {
    const nums = list.map(valueToNumber).filter(value => Number.isFinite(value))
    if (nums.length === 0) return makeNull()
    const sum = nums.reduce((acc, value) => acc + value, 0)
    return makeNumber(sum / nums.length)
  }
  if (method === 'median') {
    const nums = list.map(valueToNumber).filter(value => Number.isFinite(value))
    if (nums.length === 0) return makeNull()
    const sorted = [...nums].sort((a, b) => a - b)
    const mid = Math.floor(sorted.length / 2)
    if (sorted.length % 2 === 0) {
      return makeNumber((sorted[mid - 1] + sorted[mid]) / 2)
    }
    return makeNumber(sorted[mid])
  }
  if (method === 'stddev') {
    const nums = list.map(valueToNumber).filter(value => Number.isFinite(value))
    if (nums.length === 0) return makeNull()
    const mean = nums.reduce((acc, value) => acc + value, 0) / nums.length
    const variance = nums.reduce((acc, value) => acc + (value - mean) ** 2, 0) / nums.length
    return makeNumber(Math.sqrt(variance))
  }
  if (method === 'min' || method === 'max') {
    const nums = list.map(valueToNumber).filter(value => Number.isFinite(value))
    if (nums.length === 0) return makeNull()
    const value = method === 'min' ? Math.min(...nums) : Math.max(...nums)
    return makeNumber(value)
  }
  if (method === 'isEmpty') {
    return makeBoolean(list.length === 0)
  }
  if (method === 'length') {
    return makeNumber(list.length)
  }
  return makeNull()
}

export const evalDateMethod = (receiver: DateValue, method: string, args: Value[]): Value => {
  const value = receiver.value
  if (method === 'date') {
    const date = new Date(value.getTime())
    date.setUTCHours(0, 0, 0, 0)
    return makeDate(date)
  }
  if (method === 'format') {
    const pattern = args[0] ? valueToString(args[0]) : 'YYYY-MM-DD'
    return makeString(formatDatePattern(value, pattern))
  }
  if (method === 'time') {
    return makeString(formatTime(value))
  }
  if (method === 'relative') {
    return makeString(formatRelative(value))
  }
  if (method === 'isEmpty') {
    return makeBoolean(false)
  }
  if (method === 'year') return makeNumber(value.getUTCFullYear())
  if (method === 'month') return makeNumber(value.getUTCMonth() + 1)
  if (method === 'day') return makeNumber(value.getUTCDate())
  if (method === 'hour') return makeNumber(value.getUTCHours())
  if (method === 'minute') return makeNumber(value.getUTCMinutes())
  if (method === 'second') return makeNumber(value.getUTCSeconds())
  if (method === 'millisecond') return makeNumber(value.getUTCMilliseconds())
  return makeNull()
}

export const evalObjectMethod = (receiver: ObjectValue, method: string): Value => {
  const entries = Object.entries(receiver.value)
  if (method === 'isEmpty') return makeBoolean(entries.length === 0)
  if (method === 'keys') return makeList(entries.map(([key]) => makeString(key)))
  if (method === 'values') return makeList(entries.map(([, value]) => value))
  return makeNull()
}

export const evalFileMethod = (
  receiver: FileValue,
  method: string,
  args: Value[],
  ctx: EvalContext,
): Value => {
  const file = receiver.value
  if (method === 'asLink') {
    const display = args[0] ? valueToString(args[0]) : undefined
    const slug = file.slug ? String(file.slug) : ''
    return makeLink(slug, display)
  }
  if (method === 'hasTag') {
    const tags = args.map(arg => valueToString(arg).replace(/^#/, ''))
    const rawTags = file.frontmatter?.tags
    const fileTags = Array.isArray(rawTags) ? rawTags : typeof rawTags === 'string' ? [rawTags] : []
    // Obsidian matches nested tags: hasTag("a") is true for a/b.
    return makeBoolean(
      fileTags.some(raw => {
        const fileTag = String(raw).replace(/^#/, '')
        return tags.some(tag => fileTag === tag || fileTag.startsWith(`${tag}/`))
      }),
    )
  }
  if (method === 'inFolder') {
    const rawFolder = args[0] ? valueToString(args[0]) : ''
    const folder =
      rawFolder.startsWith('/') || rawFolder.endsWith('/')
        ? rawFolder.replace(/^\/+|\/+$/g, '')
        : rawFolder
    if (!folder) return makeBoolean(true)
    const vp = vaultPath(file)
    return makeBoolean(vp.startsWith(folder) && vp.charCodeAt(folder.length) === 47)
  }
  if (method === 'hasProperty') {
    const prop = args[0] ? valueToString(args[0]) : ''
    const fm = file.frontmatter
    return makeBoolean(Boolean(fm && prop in fm))
  }
  if (method === 'hasLink') {
    const arg = args[0] ?? makeNull()
    const targetSlug = resolveLinkSlugFromValue(arg, ctx)
    if (!targetSlug || !Array.isArray(file.links)) return makeBoolean(false)
    for (let i = 0; i < file.links.length; i++) {
      if (String(file.links[i]) === targetSlug) return makeBoolean(true)
    }
    return makeBoolean(false)
  }
  return makeNull()
}

export const evalLinkMethod = (
  receiver: LinkValue,
  method: string,
  args: Value[],
  ctx: EvalContext,
): Value => {
  if (method === 'asFile') {
    const file = findFileByTarget(receiver.value, ctx)
    return file ? makeFile(file) : makeNull()
  }
  if (method === 'linksTo') {
    const arg = args[0] ?? makeNull()
    const targetSlug = resolveLinkSlugFromValue(arg, ctx)
    const receiverSlug = resolveLinkSlugFromText(receiver.value, ctx)
    if (!targetSlug || !receiverSlug) return makeBoolean(false)
    return makeBoolean(receiverSlug === targetSlug)
  }
  return makeNull()
}
