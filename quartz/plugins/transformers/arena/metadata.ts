import type { Element, ElementContent } from 'hast'
import { toString } from 'hast-util-to-string'
import yaml from 'js-yaml'

const isElement = (node: ElementContent): node is Element => node.type === 'element'
const isUl = (node: ElementContent): node is Element => isElement(node) && node.tagName === 'ul'
const isLi = (node: ElementContent): node is Element => isElement(node) && node.tagName === 'li'

const COORDINATE_NUMBER_PATTERN = /-?\d+(?:\.\d+)?/g

export const parseCoordinateMetadata = (value: string): { lat: number; lon: number } | null => {
  const matches = value.match(COORDINATE_NUMBER_PATTERN)
  if (!matches || matches.length < 2) {
    return null
  }

  const lat = Number.parseFloat(matches[0])
  const lon = Number.parseFloat(matches[1])

  if (!Number.isFinite(lon) || !Number.isFinite(lat)) {
    return null
  }

  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) {
    return null
  }

  return { lon, lat }
}

export const getFirstTextContent = (node: Element): string => {
  for (const child of node.children) {
    if (child.type === 'text') {
      const value = child.value.trim()
      if (value.length > 0) {
        return value
      }
      continue
    }
    if (isElement(child)) {
      const text = getFirstTextContent(child)
      if (text.trim().length > 0) {
        return text
      }
    }
  }
  return ''
}

export const extractNestedList = (li: Element): Element | null => {
  for (const child of li.children) {
    if (isUl(child)) return child
  }
  return null
}

const appendListToYaml = (list: Element, indent: number, lines: string[]): void => {
  const indentStr = '  '.repeat(indent)

  for (const child of list.children) {
    if (!isLi(child)) continue

    const nested = extractNestedList(child)
    let rawText = ''
    for (const ch of child.children as ElementContent[]) {
      if (isElement(ch) && ch.tagName === 'ul') continue
      rawText += toString(ch)
    }
    rawText = rawText.trim()

    if (nested) {
      if (rawText.length > 0) {
        lines.push(`${indentStr}${rawText}`)
      } else {
        lines.push(`${indentStr}-`)
      }
      appendListToYaml(nested, indent + 1, lines)
      continue
    }

    if (rawText.length === 0) continue
    const normalized = rawText.replace(/^-+\s*/, '').trim()

    if (normalized.includes(':')) {
      lines.push(`${indentStr}${normalized}`)
    } else {
      lines.push(`${indentStr}- ${normalized}`)
    }
  }
}

export const extractMetadataFromList = (
  list: Element,
): { metadata?: Record<string, unknown>; tags?: string[] } => {
  if (list.children.length === 0) return {}

  const firstItem = list.children.find(isLi)
  if (!firstItem) return {}

  const label = getFirstTextContent(firstItem).trim().toLowerCase()
  if (!/^\[meta\](?:\s*[:\-–—])?$/.test(label)) return {}

  const metaList = extractNestedList(firstItem)
  if (!metaList || metaList.children.length === 0) return {}

  const metadata: Record<string, unknown> = {}
  const tags: string[] = []
  const seenTags = new Set<string>()

  const pushTags = (values: string[]) => {
    for (const entry of values) {
      const trimmed = entry.trim()
      if (trimmed.length === 0) continue
      if (seenTags.has(trimmed)) continue
      seenTags.add(trimmed)
      tags.push(trimmed)
    }
  }

  const extractTagTokens = (raw: string): string[] => {
    let normalized = raw.trim()
    if (normalized.length === 0) return []
    normalized = normalized.replace(/^[-•]\s*/, '')

    const first = normalized.charAt(0)
    const last = normalized.charAt(normalized.length - 1)
    if ((first === '[' && last === ']') || (first === '(' && last === ')')) {
      normalized = normalized.slice(1, -1)
    }

    const rawTokens = normalized
      .split(/[\n,;|]/)
      .map(segment => segment.trim())
      .filter(segment => segment.length > 0)

    const cleaned: string[] = []
    for (const token of rawTokens.length > 0 ? rawTokens : [normalized]) {
      const stripped = token
        .replace(/^[-•]\s*/, '')
        .replace(/^['"]/, '')
        .replace(/['"]$/, '')
        .trim()
      if (stripped.length > 0) cleaned.push(stripped)
    }

    return cleaned
  }

  for (const item of metaList.children) {
    if (!isLi(item)) continue
    const raw = getFirstTextContent(item).trim()
    const sublist = extractNestedList(item)

    let keySource: string | undefined
    let value: string | undefined

    if (raw.length > 0) {
      const delimiterIndex = raw.indexOf(':')
      if (delimiterIndex !== -1) {
        keySource = raw.slice(0, delimiterIndex).trim()
        value = raw.slice(delimiterIndex + 1).trim()
      } else if (sublist) {
        const normalized = raw.trim().toLowerCase()
        if (normalized === 'tags' || normalized === '[tags]') {
          keySource = 'tags'
          value = ''
        } else {
          keySource = normalized
          value = ''
        }
      }
    } else if (sublist) {
      keySource = 'tags'
      value = ''
    }

    if (!keySource) continue

    const normalizedKey = keySource.toLowerCase().replace(/\s+/g, '_')

    if (normalizedKey === 'tags') {
      const candidateStrings: string[] = []
      if (typeof value === 'string' && value.length > 0) {
        candidateStrings.push(value)
      }
      if (sublist) {
        for (const child of sublist.children) {
          if (!isLi(child)) continue
          const tagText = getFirstTextContent(child).trim()
          if (tagText.length > 0) candidateStrings.push(tagText)
        }
      }
      for (const candidate of candidateStrings) {
        const tokens = extractTagTokens(candidate)
        if (tokens.length > 0) pushTags(tokens)
      }
      continue
    }

    if (sublist && !value) {
      const yamlLines: string[] = []
      appendListToYaml(sublist, 0, yamlLines)
      const yamlSource = yamlLines.join('\n')

      if (yamlSource.trim().length > 0) {
        const parsed = yaml.load(yamlSource)
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
          metadata[normalizedKey] = parsed
          continue
        }
      }
    }

    if (!value || value.length === 0) continue
    const key = normalizedKey
    if (key.length === 0) continue
    metadata[key] = value
  }

  const result: { metadata?: Record<string, unknown>; tags?: string[] } = {}
  if (Object.keys(metadata).length > 0) result.metadata = metadata
  if (tags.length > 0) result.tags = tags
  return result
}

export const normalizeChannelMetadata = (
  metadata: Record<string, unknown>,
): Record<string, string | boolean> => {
  const normalized: Record<string, string | boolean> = {}
  for (const [key, value] of Object.entries(metadata)) {
    if (typeof value === 'string' || typeof value === 'boolean') {
      normalized[key] = value
    } else if (typeof value === 'number' && Number.isFinite(value)) {
      normalized[key] = String(value)
    }
  }
  return normalized
}
