import { escapeHTML } from '../../util/escape'
import {
  type FullSlug,
  getFileExtension,
  isFilePath,
  isFullSlug,
  resolveRelative,
  slugifyFilePath,
  splitAnchor,
} from '../../util/path'
import { isRecord, readNumber, readString } from '../../util/type-guards'

export interface MarkerData {
  lat: number
  lon: number
  title: string
  slug: FullSlug
  icon?: string
  color?: string
  popupFields: Record<string, unknown>
}

export interface MapConfig {
  defaultZoom: number
  defaultCenter?: [number, number]
  clustering: boolean
}

export type MapProperties = Record<string, { displayName?: string }>

export interface BaseMapData {
  markers: MarkerData[]
  config: MapConfig
  currentSlug: FullSlug
  properties: MapProperties
}

export interface BaseMapAttributes {
  markers?: string
  config?: string
  currentSlug?: string
  properties?: string
}

type IconToken = { kind: 'icon'; value: string }

function parseJson(raw: string | undefined): unknown {
  if (raw === undefined) return undefined
  try {
    return JSON.parse(raw)
  } catch {
    return undefined
  }
}

function isIconToken(value: unknown): value is IconToken {
  return isRecord(value) && value.kind === 'icon' && typeof value.value === 'string'
}

function parseMarker(value: unknown): MarkerData | undefined {
  if (!isRecord(value)) return undefined

  const lat = readNumber(value, 'lat')
  const lon = readNumber(value, 'lon')
  const title = readString(value, 'title')
  const slug = readString(value, 'slug')
  if (
    lat === undefined ||
    lon === undefined ||
    !Number.isFinite(lat) ||
    !Number.isFinite(lon) ||
    lat < -90 ||
    lat > 90 ||
    lon < -180 ||
    lon > 180 ||
    title === undefined ||
    slug === undefined ||
    slug.length === 0 ||
    !isFullSlug(slug)
  ) {
    return undefined
  }

  return {
    lat,
    lon,
    title,
    slug,
    icon: readString(value, 'icon'),
    color: readString(value, 'color'),
    popupFields: isRecord(value.popupFields) ? value.popupFields : {},
  }
}

function parseMapConfig(value: unknown): MapConfig {
  const defaultZoom = isRecord(value) ? readNumber(value, 'defaultZoom') : undefined
  const rawCenter = isRecord(value) ? value.defaultCenter : undefined
  const defaultCenter: [number, number] | undefined =
    Array.isArray(rawCenter) &&
    rawCenter.length === 2 &&
    typeof rawCenter[0] === 'number' &&
    typeof rawCenter[1] === 'number' &&
    Number.isFinite(rawCenter[0]) &&
    Number.isFinite(rawCenter[1]) &&
    rawCenter[0] >= -90 &&
    rawCenter[0] <= 90 &&
    rawCenter[1] >= -180 &&
    rawCenter[1] <= 180
      ? [rawCenter[0], rawCenter[1]]
      : undefined

  return {
    defaultZoom:
      defaultZoom !== undefined &&
      Number.isFinite(defaultZoom) &&
      defaultZoom >= 0 &&
      defaultZoom <= 24
        ? defaultZoom
        : 12,
    defaultCenter,
    clustering: isRecord(value) && typeof value.clustering === 'boolean' ? value.clustering : true,
  }
}

function parseMapProperties(value: unknown): MapProperties {
  if (!isRecord(value)) return {}

  const properties: MapProperties = {}
  for (const [key, metadata] of Object.entries(value)) {
    if (!isRecord(metadata)) continue
    const displayName = readString(metadata, 'displayName')
    properties[key] = displayName === undefined ? {} : { displayName }
  }
  return properties
}

export function readBaseMapData(attributes: BaseMapAttributes): BaseMapData | undefined {
  const rawMarkers = parseJson(attributes.markers)
  const currentSlug = attributes.currentSlug
  if (
    !Array.isArray(rawMarkers) ||
    currentSlug === undefined ||
    currentSlug.length === 0 ||
    !isFullSlug(currentSlug)
  ) {
    return undefined
  }

  const markers: MarkerData[] = []
  for (const value of rawMarkers) {
    const marker = parseMarker(value)
    if (marker === undefined) return undefined
    markers.push(marker)
  }

  return {
    markers,
    config: parseMapConfig(parseJson(attributes.config)),
    currentSlug,
    properties: parseMapProperties(parseJson(attributes.properties)),
  }
}

export function readPopupFields(value: unknown): Record<string, unknown> {
  if (isRecord(value)) return value
  if (typeof value !== 'string') return {}
  const parsed = parseJson(value)
  return isRecord(parsed) ? parsed : {}
}

function parseWikilinkValue(raw: string) {
  let text = raw.trim()
  if (!text.startsWith('[[')) {
    if (text.startsWith('![[') && text.endsWith(']]')) {
      text = text.slice(1)
    } else {
      return undefined
    }
  }
  if (!text.endsWith(']]')) return undefined

  const inner = text.slice(2, -2)
  let target = ''
  let alias: string | undefined
  let escaped = false
  for (let index = 0; index < inner.length; index += 1) {
    const character = inner[index]
    if (escaped) {
      target += character
      escaped = false
    } else if (character === '\\') {
      escaped = true
    } else if (character === '|' && alias === undefined) {
      alias = inner
        .slice(index + 1)
        .replace(/\\\|/g, '|')
        .trim()
      break
    } else {
      target += character
    }
  }

  const [path, anchor] = splitAnchor(target.trim())
  return {
    target: path,
    alias: alias && alias.length > 0 ? alias : undefined,
    anchor: anchor.length > 0 ? anchor : undefined,
  }
}

function resolveWikilinkSlug(target: string, currentSlug: FullSlug): FullSlug | undefined {
  if (target.length === 0) return currentSlug

  const normalized = target.replace(/\\/g, '/').replace(/^\/+/, '')
  const candidate = normalized.endsWith('/')
    ? `${normalized}index.md`
    : getFileExtension(normalized)
      ? normalized
      : `${normalized}.md`
  if (!isFilePath(candidate)) return undefined

  return slugifyFilePath(candidate)
}

function splitIconClasses(raw: string): string[] {
  return raw.trim().split(/\s+/).filter(Boolean)
}

function buildIconClassList(raw: string): string[] {
  const parts = splitIconClasses(raw)
  if (parts.length === 0) return []
  if (parts.length > 1) {
    return parts.every(value => /^[a-z_][a-z0-9_-]*$/i.test(value)) ? parts : []
  }

  const separator = parts[0].lastIndexOf(':')
  const name = separator >= 0 ? parts[0].slice(separator + 1) : parts[0]
  if (!/^[a-z_][a-z0-9_-]*$/i.test(name)) return []
  return [name.startsWith('icon-') ? name : `icon-${name}`]
}

function renderIconHtml(raw: string): string {
  const classes = buildIconClassList(raw)
  return classes.length === 0 ? '' : `<i class="${classes.join(' ')}" aria-hidden="true"></i>`
}

function formatPropertyValue(value: unknown, currentSlug: FullSlug): string {
  if (value === undefined || value === null) return ''
  if (isIconToken(value)) return renderIconHtml(value.value)
  if (Array.isArray(value)) {
    return value.map(item => formatPropertyValue(item, currentSlug)).join(', ')
  }
  if (value instanceof Date) return value.toISOString().split('T')[0]
  if (typeof value !== 'string') return escapeHTML(String(value))

  const wikilink = parseWikilinkValue(value)
  if (!wikilink) return escapeHTML(value)

  const slug = resolveWikilinkSlug(wikilink.target, currentSlug)
  if (!slug) return escapeHTML(value)

  const hrefBase = resolveRelative(currentSlug, slug)
  const href = wikilink.anchor ? `${hrefBase}${wikilink.anchor}` : hrefBase
  const dataSlug = wikilink.anchor ? `${slug}${wikilink.anchor}` : slug
  const label = wikilink.alias ?? (wikilink.target.length > 0 ? wikilink.target : currentSlug)
  return `<a href="${escapeHTML(href)}" class="internal" data-slug="${escapeHTML(dataSlug)}">${escapeHTML(label)}</a>`
}

export function createPopupContent(
  marker: MarkerData,
  currentSlug: FullSlug,
  properties: MapProperties,
): string {
  const href = resolveRelative(currentSlug, marker.slug)
  const fields = Object.entries(marker.popupFields).flatMap(([key, value]) => {
    const formattedValue = formatPropertyValue(value, currentSlug)
    if (formattedValue.length === 0) return []

    const displayName =
      properties[key]?.displayName ??
      key
        .replace(/^(note|file)\./, '')
        .split('.')
        .pop()
        ?.replace(/[-_]/g, ' ') ??
      key
    return [
      `<div class="base-map-popup-field"><span class="base-map-popup-label">${escapeHTML(displayName)}:</span> <span class="base-map-popup-value">${formattedValue}</span></div>`,
    ]
  })
  const metadata =
    fields.length === 0 ? '' : `<div class="base-map-popup-meta">${fields.join('')}</div>`
  return `<div class="base-map-popup"><a href="${escapeHTML(href)}" class="base-map-popup-title" data-slug="${escapeHTML(marker.slug)}">${escapeHTML(marker.title)}</a>${metadata}</div>`
}
