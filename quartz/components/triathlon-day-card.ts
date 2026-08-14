import type { Element } from 'hast'
import { h, s } from 'hastscript'
import type { QuartzPluginData } from '../plugins/vfile'
import { joinSegments, slugAnchor } from '../util/path'
import {
  buildDayCard,
  type DayCardExtras,
  type DayCardPayload,
  type DetailCtx,
  type TriNodeFactory,
  parseExcludedActivityIds,
} from '../util/triathlon-card'
import { triathlonDaySlug } from '../util/triathlon-date-route'
import { DEFAULT_TRIATHLON_PRESENTATION } from '../util/triathlon-presentation'
import {
  parseTriathlonTraceSettings,
  serializeTriathlonTraceSettings,
  triathlonTraceEnabled,
  type TriathlonTraceSettings,
} from '../util/triathlon-trace-settings'

const TRIATHLON_SPORT_ANCHOR: Record<string, NonNullable<DayCardExtras['sport']>> = {
  swim: 'swim',
  bike: 'bike',
  cycling: 'bike',
  cycle: 'bike',
  run: 'run',
  walk: 'walk',
}
const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every(segment => typeof segment === 'string')
const TRIATHLON_EMBED_RE = /!\[\[triathlon#([^\]|]+)(?:\|[^\]]*)?\]\]/g

export type TriathlonEmbedAnchor = {
  date: string
  sport?: NonNullable<DayCardExtras['sport']>
  excludedActivityIds?: string[]
  settings?: TriathlonTraceSettings
}

export const triathlonEmbedAnchor = (value: string | undefined): TriathlonEmbedAnchor | null => {
  if (!value) return null
  let segments: unknown
  try {
    segments = JSON.parse(value)
  } catch {
    return null
  }
  if (!isStringArray(segments) || segments.length < 2) return null
  const [date, ...options] = segments
  if (triathlonDaySlug(date) === null) return null

  let sport: TriathlonEmbedAnchor['sport']
  let excludedActivityIds: string[] | undefined
  let settings: TriathlonTraceSettings | undefined
  for (const option of options) {
    const activityKind = TRIATHLON_SPORT_ANCHOR[option]
    if (activityKind) {
      if (sport) return null
      sport = activityKind
      continue
    }
    if (option.startsWith('filter=')) {
      if (excludedActivityIds) return null
      const parsed = parseExcludedActivityIds(option)
      if (parsed.length === 0) return null
      excludedActivityIds = parsed
      continue
    }
    if (option.startsWith('settings=')) {
      if (settings) return null
      const parsed = parseTriathlonTraceSettings(option)
      if (!parsed) return null
      settings = parsed
      continue
    }
    return null
  }

  return {
    date,
    ...(sport ? { sport } : {}),
    ...(excludedActivityIds ? { excludedActivityIds } : {}),
    ...(settings ? { settings } : {}),
  }
}

export const triathlonEmbedAnchorFromSource = (
  value: string | undefined,
  source: string | undefined,
): TriathlonEmbedAnchor | null => {
  const parsed = triathlonEmbedAnchor(value)
  if (parsed || !value || !source) return parsed

  for (const match of source.matchAll(TRIATHLON_EMBED_RE)) {
    const segments = match[1].split('#').map(segment => segment.trim())
    if (JSON.stringify(segments.map(segment => slugAnchor(segment))) !== value) continue
    const recovered = triathlonEmbedAnchor(JSON.stringify(segments))
    if (recovered) return recovered
  }
  return null
}

export const triathlonEmbedDayHref = (root: string, date: string): string | null => {
  const daySlug = triathlonDaySlug(date)
  return daySlug ? joinSegments(root, daySlug) : null
}

export const triathlonCardFactory: TriNodeFactory<Element> = {
  presentation: DEFAULT_TRIATHLON_PRESENTATION,
  el: (tag, cls, text, attrs) =>
    h(tag, { ...(cls ? { class: cls } : {}), ...attrs }, text === undefined ? [] : [text]),
  svg: (tag, attrs) => s(tag, attrs),
  add: (parent, ...children) => {
    parent.children.push(...children)
  },
}

export const triathlonDayExtras = (page: QuartzPluginData, date: string): DayCardExtras => {
  const extras: DayCardExtras = {}
  const location = page.frontmatter?.['location']
  if (typeof location === 'string' && location !== '') extras.location = location
  const day = page.tracking?.days.find(entry => entry.date === date)
  const event = day?.event ?? (day?.race ? 'race' : null)
  if (event) extras.event = event
  return extras
}

export const triathlonDayProps = (extras: DayCardExtras, date: string): Record<string, string> => {
  const props: Record<string, string> = { 'data-triathlon-date': date }
  if (extras.location) props['data-triathlon-loc'] = extras.location
  if (extras.event) props['data-triathlon-event'] = extras.event
  if (extras.sport) props['data-triathlon-sport'] = extras.sport
  if (extras.excludedActivityIds?.length)
    props['data-triathlon-filter'] = extras.excludedActivityIds.join('&')
  if (extras.settings)
    props['data-triathlon-settings'] = serializeTriathlonTraceSettings(extras.settings)
  if (extras.expanded) props['data-triathlon-expanded'] = '1'
  if (extras.embedded) props['data-triathlon-embedded'] = '1'
  if (extras.dateHref) props['data-triathlon-date-href'] = extras.dateHref
  return props
}

export const filterTriathlonTraceElements = (
  root: Element,
  settings: TriathlonTraceSettings | undefined,
): void => {
  root.children = root.children.filter(child => {
    if (child.type !== 'element') return true
    const trace = child.properties.dataTriTrace
    if (typeof trace === 'string' && !triathlonTraceEnabled(settings, trace)) return false
    filterTriathlonTraceElements(child, settings)
    return true
  })
}

export const triathlonDayCard = (
  date: string,
  payload: DayCardPayload | null,
  extras: DayCardExtras,
  ctx: DetailCtx,
): Element => {
  const card = buildDayCard(triathlonCardFactory, date, payload, extras, undefined, ctx)
  filterTriathlonTraceElements(card, extras.settings)
  return card
}
