import type { ActivityKind } from '../../plugins/stores/strava'
import type {
  TriathlonPreview,
  TriathlonPreviewActivity,
  TriathlonPreviewTarget,
} from '../../util/triathlon-preview'
import {
  clock,
  dist,
  distCombined,
  dur,
  formatElevationGain,
  rate,
} from '../../util/triathlon-card'
import { triathlonActivityAnchor } from '../../util/triathlon-date-route'
import {
  DEFAULT_TRIATHLON_PRESENTATION,
  distanceSystemFromStoredUnit,
  TRI_UNIT_KEY,
  type TriathlonPresentation,
} from '../../util/triathlon-presentation'

const TIMED_ONLY = new Set<ActivityKind>(['strength', 'yoga', 'treatment', 'sauna'])
const CLIMB_SPORTS = new Set<ActivityKind>(['bike', 'run', 'walk'])
const SVG_NS = 'http://www.w3.org/2000/svg'
const LONG_DATE = new Intl.DateTimeFormat('en-US', {
  weekday: 'short',
  month: 'short',
  day: 'numeric',
  year: 'numeric',
  timeZone: 'UTC',
})

export function triathlonPreviewPresentation(): TriathlonPresentation {
  try {
    const distance = distanceSystemFromStoredUnit(localStorage.getItem(TRI_UNIT_KEY))
    return { ...DEFAULT_TRIATHLON_PRESENTATION, distance }
  } catch {
    return DEFAULT_TRIATHLON_PRESENTATION
  }
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag)
  if (className) node.className = className
  if (text !== undefined) node.textContent = text
  return node
}

function link(href: string, text: string, className?: string): HTMLAnchorElement {
  const anchor = el('a', className, text)
  anchor.href = href
  return anchor
}

const longDate = (iso: string): string => {
  const [year, month, day] = iso.split('-').map(Number)
  return LONG_DATE.format(new Date(Date.UTC(year, month - 1, day)))
}

const signed = (value: number): string =>
  `${value > 0 ? '+' : value < 0 ? '−' : ''}${Math.abs(value).toFixed(1)}`

function activityStats(p: TriathlonPresentation, a: TriathlonPreviewActivity): [string, string][] {
  const rows: [string, string][] = []
  if (!TIMED_ONLY.has(a.sport) && a.distanceKm > 0) {
    rows.push(['distance', dist(p, a.distanceKm, a.sport)])
    rows.push(['time', dur(a.movingTimeS)])
    if (a.movingTimeS > 0)
      rows.push([
        a.sport === 'bike' ? 'speed' : 'pace',
        a.swimPaceSPer100m != null
          ? `${clock(a.swimPaceSPer100m)} /100m`
          : rate(p, a.sport, a.distanceKm, a.movingTimeS),
      ])
  } else rows.push(['time', dur(a.movingTimeS)])
  const watts = a.npWatts ?? a.avgWatts
  if (watts != null) rows.push([a.npWatts != null ? 'NP' : 'avg power', `${Math.round(watts)} W`])
  if (a.avgHr != null) rows.push(['avg hr', `${Math.round(a.avgHr)} bpm`])
  if (CLIMB_SPORTS.has(a.sport) && a.elevationM > 0)
    rows.push(['elevation', formatElevationGain(p, a.elevationM)])
  if (a.load != null) rows.push(['load', String(Math.round(a.load))])
  return rows
}

function dayStats(p: TriathlonPresentation, preview: TriathlonPreview): [string, string][] {
  const { activities } = preview
  const rows: [string, string][] = []
  if (activities.length > 0) {
    rows.push(['sessions', String(activities.length)])
    rows.push(['time', dur(activities.reduce((sum, a) => sum + a.movingTimeS, 0))])
    const km = activities.reduce((sum, a) => sum + (TIMED_ONLY.has(a.sport) ? 0 : a.distanceKm), 0)
    if (km > 0) rows.push(['distance', distCombined(p, km)])
  }
  if (preview.load != null && preview.load > 0)
    rows.push(['load', String(Math.round(preview.load))])
  if (preview.form) rows.push(['form', signed(preview.form.tsb)])
  return rows
}

const sessionSummary = (p: TriathlonPresentation, a: TriathlonPreviewActivity): string =>
  TIMED_ONLY.has(a.sport) || a.distanceKm <= 0
    ? dur(a.movingTimeS)
    : `${dist(p, a.distanceKm, a.sport)} · ${dur(a.movingTimeS)}`

function traceFigure(activity: TriathlonPreviewActivity): SVGSVGElement | null {
  if (!activity.trace) return null
  const svg = document.createElementNS(SVG_NS, 'svg')
  svg.classList.add('triathlon-popover-trace')
  svg.dataset.sport = activity.sport
  svg.setAttribute('viewBox', '-6 -6 112 112')
  svg.setAttribute('aria-hidden', 'true')
  const path = document.createElementNS(SVG_NS, 'path')
  path.setAttribute('d', activity.trace)
  path.setAttribute('vector-effect', 'non-scaling-stroke')
  svg.appendChild(path)
  return svg
}

export function renderTriathlonPreview(
  preview: TriathlonPreview,
  target: TriathlonPreviewTarget,
  dayHref: string,
  p: TriathlonPresentation,
  popoverInner: HTMLElement,
) {
  const focus = target.activityId
    ? preview.activities.find(a => String(a.id) === target.activityId)
    : undefined
  const activityHref = (a: TriathlonPreviewActivity) =>
    `${dayHref}#${triathlonActivityAnchor(a.id) ?? ''}`
  popoverInner.dataset.contentType = 'text/x-triathlon'

  const card = el('article', 'triathlon-popover-card')
  const header = el('header', 'triathlon-popover-header')
  const title = el('h2', 'triathlon-popover-title')
  title.appendChild(
    link(focus ? activityHref(focus) : dayHref, focus ? focus.name : longDate(preview.date)),
  )
  header.appendChild(title)

  const meta = el('ul', 'triathlon-popover-meta')
  meta.ariaLabel = focus ? 'activity' : 'day'
  if (focus) {
    const sport = el('li', 'triathlon-popover-sport', focus.sport)
    sport.dataset.sport = focus.sport
    meta.appendChild(sport)
    meta.appendChild(el('li', undefined, longDate(preview.date)))
  }
  const location = (focus ?? preview.activities[0])?.location
  if (location) meta.appendChild(el('li', undefined, location))
  if (focus?.virtual) meta.appendChild(el('li', undefined, 'virtual'))
  if (preview.activities.length === 0) meta.appendChild(el('li', undefined, 'rest day'))
  if (preview.event) meta.appendChild(el('li', 'triathlon-popover-event', preview.event))
  header.appendChild(meta)
  card.appendChild(header)

  const traced =
    focus ??
    [...preview.activities]
      .filter(a => a.trace)
      .sort((left, right) => right.distanceKm - left.distanceKm)[0]
  const trace = traced ? traceFigure(traced) : null
  if (trace) {
    card.classList.add('has-trace')
    card.appendChild(trace)
  }

  const stats = el('dl', 'triathlon-popover-stats')
  for (const [label, value] of focus ? activityStats(p, focus) : dayStats(p, preview)) {
    const cell = el('div')
    cell.append(el('dt', undefined, label), el('dd', undefined, value))
    stats.appendChild(cell)
  }
  if (stats.childElementCount > 0) card.appendChild(stats)

  const sessions = preview.activities.filter(a => a !== focus)
  if (sessions.length > 0) {
    const section = el('section', 'triathlon-popover-sessions')
    section.ariaLabel = focus ? 'same day' : 'sessions'
    if (focus) section.appendChild(el('h3', undefined, 'same day'))
    const list = el('ol')
    for (const a of sessions) {
      const row = el('li')
      const anchor = link(activityHref(a), '')
      const swatch = el('span', 'triathlon-popover-sport', a.sport)
      swatch.dataset.sport = a.sport
      anchor.append(
        swatch,
        el('span', 'triathlon-popover-session-name', a.name),
        el('span', 'triathlon-popover-session-figures', sessionSummary(p, a)),
      )
      row.appendChild(anchor)
      list.appendChild(row)
    }
    section.appendChild(list)
    card.appendChild(section)
  }

  const source = el('p', 'triathlon-popover-source')
  source.appendChild(link(dayHref, 'triathlon log'))
  card.appendChild(source)
  popoverInner.appendChild(card)
}
