import type {
  TriathlonCalendarPreview,
  TriathlonPreview,
  TriathlonPreviewActivity,
  TriathlonPreviewTarget,
} from '../../util/triathlon-preview'
import { SPORT_ICON, type ActivityKind } from '../../plugins/stores/strava'
import {
  computeTriathlonCalcTimes,
  formatDurationClock,
  TRI_RACE_DISTANCES,
  type CalcShare,
} from '../../util/triathlon-calculator'
import { clock, dist, distCombined, dur, KM_TO_MI } from '../../util/triathlon-card'
import { triathlonActivityAnchor } from '../../util/triathlon-date-route'
import {
  DEFAULT_TRIATHLON_PRESENTATION,
  distanceSystemFromStoredUnit,
  TRI_UNIT_KEY,
  type TriathlonPresentation,
} from '../../util/triathlon-presentation'

const TIMED_ONLY = new Set<ActivityKind>(['strength', 'yoga', 'treatment', 'sauna'])
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

const sessionSummary = (p: TriathlonPresentation, a: TriathlonPreviewActivity): string =>
  TIMED_ONLY.has(a.sport) || a.distanceKm <= 0
    ? dur(a.movingTimeS)
    : `${dist(p, a.distanceKm, a.sport)} · ${dur(a.movingTimeS)}`

const ICONS: Record<string, string[]> = {
  day: ['M6 3h8l4 4v14H6z', 'M14 3v5h5'],
  calendar: ['M5 5h14v15H5z', 'M5 9h14', 'M9 3v4', 'M15 3v4'],
  calculator: [
    'M7 3h10v18H7z',
    'M9 7h6',
    'M9 11h.01',
    'M12 11h.01',
    'M15 11h.01',
    'M9 15h.01',
    'M12 15h.01',
    'M15 15h.01',
  ],
}

const isSport = (name: string): name is ActivityKind =>
  Object.prototype.hasOwnProperty.call(SPORT_ICON, name)

function icon(name: string): SVGSVGElement {
  const sport = isSport(name)
  const glyph = document.createElementNS(SVG_NS, 'svg')
  glyph.setAttribute('viewBox', '0 0 24 24')
  glyph.setAttribute('aria-hidden', 'true')
  if (sport) glyph.dataset.sport = name
  if (name === 'yoga' || name === 'treatment') glyph.dataset.solid = ''
  for (const d of sport ? SPORT_ICON[name] : (ICONS[name] ?? ICONS.day)) {
    const path = document.createElementNS(SVG_NS, 'path')
    path.setAttribute('d', d)
    glyph.appendChild(path)
  }
  return glyph
}

export function miniatureList(items: string[]): HTMLElement {
  const list = el('ul', 'document-popover-list')
  for (const item of items) list.appendChild(el('li', undefined, item))
  return list
}

export function windowPreview(options: {
  href: string
  title: string
  icon: string
  meta: string[]
  trace: SVGSVGElement | null
  body: HTMLElement | null
}) {
  const card = el('article', 'document-popover-card')
  card.dataset.type = 'triathlon'
  if (options.trace) card.dataset.trace = ''

  const anchor = link(options.href, '', 'document-popover-window')
  const frame = el('div', 'document-popover-frame')
  frame.ariaHidden = 'true'
  const surface = el('div', 'document-popover-surface')
  if (options.trace) {
    const figure = el('div', 'document-popover-route')
    figure.appendChild(options.trace)
    surface.appendChild(figure)
  } else if (options.body) surface.appendChild(options.body)
  else frame.classList.add('is-empty')
  frame.appendChild(surface)

  const badge = el('span', 'document-popover-badge')
  badge.appendChild(icon(options.icon))
  anchor.append(frame, badge, el('span', 'document-popover-title', options.title))
  card.appendChild(anchor)

  const meta = el('ul', 'document-popover-meta')
  for (const item of options.meta) meta.appendChild(el('li', undefined, item))
  card.appendChild(meta)
  return card
}

function traceFigure(activity: TriathlonPreviewActivity): SVGSVGElement | null {
  if (!activity.trace) return null
  const svg = document.createElementNS(SVG_NS, 'svg')
  svg.dataset.sport = activity.sport
  svg.setAttribute('viewBox', '-6 -6 112 112')
  svg.setAttribute('aria-hidden', 'true')
  const path = document.createElementNS(SVG_NS, 'path')
  path.setAttribute('d', activity.trace)
  path.setAttribute('vector-effect', 'non-scaling-stroke')
  svg.appendChild(path)
  return svg
}

function poolFigure(activity: TriathlonPreviewActivity): SVGSVGElement | null {
  if (!activity.pool) return null
  const svg = document.createElementNS(SVG_NS, 'svg')
  svg.dataset.sport = activity.sport
  svg.dataset.chart = 'pool'
  svg.setAttribute('viewBox', '0 0 100 40')
  svg.setAttribute('preserveAspectRatio', 'none')
  svg.setAttribute('aria-hidden', 'true')
  for (const [stroke, d] of Object.entries(activity.pool)) {
    if (!d) continue
    const path = document.createElementNS(SVG_NS, 'path')
    path.dataset.stroke = stroke
    path.setAttribute('d', d)
    svg.appendChild(path)
  }
  return svg
}

const longestTrace = (
  activities: readonly TriathlonPreviewActivity[],
): TriathlonPreviewActivity | undefined =>
  [...activities]
    .filter(activity => activity.trace)
    .sort((left, right) => right.distanceKm - left.distanceKm)[0]

function dayMeta(
  p: TriathlonPresentation,
  preview: TriathlonPreview,
  focus?: TriathlonPreviewActivity,
): string[] {
  if (focus) {
    const items = [focus.sport, sessionSummary(p, focus)]
    const watts = focus.npWatts ?? focus.avgWatts
    if (watts != null) items.push(`${Math.round(watts)} W`)
    if (focus.virtual) items.push('virtual')
    return items
  }
  const items = ['day']
  if (preview.activities.length > 0) items.push(String(preview.activities.length))
  const km = preview.activities.reduce(
    (sum, activity) => sum + (TIMED_ONLY.has(activity.sport) ? 0 : activity.distanceKm),
    0,
  )
  if (km > 0) items.push(distCombined(p, km))
  if (preview.load != null && preview.load > 0) items.push(`load ${Math.round(preview.load)}`)
  if (preview.activities.length === 0) items.push('rest')
  if (preview.event) items.push(preview.event)
  return items
}

function mount(popoverInner: HTMLElement, card: HTMLElement, contentType: string) {
  popoverInner.dataset.contentType = contentType
  popoverInner.appendChild(card)
}

export function renderTriathlonPreview(
  preview: TriathlonPreview,
  target: Extract<TriathlonPreviewTarget, { kind: 'day' }>,
  dayHref: string,
  p: TriathlonPresentation,
  popoverInner: HTMLElement,
) {
  const focus = target.activityId
    ? preview.activities.find(activity => String(activity.id) === target.activityId)
    : undefined
  const href = focus ? `${dayHref}#${triathlonActivityAnchor(focus.id) ?? ''}` : dayHref
  const traced = focus?.trace ? focus : focus ? undefined : longestTrace(preview.activities)
  const body =
    traced || focus
      ? null
      : preview.activities.length > 0
        ? miniatureList(preview.activities.map(activity => `${activity.sport} · ${activity.name}`))
        : null
  mount(
    popoverInner,
    windowPreview({
      href,
      title: focus ? focus.name : longDate(preview.date),
      icon: (focus ?? traced)?.sport ?? preview.activities[0]?.sport ?? 'day',
      meta: dayMeta(p, preview, focus),
      trace: traced ? traceFigure(traced) : focus ? poolFigure(focus) : null,
      body,
    }),
    'text/x-triathlon',
  )
  if (traced && preview.activities.length > 1) {
    popoverInner
      .querySelector('.document-popover-card')
      ?.appendChild(
        miniatureList(
          preview.activities.map(activity => `${activity.sport} · ${sessionSummary(p, activity)}`),
        ),
      )
  }
}

export function renderTriathlonComparisonPreview(
  activities: readonly TriathlonPreviewActivity[],
  href: string,
  p: TriathlonPresentation,
  popoverInner: HTMLElement,
) {
  const traced = longestTrace(activities)
  mount(
    popoverInner,
    windowPreview({
      href,
      title: 'comparison',
      icon: (traced ?? activities[0])?.sport ?? 'day',
      meta: ['comparison', String(activities.length)],
      trace: traced ? traceFigure(traced) : null,
      body: traced
        ? null
        : miniatureList(activities.map(activity => `${activity.sport} · ${activity.name}`)),
    }),
    'text/x-triathlon-comparison',
  )
  if (traced) {
    popoverInner
      .querySelector('.document-popover-card')
      ?.appendChild(
        miniatureList(
          activities.map(activity => `${activity.sport} · ${sessionSummary(p, activity)}`),
        ),
      )
  }
}

export function renderTriathlonCalendarPreview(
  preview: TriathlonCalendarPreview,
  target: Extract<TriathlonPreviewTarget, { kind: 'calendar' }>,
  href: string,
  popoverInner: HTMLElement,
) {
  const season =
    (target.year != null ? preview.seasons.find(entry => entry.year === target.year) : undefined) ??
    preview.seasons[preview.seasons.length - 1]
  const events = [...(season?.events ?? [])].sort((left, right) =>
    (left.date ?? '9999').localeCompare(right.date ?? '9999'),
  )
  const shown = events.slice(0, 6)
  mount(
    popoverInner,
    windowPreview({
      href,
      title: season ? String(season.year) : 'calendar',
      icon: 'calendar',
      meta: ['calendar', target.view, ...(season ? [String(events.length)] : [])],
      trace: null,
      body:
        shown.length > 0
          ? miniatureList(
              shown.map(event => (event.date ? `${event.date} · ${event.name}` : event.name)),
            )
          : null,
    }),
    'text/x-triathlon-calendar',
  )
}

export function renderTriathlonCalculatorPreview(
  share: CalcShare,
  href: string,
  popoverInner: HTMLElement,
) {
  const [label, swimKm, bikeKm, runKm] =
    TRI_RACE_DISTANCES[share.presetIdx] ?? TRI_RACE_DISTANCES[1]
  const times = computeTriathlonCalcTimes({
    swimKm,
    bikeKm,
    runKm,
    swimPaceSec: share.swimPaceSec,
    t1Sec: share.t1Sec,
    bikeMph: share.bikeMph,
    t2Sec: share.t2Sec,
    runPaceSec: share.runPaceSec,
  })
  const imperial = share.unit === 'i'
  const rows: [string, string][] = [
    ['swim', `${clock(share.swimPaceSec)} /100m`],
    [
      'bike',
      `${(imperial ? share.bikeMph : share.bikeMph / KM_TO_MI).toFixed(1)} ${imperial ? 'mph' : 'km/h'}`,
    ],
    [
      'run',
      `${clock(imperial ? share.runPaceSec : share.runPaceSec * KM_TO_MI)} ${imperial ? '/mi' : '/km'}`,
    ],
    ['finish', formatDurationClock(times.totalSec)],
  ]
  const table = el('table', 'document-popover-table')
  const body = table.appendChild(el('tbody'))
  for (const [name, value] of rows) {
    const row = el('tr')
    row.append(el('td', undefined, name), el('td', undefined, value))
    body.appendChild(row)
  }
  mount(
    popoverInner,
    windowPreview({
      href,
      title: label,
      icon: 'calculator',
      meta: [
        'calculator',
        share.mode === 'a' ? 'average' : 'projected',
        formatDurationClock(times.totalSec),
      ],
      trace: null,
      body: table,
    }),
    'text/x-triathlon-calculator',
  )
}
