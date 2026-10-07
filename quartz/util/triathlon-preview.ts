import type { ActivitySummary, DailyPoint } from '../plugins/stores/analytics'
import type { ActivityKind, StravaActivityDetail } from '../plugins/stores/strava'
import type { FullSlug } from './path'
import {
  triathlonActivityAnchor,
  triathlonDateFromSlug,
  triathlonDaySlug,
} from './triathlon-date-route'
import { TRIATHLON_HOSTNAME, triathlonAssetPathname, triathlonHostUrl } from './triathlon-host'
import { activityGpsSegments, simplifyMapRoute } from './triathlon-map-route'
import { isRecord } from './type-guards'

// The day page and its detail shards weigh megabytes; a link preview reads this ~1 KB summary instead.
export const TRIATHLON_PREVIEW_KIND = 'triathlon-preview-v1'

const PREVIEW_KINDS: readonly ActivityKind[] = [
  'swim',
  'bike',
  'run',
  'strength',
  'walk',
  'yoga',
  'treatment',
  'sauna',
]
const TRACE_BOX = 100
const TRACE_MIN_EXTENT_M = 150

export interface TriathlonPreviewActivity {
  id: number
  sport: ActivityKind
  name: string
  distanceKm: number
  movingTimeS: number
  elevationM: number
  avgHr: number | null
  /** Device-measured power only; Strava estimates stay out of the preview. */
  avgWatts: number | null
  npWatts: number | null
  swimPaceSPer100m: number | null
  /** Garden-computed training load, not a provider measurement. */
  load: number | null
  location: string | null
  virtual: boolean
  /** Route shape as SVG path data in a 100×100 box; no coordinates leave the build. */
  trace: string | null
}

export interface TriathlonPreview {
  kind: typeof TRIATHLON_PREVIEW_KIND
  date: string
  event: string | null
  load: number | null
  form: { ctl: number; atl: number; tsb: number } | null
  activities: TriathlonPreviewActivity[]
}

export interface TriathlonPreviewTarget {
  date: string
  activityId: string | null
}

export const triathlonPreviewSlug = (date: string): FullSlug =>
  `static/triathlon/preview/${date}` as FullSlug

export const triathlonPreviewUrl = (date: string, reference: string): URL =>
  new URL(`/${triathlonPreviewSlug(date)}.json`, reference)

/** The day page on the reference's host: `/triathlon/on/…` on the apex, `/on/…` on the microsite. */
export function triathlonPreviewDayHref(date: string, reference: string): string {
  const url = new URL(`/${triathlonDaySlug(date) ?? ''}`, reference)
  return url.hostname === TRIATHLON_HOSTNAME ? triathlonHostUrl(url) : url.toString()
}

/** Resolves `/triathlon/on/YYYY/MM/DD[#tri-activity-ID]` on the apex and `/on/…` on the microsite. */
export function triathlonPreviewTarget(
  href: string,
  reference?: string,
): TriathlonPreviewTarget | null {
  if (!URL.canParse(href, reference)) return null
  const url = new URL(href, reference)
  const pathname =
    url.hostname === TRIATHLON_HOSTNAME ? triathlonAssetPathname(url.pathname) : url.pathname
  const date = triathlonDateFromSlug(pathname.replace(/^\/+|\/+$/g, '').replace(/\.html?$/, ''))
  if (!date) return null
  const activityId = /(\d+)$/.exec(url.hash)?.[1]
  return {
    date,
    activityId:
      activityId && `#${triathlonActivityAnchor(activityId)}` === url.hash ? activityId : null,
  }
}

const round = (value: number, digits: number): number => {
  const scale = 10 ** digits
  return Math.round(value * scale) / scale
}

function previewTrace(detail: StravaActivityDetail): string | null {
  const segments = activityGpsSegments(detail)
  if (segments.length === 0) return null
  const lat = segments[0][0].lat
  const scaleX = Math.cos((lat * Math.PI) / 180) * 111_320
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity
  for (const segment of segments)
    for (const point of segment) {
      const x = point.lng * scaleX
      const y = -point.lat * 111_320
      minX = Math.min(minX, x)
      maxX = Math.max(maxX, x)
      minY = Math.min(minY, y)
      maxY = Math.max(maxY, y)
    }
  const extent = Math.max(maxX - minX, maxY - minY)
  if (!Number.isFinite(extent) || extent < TRACE_MIN_EXTENT_M) return null
  const scale = TRACE_BOX / extent
  const offsetX = (TRACE_BOX - (maxX - minX) * scale) / 2
  const offsetY = (TRACE_BOX - (maxY - minY) * scale) / 2
  const commands: string[] = []
  for (const segment of segments) {
    const points = simplifyMapRoute(segment, extent / 160)
    if (points.length < 2) continue
    points.forEach((point, index) => {
      const x = round((point.lng * scaleX - minX) * scale + offsetX, 1)
      const y = round((-point.lat * 111_320 - minY) * scale + offsetY, 1)
      commands.push(`${index === 0 ? 'M' : 'L'}${x} ${y}`)
    })
  }
  return commands.length > 1 ? commands.join('') : null
}

const positive = (value: number | null | undefined): number | null =>
  value != null && Number.isFinite(value) && value > 0 ? value : null

export function buildTriathlonPreview(input: {
  date: string
  details: readonly StravaActivityDetail[]
  activities: ReadonlyMap<number, Pick<ActivitySummary, 'load'>>
  daily: Pick<DailyPoint, 'load' | 'ctl' | 'atl' | 'tsb'> | null
  event: string | null
}): TriathlonPreview {
  const activities = [...input.details]
    .sort((left, right) => left.start.localeCompare(right.start))
    .map(
      (detail): TriathlonPreviewActivity => ({
        id: detail.id,
        sport: detail.sport,
        name: detail.name,
        distanceKm: round(detail.distanceKm, 3),
        movingTimeS: Math.round(detail.movingTimeS),
        elevationM: Math.round(detail.elevationM),
        avgHr: positive(detail.avgHr),
        avgWatts: detail.deviceWatts ? positive(detail.avgWatts) : null,
        npWatts: detail.deviceWatts ? positive(detail.npWatts) : null,
        swimPaceSPer100m: detail.sport === 'swim' ? positive(detail.swimPaceSPer100m) : null,
        load: input.activities.get(detail.id)?.load ?? null,
        location: detail.location,
        virtual: detail.virtual === true,
        trace: previewTrace(detail),
      }),
    )
  return {
    kind: TRIATHLON_PREVIEW_KIND,
    date: input.date,
    event: input.event,
    load: input.daily ? round(input.daily.load, 1) : null,
    form: input.daily
      ? {
          ctl: round(input.daily.ctl, 1),
          atl: round(input.daily.atl, 1),
          tsb: round(input.daily.tsb, 1),
        }
      : null,
    activities,
  }
}

const finiteOrNull = (value: unknown): value is number | null =>
  value === null || (typeof value === 'number' && Number.isFinite(value))

function isPreviewActivity(value: unknown): value is TriathlonPreviewActivity {
  return (
    isRecord(value) &&
    typeof value.id === 'number' &&
    PREVIEW_KINDS.includes(value.sport as ActivityKind) &&
    typeof value.name === 'string' &&
    typeof value.distanceKm === 'number' &&
    typeof value.movingTimeS === 'number' &&
    typeof value.elevationM === 'number' &&
    finiteOrNull(value.avgHr) &&
    finiteOrNull(value.avgWatts) &&
    finiteOrNull(value.npWatts) &&
    finiteOrNull(value.swimPaceSPer100m) &&
    finiteOrNull(value.load) &&
    (value.location === null || typeof value.location === 'string') &&
    typeof value.virtual === 'boolean' &&
    (value.trace === null || typeof value.trace === 'string')
  )
}

export function readTriathlonPreview(value: unknown, date: string): TriathlonPreview | null {
  if (!isRecord(value) || value.kind !== TRIATHLON_PREVIEW_KIND || value.date !== date) return null
  const { form } = value
  const validForm =
    form === null ||
    (isRecord(form) && [form.ctl, form.atl, form.tsb].every(n => typeof n === 'number'))
  if (
    (value.event !== null && typeof value.event !== 'string') ||
    !finiteOrNull(value.load) ||
    !validForm ||
    !Array.isArray(value.activities) ||
    !value.activities.every(isPreviewActivity)
  )
    return null
  return value as unknown as TriathlonPreview
}
