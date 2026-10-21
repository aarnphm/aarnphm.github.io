import CALENDAR_DETAILS from '../triathlon-calendar-details.json'
import { isRecord } from '../type-guards'

export type TriathlonCalendarSeries =
  | 'ironman'
  | 't100'
  | 'supertri'
  | 'hyrox'
  | 'tcs'
  | 'b4bh'
  | 'other'
export type TriathlonCalendarLeg = 'swim' | 'bike' | 'run'
type TriathlonCalendarResultSegment = TriathlonCalendarLeg | 'T1' | 'T2' | 'overall'

export interface TriathlonCalendarResults extends Record<
  TriathlonCalendarResultSegment,
  string | null
> {
  distance: string | null
}

export interface TriathlonCalendarCourse {
  leg: TriathlonCalendarLeg
  title: string
  source: string
  edition: number | null
  mapUrl: string | null
  gpxUrl: string | null
  /** Length of the plotted GPS route. */
  distanceM: number
  /** Organiser's advertised distance, when it differs from the plotted route. */
  raceDistanceM: number | null
  laps: number | null
  aidStationsPerLap: number | null
  elevationGainM: number | null
  /** Google encoded polyline, precision 5. */
  polyline: string
  /** Elevation in metres at even distance steps; empty for swims. */
  profile: number[]
  aidM: number[]
}

export interface TriathlonCalendarScheduleItem {
  start: string
  end: string | null
  activity: string
  location: string | null
  race: boolean
}

export interface TriathlonCalendarScheduleDay {
  /** Days from the published edition's race day. */
  offset: number
  items: TriathlonCalendarScheduleItem[]
}

export interface TriathlonCalendarSchedule {
  source: string
  /** Race day of the edition the schedule was published for. */
  raceDate: string
  days: TriathlonCalendarScheduleDay[]
}

export interface TriathlonCalendarDetails {
  sources: string[]
  fetched: string | null
  venue: string | null
  venueMap: string | null
  athleteGuide: string | null
  terrain: Partial<Record<TriathlonCalendarLeg, string>>
  airC: [number, number] | null
  waterC: [number, number] | null
  course: TriathlonCalendarCourse[]
  schedule: TriathlonCalendarSchedule | null
  schedulePending: boolean
  /**
   * Fields a later sync failed to extract and carried over instead, keyed by field label
   * (`bike course`, `schedule`, `swim terrain`, `air`, `water`, `venue`) with the date of the sync
   * that last extracted them.
   */
  kept: Record<string, string>
}

export interface TriathlonCalendarEvent {
  id: string
  name: string
  location: string
  kind: 'triathlon' | 'hyrox' | 'running' | 'cycling'
  series: TriathlonCalendarSeries
  format: string
  url: string
  date: string | null
  endDate: string | null
  note: string | null
  details: TriathlonCalendarDetails | null
  results: TriathlonCalendarResults | null
  participated: boolean
}

export interface TriathlonCalendar {
  year: number
  checked: string | null
  /** Date the course and schedule details were extracted from the official pages. */
  fetched: string | null
  events: TriathlonCalendarEvent[]
}

export const TRIATHLON_CALENDAR_LEGS: readonly TriathlonCalendarLeg[] = ['swim', 'bike', 'run']
export const TRIATHLON_CALENDAR_RESULT_SEGMENTS: readonly TriathlonCalendarResultSegment[] = [
  'overall',
  'swim',
  'T1',
  'bike',
  'T2',
  'run',
]

const SERIES_HOSTS: readonly [string, TriathlonCalendarSeries][] = [
  ['ironman.com', 'ironman'],
  ['t100triathlon.com', 't100'],
  ['supertri.com', 'supertri'],
  ['hyrox.com', 'hyrox'],
  ['torontowaterfrontmarathon.com', 'tcs'],
  ['bikeforbrainhealth.ca', 'b4bh'],
]

export const calendarSeries = (url: string): TriathlonCalendarSeries => {
  const host = new URL(url).hostname
  return (
    SERIES_HOSTS.find(([domain]) => host === domain || host.endsWith(`.${domain}`))?.[1] ?? 'other'
  )
}

export const civilDate = (value: unknown): value is string => {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false
  if (value.startsWith('0000')) return false
  const date = new Date(`${value}T00:00:00.000Z`)
  return Number.isFinite(date.valueOf()) && date.toISOString().slice(0, 10) === value
}

export const text = (value: unknown): string | null =>
  typeof value === 'string' && value.trim() ? value.trim() : null

const resultDuration = (value: unknown): string | null =>
  typeof value === 'string' && /^\d+:[0-5]\d:[0-5]\d$/.test(value) ? value : null

export const parseCalendarResults = (value: unknown): TriathlonCalendarResults | null => {
  if (!isRecord(value)) return null
  const results: TriathlonCalendarResults = {
    distance: text(value.distance),
    overall: resultDuration(value.overall),
    swim: resultDuration(value.swim),
    T1: resultDuration(value.T1),
    bike: resultDuration(value.bike),
    T2: resultDuration(value.T2),
    run: resultDuration(value.run),
  }
  return TRIATHLON_CALENDAR_RESULT_SEGMENTS.some(segment => results[segment] !== null)
    ? results
    : null
}

export const sourceUrl = (value: unknown): string | null => {
  const source = text(value)
  if (!source || !URL.canParse(source)) return null
  for (const character of source) {
    const code = character.charCodeAt(0)
    if (code < 32 || code === 127) return null
  }
  const url = new URL(source)
  if ((url.protocol !== 'https:' && url.protocol !== 'http:') || url.username || url.password)
    return null
  return url.href
}

const clockTime = (value: unknown): value is string =>
  typeof value === 'string' && /^(?:[01]\d|2[0-3]):[0-5]\d$/.test(value)

const finiteNumbers = (value: unknown): number[] =>
  Array.isArray(value)
    ? value.filter((item): item is number => typeof item === 'number' && Number.isFinite(item))
    : []

const celsiusRange = (value: unknown): [number, number] | null => {
  const [low, high, ...rest] = finiteNumbers(value)
  return low !== undefined && high !== undefined && rest.length === 0 && low <= high
    ? [low, high]
    : null
}

const isLeg = (value: unknown): value is TriathlonCalendarLeg =>
  value === 'swim' || value === 'bike' || value === 'run'

const positiveInteger = (value: unknown): number | null =>
  typeof value === 'number' && Number.isInteger(value) && value > 0 ? value : null

const parseCourse = (value: unknown): TriathlonCalendarCourse | null => {
  if (!isRecord(value) || !isLeg(value.leg)) return null
  const title = text(value.title)
  const source = sourceUrl(value.source)
  const polyline = text(value.polyline)
  const distanceM = value.distanceM
  if (
    !title ||
    !source ||
    !polyline ||
    typeof distanceM !== 'number' ||
    !Number.isFinite(distanceM) ||
    !(distanceM > 0)
  )
    return null
  const gain = value.elevationGainM
  const edition = positiveInteger(value.edition)
  return {
    leg: value.leg,
    title,
    source,
    edition: edition !== null && edition >= 1000 && edition <= 9999 ? edition : null,
    mapUrl: sourceUrl(value.mapUrl),
    gpxUrl:
      typeof value.gpxUrl === 'string' &&
      /^\/triathlon\/routes\/[a-z0-9-]+\.gpx$/.test(value.gpxUrl)
        ? value.gpxUrl
        : sourceUrl(value.gpxUrl),
    distanceM,
    raceDistanceM: positiveInteger(value.raceDistanceM),
    laps: positiveInteger(value.laps),
    aidStationsPerLap: positiveInteger(value.aidStationsPerLap),
    elevationGainM: typeof gain === 'number' && gain >= 0 ? gain : null,
    polyline,
    profile: finiteNumbers(value.profile),
    aidM: finiteNumbers(value.aidM).filter(distance => distance > 0 && distance < distanceM),
  }
}

const parseScheduleItem = (value: unknown): TriathlonCalendarScheduleItem | null => {
  if (!isRecord(value) || !clockTime(value.start)) return null
  const activity = text(value.activity)
  if (!activity) return null
  return {
    start: value.start,
    end: clockTime(value.end) ? value.end : null,
    activity,
    location: text(value.location),
    race: value.race === true,
  }
}

const parseSchedule = (value: unknown): TriathlonCalendarSchedule | null => {
  if (!isRecord(value) || !civilDate(value.raceDate) || !Array.isArray(value.days)) return null
  const source = sourceUrl(value.source)
  if (!source) return null
  const days: TriathlonCalendarScheduleDay[] = []
  for (const day of value.days) {
    if (!isRecord(day) || !Number.isInteger(day.offset) || !Array.isArray(day.items)) continue
    const items = day.items
      .map(parseScheduleItem)
      .filter((item): item is TriathlonCalendarScheduleItem => item !== null)
    if (items.length > 0) days.push({ offset: Number(day.offset), items })
  }
  return days.length > 0 ? { source, raceDate: value.raceDate, days } : null
}

export const parseCalendarDetails = (value: unknown): TriathlonCalendarDetails | null => {
  if (!isRecord(value)) return null
  const terrain: Partial<Record<TriathlonCalendarLeg, string>> = {}
  if (isRecord(value.terrain))
    for (const leg of TRIATHLON_CALENDAR_LEGS) {
      const label = text(value.terrain[leg])
      if (label) terrain[leg] = label
    }
  return {
    sources: Array.isArray(value.sources)
      ? value.sources.map(sourceUrl).filter((source): source is string => source !== null)
      : [],
    fetched: civilDate(value.fetched) ? value.fetched : null,
    venue: text(value.venue),
    venueMap: sourceUrl(value.venueMap),
    athleteGuide: sourceUrl(value.athleteGuide),
    terrain,
    airC: celsiusRange(value.airC),
    waterC: celsiusRange(value.waterC),
    course: Array.isArray(value.course)
      ? value.course
          .map(parseCourse)
          .filter((course): course is TriathlonCalendarCourse => course !== null)
      : [],
    schedule: parseSchedule(value.schedule),
    schedulePending: value.schedulePending === true,
    kept: isRecord(value.kept)
      ? Object.fromEntries(
          Object.entries(value.kept).filter((entry): entry is [string, string] =>
            civilDate(entry[1]),
          ),
        )
      : {},
  }
}

export const CALENDAR_DETAILS_FETCHED = civilDate(CALENDAR_DETAILS.fetched)
  ? CALENDAR_DETAILS.fetched
  : null
export const CALENDAR_EVENT_DETAILS = new Map<string, TriathlonCalendarDetails>()
for (const [url, value] of Object.entries(CALENDAR_DETAILS.events as Record<string, unknown>)) {
  const details = parseCalendarDetails(value)
  if (details) CALENDAR_EVENT_DETAILS.set(url, details)
}

/** Decodes a Google encoded polyline (precision 5) into [lat, lng] pairs. */
export const decodeCalendarPolyline = (encoded: string): [number, number][] => {
  const points: [number, number][] = []
  let index = 0
  let lat = 0
  let lng = 0
  const next = (): number | null => {
    let result = 0
    let shift = 0
    let byte: number
    do {
      if (index >= encoded.length) return null
      byte = encoded.charCodeAt(index++) - 63
      result |= (byte & 0x1f) << shift
      shift += 5
    } while (byte >= 0x20)
    return result & 1 ? ~(result >> 1) : result >> 1
  }
  while (index < encoded.length) {
    const deltaLat = next()
    const deltaLng = next()
    if (deltaLat === null || deltaLng === null) break
    lat += deltaLat
    lng += deltaLng
    points.push([lat / 1e5, lng / 1e5])
  }
  return points
}

const shiftDate = (date: string, days: number): string => {
  const shifted = new Date(`${date}T00:00:00.000Z`)
  shifted.setUTCDate(shifted.getUTCDate() + days)
  return shifted.toISOString().slice(0, 10)
}

interface CalendarScheduleDate {
  date: string
  race: boolean
  items: TriathlonCalendarScheduleItem[]
}

/** Places each published schedule day on this event's calendar, keyed by race-day offset. */
export const calendarScheduleDates = (event: TriathlonCalendarEvent): CalendarScheduleDate[] => {
  const schedule = event.details?.schedule
  if (!schedule || !event.date) return []
  return schedule.days.map(day => ({
    date: shiftDate(event.date!, day.offset),
    race: day.offset === 0,
    items: day.items,
  }))
}

/** Year of the edition a schedule was published for; differs from the event year when projected. */
export const calendarScheduleEdition = (schedule: TriathlonCalendarSchedule): number =>
  Number(schedule.raceDate.slice(0, 4))

export const calendarRaceStart = (
  event: TriathlonCalendarEvent,
): TriathlonCalendarScheduleItem | null =>
  event.details?.schedule?.days.find(day => day.offset === 0)?.items.find(item => item.race) ?? null
