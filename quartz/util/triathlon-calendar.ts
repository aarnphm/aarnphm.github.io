import { TRIATHLON_CALENDAR_CATALOG } from './triathlon-calendar-data'
import CALENDAR_DETAILS from './triathlon-calendar-details.json'
import { isRecord, type UnknownRecord } from './type-guards'

export type TriathlonCalendarSeries =
  | 'ironman'
  | 't100'
  | 'supertri'
  | 'hyrox'
  | 'tcs'
  | 'b4bh'
  | 'other'
export type TriathlonCalendarLeg = 'swim' | 'bike' | 'run'
export type TriathlonCalendarResultSegment = TriathlonCalendarLeg | 'T1' | 'T2' | 'overall'

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

const civilDate = (value: unknown): value is string => {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false
  if (value.startsWith('0000')) return false
  const date = new Date(`${value}T00:00:00.000Z`)
  return Number.isFinite(date.valueOf()) && date.toISOString().slice(0, 10) === value
}

const text = (value: unknown): string | null =>
  typeof value === 'string' && value.trim() ? value.trim() : null

const resultDuration = (value: unknown): string | null =>
  typeof value === 'string' && /^\d+:[0-5]\d:[0-5]\d$/.test(value) ? value : null

const parseCalendarResults = (value: unknown): TriathlonCalendarResults | null => {
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

const sourceUrl = (value: unknown): string | null => {
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

const CALENDAR_DETAILS_FETCHED = civilDate(CALENDAR_DETAILS.fetched)
  ? CALENDAR_DETAILS.fetched
  : null
const CALENDAR_EVENT_DETAILS = new Map<string, TriathlonCalendarDetails>()
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

export interface CalendarScheduleDate {
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

const parseEvent = (value: unknown, year: number): TriathlonCalendarEvent | null => {
  if (!isRecord(value)) return null
  const id = text(value.id)
  const name = text(value.name)
  const location = text(value.location)
  const format = text(value.format)
  const url = sourceUrl(value.url)
  const kind = value.kind
  const date = value.date ?? null
  const endDate = value.endDate ?? null
  const note = value.note == null ? null : text(value.note)
  if (
    !id ||
    !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(id) ||
    !name ||
    !location ||
    !format ||
    !url ||
    (kind !== 'triathlon' && kind !== 'hyrox' && kind !== 'running' && kind !== 'cycling') ||
    (date !== null && (!civilDate(date) || Number(date.slice(0, 4)) !== year)) ||
    (endDate !== null && (!civilDate(endDate) || date === null || endDate < date)) ||
    (value.note != null && note === null)
  ) {
    return null
  }
  const results = parseCalendarResults(value.results)
  return {
    id,
    name,
    location,
    kind,
    series: calendarSeries(url),
    format,
    url,
    date,
    endDate,
    note,
    details: parseCalendarDetails(value.details) ?? CALENDAR_EVENT_DETAILS.get(url) ?? null,
    results,
    participated: value.participated === true || results !== null,
  }
}

const calendarError = (message: string): Error =>
  new Error(
    `Triathlon calendar: ${message}. Check the calendar map and verified metadata in quartz/util/triathlon-calendar-data.ts.`,
  )

const parseSeason = (year: number, entries: UnknownRecord, results: unknown): TriathlonCalendar => {
  const season = TRIATHLON_CALENDAR_CATALOG[year]
  if (Object.keys(entries).length === 0) {
    return {
      year,
      checked: civilDate(season?.checked) ? season.checked : null,
      fetched: null,
      events: [],
    }
  }
  if (!season) throw calendarError(`year ${year} has no verified metadata catalog`)
  const { checked, sources } = season
  if (!civilDate(checked))
    throw calendarError(`year ${year} has an invalid metadata verification date`)
  const events: TriathlonCalendarEvent[] = []
  const selectedSources = new Map<string, string>()
  for (const [key, rawUrl] of Object.entries(entries)) {
    if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(key))
      throw calendarError(`event key ${JSON.stringify(key)} must use lowercase words and hyphens`)
    const url = sourceUrl(rawUrl)
    if (!url) throw calendarError(`event "${key}" must contain a valid HTTP(S) source URL`)
    const metadata = sources[url]
    if (!metadata) throw calendarError(`event "${key}" has an unknown official source: ${url}`)
    const previous = selectedSources.get(url)
    if (previous)
      throw calendarError(`events "${previous}" and "${key}" repeat the official source: ${url}`)
    const record = metadata.recordKey && isRecord(results) ? results[metadata.recordKey] : null
    const event = parseEvent(
      {
        ...metadata,
        id: `${key}-${year}`,
        url,
        results: record,
        participated: isRecord(record) && record.participated === true,
      },
      year,
    )
    if (!event) throw calendarError(`event "${key}" has invalid verified metadata for ${url}`)
    selectedSources.set(url, key)
    events.push(event)
  }
  events.sort((left, right) => {
    if (left.date === null) return right.date === null ? left.id.localeCompare(right.id) : 1
    if (right.date === null) return -1
    return left.date.localeCompare(right.date) || left.id.localeCompare(right.id)
  })
  return { year, checked, fetched: CALENDAR_DETAILS_FETCHED, events }
}

export const parseTriathlonCalendars = (value: unknown, results?: unknown): TriathlonCalendar[] => {
  if (value == null) return []
  if (!isRecord(value)) throw calendarError('calendar must map years to event URL maps')
  const calendars: TriathlonCalendar[] = []
  for (const [yearKey, entries] of Object.entries(value)) {
    if (!/^\d{4}$/.test(yearKey) || Number(yearKey) === 0)
      throw calendarError(
        `year key ${JSON.stringify(yearKey)} must be a valid four-digit calendar year`,
      )
    if (!isRecord(entries))
      throw calendarError(`year ${yearKey} must contain a map of event keys to official URLs`)
    calendars.push(parseSeason(Number(yearKey), entries, results))
  }
  return calendars.sort((left, right) => left.year - right.year)
}

export const parseTriathlonCalendar = (
  value: unknown,
  selectedYear?: number,
  results?: unknown,
): TriathlonCalendar | null => {
  if (
    selectedYear !== undefined &&
    (!Number.isInteger(selectedYear) || selectedYear < 1 || selectedYear > 9999)
  ) {
    throw calendarError(`selected year ${selectedYear} must be a valid calendar year`)
  }
  const calendars = parseTriathlonCalendars(value, results)
  if (selectedYear === undefined) return calendars.at(-1) ?? null
  const calendar = calendars.find(season => season.year === selectedYear)
  if (!calendar) throw calendarError(`selected year ${selectedYear} is not configured`)
  return calendar
}

const icalText = (value: string): string =>
  value
    .replaceAll('\\', '\\\\')
    .replace(/\r\n|\r|\n/g, '\\n')
    .replaceAll(';', '\\;')
    .replaceAll(',', '\\,')

const foldLine = (line: string): string => {
  const encoder = new TextEncoder()
  let folded = ''
  let bytes = 0
  for (const character of line) {
    const size = encoder.encode(character).length
    if (bytes + size > 75) {
      folded += '\r\n '
      bytes = 1
    }
    folded += character
    bytes += size
  }
  return folded
}

const icalDate = (date: string): string => date.replaceAll('-', '')

const exclusiveEndDate = (date: string): string => {
  const end = new Date(`${date}T00:00:00.000Z`)
  end.setUTCDate(end.getUTCDate() + 1)
  return icalDate(end.toISOString().slice(0, 10))
}

export const serializeTriathlonCalendar = (calendar: TriathlonCalendar | null): string => {
  const lines = [
    'BEGIN:VCALENDAR',
    'VERSION:2.0',
    'PRODID:-//aarnphm.xyz//Race calendar//EN',
    'CALSCALE:GREGORIAN',
    `X-WR-CALNAME:${calendar ? `${calendar.year} race calendar` : 'Race calendar'}`,
  ]
  for (const event of calendar?.events ?? []) {
    if (event.date === null) continue
    const description = [
      event.format,
      event.participated ? 'Participated' : null,
      event.note,
      `Official event: ${event.url}`,
      calendar?.checked ? `Dates checked: ${calendar.checked}` : null,
    ]
      .filter(Boolean)
      .join('\n')
    lines.push(
      'BEGIN:VEVENT',
      `UID:${event.id}@aarnphm.xyz`,
      `DTSTAMP:${icalDate(calendar?.checked ?? event.date)}T000000Z`,
      `DTSTART;VALUE=DATE:${icalDate(event.date)}`,
      `DTEND;VALUE=DATE:${exclusiveEndDate(event.endDate ?? event.date)}`,
      `SUMMARY:${icalText(event.name)}`,
      `LOCATION:${icalText(event.location)}`,
      `DESCRIPTION:${icalText(description)}`,
      `URL:${event.url}`,
      'END:VEVENT',
    )
  }
  lines.push('END:VCALENDAR')
  return `${lines.map(foldLine).join('\r\n')}\r\n`
}
