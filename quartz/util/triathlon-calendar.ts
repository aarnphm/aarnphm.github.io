import { TRIATHLON_CALENDAR_CATALOG } from './triathlon-calendar-data'
import {
  CALENDAR_DETAILS_FETCHED,
  CALENDAR_EVENT_DETAILS,
  calendarSeries,
  civilDate,
  parseCalendarDetails,
  parseCalendarResults,
  sourceUrl,
  text,
  type TriathlonCalendar,
  type TriathlonCalendarEvent,
} from './triathlon-calendar/details'
import { isRecord, type UnknownRecord } from './type-guards'

export {
  TRIATHLON_CALENDAR_LEGS,
  TRIATHLON_CALENDAR_RESULT_SEGMENTS,
  calendarRaceStart,
  calendarScheduleDates,
  calendarScheduleEdition,
  calendarSeries,
  decodeCalendarPolyline,
  parseCalendarDetails,
  type TriathlonCalendar,
  type TriathlonCalendarCourse,
  type TriathlonCalendarDetails,
  type TriathlonCalendarEvent,
  type TriathlonCalendarLeg,
  type TriathlonCalendarResults,
  type TriathlonCalendarSchedule,
  type TriathlonCalendarScheduleDay,
  type TriathlonCalendarScheduleItem,
  type TriathlonCalendarSeries,
} from './triathlon-calendar/details'
export { serializeTriathlonCalendar } from './triathlon-calendar/ical'

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
