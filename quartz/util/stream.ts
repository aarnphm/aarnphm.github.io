import type { StreamEntry } from '../plugins/transformers/stream'

export interface StreamDateParts {
  year: number
  month: number
  day: number
  yearText: string
  monthText: string
  dayText: string
  key: string
  iso: string
  timestamp: number
}

export interface StreamEntryGroup {
  id: string
  timestamp?: number
  isoDate?: string
  dateParts?: StreamDateParts
  entries: StreamEntry[]
}

export interface StreamMonthGroup {
  id: string
  year: number
  month: number
  yearText: string
  monthText: string
  label: string
  path: string
  entries: StreamEntry[]
}

export interface StreamYearGroup {
  id: string
  year: number
  yearText: string
  path: string
  entries: StreamEntry[]
  months: StreamMonthGroup[]
}

export const truthyStreamFlag = (value: unknown): boolean => {
  if (typeof value === 'boolean') return value
  if (typeof value === 'number') return value !== 0
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase()
    return (
      normalized === 'true' || normalized === '1' || normalized === 'yes' || normalized === 'on'
    )
  }
  return false
}

export const isProtectedEntry = (entry: StreamEntry): boolean =>
  truthyStreamFlag(entry.metadata?.protected)

export const isPrivateEntry = (entry: StreamEntry): boolean =>
  truthyStreamFlag(entry.metadata?.private)

export const isDraftEntry = (entry: StreamEntry): boolean => truthyStreamFlag(entry.metadata?.draft)

export const isRestrictedEntry = (entry: StreamEntry): boolean =>
  isProtectedEntry(entry) || isPrivateEntry(entry)

export const formatStreamDate = (isoDate: string | undefined): string | null => {
  if (!isoDate) return null

  const date = new Date(isoDate)
  if (Number.isNaN(date.getTime())) return null

  const formatter = new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    hour12: false,
    timeZone: 'America/Los_Angeles',
    timeZoneName: 'shortOffset',
  })

  return formatter.format(date)
}

const pad2 = (value: number): string => String(value).padStart(2, '0')

export const buildStreamOnPath = (): string => '/stream/on'

export const buildStreamYearPath = (year: number | string): string => `/stream/on/${year}`

export const buildStreamMonthPath = (year: number | string, month: number | string): string =>
  `${buildStreamYearPath(year)}/${typeof month === 'number' ? pad2(month) : month}`

export const buildStreamDayPath = (
  year: number | string,
  month: number | string,
  day: number | string,
): string => `${buildStreamMonthPath(year, month)}/${typeof day === 'number' ? pad2(day) : day}`

export const buildStreamEntryPath = (dayPath: string, entryId: string): string =>
  `${dayPath}?entry=${encodeURIComponent(entryId)}`

const normalizeToUTCStartOfDay = (date: Date) => {
  const year = date.getUTCFullYear()
  const month = date.getUTCMonth()
  const day = date.getUTCDate()
  const timestamp = Date.UTC(year, month, day)
  const iso = new Date(timestamp).toISOString()
  const monthText = pad2(month + 1)
  const dayText = pad2(day)
  const yearText = String(year)
  const key = `day-${yearText}-${monthText}-${dayText}`
  return { year, month: month + 1, day, yearText, monthText, dayText, key, iso, timestamp }
}

const deriveDayGrouping = (entry: StreamEntry): StreamDateParts | null => {
  if (entry.date) {
    const date = new Date(entry.date)
    if (!Number.isNaN(date.getTime())) {
      return normalizeToUTCStartOfDay(date)
    }
  }

  if (typeof entry.timestamp === 'number') {
    const date = new Date(entry.timestamp)
    if (!Number.isNaN(date.getTime())) {
      return normalizeToUTCStartOfDay(date)
    }
  }

  return null
}

export const getStreamEntryDateParts = (entry: StreamEntry): StreamDateParts | null =>
  deriveDayGrouping(entry)

export const getStreamDatePartsFromIso = (
  isoDate: string | undefined | null,
): StreamDateParts | null => {
  if (!isoDate) return null
  const date = new Date(isoDate)
  if (Number.isNaN(date.getTime())) return null
  return normalizeToUTCStartOfDay(date)
}

export const buildStreamDayPathFromIso = (isoDate: string | undefined | null): string | null => {
  const parts = getStreamDatePartsFromIso(isoDate)
  if (!parts) return null
  return buildStreamDayPath(parts.yearText, parts.monthText, parts.dayText)
}

export const buildStreamEntryPathFromIso = (
  isoDate: string | undefined | null,
  entryId: string,
): string | null => {
  const dayPath = buildStreamDayPathFromIso(isoDate)
  if (!dayPath) return null
  return buildStreamEntryPath(dayPath, entryId)
}

export const formatStreamMonthLabel = (year: number, month: number): string => {
  const date = new Date(Date.UTC(year, month - 1, 1))
  return new Intl.DateTimeFormat('en-US', { month: 'long', year: 'numeric', timeZone: 'UTC' })
    .format(date)
    .toLowerCase()
}

export function groupStreamEntries(entries: StreamEntry[]): StreamEntryGroup[] {
  const groups: StreamEntryGroup[] = []
  const indexByKey = new Map<string, number>()
  let fallbackCounter = 0

  for (const entry of entries) {
    const grouping = deriveDayGrouping(entry)
    const key = grouping ? grouping.key : `entry-${fallbackCounter++}`
    const existingIndex = indexByKey.get(key)

    if (existingIndex !== undefined) {
      const group = groups[existingIndex]
      group.entries.push(entry)
      if (grouping) {
        if (!group.isoDate) {
          group.isoDate = grouping.iso
        }
        if (!group.timestamp) {
          group.timestamp = grouping.timestamp
        }
        if (!group.dateParts) {
          group.dateParts = grouping
        }
      } else if (!group.isoDate && entry.date) {
        group.isoDate = entry.date
      }
      continue
    }

    const group: StreamEntryGroup = {
      id: key,
      timestamp:
        grouping?.timestamp ?? (typeof entry.timestamp === 'number' ? entry.timestamp : undefined),
      isoDate: grouping?.iso ?? entry.date,
      dateParts: grouping ?? undefined,
      entries: [entry],
    }

    indexByKey.set(key, groups.length)
    groups.push(group)
  }

  return groups
}

export function groupStreamEntriesByYear(entries: StreamEntry[]): StreamYearGroup[] {
  const years: StreamYearGroup[] = []
  const yearIndex = new Map<string, StreamYearGroup>()
  const monthIndex = new Map<string, StreamMonthGroup>()

  for (const entry of entries) {
    const parts = deriveDayGrouping(entry)
    if (!parts) continue

    let yearGroup = yearIndex.get(parts.yearText)
    if (!yearGroup) {
      yearGroup = {
        id: `year-${parts.yearText}`,
        year: parts.year,
        yearText: parts.yearText,
        path: buildStreamYearPath(parts.yearText),
        entries: [],
        months: [],
      }
      yearIndex.set(parts.yearText, yearGroup)
      years.push(yearGroup)
    }

    const monthKey = `${parts.yearText}-${parts.monthText}`
    let monthGroup = monthIndex.get(monthKey)
    if (!monthGroup) {
      monthGroup = {
        id: `month-${monthKey}`,
        year: parts.year,
        month: parts.month,
        yearText: parts.yearText,
        monthText: parts.monthText,
        label: formatStreamMonthLabel(parts.year, parts.month),
        path: buildStreamMonthPath(parts.yearText, parts.monthText),
        entries: [],
      }
      monthIndex.set(monthKey, monthGroup)
      yearGroup.months.push(monthGroup)
    }

    yearGroup.entries.push(entry)
    monthGroup.entries.push(entry)
  }

  return years
}
