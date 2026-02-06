import type { StreamEntry } from '../plugins/transformers/stream'

export interface StreamEntryGroup {
  id: string
  timestamp?: number
  isoDate?: string
  entries: StreamEntry[]
}

/**
 * Group stream entries by their parsed timestamp, preserving original order.
 * Entries without a timestamp are treated as their own singleton groups.
 */
const normalizeToUTCStartOfDay = (date: Date) => {
  const year = date.getUTCFullYear()
  const month = date.getUTCMonth()
  const day = date.getUTCDate()
  const timestamp = Date.UTC(year, month, day)
  const iso = new Date(timestamp).toISOString()
  const key = `day-${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`
  return { key, iso, timestamp }
}

const deriveDayGrouping = (entry: StreamEntry) => {
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
      entries: [entry],
    }

    indexByKey.set(key, groups.length)
    groups.push(group)
  }

  return groups
}
