import type { Root as HastRoot } from 'hast'
import { toString as hastToString } from 'hast-util-to-string'
import type { StreamEntry } from '../plugins/transformers/stream'
import {
  buildStreamDayPathFromIso,
  formatStreamDate,
  isRestrictedEntry,
  type StreamEntryGroup,
} from './stream'
import { isRecord } from './type-guards'

export interface StreamManifestEntry {
  id: string
  title: string | null
  description: string | null
  text: string
  metadata: unknown
  isoDate: string | null
  displayDate: string | null
  wordCount: number
}

export interface StreamManifestGroup {
  groupId: string
  timestamp: number | null
  isoDate: string | null
  groupSize: number
  path: string | null
  entries: StreamManifestEntry[]
}

export const getStreamEntrySearchText = (entry: StreamEntry): string => {
  const root: HastRoot = { type: 'root', children: entry.content }
  const contentText = hastToString(root)
  const titleText = entry.title ? String(entry.title) : ''
  const descriptionText = entry.description ? String(entry.description) : ''
  return [titleText, descriptionText, contentText]
    .filter(part => part.length > 0)
    .join(' ')
    .trim()
}

export const getStreamEntryWordCount = (entry: StreamEntry): number => {
  const text = getStreamEntrySearchText(entry)
  if (!text) return 0
  return text.split(/\s+/).filter(token => token.length > 0).length
}

const formatIsoAsYMD = (iso?: string | null): string | null => {
  if (!iso) return null
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return null
  const year = date.getUTCFullYear()
  const month = String(date.getUTCMonth() + 1).padStart(2, '0')
  const day = String(date.getUTCDate()).padStart(2, '0')
  return `${year}/${month}/${day}`
}

export const buildStreamManifestGroup = (group: StreamEntryGroup): StreamManifestGroup | null => {
  const isoSource =
    group.isoDate ??
    group.entries.find(entry => entry.date)?.date ??
    (group.timestamp ? new Date(group.timestamp).toISOString() : null)
  const path = buildStreamDayPathFromIso(isoSource)
  const publicEntries = group.entries.filter(entry => !isRestrictedEntry(entry))
  if (publicEntries.length === 0) return null

  return {
    groupId: group.id,
    timestamp: group.timestamp ?? null,
    isoDate: group.isoDate ?? null,
    groupSize: publicEntries.length,
    path,
    entries: publicEntries.map(entry => ({
      id: entry.id,
      title: entry.title ?? null,
      description: entry.description ?? null,
      text: getStreamEntrySearchText(entry),
      metadata: entry.metadata,
      isoDate: entry.date ?? group.isoDate ?? null,
      displayDate:
        formatIsoAsYMD(entry.date ?? group.isoDate ?? isoSource) ??
        formatStreamDate(entry.date ?? group.isoDate) ??
        null,
      wordCount: getStreamEntryWordCount(entry),
    })),
  }
}

const nullableString = (value: unknown): value is string | null =>
  typeof value === 'string' || value === null

const nullableNumber = (value: unknown): value is number | null =>
  (typeof value === 'number' && Number.isFinite(value)) || value === null

const readStreamManifestEntry = (value: unknown): StreamManifestEntry | null => {
  if (!isRecord(value)) return null
  if (typeof value.id !== 'string') return null
  if (!nullableString(value.title)) return null
  if (!nullableString(value.description)) return null
  if (typeof value.text !== 'string') return null
  if (!nullableString(value.isoDate)) return null
  if (!nullableString(value.displayDate)) return null
  if (typeof value.wordCount !== 'number' || !Number.isFinite(value.wordCount)) return null

  return {
    id: value.id,
    title: value.title,
    description: value.description,
    text: value.text,
    metadata: value.metadata,
    isoDate: value.isoDate,
    displayDate: value.displayDate,
    wordCount: value.wordCount,
  }
}

const readStreamManifestGroup = (value: unknown): StreamManifestGroup | null => {
  if (!isRecord(value)) return null
  if (typeof value.groupId !== 'string') return null
  if (!nullableNumber(value.timestamp)) return null
  if (!nullableString(value.isoDate)) return null
  if (typeof value.groupSize !== 'number' || !Number.isFinite(value.groupSize)) return null
  if (!nullableString(value.path)) return null
  if (!Array.isArray(value.entries)) return null

  const entries = value.entries.map(readStreamManifestEntry)
  if (entries.some(entry => entry === null)) return null

  return {
    groupId: value.groupId,
    timestamp: value.timestamp,
    isoDate: value.isoDate,
    groupSize: value.groupSize,
    path: value.path,
    entries: entries.filter(entry => entry !== null),
  }
}

export const parseStreamManifest = (value: string): StreamManifestGroup[] =>
  value
    .split('\n')
    .map(line => line.trim())
    .filter(line => line.length > 0)
    .map((line, index) => {
      const group = readStreamManifestGroup(JSON.parse(line))
      if (!group) throw new Error(`invalid stream manifest group at line ${index + 1}`)
      return group
    })
