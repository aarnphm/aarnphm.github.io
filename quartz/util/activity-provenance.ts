import { existsSync, readFileSync, statSync } from 'node:fs'
import { resolve } from 'node:path'
import { ACTIVITY_KINDS, type ActivityKind } from '../plugins/stores/strava/raw-and-kinds'
import { stravaId } from './strava-id'
import { isRecord } from './type-guards'

export interface ActivityProvenance {
  source: 'athlete-estimate'
  note: string
  trainingPeaksWorkoutId?: number
}

// sport, startDate and the optional Garmin id identify the activity; an entry that no longer
// matches its detail is skipped instead of stamping a note on a different recording.
interface ActivityProvenanceEntry extends ActivityProvenance {
  activityId: string
  sport: ActivityKind
  startDate: string
  garminActivityId?: string
}

const provenancePath = (contentDirectory: string): string =>
  resolve(contentDirectory, 'triathlon', 'provenance.json')

export function activityProvenanceStamp(contentDirectory: string): string {
  const path = provenancePath(contentDirectory)
  if (!existsSync(path)) return `${path}:missing`
  const info = statSync(path)
  return `${path}:${info.mtimeMs}:${info.size}`
}

const positiveId = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value) && value > 0

function isProvenance(value: unknown): value is ActivityProvenanceEntry {
  return (
    isRecord(value) &&
    value.source === 'athlete-estimate' &&
    typeof value.note === 'string' &&
    value.note.trim().length > 0 &&
    typeof value.activityId === 'string' &&
    stravaId(value.activityId) !== null &&
    ACTIVITY_KINDS.includes(value.sport as ActivityKind) &&
    (value.trainingPeaksWorkoutId === undefined || positiveId(value.trainingPeaksWorkoutId)) &&
    (value.garminActivityId === undefined ||
      (typeof value.garminActivityId === 'string' &&
        /^connect:\d+$/.test(value.garminActivityId))) &&
    typeof value.startDate === 'string' &&
    Number.isFinite(Date.parse(value.startDate))
  )
}

export function readActivityProvenance(contentDirectory: string): ActivityProvenanceEntry[] {
  const path = provenancePath(contentDirectory)
  if (!existsSync(path)) return []
  const value: unknown = JSON.parse(readFileSync(path, 'utf8'))
  if (Array.isArray(value))
    for (const entry of value) if (isRecord(entry)) entry.activityId = stravaId(entry.activityId)
  if (!Array.isArray(value) || !value.every(isProvenance))
    throw new Error(`Invalid activity provenance in ${path}`)
  const ids = new Set<string>()
  for (const entry of value) {
    if (ids.has(entry.activityId))
      throw new Error(`Duplicate activity provenance ${entry.activityId}`)
    ids.add(entry.activityId)
  }
  return value
}
