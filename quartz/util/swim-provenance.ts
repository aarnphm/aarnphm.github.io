import { existsSync, readFileSync, statSync } from 'node:fs'
import { resolve } from 'node:path'
import { isRecord } from './type-guards'

export interface SwimProvenance {
  source: 'athlete-estimate'
  note: string
  trainingPeaksWorkoutId: number
}

interface SwimProvenanceEntry extends SwimProvenance {
  activityId: number
  garminActivityId: string
  startDate: string
}

const provenancePath = (contentDirectory: string): string =>
  resolve(contentDirectory, 'triathlon', 'swim-provenance.json')

export function swimProvenanceStamp(contentDirectory: string): string {
  const path = provenancePath(contentDirectory)
  if (!existsSync(path)) return `${path}:missing`
  const info = statSync(path)
  return `${path}:${info.mtimeMs}:${info.size}`
}

const positiveId = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value) && value > 0

function isProvenance(value: unknown): value is SwimProvenanceEntry {
  return (
    isRecord(value) &&
    value.source === 'athlete-estimate' &&
    typeof value.note === 'string' &&
    value.note.trim().length > 0 &&
    positiveId(value.activityId) &&
    positiveId(value.trainingPeaksWorkoutId) &&
    typeof value.garminActivityId === 'string' &&
    /^connect:\d+$/.test(value.garminActivityId) &&
    typeof value.startDate === 'string' &&
    Number.isFinite(Date.parse(value.startDate))
  )
}

export function readSwimProvenance(contentDirectory: string): SwimProvenanceEntry[] {
  const path = provenancePath(contentDirectory)
  if (!existsSync(path)) return []
  const value: unknown = JSON.parse(readFileSync(path, 'utf8'))
  if (!Array.isArray(value) || !value.every(isProvenance))
    throw new Error(`Invalid swim provenance in ${path}`)
  const ids = new Set<number>()
  for (const entry of value) {
    if (ids.has(entry.activityId)) throw new Error(`Duplicate swim provenance ${entry.activityId}`)
    ids.add(entry.activityId)
  }
  return value
}
