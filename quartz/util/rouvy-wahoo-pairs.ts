import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import type { StravaRawCache } from '../plugins/stores/strava'
import type { WahooData } from '../plugins/stores/wahoo'
import { parseWahooFitLink, type ActivityTrackingEntry } from '../plugins/stores/tracking'
import { isRouvyActivity, isWahooStravaDuplicate, type RecordingWindow } from './rouvy-wahoo'
import { isRecord } from './type-guards'

export interface RouvyWahooPair extends RecordingWindow {
  canonicalId: number
  canonicalStartDate: string
  workoutId: number
  duplicateStravaId: number | null
  fitPath: string
  sha256: string
}

export const rouvyWahooPairsPath = (contentDirectory = 'content'): string =>
  resolve(contentDirectory, 'triathlon/wahoo/pairs.json')

export function readRouvyWahooPairs(contentDirectory = 'content'): RouvyWahooPair[] {
  let text: string
  try {
    text = readFileSync(rouvyWahooPairsPath(contentDirectory), 'utf8')
  } catch (error) {
    if (isRecord(error) && error.code === 'ENOENT') return []
    throw error
  }
  const value: unknown = JSON.parse(text)
  if (!isRecord(value) || value.version !== 1 || !Array.isArray(value.pairs))
    throw new Error('Invalid ROUVY/Wahoo pairing archive')
  const pairs: RouvyWahooPair[] = []
  for (const pair of value.pairs) {
    if (
      !isRecord(pair) ||
      typeof pair.canonicalId !== 'number' ||
      !Number.isSafeInteger(pair.canonicalId) ||
      typeof pair.workoutId !== 'number' ||
      !Number.isSafeInteger(pair.workoutId) ||
      typeof pair.canonicalStartDate !== 'string' ||
      !Number.isFinite(Date.parse(pair.canonicalStartDate)) ||
      typeof pair.startDate !== 'string' ||
      !Number.isFinite(Date.parse(pair.startDate)) ||
      typeof pair.elapsedTimeS !== 'number' ||
      !Number.isFinite(pair.elapsedTimeS) ||
      pair.elapsedTimeS <= 0 ||
      typeof pair.distanceM !== 'number' ||
      !Number.isFinite(pair.distanceM) ||
      pair.distanceM < 0 ||
      (pair.duplicateStravaId !== null &&
        (typeof pair.duplicateStravaId !== 'number' ||
          !Number.isSafeInteger(pair.duplicateStravaId))) ||
      typeof pair.fitPath !== 'string' ||
      parseWahooFitLink(`[[${pair.fitPath}]]`) !== pair.fitPath ||
      typeof pair.sha256 !== 'string' ||
      !/^[a-f0-9]{64}$/.test(pair.sha256)
    )
      throw new Error('Invalid ROUVY/Wahoo pairing record')
    if (
      pairs.some(
        other => other.canonicalId === pair.canonicalId || other.workoutId === pair.workoutId,
      )
    )
      throw new Error('Ambiguous ROUVY/Wahoo pairing archive')
    pairs.push({
      canonicalId: pair.canonicalId,
      canonicalStartDate: pair.canonicalStartDate,
      workoutId: pair.workoutId,
      duplicateStravaId: pair.duplicateStravaId,
      fitPath: pair.fitPath,
      sha256: pair.sha256,
      startDate: pair.startDate,
      elapsedTimeS: pair.elapsedTimeS,
      distanceM: pair.distanceM,
    })
  }
  return pairs
}

export function pairedActivityTracking(
  pairs: readonly RouvyWahooPair[],
  entries: readonly ActivityTrackingEntry[],
  strava: StravaRawCache | null,
): ActivityTrackingEntry[] {
  const result = new Map(entries.map(entry => [entry.activityId, entry]))
  for (const pair of pairs) {
    const canonical = strava?.activities[String(pair.canonicalId)]
    if (
      !canonical ||
      !isRouvyActivity(canonical) ||
      Date.parse(canonical.startDate) !== Date.parse(pair.canonicalStartDate)
    )
      continue
    const existing = result.get(pair.canonicalId)
    result.set(pair.canonicalId, {
      activityId: pair.canonicalId,
      garminActivityId: null,
      virtual: true,
      ...existing,
      wahooFitPath: existing?.wahooFitPath ?? pair.fitPath,
      power: existing?.power ?? 'wahoo',
    })
  }
  return [...result.values()]
}

export function withoutPairedWahooDuplicates(
  strava: StravaRawCache | null,
  pairs: readonly RouvyWahooPair[],
  entries: readonly ActivityTrackingEntry[],
  wahoo: WahooData | null,
): StravaRawCache | null {
  if (!strava) return null
  const activities = { ...strava.activities }
  for (const pair of pairs) {
    const canonical = activities[String(pair.canonicalId)]
    const entry = entries.find(value => value.activityId === pair.canonicalId)
    const route = strava.streams?.[String(pair.canonicalId)]
    const source = Object.values(wahoo?.activities ?? {}).find(
      value => 'path' in value.sourceFile && value.sourceFile.path === pair.fitPath,
    )
    if (
      !canonical ||
      !isRouvyActivity(canonical) ||
      !route?.time ||
      route.time.length < 2 ||
      route.latlng.length !== route.time.length ||
      Date.parse(canonical.startDate) !== Date.parse(pair.canonicalStartDate) ||
      entry?.wahooFitPath !== pair.fitPath ||
      source?.sourceFile.sha256 !== pair.sha256
    )
      continue
    const duplicates = Object.values(activities).filter(
      activity => activity.id !== pair.canonicalId && isWahooStravaDuplicate(activity, pair),
    )
    if (duplicates.length === 1 && duplicates[0].id === pair.duplicateStravaId)
      delete activities[String(pair.duplicateStravaId)]
  }
  return { ...strava, activities }
}
