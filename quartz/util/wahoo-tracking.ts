import { readFileSync, statSync } from 'node:fs'
import { resolve } from 'node:path'
import type { ActivityTrackingEntry } from '../plugins/stores/tracking'
import type { ProcessedContent } from '../plugins/vfile'
import type { ChangeEvent } from '../types/plugin'
import type { FilePath } from './path'
import { normalizeKind, type StravaRawCache } from '../plugins/stores/strava'
import { parseWahooFitLink } from '../plugins/stores/tracking'
import {
  matchWahooActivity,
  normalizeWahooSport,
  type WahooData,
  type WahooLocalFitActivity,
} from '../plugins/stores/wahoo'
import { decodeWahooFit, wahooFitSha256, type WahooFitData } from './wahoo-fit'

const fitMemo = new Map<
  string,
  { stamp: string; data: WahooFitData; sha256: string; size: number }
>()

export function trackedWahooChangeEvents(
  content: ProcessedContent[],
  events: ChangeEvent[],
  contentDirectory: string,
): ChangeEvent[] {
  const touched = new Set(events.map(event => resolve(event.path)))
  const owners = content.filter(([, file]) =>
    file.data.tracking?.activities.some(
      entry =>
        entry.wahooFitPath &&
        (touched.has(resolve(contentDirectory, entry.wahooFitPath)) ||
          touched.has(resolve(file.data.filePath!))),
    ),
  )
  for (const event of events)
    if (
      event.file &&
      event.previousFile?.data.tracking?.activities.some(entry => entry.wahooFitPath)
    ) {
      const owner = content.find(([, file]) => file.data.slug === event.file!.data.slug)
      if (owner) owners.push(owner)
    }
  if (owners.length === 0) return events
  const slugs = new Set<string>(owners.map(([, file]) => file.data.slug!))
  const affected = content.filter(
    ([, file]) => slugs.has(file.data.slug!) || file.data.links?.some(link => slugs.has(link)),
  )
  return [
    ...events,
    ...affected.flatMap(([, file]): ChangeEvent[] =>
      touched.has(resolve(file.data.filePath!))
        ? []
        : [{ type: 'change', path: file.data.filePath as FilePath, file }],
    ),
  ]
}

function fitFilePath(path: string, contentDirectory: string): string {
  if (parseWahooFitLink(`[[${path}]]`) !== path)
    throw new Error(`Invalid tracked Wahoo FIT: ${path}`)
  return resolve(contentDirectory, path)
}

export function wahooTrackingStamp(
  entries: readonly ActivityTrackingEntry[],
  contentDirectory = 'content',
): string {
  return entries
    .flatMap(entry => {
      if (!entry.wahooFitPath) return []
      const path = fitFilePath(entry.wahooFitPath, contentDirectory)
      try {
        const info = statSync(path)
        return [`${path}:${info.mtimeMs}:${info.size}`]
      } catch {
        return [`${path}:missing`]
      }
    })
    .join('|')
}

export function loadTrackedWahooFits(
  cache: WahooData | null,
  strava: StravaRawCache | null,
  entries: readonly ActivityTrackingEntry[],
  contentDirectory = 'content',
): WahooData | null {
  const linked = entries.filter(
    entry => entry.wahooFitPath && strava?.activities[String(entry.activityId)],
  )
  if (linked.length === 0) return cache
  const result: WahooData = {
    lastSync: cache?.lastSync ?? 0,
    activities: { ...cache?.activities },
    streams: { ...cache?.streams },
    gearShifts: { ...cache?.gearShifts },
    cyclingDynamics: { ...cache?.cyclingDynamics },
    summitSegments: { ...cache?.summitSegments },
  }
  for (const entry of linked) {
    const fitPath = entry.wahooFitPath!
    const path = fitFilePath(fitPath, contentDirectory)
    const info = statSync(path)
    const stamp = `${info.mtimeMs}:${info.size}`
    let decoded = fitMemo.get(path)
    if (decoded?.stamp !== stamp) {
      const bytes = readFileSync(path)
      decoded = {
        stamp,
        data: decodeWahooFit(bytes),
        sha256: wahooFitSha256(bytes),
        size: bytes.length,
      }
      fitMemo.set(path, decoded)
    }
    const fit = decoded.data
    const sport = normalizeWahooSport(-1, fit.sport)
    if (sport !== 'bike') throw new Error(`Tracked Wahoo FIT must be cycling: ${fitPath}`)
    const original = strava!.activities[String(entry.activityId)]
    if (normalizeKind(original.sportType) !== 'bike')
      throw new Error(`Tracked Wahoo FIT requires a cycling activity: ${entry.activityId}`)
    // Cloud exports can re-encode the same recording; compare its session identity too.
    const duplicate = Object.values(result.activities).find(
      activity =>
        activity.sourceFile.sha256 === decoded.sha256 ||
        (activity.sourceDevice === fit.sourceDevice &&
          Date.parse(activity.startDate) === Date.parse(fit.startDate) &&
          activity.distanceM != null &&
          activity.distanceM === fit.distanceM &&
          activity.movingTimeS === fit.movingTimeS &&
          activity.elapsedTimeS === fit.elapsedTimeS &&
          activity.metrics.avgPower === fit.metrics.avgPower &&
          activity.metrics.totalWorkKJ === fit.metrics.totalWorkKJ),
    )
    const id = duplicate?.id ?? `wahoo-fit:${fitPath}`
    const localOffsetMs = Date.parse(original.startDateLocal) - Date.parse(original.startDate)
    const activity: WahooLocalFitActivity = {
      id,
      name: duplicate?.name ?? null,
      sport,
      startDate: fit.startDate,
      startDateLocal: new Date(Date.parse(fit.startDate) + localOffsetMs).toISOString(),
      distanceM: fit.distanceM,
      movingTimeS: fit.movingTimeS,
      elapsedTimeS: fit.elapsedTimeS,
      sourceDevice: fit.sourceDevice,
      sourceFile: {
        path: fitPath,
        sha256: decoded.sha256,
        byteLength: decoded.size,
        profileVersion: fit.profileVersion,
      },
      sweatLoss: fit.sweatLoss,
      metrics: fit.metrics,
    }
    result.activities[id] = activity
    result.streams[id] = fit.streams
    result.gearShifts[id] = fit.gearShifts
    result.cyclingDynamics[id] = fit.cyclingDynamics
    result.summitSegments[id] = fit.summitSegments
    if (!matchWahooActivity(original, sport, result, fitPath))
      throw new Error(`Tracked Wahoo FIT does not match activity ${entry.activityId}: ${fitPath}`)
  }
  return result
}
