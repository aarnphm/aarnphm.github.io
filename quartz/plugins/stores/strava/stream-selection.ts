import type { ActivityTrackingEntry } from '../tracking'
import {
  matchGarminActivity,
  type GarminActivityMatch,
  type GarminCache,
  type GarminStreams,
} from '../garmin'
import { matchWahooActivity, type WahooActivityMatch, type WahooData } from '../wahoo'
import {
  normalizeKind,
  POWER_SUMMARY_FIELDS,
  WAHOO_SUMMARY_FIELDS,
  WAHOO_TELEMETRY_MAX_DISTANCE_S,
  type ActivityHeartRate,
  type ActivityKind,
  type RawStravaActivity,
  type RawStravaActivityDetail,
  type StravaRawCache,
  type StravaStreams,
} from './raw-and-kinds'
import {
  avgPos,
  hasPositive,
  maxPos,
  nearestElapsedIndex,
  positiveCount,
  roundPos,
  timedMetricAt,
  timedNullableMetricAt,
  timedStreamAlignment,
} from './stream-primitives'

export function isWahooPowerSummary(field: (typeof WAHOO_SUMMARY_FIELDS)[number][0]): boolean {
  return (
    field === 'averageWatts' ||
    field === 'weightedAverageWatts' ||
    field === 'maxWatts' ||
    field === 'kilojoules'
  )
}

export function withWahooSummary(
  activity: RawStravaActivity,
  detail: RawStravaActivityDetail | undefined,
  match: WahooActivityMatch,
  power?: ActivityTrackingEntry['power'],
): RawStravaActivity {
  const projected = { ...activity }
  for (const [field, metric] of WAHOO_SUMMARY_FIELDS) {
    if (isWahooPowerSummary(field) && (power === 'strava' || power === 'garmin')) continue
    if (field === 'calories') projected.calories ??= detail?.calories ?? undefined
    const value = match.activity.metrics[metric]
    if (
      (projected[field] == null || (power === 'wahoo' && isWahooPowerSummary(field))) &&
      value != null &&
      Number.isFinite(value)
    )
      projected[field] = value
  }
  if (
    (power == null || power === 'wahoo') &&
    (power === 'wahoo' || activity.averageWatts == null || activity.deviceWatts == null) &&
    match.activity.metrics.avgPower != null
  )
    projected.deviceWatts = true
  return projected
}

export function mergeWahooTelemetry(
  activity: RawStravaActivity,
  original: StravaStreams | GarminStreams | undefined,
  match: WahooActivityMatch,
  wahoo: WahooData,
  power?: ActivityTrackingEntry['power'],
): StravaStreams | undefined {
  const time = original?.time
  const stream = wahoo.streams[match.activity.id]
  if (!original || !time?.length || !stream) return original
  const offsetS = (Date.parse(match.activity.startDate) - Date.parse(activity.startDate)) / 1_000
  const merged: StravaStreams = { ...original }
  for (const key of ['watts', 'heartrate', 'cadence'] as const) {
    if (key === 'watts' && (power === 'strava' || power === 'garmin')) continue
    const values = stream[key]
    if (values.length !== stream.time.length) continue
    const samples = stream.time.map((elapsedS, index) => ({
      elapsedS: elapsedS + offsetS,
      value: values[index],
    }))
    merged[key] = time.map(
      (elapsedS, index) =>
        timedNullableMetricAt(samples, elapsedS, WAHOO_TELEMETRY_MAX_DISTANCE_S) ??
        original[key]?.[index] ??
        0,
    )
  }
  return merged
}

function streamQuality(streams: StravaStreams | GarminStreams | undefined): number {
  if (!streams) return 0
  const channels =
    (streams.latlng.length >= 2 ? 1 : 0) +
    (streams.altitude.length > 0 ? 1 : 0) +
    (streams.distance.length > 0 ? 1 : 0) +
    (hasPositive(streams.heartrate) ? 1 : 0) +
    (hasPositive(streams.cadence) ? 1 : 0) +
    (hasPositive(streams.watts) ? 1 : 0)
  return (
    channels * 10_000 +
    streams.latlng.length +
    streams.altitude.length +
    streams.distance.length +
    positiveCount(streams.heartrate) +
    positiveCount(streams.cadence) +
    positiveCount(streams.watts)
  )
}

function hasCompleteTimedRoute(streams: StravaStreams | undefined): boolean {
  const length = streams?.time?.length ?? 0
  return (
    length >= 2 &&
    streams?.latlng.length === length &&
    streams.distance.length === length &&
    streams.altitude.length === length
  )
}

export function selectStreams(
  strava: StravaStreams | undefined,
  match: GarminActivityMatch | null,
  garmin: GarminCache | null,
  activity?: RawStravaActivity,
): StravaStreams | GarminStreams | undefined {
  const fromGarmin = match ? garmin?.streams?.[match.activity.id] : undefined
  if (activity && match && fromGarmin && activity.sportType.startsWith('Virtual')) {
    if (hasCompleteTimedRoute(strava)) return strava
    const alignment = timedStreamAlignment(fromGarmin)
    if (!alignment || fromGarmin.latlng.length !== alignment.time.length) return strava
    const offsetS = (Date.parse(match.activity.startDate) - Date.parse(activity.startDate)) / 1_000
    const endS = activity.elapsedTime || activity.movingTime
    const indices = alignment.time.flatMap((time, index) => {
      const elapsedS = time + offsetS
      return elapsedS >= 0 && elapsedS <= endS ? [index] : []
    })
    if (indices.length < 2) return strava
    const time = indices.map(index => alignment.time[index] + offsetS)
    const telemetry = (key: 'watts' | 'heartrate' | 'cadence'): number[] => {
      const values = strava?.[key]
      const samples =
        values && strava?.time?.length === values.length
          ? strava.time.map((elapsedS, index) => ({ elapsedS, value: values[index] }))
          : []
      return indices.map(
        (index, position) =>
          timedNullableMetricAt(samples, time[position], 2.5) ?? fromGarmin[key]?.[index] ?? 0,
      )
    }
    return {
      time,
      latlng: indices.map(index => fromGarmin.latlng[index]),
      altitude: indices.map(index => fromGarmin.altitude[index]),
      distance: indices.map(index => fromGarmin.distance[index]),
      watts: telemetry('watts'),
      heartrate: telemetry('heartrate'),
      cadence: telemetry('cadence'),
    }
  }
  if (activity && normalizeKind(activity.sportType) === 'bike') {
    if (!fromGarmin || !match) return strava
    const offsetS = (Date.parse(match.activity.startDate) - Date.parse(activity.startDate)) / 1_000
    const garminTime = fromGarmin.time
    const stravaTime = strava?.time
    if (!Number.isFinite(offsetS) || !garminTime?.length) return strava
    if (!strava || !stravaTime?.length) {
      const endS = activity.elapsedTime || activity.movingTime
      const indices = garminTime.flatMap((time, index) =>
        time + offsetS >= 0 && time + offsetS <= endS ? [index] : [],
      )
      const aligned: StravaStreams = {
        time: indices.map(index => garminTime[index] + offsetS),
        latlng:
          fromGarmin.latlng.length === garminTime.length
            ? indices.map(index => fromGarmin.latlng[index])
            : [],
        altitude:
          fromGarmin.altitude.length === garminTime.length
            ? indices.map(index => fromGarmin.altitude[index])
            : [],
        distance:
          fromGarmin.distance.length === garminTime.length
            ? indices.map(index => fromGarmin.distance[index])
            : [],
      }
      for (const key of ['watts', 'heartrate', 'cadence'] as const) {
        const values = fromGarmin[key]
        if (values?.length === garminTime.length) aligned[key] = indices.map(index => values[index])
      }
      return aligned
    }
    const merged: StravaStreams = { ...strava }
    for (const key of ['distance', 'altitude', 'watts', 'heartrate', 'cadence'] as const) {
      const primary = strava[key]
      const fallback = fromGarmin[key]
      if (primary?.length === stravaTime.length && primary.every(Number.isFinite)) continue
      if (fallback?.length !== garminTime.length) continue
      const samples = garminTime.map((time, index) => ({
        elapsedS: time + offsetS,
        value: Number.isFinite(fallback[index]) ? fallback[index] : null,
      }))
      merged[key] = stravaTime.map((time, index) =>
        primary?.length === stravaTime.length && Number.isFinite(primary[index])
          ? primary[index]
          : (timedNullableMetricAt(samples, time, WAHOO_TELEMETRY_MAX_DISTANCE_S) ?? 0),
      )
    }
    if (
      strava.latlng.length !== stravaTime.length &&
      fromGarmin.latlng.length === garminTime.length
    ) {
      const time = garminTime.map(value => value + offsetS)
      const positions = stravaTime.map(value => {
        const index = nearestElapsedIndex(time, value)
        return Math.abs(time[index] - value) <= WAHOO_TELEMETRY_MAX_DISTANCE_S
          ? fromGarmin.latlng[index]
          : null
      })
      if (positions.every(position => position != null)) merged.latlng = positions
    }
    return merged
  }
  return streamQuality(fromGarmin) > streamQuality(strava) ? fromGarmin : strava
}

export function applyActivityTracking(
  cache: StravaRawCache | null,
  garmin: GarminCache | null,
  entries: readonly ActivityTrackingEntry[],
  wahoo: WahooData | null = null,
): StravaRawCache | null {
  if (!cache || entries.length === 0) return cache
  const activities = { ...cache.activities }
  const streams = { ...cache.streams }
  for (const entry of entries) {
    const id = String(entry.activityId)
    const activity = cache.activities[id]
    if (!activity) continue
    const sport = normalizeKind(activity.sportType)
    if (sport !== 'bike' && sport !== 'run') continue
    const match = matchGarminActivity(activity, sport, garmin, entry.garminActivityId)
    const garminDistance = match?.activity.distanceM
    const distance = Number.isFinite(activity.distance) ? activity.distance : (garminDistance ?? 0)
    if (entry.virtual)
      activities[id] = {
        ...activity,
        sportType: sport === 'bike' ? 'VirtualRide' : 'VirtualRun',
        distance,
        averageSpeed: activity.movingTime > 0 ? distance / activity.movingTime : 0,
        totalElevationGain: match?.activity.metrics.totalAscentM ?? activity.totalElevationGain,
      }
    const original = cache.streams?.[id]
    const fromGarmin = match ? garmin?.streams?.[match.activity.id] : undefined
    const alignment = timedStreamAlignment(fromGarmin)
    if (
      entry.virtual &&
      !hasCompleteTimedRoute(original) &&
      match &&
      original?.time?.length &&
      alignment
    ) {
      const offsetS =
        (Date.parse(match.activity.startDate) - Date.parse(activity.startDate)) / 1_000
      const samples = alignment.time.map((time, index) => ({
        elapsedS: time + offsetS,
        value: alignment.distance[index],
      }))
      const altitudeSamples =
        fromGarmin?.altitude.length === alignment.time.length
          ? alignment.time.map((time, index) => ({
              elapsedS: time + offsetS,
              value: fromGarmin.altitude[index],
            }))
          : []
      streams[id] = {
        ...original,
        latlng: [],
        altitude: original.time.map(time => timedMetricAt(altitudeSamples, time) ?? 0),
        distance: original.time.map(time => timedMetricAt(samples, time) ?? 0),
      }
    }
    const wahooMatch =
      sport === 'bike' ? matchWahooActivity(activity, sport, wahoo, entry.wahooFitPath) : null
    if (wahooMatch && wahoo) {
      activities[id] = withWahooSummary(
        activities[id],
        cache.activityDetails?.[id],
        wahooMatch,
        entry.power,
      )
      const base =
        streams[id] ?? (match ? selectStreams(undefined, match, garmin, activities[id]) : undefined)
      const merged = mergeWahooTelemetry(activity, base, wahooMatch, wahoo, entry.power)
      if (merged) streams[id] = merged
    }
    if (entry.power === 'garmin' && match) {
      const projected = { ...activities[id] }
      for (const [field, metric] of POWER_SUMMARY_FIELDS) {
        const value = match.activity.metrics[metric]
        if (value != null && Number.isFinite(value)) projected[field] = value
      }
      if (match.activity.metrics.avgPower != null) projected.deviceWatts = true
      activities[id] = projected
      const base = streams[id] ?? selectStreams(undefined, match, garmin, projected)
      if (
        base?.time?.length &&
        fromGarmin?.time?.length &&
        fromGarmin.watts?.length === fromGarmin.time.length
      ) {
        const offsetS =
          (Date.parse(match.activity.startDate) - Date.parse(activity.startDate)) / 1_000
        const samples = fromGarmin.time.map((time, index) => ({
          elapsedS: time + offsetS,
          value: fromGarmin.watts?.[index] ?? null,
        }))
        streams[id] = {
          ...base,
          watts: base.time.map(
            (time, index) =>
              timedNullableMetricAt(samples, time, WAHOO_TELEMETRY_MAX_DISTANCE_S) ??
              base.watts?.[index] ??
              0,
          ),
        }
      }
    }
    if (!entry.virtual && sport === 'bike') {
      const selected = selectStreams(streams[id], match, garmin, activities[id])
      if (selected) streams[id] = selected
    }
  }
  return { ...cache, activities, streams }
}

export function selectEffortStreams(
  strava: StravaStreams | undefined,
  selected: StravaStreams | GarminStreams | undefined,
): StravaStreams | GarminStreams | undefined {
  return strava?.time?.length && strava.time !== selected?.time ? strava : selected
}

export function resolveActivityHeartRate(
  a: RawStravaActivity,
  sport: ActivityKind,
  selectedStreams: StravaStreams | GarminStreams | undefined,
  garminMatch: GarminActivityMatch | null,
  garmin: GarminCache | null,
): ActivityHeartRate {
  const selectedHr = selectedStreams?.heartrate ?? []
  const stravaAvg = roundPos(a.averageHeartrate) ?? avgPos(selectedHr)
  const stravaMax = roundPos(a.maxHeartrate) ?? maxPos(selectedHr)
  const garminStream = garminMatch
    ? (garmin?.streams?.[garminMatch.activity.id]?.heartrate ?? [])
    : []
  const metrics = garminMatch?.activity.metrics
  const garminAvg = roundPos(metrics?.avgHeartRate) ?? avgPos(garminStream)
  const garminMax = roundPos(metrics?.maxHeartRate) ?? maxPos(garminStream)

  if (sport === 'run' && (garminAvg != null || garminMax != null))
    return {
      avgHr: garminAvg ?? stravaAvg,
      maxHr: garminMax ?? stravaMax,
      stream: hasPositive(garminStream) ? garminStream : selectedHr,
    }

  return { avgHr: stravaAvg ?? garminAvg, maxHr: stravaMax ?? garminMax, stream: selectedHr }
}
