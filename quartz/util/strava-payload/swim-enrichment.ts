import type { AppleCache, AppleSwim } from '../../plugins/stores/apple'
import type { GarminCache, GarminSwimData } from '../../plugins/stores/garmin'
import type {
  ActivityAnalysisRange,
  StravaActivityDetail,
  StravaPayload,
  SwimActivityInterval,
} from '../../plugins/stores/strava'
import { matchAppleSwims, matchAppleSwimTelemetry } from '../apple-swim-match'
import { swimPaceSeconds, swimStrokeRate } from '../swim-metrics'
import { buildSwimPowerEstimate, openWaterPowerIntervals } from '../swim-power'

export function swimActivityIntervals(swim: AppleSwim): {
  durationS: number | null
  intervals: SwimActivityInterval[]
} {
  const raw = (swim.intervals ?? [])
    .slice()
    .sort((a, b) => a.start.localeCompare(b.start) || a.end.localeCompare(b.end))
  const firstStart = raw[0]?.start
  const startMs = Date.parse(swim.start ?? firstStart ?? '')
  if (!Number.isFinite(startMs)) return { durationS: null, intervals: [] }
  const intervalEndMs = raw.reduce((latest, interval) => {
    const end = Date.parse(interval.end)
    return Number.isFinite(end) ? Math.max(latest, end) : latest
  }, startMs)
  const workoutEndMs = Date.parse(swim.end ?? '')
  const endMs = Number.isFinite(workoutEndMs)
    ? Math.max(workoutEndMs, intervalEndMs)
    : intervalEndMs
  const durationS = endMs > startMs ? Math.round((endMs - startMs) / 1000) : null
  let distanceM = 0
  const intervals: SwimActivityInterval[] = []
  for (const interval of raw) {
    const intervalStartMs = Date.parse(interval.start)
    const intervalEndMs = Date.parse(interval.end)
    if (
      !Number.isFinite(intervalStartMs) ||
      !Number.isFinite(intervalEndMs) ||
      intervalEndMs <= intervalStartMs ||
      intervalEndMs <= startMs ||
      !Number.isFinite(interval.distanceM) ||
      interval.distanceM <= 0
    )
      continue
    const timestampDurationS = (intervalEndMs - intervalStartMs) / 1000
    const exportedDurationS =
      interval.durationS != null && Number.isFinite(interval.durationS) && interval.durationS > 0
        ? interval.durationS
        : null
    const activeTimeS = exportedDurationS ?? timestampDurationS
    const startElapsedS =
      Math.round(
        Math.max(
          0,
          interval.startElapsedS != null &&
            Number.isFinite(interval.startElapsedS) &&
            interval.startElapsedS >= 0
            ? interval.startElapsedS
            : (intervalStartMs - startMs) / 1000,
        ) * 10,
      ) / 10
    const exportedEndElapsedS =
      interval.endElapsedS != null &&
      Number.isFinite(interval.endElapsedS) &&
      interval.endElapsedS > startElapsedS
        ? interval.endElapsedS
        : null
    const endElapsedS =
      Math.round(
        Math.max(
          0,
          exportedEndElapsedS ??
            (exportedDurationS != null
              ? startElapsedS + exportedDurationS
              : (intervalEndMs - startMs) / 1000),
        ) * 10,
      ) / 10
    distanceM += interval.distanceM
    intervals.push({
      startElapsedS,
      endElapsedS,
      distanceM: Math.round(interval.distanceM * 10) / 10,
      durationS: Math.round(activeTimeS * 10) / 10,
      cumulativeDistanceM: Math.round(distanceM * 10) / 10,
      paceSPer100m: swimPaceSeconds(interval.distanceM, activeTimeS),
      strokeCount:
        interval.stroke !== 'kickboard' && interval.strokeCount != null
          ? Math.round(interval.strokeCount * 10) / 10
          : null,
      strokeTimeS:
        interval.stroke !== 'kickboard' && interval.strokeTimeS != null
          ? Math.round(interval.strokeTimeS * 10) / 10
          : null,
      strokeRateSpm:
        interval.stroke === 'kickboard'
          ? null
          : swimStrokeRate(interval.strokeCount ?? 0, interval.strokeTimeS ?? 0),
      stroke: interval.stroke,
    })
  }
  return { durationS, intervals }
}

const SWIM_INTERVAL_COVERAGE_MIN = 0.9

const intervalSwimPace = (
  totalM: number,
  intervals: readonly SwimActivityInterval[],
): number | null => {
  if (!Number.isFinite(totalM) || totalM <= 0) return null
  let distanceM = 0
  let strokeTimeS = 0
  for (const interval of intervals) {
    if (interval.paceSPer100m == null) continue
    distanceM += interval.distanceM
    strokeTimeS += interval.durationS
  }
  if (distanceM / totalM < SWIM_INTERVAL_COVERAGE_MIN) return null
  return swimPaceSeconds(distanceM, strokeTimeS)
}

function garminSwimIntervals(swim: GarminSwimData | undefined): SwimActivityInterval[] {
  if (!swim) return []
  let cumulativeDistanceM = 0
  return swim.lengths.map(length => {
    cumulativeDistanceM += length.distanceM
    return {
      ...length,
      cumulativeDistanceM: Math.round(cumulativeDistanceM * 10) / 10,
      paceSPer100m: swimPaceSeconds(length.distanceM, length.durationS),
    }
  })
}

function garminSwimAnalysisRanges(swim: GarminSwimData | undefined): ActivityAnalysisRange[] {
  if (!swim) return []
  const ranges: ActivityAnalysisRange[] = []
  let cumulativeDistanceKm = 0
  for (const lap of swim.laps ?? []) {
    if (
      !Number.isFinite(lap.startElapsedS) ||
      !Number.isFinite(lap.endElapsedS) ||
      lap.startElapsedS < 0 ||
      lap.endElapsedS <= lap.startElapsedS ||
      !Number.isFinite(lap.distanceM) ||
      lap.distanceM <= 0 ||
      !Number.isFinite(lap.durationS) ||
      lap.durationS <= 0
    )
      continue
    const paceSPer100m = swimPaceSeconds(lap.distanceM, lap.durationS)
    if (paceSPer100m == null) continue
    const distanceKm = lap.distanceM / 1_000
    const startDistanceKm = cumulativeDistanceKm
    cumulativeDistanceKm += distanceKm
    ranges.push({
      kind: 'lap',
      id: `garmin-swim-lap:${ranges.length + 1}`,
      label: `Lap ${ranges.length + 1}`,
      startElapsedS: lap.startElapsedS,
      endElapsedS: lap.endElapsedS,
      startDistanceKm,
      endDistanceKm: cumulativeDistanceKm,
      durationS: lap.durationS,
      movingTimeS: lap.durationS,
      distanceKm,
      elevationGainM: swim.location === 'openWater' ? lap.elevationGainM : null,
      averageSpeedKph: 360 / paceSPer100m,
      averageHeartRate: lap.averageHeartRate,
      averageWatts: null,
      averageCadence: lap.strokeRateSpm,
    })
  }
  return ranges
}

function garminSwimStrokeDistances(swim: GarminSwimData | undefined): Record<string, number> {
  const strokes: Record<string, number> = {}
  for (const length of swim?.lengths ?? []) {
    if (!length.stroke || !Number.isFinite(length.distanceM) || length.distanceM <= 0) continue
    strokes[length.stroke] = (strokes[length.stroke] ?? 0) + length.distanceM
  }
  return strokes
}

function projectSwimHeartRateDistance(
  detail: StravaActivityDetail,
  intervals: readonly SwimActivityInterval[],
): void {
  if (
    intervals.length === 0 ||
    detail.heartRateTrace.some(point => Number.isFinite(point.distanceKm) && point.distanceKm > 0)
  )
    return
  const measured = intervals
    .filter(
      interval =>
        Number.isFinite(interval.startElapsedS) &&
        Number.isFinite(interval.endElapsedS) &&
        interval.endElapsedS > interval.startElapsedS &&
        Number.isFinite(interval.distanceM) &&
        interval.distanceM > 0 &&
        Number.isFinite(interval.cumulativeDistanceM) &&
        interval.cumulativeDistanceM >= interval.distanceM,
    )
    .toSorted((left, right) => left.startElapsedS - right.startElapsedS)
  if (measured.length === 0) return
  detail.heartRateTrace = detail.heartRateTrace.map(point => {
    if (!Number.isFinite(point.elapsedS) || point.elapsedS < 0) return point
    let completedDistanceM = 0
    for (const interval of measured) {
      const startDistanceM = Math.max(
        completedDistanceM,
        interval.cumulativeDistanceM - interval.distanceM,
      )
      if (point.elapsedS < interval.startElapsedS)
        return { ...point, distanceKm: completedDistanceM / 1_000 }
      if (point.elapsedS <= interval.endElapsedS) {
        const fraction =
          (point.elapsedS - interval.startElapsedS) /
          (interval.endElapsedS - interval.startElapsedS)
        return { ...point, distanceKm: (startDistanceM + interval.distanceM * fraction) / 1_000 }
      }
      completedDistanceM = Math.max(completedDistanceM, interval.cumulativeDistanceM)
    }
    return { ...point, distanceKm: completedDistanceM / 1_000 }
  })
}

export function enrichSwimMetrics(
  payload: StravaPayload,
  apple: AppleCache | null,
  garmin: GarminCache | null,
): void {
  const details = Object.values(payload.details).filter(
    (detail): detail is StravaActivityDetail => detail.sport === 'swim',
  )
  const swims = Object.values(apple?.swims ?? {})
  const candidates = details.map(detail => ({
    id: detail.id,
    date: detail.date,
    start: detail.start,
    distanceM: detail.distanceKm * 1_000,
  }))
  const matches = matchAppleSwims(swims, candidates)
  const telemetryMatches = matchAppleSwimTelemetry(swims, candidates)

  payload.swimTrend = []
  for (const detail of details) {
    const garminSwim = detail.garmin ? garmin?.swims?.[detail.garmin.activityId] : undefined
    const garminIntervals = garminSwimIntervals(garminSwim)
    const garminAnalysisRanges = garminSwimAnalysisRanges(garminSwim)
    const garminStrokes = garminSwimStrokeDistances(garminSwim)
    const swim = matches.get(detail.id)
    const telemetry = telemetryMatches.get(detail.id) ?? swim
    const activity = swim ? swimActivityIntervals(swim) : null
    const intervals = garminIntervals.length > 0 ? garminIntervals : (activity?.intervals ?? [])
    const intervalDistanceM =
      garminIntervals.length > 0 ? (garminSwim?.distanceM ?? 0) : (swim?.totalM ?? 0)
    const strokePace = intervalSwimPace(intervalDistanceM, intervals)
    const activePace = garminSwim
      ? swimPaceSeconds(garminSwim.distanceM ?? 0, garminSwim.activeTimeS ?? 0)
      : swim
        ? swimPaceSeconds(swim.totalM, swim.activeTimeS ?? 0)
        : null
    const movingPace = swimPaceSeconds(detail.distanceKm * 1_000, detail.movingTimeS)
    detail.swimPaceSPer100m = strokePace ?? activePace ?? movingPace
    detail.swimPaceSource =
      strokePace != null
        ? 'stroke'
        : activePace != null
          ? 'active'
          : movingPace != null
            ? 'moving'
            : null
    if (Object.keys(garminStrokes).length > 0) detail.strokes = garminStrokes
    else {
      const strokeDistribution = [swim, telemetry].find(
        candidate => candidate != null && Object.keys(candidate.strokes).length > 0,
      )
      if (strokeDistribution) detail.strokes = strokeDistribution.strokes
    }
    const matchedStrokeRate = swim
      ? swimStrokeRate(swim.strokeCount ?? 0, swim.strokeTimeS ?? 0)
      : null
    const telemetryStrokeRate = telemetry
      ? swimStrokeRate(telemetry.strokeCount ?? 0, telemetry.strokeTimeS ?? 0)
      : null
    detail.strokeCount =
      garminSwim?.strokeCount ??
      (matchedStrokeRate != null
        ? (swim?.strokeCount ?? null)
        : telemetryStrokeRate != null
          ? (telemetry?.strokeCount ?? null)
          : (swim?.strokeCount ?? telemetry?.strokeCount ?? null))
    detail.strokeRateSpm =
      garminSwim?.strokeRateSpm ??
      matchedStrokeRate ??
      telemetryStrokeRate ??
      (detail.avgCadence != null && detail.avgCadence > 0 ? detail.avgCadence : null)
    detail.swimDurationS = garminSwim?.elapsedTimeS ?? activity?.durationS ?? null
    detail.swimIntervals = intervals
    detail.swimPower = buildSwimPowerEstimate(
      detail.provenance ? 'reconstructed' : garminIntervals.length > 0 ? 'garmin' : 'apple',
      intervals,
    )
    if (garminAnalysisRanges.length > 0)
      detail.analysisRanges = [
        ...detail.analysisRanges.filter(range => range.kind !== 'lap'),
        ...garminAnalysisRanges,
      ]
    projectSwimHeartRateDistance(detail, intervals)
    detail.swimLocation = garminSwim?.location ?? telemetry?.location ?? null
    if (detail.swimLocation === 'openWater' && detail.route.length >= 2)
      detail.swimPower = buildSwimPowerEstimate(
        'route',
        openWaterPowerIntervals(detail.route.filter(p => p.elapsedS <= detail.elapsedTimeS)),
        'ground-speed',
      )
    detail.waterTemperatureC = telemetry?.waterTemperatureC ?? null
    if (detail.swimPaceSPer100m == null && detail.strokeRateSpm == null) continue
    payload.swimTrend.push({
      id: detail.id,
      date: detail.date,
      start: detail.start,
      paceSPer100m: detail.swimPaceSPer100m,
      paceSource: detail.swimPaceSource,
      strokeRateSpm: detail.strokeRateSpm,
    })
  }
  payload.swimTrend.sort((a, b) => a.start.localeCompare(b.start) || a.id - b.id)
}
