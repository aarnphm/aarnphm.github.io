import type { CriticalPowerAnchor, CriticalPowerEstimate } from '../critical-power'
import type { OuraCache } from '../oura'
import type { ActivityTrackingEntry } from '../tracking'
import type { WeatherCache } from '../weather'
import type { StravaActivityDetail } from './activity-types'
import type { PowerCurvePoint } from './efforts-and-curves'
import { buildCyclingIntensityTrace } from '../../../util/cycling-intensity'
import { buildCyclingPowerTrace } from '../../../util/cycling-power'
import { estimateWahooCyclingStamina } from '../../../util/cycling-stamina'
import {
  buildCyclingTorqueTrace,
  cyclingMechanics,
  type CyclingTorqueSample,
} from '../../../util/cycling-torque'
import { applyHeartRatePhysiology } from '../../../util/heart-rate-physiology'
import { latestProviderSync } from '../../../util/provider-sync'
import { buildRunHeartCostTrace, buildRunStrideCadenceTrace } from '../../../util/run-mechanics'
import {
  matchGarminHeartRateActivity,
  matchGarminTrainingEffectActivity,
  type GarminActivityMatch,
  type GarminCache,
  type GarminStreams,
} from '../garmin'
import { matchWahooActivity, type WahooActivityMatch, type WahooData } from '../wahoo'
import { activityClimbSegments } from './climbs-and-ranges'
import {
  activityCyclingDynamics,
  activityGearShifts,
  cyclingPowerStreamsOnTerrain,
  cyclingPowerTerrain,
} from './cycling-mechanics'
import { activityMetricSamples } from './metric-samples'
import { projectDetail } from './project-detail'
import {
  activityComputer,
  activityDevice,
  activityFueling,
  activityRestingHeartRate,
  activityRunWalk,
  activityWeight,
  garminRespirationProjectionModel,
  garminVerification,
  nearestSameDayWeather,
} from './provider-metadata'
import {
  WAHOO_SUMMARY_FIELDS,
  type ActivityHeartRate,
  type ActivityKind,
  type RawStravaActivity,
  type StravaRawCache,
  type StravaStreams,
} from './raw-and-kinds'
import {
  avgPos,
  inferRouteHome,
  round,
  timedMetricAt,
  timedStreamAlignment,
} from './stream-primitives'
import {
  isWahooPowerSummary,
  resolveActivityHeartRate,
  selectEffortStreams,
  withWahooSummary,
} from './stream-selection'
import { calculateActivityExerciseLoad, calculateActivityIntensityFactor } from './training-load'

/** Payload-wide state that projectActivityDetail reads: provider matches, streams, bounds and power models. */
export interface ActivityProjectionScope {
  cache: StravaRawCache
  originalCache: StravaRawCache
  oura: OuraCache | null
  garmin: GarminCache | null
  weather: WeatherCache | null | undefined
  wahoo: WahooData | null | undefined
  trackingById: ReadonlyMap<ActivityTrackingEntry['activityId'], ActivityTrackingEntry>
  resolveGarmin: (activity: RawStravaActivity, sport: ActivityKind) => GarminActivityMatch | null
  garminMatches: ReadonlyMap<string, GarminActivityMatch | null>
  garminTrainingEffectMatches: ReadonlyMap<string, GarminActivityMatch | null>
  garminHeartRateMatches: ReadonlyMap<string, GarminActivityMatch | null>
  wahooMatches: ReadonlyMap<string, WahooActivityMatch | null>
  selectedStreams: ReadonlyMap<string, StravaStreams | GarminStreams | undefined>
  heartRates: ReadonlyMap<string, ActivityHeartRate>
  respirationProjection: ReturnType<typeof garminRespirationProjectionModel>
  wahooStamina: ReturnType<typeof estimateWahooCyclingStamina>
  generatedAt: ReturnType<typeof latestProviderSync>
  hrBounds: number[]
  powerBounds: number[]
  ftp: number | null
  inputMaxHeartRate: number | null | undefined
  inputLactateThresholdHeartRate: number | null | undefined
  hrmax: number
  criticalPower: CriticalPowerEstimate | null
  criticalPowerYear: CriticalPowerEstimate | null
  home: ReturnType<typeof inferRouteHome>
  powerCurves: ReadonlyMap<string, PowerCurvePoint[]>
  activityCriticalPowers: ReadonlyMap<string, CriticalPowerEstimate>
  criticalPowerAnchorsByActivity: ReadonlyMap<string, CriticalPowerAnchor[]>
  torqueSamplesByActivity: ReadonlyMap<string, CyclingTorqueSample[]>
}

export function projectActivityDetail(
  scope: ActivityProjectionScope,
  a: RawStravaActivity,
  sport: ActivityKind,
): StravaActivityDetail {
  const {
    cache,
    originalCache,
    oura,
    garmin,
    weather,
    wahoo,
    trackingById,
    resolveGarmin,
    garminMatches,
    garminTrainingEffectMatches,
    garminHeartRateMatches,
    wahooMatches,
    selectedStreams,
    heartRates,
    respirationProjection,
    wahooStamina,
    generatedAt,
    hrBounds,
    powerBounds,
    ftp,
    inputMaxHeartRate,
    inputLactateThresholdHeartRate,
    hrmax,
    criticalPower,
    criticalPowerYear,
    home,
    powerCurves,
    activityCriticalPowers,
    criticalPowerAnchorsByActivity,
    torqueSamplesByActivity,
  } = scope
  const id = String(a.id)
  const garminMatch = garminMatches.get(id) ?? resolveGarmin(a, sport)
  const garminTrainingEffectMatch =
    trackingById.get(a.id)?.garminActivityId != null
      ? garminMatch
      : (garminTrainingEffectMatches.get(id) ??
        matchGarminTrainingEffectActivity(a, sport, garmin, garminMatch))
  const garminHeartRateMatch =
    garminHeartRateMatches.get(id) ?? matchGarminHeartRateActivity(a, sport, garmin)
  const original = originalCache.activities[id] ?? a
  const wahooMatch =
    wahooMatches.get(id) ??
    matchWahooActivity(original, sport, wahoo ?? null, trackingById.get(a.id)?.wahooFitPath)
  const climbs = activityClimbSegments(
    sport,
    garminMatch,
    garmin,
    trackingById.get(a.id)?.virtual ? null : wahooMatch,
    wahoo ?? null,
  )
  const selectedStream = selectedStreams.get(id)
  const exactWeather = weather?.activities[id]
  const usesRouteLessWeather =
    (sport === 'swim' || sport === 'strength') &&
    (!selectedStream || selectedStream.latlng.length < 2)
  const activityWeather = a.sportType.startsWith('Virtual')
    ? undefined
    : (exactWeather ??
      (weather && usesRouteLessWeather ? nearestSameDayWeather(weather, a) : undefined))
  const detail = projectDetail(
    wahooMatch
      ? withWahooSummary(a, cache.activityDetails?.[id], wahooMatch, trackingById.get(a.id)?.power)
      : a,
    sport,
    selectedStream,
    selectEffortStreams(cache.streams?.[id], selectedStream),
    cache.streams?.[id],
    heartRates.get(id) ??
      resolveActivityHeartRate(a, sport, selectedStream, garminHeartRateMatch, garmin),
    activityMetricSamples(
      a,
      sport,
      garminMatch,
      garmin,
      wahooMatch,
      wahoo ?? null,
      wahooMatch ? (wahooStamina.get(wahooMatch.activity.id) ?? null) : null,
      respirationProjection,
    ),
    activityGearShifts(a, garminMatch, garmin, wahooMatch, wahoo ?? null),
    activityCyclingDynamics(a, garminMatch, garmin, wahooMatch, wahoo ?? null),
    activityRunWalk(sport, garminMatch, garmin),
    activityWeather,
    weather?.attribution ?? null,
    weather?.uvCalibration ?? null,
    generatedAt,
    a.sportType.startsWith('Virtual') ? undefined : cache.geo?.[String(a.id)],
    activityFueling(garminMatch, wahooMatch),
    garminVerification(original, garminMatch, garminTrainingEffectMatch),
    activityComputer(sport, garminMatch, wahooMatch),
    activityDevice(sport, a, garminMatch),
    activityWeight(garmin, a),
    cache.activityDetails?.[id],
    climbs,
    hrBounds,
    powerBounds,
    ftp,
    inputLactateThresholdHeartRate ?? null,
    activityRestingHeartRate(oura, a.startDateLocal.slice(0, 10)),
    criticalPowerYear ?? criticalPower,
    a.sportType.startsWith('Virtual') ? null : home,
    powerCurves.get(id),
    activityCriticalPowers.get(id) ?? null,
  )
  if (sport === 'bike') {
    const originalStreams = originalCache.streams?.[id]
    const terrain = cyclingPowerTerrain(a, originalStreams, garminMatch, garmin)
    const nativePowerOffsetS = wahooMatch
      ? (Date.parse(wahooMatch.activity.startDate) - Date.parse(a.startDate)) / 1_000
      : 0
    const nativePowerTrace = wahooMatch
      ? buildCyclingPowerTrace({
          source: 'wahoo',
          terrainSource: terrain?.source ?? 'wahoo',
          streams: cyclingPowerStreamsOnTerrain(
            wahoo?.streams[wahooMatch.activity.id],
            nativePowerOffsetS,
            terrain,
          ),
          startOffsetS: nativePowerOffsetS,
          elapsedTimeS: detail.elapsedTimeS,
        })
      : null
    detail.cyclingPowerTrace =
      nativePowerTrace ??
      (original.deviceWatts && originalStreams?.time && originalStreams.watts
        ? buildCyclingPowerTrace({
            source: 'strava',
            terrainSource: terrain?.source ?? 'strava',
            streams: cyclingPowerStreamsOnTerrain(
              { ...originalStreams, time: originalStreams.time, watts: originalStreams.watts },
              0,
              terrain,
            ),
            startOffsetS: 0,
            elapsedTimeS: detail.elapsedTimeS,
          })
        : null)
    detail.criticalPowerAnchors = criticalPowerAnchorsByActivity.get(id) ?? []
    const samples = torqueSamplesByActivity.get(id) ?? []
    detail.cyclingTorque = buildCyclingTorqueTrace(samples, detail.elapsedTimeS, detail.route)
    for (const effort of detail.bestEfforts?.power ?? [])
      if (effort.startElapsedS != null && effort.endElapsedS != null)
        effort.mechanics = cyclingMechanics(samples, effort.startElapsedS, effort.endElapsedS)
  }
  if (sport === 'run') {
    const originalStreams = originalCache.streams?.[id]
    detail.runStrideCadence = buildRunStrideCadenceTrace(originalStreams)
    detail.runHeartCost = buildRunHeartCostTrace(originalStreams, detail.elapsedTimeS)
  }
  const computerOverride = trackingById.get(a.id)?.computer?.trim()
  if (sport === 'bike' && computerOverride) detail.computerOverride = computerOverride
  if (original.gearId)
    detail.equipment = {
      id: original.gearId,
      name: cache.gear?.[original.gearId]?.name ?? null,
      source: 'strava',
    }
  applyHeartRatePhysiology(detail, inputMaxHeartRate ?? hrmax)
  if (!detail.virtual && trackingById.get(a.id)?.garminActivityId != null)
    detail.distanceSource = 'strava'
  const powerWatts = trackingById.get(a.id)?.power
    ? cache.streams?.[id]?.watts
    : originalCache.streams?.[id]?.watts
  if (detail.powerWithoutZeros && powerWatts?.some(value => value > 0))
    detail.powerWithoutZeros.avgWatts = avgPos(powerWatts)
  const fileName = (path: string | null): string | null => path?.split(/[\\/]/).at(-1) || null
  detail.sources = [{ provider: 'strava', activityId: id, name: original.name, fileName: null }]
  if (garminMatch)
    detail.sources.push({
      provider: 'garmin',
      activityId: garminMatch.activity.id,
      name: garminMatch.activity.name,
      fileName: fileName(garminMatch.activity.sourceFile),
    })
  if (
    garminTrainingEffectMatch &&
    garminTrainingEffectMatch.activity.id !== garminMatch?.activity.id
  )
    detail.sources.push({
      provider: 'garmin',
      activityId: garminTrainingEffectMatch.activity.id,
      name: garminTrainingEffectMatch.activity.name,
      fileName: fileName(garminTrainingEffectMatch.activity.sourceFile),
    })
  if (wahooMatch) {
    const source = wahooMatch.activity
    detail.sources.push({
      provider: 'wahoo',
      activityId: source.id,
      name: source.name,
      fileName:
        'path' in source.sourceFile
          ? fileName(source.sourceFile.path)
          : fileName(source.sourceFile.url.split(/[?#]/)[0]),
    })
  }
  if (wahooMatch) {
    const source = wahooMatch.activity
    detail.wahoo = {
      activityId: source.id,
      fitPath: 'path' in source.sourceFile ? source.sourceFile.path : null,
      sha256: source.sourceFile.sha256,
      sourceDevice: source.sourceDevice,
      startOffsetS: (Date.parse(source.startDate) - Date.parse(original.startDate)) / 1_000,
      distanceM: source.distanceM,
      metrics: { ...source.metrics },
      summarySources: Object.fromEntries(
        WAHOO_SUMMARY_FIELDS.flatMap(([rawField, metric, field]) =>
          source.metrics[metric] == null ||
          !Number.isFinite(source.metrics[metric]) ||
          (isWahooPowerSummary(rawField) &&
            (trackingById.get(a.id)?.power === 'strava' ||
              trackingById.get(a.id)?.power === 'garmin')) ||
          (!(trackingById.get(a.id)?.power === 'wahoo' && isWahooPowerSummary(rawField)) &&
            (original[rawField] != null ||
              (rawField === 'calories' && originalCache.activityDetails?.[id]?.calories != null)))
            ? []
            : [[field, 'wahoo']],
        ),
      ),
      streamFallback: originalCache.streams?.[id]?.time?.length
        ? 'strava'
        : garminMatch && garmin?.streams?.[garminMatch.activity.id]?.time?.length
          ? 'garmin'
          : null,
    }
    if (
      detail.sport === 'bike' &&
      (trackingById.get(a.id)?.power == null || trackingById.get(a.id)?.power === 'wahoo')
    )
      detail.cyclingIntensityTrace = buildCyclingIntensityTrace({
        streams: wahoo?.streams[source.id],
        metrics: source.metrics,
        startOffsetS: detail.wahoo.startOffsetS,
        elapsedTimeS: detail.elapsedTimeS,
        route: detail.route,
        athleteFtp: ftp,
      })
    detail.distanceSource = 'strava'
    detail.calculatedIntensityFactor = calculateActivityIntensityFactor(
      detail,
      null,
      ftp,
      inputLactateThresholdHeartRate ?? null,
    )
    detail.calculatedExerciseLoad = calculateActivityExerciseLoad(detail)
    const merged = cache.streams?.[id]
    if (trackingById.get(a.id)?.wahooFitPath && merged?.time)
      for (const range of detail.analysisRanges) {
        for (const [field, channel] of [
          ['averageWatts', 'watts'],
          ['averageHeartRate', 'heartrate'],
          ['averageCadence', 'cadence'],
        ] as const) {
          const values = merged[channel]
          if (!values || values.length !== merged.time.length) continue
          let sum = 0
          let duration = 0
          for (let index = 0; index < merged.time.length - 1; index++) {
            const seconds =
              Math.min(range.endElapsedS, merged.time[index + 1]) -
              Math.max(range.startElapsedS, merged.time[index])
            if (
              seconds <= 0 ||
              !Number.isFinite(values[index]) ||
              (channel === 'heartrate' && values[index] <= 0)
            )
              continue
            sum += values[index] * seconds
            duration += seconds
          }
          if (duration > 0) range[field] = round(sum / duration, 1)
        }
      }
  }
  if (trackingById.get(a.id)?.virtual && garminMatch) {
    const metrics = garminMatch.activity.metrics
    detail.distanceSource = Number.isFinite(original.distance) ? 'strava' : 'garmin'
    detail.elevationM = metrics.totalAscentM ?? detail.elevationM
    detail.descentM = metrics.totalDescentM ?? detail.descentM
  }
  if (trackingById.get(a.id)?.virtual || detail.wahoo) {
    const alignment = timedStreamAlignment(cache.streams?.[id])
    if (alignment) {
      const distanceSamples = alignment.time.map((elapsedS, index) => ({
        elapsedS,
        value: alignment.distance[index],
      }))
      const distanceKmAt = (elapsedS: number) =>
        round((timedMetricAt(distanceSamples, elapsedS) ?? 0) / 1_000, 3)
      for (const shift of detail.gearShifts) shift.distanceKm = distanceKmAt(shift.elapsedS)
      if (detail.cyclingDynamics) {
        detail.cyclingDynamics.distanceKm = detail.cyclingDynamics.elapsedS.map(distanceKmAt)
        for (const change of detail.cyclingDynamics.positionChanges)
          change.distanceKm = distanceKmAt(change.elapsedS)
      }
    }
  }
  return detail
}
