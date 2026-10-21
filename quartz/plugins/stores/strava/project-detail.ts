import type { CriticalPowerEstimate } from '../critical-power'
import type { GarminRunWalkData, GarminStreams } from '../garmin'
import type { WeatherCache } from '../weather'
import type { ActivityCyclingDynamics, ActivityGearShift } from './cycling-mechanics'
import {
  buildActivityEnvironment,
  type GardenApparentWindEstimate,
  type GardenEnvironmentEstimate,
} from '../../../util/activity-environment'
import { parseActivityProviderReports } from '../../../util/activity-provider-reports'
import { applyGardenUvCalibration } from '../../../util/activity-uv-score'
import { runBestEfforts } from '../../../util/run-best-efforts'
import { rawMapRouteSegments } from '../../../util/triathlon-map-route'
import { buildWalkPowerEstimate, type WalkPowerEstimate } from '../../../util/walk-power'
import { calculateAnaerobicPowerEstimate, type StravaActivityDetail } from './activity-types'
import {
  cyclingClimbEfforts,
  projectAnalysisRanges,
  projectRunSplits,
  type ActivityClimbSegment,
} from './climbs-and-ranges'
import {
  calculateCyclingPerformanceCondition,
  estimatedPerformanceConditionAt,
  performanceConditionAt,
  projectHeartRateTrace,
  type ActivityPerformanceConditionTrace,
} from './detail-streams'
import {
  distanceBestEfforts,
  effortTimeline,
  maxSpeedKph,
  meanMaxCurve,
  powerBestEfforts,
  type PowerCurvePoint,
} from './efforts-and-curves'
import { nativeRunDynamicsAt, nativeThermalAt, type ActivityMetricSamples } from './metric-samples'
import {
  temperatureAt,
  type ActivityFueling,
  type ActivityWeight,
  type GarminVerification,
} from './provider-metadata'
import {
  ROUTE_POINTS,
  WAHOO_TELEMETRY_MAX_DISTANCE_S,
  type ActivityComputer,
  type ActivityDevice,
  type ActivityHeartRate,
  type ActivityKind,
  type RawStravaActivity,
  type RawStravaActivityDetail,
  type StravaMapPoint,
  type StravaRoutePoint,
  type StravaStreams,
} from './raw-and-kinds'
import {
  avgPos,
  cleanAltitude,
  durationZoneTimes,
  localSpeedKph,
  nullableRound,
  powerHistogram,
  privateRouteBounds,
  round,
  roundPos,
  routeElapsedSeconds,
  sampleIndicesWithRequired,
  THERMAL_SAMPLE_MAX_DISTANCE_S,
  timedMetricAt,
  timedNullableMetricAt,
  timedStreamAlignment,
  zoneTimes,
  type TimedMetricSample,
} from './stream-primitives'

export function projectDetail(
  a: RawStravaActivity,
  sport: ActivityKind,
  streams: StravaStreams | GarminStreams | undefined,
  effortStreams: StravaStreams | GarminStreams | undefined,
  stravaStreams: StravaStreams | undefined,
  heartRate: ActivityHeartRate,
  metricSamples: ActivityMetricSamples,
  gearShifts: ActivityGearShift[],
  cyclingDynamics: ActivityCyclingDynamics | null,
  runWalk: GarminRunWalkData | null,
  weather: WeatherCache['activities'][string] | undefined,
  weatherAttribution: WeatherCache['attribution'],
  uvCalibration: WeatherCache['uvCalibration'],
  generatedAt: number,
  geo: string | undefined,
  fueling: ActivityFueling | null,
  garmin: GarminVerification | null,
  computer: ActivityComputer | null,
  device: ActivityDevice | null,
  weight: ActivityWeight | null,
  rawDetail: RawStravaActivityDetail | undefined,
  climbs: ActivityClimbSegment[],
  hrBounds: number[],
  powerBounds: number[],
  ftpWatts: number | null,
  lactateThresholdHeartRateBpm: number | null,
  restingHeartRateBpm: number | null,
  anaerobicPowerModel: Pick<CriticalPowerEstimate, 'criticalPowerWatts' | 'wPrimeJoules'> | null,
  home: [number, number] | null,
  powerCurve: PowerCurvePoint[] | undefined,
  activityCriticalPower: CriticalPowerEstimate | null,
): StravaActivityDetail {
  const route: StravaRoutePoint[] = []
  let mapRoute: StravaMapPoint[][] = []
  const analysis = projectAnalysisRanges(
    a,
    rawDetail,
    effortStreams,
    climbs,
    effortStreams === stravaStreams,
  )
  // Keep recorded lap endpoints on their source timeline when Garmin's cached HR is coarser.
  const lapHeartRateStreams = analysis.ranges.some(range => range.heartRateChange)
    ? stravaStreams
    : undefined
  // Walk physiology needs the recorded session timeline, including privacy-trimmed route sections.
  const walkHeartRateStreams =
    sport === 'walk' &&
    stravaStreams?.time &&
    stravaStreams.time.length >= 2 &&
    stravaStreams.heartrate?.length === stravaStreams.time.length &&
    stravaStreams.heartrate.some(value => value > 0)
      ? stravaStreams
      : undefined
  const heartRateTraceStreams = walkHeartRateStreams ?? lapHeartRateStreams
  const estimateWalkPower = (
    input: StravaStreams | GarminStreams | undefined,
    inputSource: WalkPowerEstimate['inputSource'],
  ): WalkPowerEstimate | null =>
    sport === 'walk' &&
    !a.deviceWatts &&
    garmin?.avgPower == null &&
    weight?.date === a.startDateLocal.slice(0, 10) &&
    input?.time
      ? buildWalkPowerEstimate({
          inputSource,
          weight: { ...weight, source: 'garmin' },
          streams: { time: input.time, distance: input.distance, altitude: input.altitude },
          elapsedTimeS: a.elapsedTime > 0 ? a.elapsedTime : a.movingTime,
        })
      : null
  const walkPower =
    estimateWalkPower(stravaStreams, 'strava') ??
    (streams !== stravaStreams ? estimateWalkPower(streams, 'garmin') : null)
  let minAlt = 0
  let maxAlt = 0
  let ascentM = 0
  let descentM = 0
  let environment: GardenEnvironmentEstimate | null = null
  let apparentWind: GardenApparentWindEstimate | null = null
  const temperatureSeries = weather?.temperatureSeries ?? []
  const fallbackTemperatureC = weather?.temperatureC ?? a.averageTemp ?? null
  const mapLatlng = streams?.latlng ?? []
  const mapTime = streams
    ? (routeElapsedSeconds(streams, a.movingTime) ??
      Array.from({ length: mapLatlng.length }, (_, index) => index))
    : null
  if (
    mapRoute.length === 0 &&
    streams &&
    mapTime &&
    mapLatlng.length >= 2 &&
    streams.distance.length === mapLatlng.length
  ) {
    const [mapLo, mapHi] = privateRouteBounds(mapLatlng, home)
    mapRoute = rawMapRouteSegments(mapLatlng, streams.distance, mapTime, mapLo, mapHi)
  }
  const timedEffort = timedStreamAlignment(effortStreams)
  const hasEffortPower = effortStreams?.watts?.some(value => value > 0) ?? false
  const timeline =
    sport === 'bike' || hasEffortPower ? effortTimeline(effortStreams, a.movingTime) : null
  const nativePerformanceCondition =
    (sport === 'run' || sport === 'bike') &&
    metricSamples.performanceCondition.filter(sample => sample.value != null).length >= 2
  const estimatedPerformanceCondition =
    sport === 'bike' && !nativePerformanceCondition && timeline
      ? calculateCyclingPerformanceCondition(
          timeline.watts,
          timeline.wattsObserved,
          timeline.heartRate,
          ftpWatts,
          lactateThresholdHeartRateBpm,
          restingHeartRateBpm,
        )
      : null
  const performanceConditionTrace: ActivityPerformanceConditionTrace | null =
    nativePerformanceCondition
      ? { source: 'garmin', method: 'garmin-native' }
      : estimatedPerformanceCondition
        ? {
            source: 'garden-estimate',
            method: estimatedPerformanceCondition.method,
            ftpWatts: estimatedPerformanceCondition.ftpWatts,
            lactateThresholdHeartRateBpm:
              estimatedPerformanceCondition.lactateThresholdHeartRateBpm,
            restingHeartRateBpm: estimatedPerformanceCondition.restingHeartRateBpm,
            windowSeconds: estimatedPerformanceCondition.windowSeconds,
          }
        : null
  const routeStreams =
    timedEffort && timedEffort.streams.latlng.length === timedEffort.time.length
      ? timedEffort.streams
      : streams
  const routeTime = routeStreams ? routeElapsedSeconds(routeStreams, a.movingTime) : null
  const latlng = routeTime ? (routeStreams?.latlng ?? []) : []
  const cadStream = routeStreams?.cadence ?? []
  if (routeStreams && routeTime && latlng.length >= 2) {
    const altitude = cleanAltitude(routeStreams.altitude)
    const distance: number[] = []
    let previousDistance = 0
    for (const value of routeStreams.distance) {
      previousDistance = Math.max(previousDistance, Number.isFinite(value) ? value : 0)
      distance.push(previousDistance)
    }
    const watts = routeStreams.watts ?? []
    const routeHrStream = routeStreams.heartrate ?? []
    const speedKph = localSpeedKph(routeTime, distance)
    if (weather?.activityId === a.id) {
      const result = buildActivityEnvironment({
        activityId: a.id,
        elapsedTimeS: a.elapsedTime > 0 ? a.elapsedTime : a.movingTime,
        movingTimeS: a.movingTime,
        timeS: routeTime,
        distanceM: distance,
        latlng,
        openWater: sport === 'swim',
        weather,
        attribution: weatherAttribution,
        computedAt: generatedAt,
      })
      environment = result.environment
      apparentWind = result.apparentWind
    }
    let ascent = 0
    let descent = 0
    for (let i = 1; i < altitude.length; i++) {
      const delta = altitude[i] - altitude[i - 1]
      if (delta > 0) ascent += delta
      else descent -= delta
    }
    ascentM = Math.round(ascent)
    descentM = Math.round(descent)
    const [lo0, hi0] = privateRouteBounds(latlng, home)
    const requiredIndices = routeStreams === effortStreams ? analysis.boundaryIndices : []
    const idx = sampleIndicesWithRequired(lo0, hi0, ROUTE_POINTS, requiredIndices)
    let sumLat = 0
    let sumLng = 0
    for (const i of idx) {
      sumLat += latlng[i][0]
      sumLng += latlng[i][1]
    }
    const meanLat = sumLat / idx.length
    const meanLng = sumLng / idx.length
    const cosLat = Math.cos((meanLat * Math.PI) / 180)
    const xs = idx.map(i => (latlng[i][1] - meanLng) * cosLat)
    const ys = idx.map(i => latlng[i][0] - meanLat)
    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    const minY = Math.min(...ys)
    const maxY = Math.max(...ys)
    const span = Math.max(maxX - minX, maxY - minY) || 1
    const offX = (1 - (maxX - minX) / span) / 2
    const offY = (1 - (maxY - minY) / span) / 2
    const alts = idx.map(i => altitude[i] ?? 0)
    minAlt = round(Math.min(...alts), 1)
    maxAlt = round(Math.max(...alts), 1)
    idx.forEach((i, k) => {
      const elapsedS = routeTime[i]
      const temperatureC =
        timedNullableMetricAt(
          metricSamples.deviceTemperatureC,
          elapsedS,
          WAHOO_TELEMETRY_MAX_DISTANCE_S,
        ) ??
        temperatureAt(temperatureSeries, elapsedS) ??
        fallbackTemperatureC
      const rightPowerPct = timedNullableMetricAt(
        metricSamples.rightBalance,
        elapsedS,
        WAHOO_TELEMETRY_MAX_DISTANCE_S,
      )
      const staminaAt = (samples: TimedMetricSample[]): number | null =>
        samples.length > 0 &&
        elapsedS >= samples[0].elapsedS &&
        elapsedS <= samples[samples.length - 1].elapsedS
          ? timedNullableMetricAt(samples, elapsedS, THERMAL_SAMPLE_MAX_DISTANCE_S)
          : null
      const stamina = staminaAt(metricSamples.stamina)
      const potentialStamina = staminaAt(metricSamples.potentialStamina)
      const respiration = timedMetricAt(metricSamples.respiration, elapsedS)
      const muscleOxygen = timedMetricAt(metricSamples.muscleOxygenPercent, elapsedS)
      const thermal = nativeThermalAt(metricSamples, elapsedS)
      const performanceCondition = estimatedPerformanceCondition
        ? estimatedPerformanceConditionAt(estimatedPerformanceCondition.samples, elapsedS)
        : nativePerformanceCondition
          ? performanceConditionAt(metricSamples.performanceCondition, elapsedS)
          : null
      const runDynamics = nativeRunDynamicsAt(metricSamples, elapsedS)
      route.push({
        x: round((xs[k] - minX) / span + offX, 4),
        y: round((ys[k] - minY) / span + offY, 4),
        d: round((distance[i] ?? 0) / 1000, 3),
        alt: round(alts[k], 1),
        w: Math.round(watts[i] ?? 0),
        hr: Math.round(routeHrStream[i] ?? 0),
        cad: Math.round(cadStream[i] ?? 0),
        ...(rightPowerPct == null ? {} : { rightPowerPct: round(rightPowerPct, 1) }),
        stamina: stamina == null ? null : round(stamina, 1),
        potentialStamina: potentialStamina == null ? null : round(potentialStamina, 1),
        resp: respiration == null ? null : round(respiration, 1),
        muscleOxygenPct: muscleOxygen == null ? null : round(muscleOxygen, 1),
        tempC: temperatureC == null ? null : round(temperatureC, 1),
        ...thermal,
        performanceCondition,
        ...runDynamics,
        lat: round(latlng[i][0], 5),
        lng: round(latlng[i][1], 5),
        elapsedS: round(elapsedS, 3),
        speedKph: speedKph[i] ?? 0,
      })
    })
  }
  const wFull = (a.sportType.startsWith('Virtual') ? effortStreams?.watts : streams?.watts) ?? []
  const hasHr = heartRate.stream.some(v => v > 0)
  const hasW = wFull.some(v => v > 0)
  const avgWattsWithoutZeros = hasW ? avgPos(wFull) : roundPos(garmin?.avgPower)
  const powerWithoutZeros =
    sport === 'bike' && avgWattsWithoutZeros != null
      ? {
          avgWatts: avgWattsWithoutZeros,
          powerZones: hasW && powerBounds.length > 0 ? zoneTimes(wFull, powerBounds, false) : null,
          powerHist: hasW ? powerHistogram(wFull, false) : null,
        }
      : null
  const elapsedTimeline =
    timeline && effortStreams && 'time' in effortStreams && effortStreams.time?.length
      ? timeline
      : null
  const runDistanceEfforts = sport === 'run' ? runBestEfforts(effortStreams) : []
  return {
    id: a.id,
    virtual: a.sportType.startsWith('Virtual'),
    commute: a.commute === true,
    sport,
    name: a.name,
    date: a.startDateLocal.slice(0, 10),
    start: a.startDate,
    distanceKm: round(a.distance / 1000, sport === 'swim' ? 3 : 1),
    movingTimeS: a.movingTime,
    elapsedTimeS: a.elapsedTime > 0 ? a.elapsedTime : a.movingTime,
    maxSpeedKph: a.maxSpeed == null ? maxSpeedKph(timeline) : round(a.maxSpeed * 3.6, 1),
    elevationM: ascentM,
    avgHr: nullableRound(a.averageHeartrate ?? garmin?.avgHeartRate ?? heartRate.avgHr, 0),
    maxHr: nullableRound(a.maxHeartrate ?? heartRate.maxHr, 0),
    avgWatts: nullableRound(a.averageWatts ?? garmin?.avgPower ?? null, 0),
    npWatts: nullableRound(a.weightedAverageWatts ?? garmin?.normalizedPower ?? null, 0),
    maxWatts: nullableRound(a.maxWatts ?? garmin?.maxPower ?? null, 0),
    kilojoules: nullableRound(a.kilojoules ?? garmin?.totalWorkKJ ?? null, 0),
    deviceWatts: a.deviceWatts === true || (a.averageWatts == null && garmin?.avgPower != null),
    avgCadence:
      a.averageCadence != null
        ? Math.round(a.averageCadence)
        : (roundPos(garmin?.avgCadence) ?? avgPos(cadStream)),
    sufferScore: a.sufferScore != null ? Math.round(a.sufferScore) : null,
    calories:
      a.calories != null
        ? Math.round(a.calories)
        : rawDetail?.calories != null
          ? Math.round(rawDetail.calories)
          : (garmin?.totalCalories ?? null),
    deviceTemperatureC: a.averageTemp != null ? round(a.averageTemp, 1) : null,
    ambientTemperatureC:
      environment?.summary.averageAmbientTemperatureC ?? weather?.temperatureC ?? null,
    windKph: weather?.windKph ?? null,
    windDir: weather?.windDir ?? null,
    windDirDeg: weather?.windDirDeg ?? null,
    windGustKph: weather?.windGustKph ?? null,
    averageRelativeHumidityPct: weather?.averageRelativeHumidityPct ?? null,
    relativeHumidityProvenance: weather?.relativeHumidityProvenance ?? null,
    location: geo ?? null,
    fueling,
    strength: null,
    sauna: null,
    garmin,
    computer,
    device,
    staminaTrace: metricSamples.staminaTrace,
    performanceConditionTrace,
    calculatedIntensityFactor: null,
    calculatedExerciseLoad: null,
    anaerobicPowerEstimate:
      sport === 'bike' && hasEffortPower && timeline
        ? calculateAnaerobicPowerEstimate(timeline.watts, a.movingTime, anaerobicPowerModel)
        : null,
    calculatedTrainingEffect: null,
    gearShifts,
    cyclingDynamics,
    runWalk,
    route,
    walkPower,
    heartRateTrace:
      route.length >= 2 && sport !== 'walk'
        ? []
        : projectHeartRateTrace(
            sport,
            heartRateTraceStreams ?? streams,
            heartRateTraceStreams
              ? { ...heartRate, stream: heartRateTraceStreams.heartrate ?? [] }
              : heartRate,
            metricSamples,
            analysis.ranges.flatMap(range => [range.startElapsedS, range.endElapsedS]),
          ),
    mapRoute,
    analysisRanges: analysis.ranges,
    runSplitsMetric: sport === 'run' ? projectRunSplits(rawDetail?.splitsMetric) : [],
    runSplitsStandard: sport === 'run' ? projectRunSplits(rawDetail?.splitsStandard) : [],
    runPaceZones: null,
    minAlt,
    maxAlt,
    descentM,
    hrZones:
      hasHr && hrBounds.length > 0
        ? durationZoneTimes(heartRate.stream, hrBounds, a.movingTime)
        : null,
    powerZones: hasW && powerBounds.length > 0 ? zoneTimes(wFull, powerBounds, true) : null,
    powerHist: hasW ? powerHistogram(wFull) : null,
    powerWithoutZeros,
    powerCurve: hasEffortPower && timeline ? (powerCurve ?? meanMaxCurve(timeline)) : null,
    powerCurveWeight:
      hasEffortPower && timeline && weight ? { ...weight, source: 'garmin' } : undefined,
    activityCriticalPower,
    bestEfforts:
      sport === 'bike' && (elapsedTimeline || climbs.length > 0)
        ? {
            weightKg: weight?.kg ?? null,
            weightDate: weight?.date ?? null,
            distance: elapsedTimeline ? distanceBestEfforts(elapsedTimeline) : [],
            power:
              elapsedTimeline && hasEffortPower
                ? powerBestEfforts(elapsedTimeline, weight?.kg ?? null)
                : [],
            climbs: cyclingClimbEfforts(climbs, weight?.kg ?? null),
          }
        : runDistanceEfforts.length > 0
          ? {
              distanceSource: 'calculated',
              weightKg: null,
              weightDate: null,
              distance: runDistanceEfforts,
              power: [],
              climbs: [],
            }
          : null,
    strokeCount: null,
    strokeRateSpm: null,
    swimPaceSPer100m: null,
    swimPaceSource: null,
    swimDurationS: null,
    swimIntervals: [],
    swimLocation: null,
    waterTemperatureC: null,
    analyses: {
      native: parseActivityProviderReports(
        rawDetail?.description ?? null,
        a.id,
        rawDetail?.fetchedAt ?? 0,
      ),
      derived: {
        environment,
        uvScore: environment
          ? applyGardenUvCalibration(environment, uvCalibration, generatedAt)
          : null,
        apparentWind,
      },
    },
  }
}
