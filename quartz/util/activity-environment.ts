import type {
  WeatherActivity,
  WeatherAttribution,
  WeatherRouteHour,
} from '../plugins/stores/weather'

export interface GardenEstimateProvenance {
  source: 'garden-estimate'
  formulaId: string
  formulaVersion: 1
  inputVersion: 'weatherkit-route-hour-v1+strava-stream-v1'
  normalizationVersion: 1
  computedAt: number
  inputAsOf: number
  temporalSamplingModel: 'weatherkit-hourly-piecewise-constant'
  spatialSamplingModel: 'route-coordinate-nearest-hour-overlap-midpoint'
}

export interface GardenEnvironmentCoverage {
  weatherPct: number
  uvPct: number
  temperaturePct: number
  cloudPct: number
  daylightPct: number
}

export interface GardenEnvironmentSummary {
  averageUvIndex: number | null
  peakUvIndex: number | null
  uviHours: number | null
  ambientSed: number | null
  averageAmbientTemperatureC: number | null
  averageCloudCoverPct: number | null
  daylightCoveragePct: number
  weatherCoveragePct: number
  coveredDurationS: number
  elapsedDurationS: number
}

export interface GardenUvDoseClocks {
  elapsedSed: number | null
  movingTelemetrySed: number | null
}

export interface GardenEnvironmentSample {
  elapsedS: number
  distanceKm: number
  uvIndex: number | null
  cumulativeSed: number | null
  cumulativeMovingTelemetrySed: number | null
  ambientTemperatureC: number | null
  cloudCoverPct: number | null
  headwindKph: number | null
  crosswindKph: number | null
  apparentAirSpeedKph: number | null
  yawDeg: number | null
}

export interface GardenEnvironmentEstimate extends GardenEstimateProvenance {
  formulaId: 'garden-environment-v1'
  summary: GardenEnvironmentSummary
  doseClocks: GardenUvDoseClocks
  coverage: GardenEnvironmentCoverage
  samples: GardenEnvironmentSample[]
  attribution: WeatherAttribution | null
}

export interface GardenApparentWindSummary {
  headwindSharePct: number
  headwindTimeS: number
  tailwindTimeS: number
  longestHeadwindS: number
  averageHeadwindKph: number
  averageCrosswindKph: number
  maximumHeadwindKph: number
  maximumCrosswindKph: number
  averageGroundSpeedKph: number
  averageApparentAirSpeedKph: number
  apparentAirRatio: number
  averageYawDeg: number
  coveragePct: number
}

export interface GardenApparentWindEstimate extends GardenEstimateProvenance {
  formulaId: 'garden-apparent-wind-v1'
  summary: GardenApparentWindSummary
  coverage: { windPct: number }
}

export interface GardenUvScore extends GardenEstimateProvenance {
  formulaId: 'garden-uv-score-v1'
  score: number
  severity: 'negligible' | 'low' | 'moderate' | 'high' | 'serious' | 'extreme'
  doseClock: 'elapsed' | 'moving-telemetry'
  doseSed: number
  coefficientSed: number
  calibrationVersion: 1
}

export interface ActivityEnvironmentInput {
  activityId: number
  elapsedTimeS: number
  movingTimeS: number
  timeS: readonly number[]
  distanceM: readonly number[]
  latlng: readonly [number, number][]
  weather: WeatherActivity
  attribution: WeatherAttribution | null
  computedAt: number
}

export interface ActivityEnvironmentResult {
  environment: GardenEnvironmentEstimate | null
  apparentWind: GardenApparentWindEstimate | null
}

interface MetricAggregate {
  coveredS: number
  weightedTotal: number
  peak: number | null
}

interface WindInterval {
  startS: number
  endS: number
  durationS: number
  groundSpeedKph: number
  headwindKph: number
  crosswindKph: number
  apparentAirSpeedKph: number
  yawDeg: number
}

const round = (value: number, digits = 2): number => {
  const factor = 10 ** digits
  const rounded = Math.round(value * factor) / factor
  return Object.is(rounded, -0) ? 0 : rounded
}

const percentage = (coveredS: number, durationS: number): number =>
  durationS > 0 ? round(Math.min(1, Math.max(0, coveredS / durationS)) * 100, 1) : 0

const validRouteHours = (weather: WeatherActivity, durationS: number): WeatherRouteHour[] =>
  (weather.routeHours ?? [])
    .filter(
      hour =>
        Number.isFinite(hour.elapsedStartS) &&
        Number.isFinite(hour.elapsedEndS) &&
        hour.elapsedStartS >= 0 &&
        hour.elapsedEndS > hour.elapsedStartS &&
        hour.elapsedEndS <= durationS + 1,
    )
    .slice()
    .sort((left, right) => left.elapsedStartS - right.elapsedStartS)

const metricAggregate = (
  hours: readonly WeatherRouteHour[],
  value: (hour: WeatherRouteHour) => number | null,
): MetricAggregate => {
  let coveredS = 0
  let weightedTotal = 0
  let peak: number | null = null
  let coveredUntilS = 0
  for (const hour of hours) {
    const metric = value(hour)
    if (metric == null || !Number.isFinite(metric)) continue
    const startS = Math.max(coveredUntilS, hour.elapsedStartS)
    const durationS = Math.max(0, hour.elapsedEndS - startS)
    if (durationS <= 0) continue
    coveredUntilS = Math.max(coveredUntilS, hour.elapsedEndS)
    coveredS += durationS
    weightedTotal += metric * durationS
    peak = Math.max(peak ?? metric, metric)
  }
  return { coveredS, weightedTotal, peak }
}

const conditionAt = (
  hours: readonly WeatherRouteHour[],
  elapsedS: number,
): WeatherRouteHour | null =>
  hours.find(
    (hour, index) =>
      elapsedS >= hour.elapsedStartS &&
      (elapsedS < hour.elapsedEndS ||
        (index === hours.length - 1 && elapsedS <= hour.elapsedEndS + 1)),
  ) ?? null

const cumulativeSedAt = (hours: readonly WeatherRouteHour[], elapsedS: number): number => {
  let joules = 0
  for (const hour of hours) {
    if (hour.uvIndex == null || elapsedS <= hour.elapsedStartS) continue
    const endS = Math.min(elapsedS, hour.elapsedEndS)
    if (endS > hour.elapsedStartS) joules += hour.uvIndex * (endS - hour.elapsedStartS)
  }
  return round(joules / 4_000, 3)
}

const radians = (degrees: number): number => (degrees * Math.PI) / 180

const degrees = (radiansValue: number): number => (radiansValue * 180) / Math.PI

const haversineMeters = (left: [number, number], right: [number, number]): number => {
  const earthRadiusM = 6_371_000
  const deltaLatitude = radians(right[0] - left[0])
  const deltaLongitude = radians(right[1] - left[1])
  const latitudeA = radians(left[0])
  const latitudeB = radians(right[0])
  const a =
    Math.sin(deltaLatitude / 2) ** 2 +
    Math.cos(latitudeA) * Math.cos(latitudeB) * Math.sin(deltaLongitude / 2) ** 2
  return earthRadiusM * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
}

const bearingDegrees = (left: [number, number], right: [number, number]): number => {
  const latitudeA = radians(left[0])
  const latitudeB = radians(right[0])
  const deltaLongitude = radians(right[1] - left[1])
  const y = Math.sin(deltaLongitude) * Math.cos(latitudeB)
  const x =
    Math.cos(latitudeA) * Math.sin(latitudeB) -
    Math.sin(latitudeA) * Math.cos(latitudeB) * Math.cos(deltaLongitude)
  return (degrees(Math.atan2(y, x)) + 360) % 360
}

const routeLength = (input: ActivityEnvironmentInput): number =>
  Math.min(input.timeS.length, input.distanceM.length, input.latlng.length)

const windIntervals = (
  input: ActivityEnvironmentInput,
  hours: readonly WeatherRouteHour[],
): { intervals: (WindInterval | null)[]; validDurationS: number } => {
  const length = routeLength(input)
  const intervals: (WindInterval | null)[] = Array.from({ length }, () => null)
  let validDurationS = 0
  for (let index = 1; index < length; index += 1) {
    const startS = input.timeS[index - 1]
    const endS = input.timeS[index]
    const durationS = endS - startS
    const distanceDeltaM = input.distanceM[index] - input.distanceM[index - 1]
    if (
      !Number.isFinite(startS) ||
      !Number.isFinite(endS) ||
      !Number.isFinite(distanceDeltaM) ||
      durationS <= 0 ||
      durationS > 30 ||
      distanceDeltaM <= 0
    )
      continue
    const groundSpeedMps = distanceDeltaM / durationS
    if (groundSpeedMps < 3) continue
    const gpsDistanceM = haversineMeters(input.latlng[index - 1], input.latlng[index])
    if (gpsDistanceM > Math.max(250, distanceDeltaM * 3 + 50)) continue
    const weather = conditionAt(hours, startS + durationS / 2)
    if (weather?.windSpeedKph == null || weather.windDirectionDeg == null) continue
    const windSpeedMps = weather.windSpeedKph / 3.6
    const courseBearing = bearingDegrees(input.latlng[index - 1], input.latlng[index])
    const relativeBearing = radians(weather.windDirectionDeg - courseBearing)
    const headwindMps = windSpeedMps * Math.cos(relativeBearing)
    const crosswindMps = windSpeedMps * Math.sin(relativeBearing)
    const apparentMps = Math.hypot(groundSpeedMps + headwindMps, crosswindMps)
    const yawDeg = degrees(Math.atan2(crosswindMps, groundSpeedMps + headwindMps))
    intervals[index] = {
      startS,
      endS,
      durationS,
      groundSpeedKph: groundSpeedMps * 3.6,
      headwindKph: headwindMps * 3.6,
      crosswindKph: crosswindMps * 3.6,
      apparentAirSpeedKph: apparentMps * 3.6,
      yawDeg,
    }
    validDurationS += durationS
  }
  return { intervals, validDurationS }
}

interface MovingTelemetryDose {
  totalSed: number | null
  cumulativeSed: (number | null)[]
}

const uvSecondsForInterval = (
  hours: readonly WeatherRouteHour[],
  startS: number,
  endS: number,
): number | null => {
  let coveredUntilS = startS
  let weightedUvSeconds = 0
  for (const hour of hours) {
    if (hour.elapsedEndS <= coveredUntilS || hour.elapsedStartS >= endS) continue
    if (hour.elapsedStartS - coveredUntilS > 1 || hour.uvIndex == null) return null
    const overlapStartS = Math.max(coveredUntilS, startS, hour.elapsedStartS)
    const overlapEndS = Math.min(endS, hour.elapsedEndS)
    if (overlapEndS <= overlapStartS) continue
    weightedUvSeconds += hour.uvIndex * (overlapEndS - overlapStartS)
    coveredUntilS = overlapEndS
    if (endS - coveredUntilS <= 1) break
  }
  return endS - coveredUntilS <= 1 ? weightedUvSeconds : null
}

const movingTelemetryDose = (
  input: ActivityEnvironmentInput,
  hours: readonly WeatherRouteHour[],
): MovingTelemetryDose => {
  const length = routeLength(input)
  const cumulativeUvSeconds = Array.from({ length }, () => 0)
  let observedMovingS = 0
  let weightedUvSeconds = 0
  for (let index = 1; index < length; index += 1) {
    const startS = input.timeS[index - 1]
    const endS = input.timeS[index]
    const durationS = endS - startS
    const distanceDeltaM = input.distanceM[index] - input.distanceM[index - 1]
    if (
      !Number.isFinite(startS) ||
      !Number.isFinite(endS) ||
      !Number.isFinite(distanceDeltaM) ||
      durationS <= 0 ||
      distanceDeltaM <= 0
    ) {
      cumulativeUvSeconds[index] = weightedUvSeconds
      continue
    }
    const intervalUvSeconds = uvSecondsForInterval(hours, startS, endS)
    if (intervalUvSeconds == null)
      return { totalSed: null, cumulativeSed: Array.from({ length }, () => null) }
    observedMovingS += durationS
    weightedUvSeconds += intervalUvSeconds
    cumulativeUvSeconds[index] = weightedUvSeconds
  }
  if (observedMovingS <= 0 || input.movingTimeS <= 0)
    return { totalSed: null, cumulativeSed: Array.from({ length }, () => null) }
  const normalization = input.movingTimeS / observedMovingS / 4_000
  return {
    totalSed: round(weightedUvSeconds * normalization, 3),
    cumulativeSed: cumulativeUvSeconds.map(value => round(value * normalization, 3)),
  }
}

const nearestRouteIndex = (timeS: readonly number[], length: number, targetS: number): number => {
  let closest = 0
  let closestDistance = Number.POSITIVE_INFINITY
  for (let index = 0; index < length; index += 1) {
    const distance = Math.abs(timeS[index] - targetS)
    if (distance >= closestDistance) continue
    closest = index
    closestDistance = distance
  }
  return closest
}

const sampleIndices = (
  input: ActivityEnvironmentInput,
  hours: readonly WeatherRouteHour[],
  wind: readonly (WindInterval | null)[],
  maximum: number,
): number[] => {
  const length = routeLength(input)
  if (length === 0) return []
  const required = new Set([0, length - 1])
  for (const [hourIndex, hour] of hours.entries()) {
    required.add(nearestRouteIndex(input.timeS, length, hour.elapsedStartS))
    required.add(nearestRouteIndex(input.timeS, length, hour.elapsedEndS))
    const next = hours[hourIndex + 1]
    if (next && next.elapsedStartS - hour.elapsedEndS > 1)
      required.add(
        nearestRouteIndex(
          input.timeS,
          length,
          hour.elapsedEndS + (next.elapsedStartS - hour.elapsedEndS) / 2,
        ),
      )
  }
  for (let index = 1; index < wind.length; index += 1) {
    if ((wind[index - 1] == null) === (wind[index] == null)) continue
    required.add(index - 1)
    required.add(index)
  }
  const essential = [...required].sort((left, right) => left - right)
  if (essential.length >= maximum) {
    const selected = new Set([essential[0], essential.at(-1) ?? essential[0]])
    for (let slot = 1; slot < maximum - 1; slot += 1)
      selected.add(essential[Math.round((slot / (maximum - 1)) * (essential.length - 1))])
    return [...selected].sort((left, right) => left - right)
  }
  const budget = Math.max(0, maximum - required.size)
  const stride = budget > 0 ? Math.max(1, Math.ceil(length / budget)) : length
  for (let index = 0; index < length; index += stride) required.add(index)
  return [...required].sort((left, right) => left - right)
}

const summarizeWind = (
  input: ActivityEnvironmentInput,
  intervals: readonly (WindInterval | null)[],
  validDurationS: number,
  provenance: GardenEstimateProvenance,
): GardenApparentWindEstimate | null => {
  if (validDurationS <= 0) return null
  let headwindTimeS = 0
  let tailwindTimeS = 0
  let longestHeadwindS = 0
  let currentHeadwindS = 0
  let previousEndS: number | null = null
  let headwindTotal = 0
  let crosswindTotal = 0
  let groundSpeedTotal = 0
  let apparentSpeedTotal = 0
  let yawTotal = 0
  let maximumHeadwindKph = 0
  let maximumCrosswindKph = 0
  for (const interval of intervals) {
    if (!interval) {
      currentHeadwindS = 0
      previousEndS = null
      continue
    }
    if (interval.headwindKph > 0) {
      headwindTimeS += interval.durationS
      currentHeadwindS =
        previousEndS != null && interval.startS - previousEndS <= 1
          ? currentHeadwindS + interval.durationS
          : interval.durationS
      longestHeadwindS = Math.max(longestHeadwindS, currentHeadwindS)
    } else {
      if (interval.headwindKph < 0) tailwindTimeS += interval.durationS
      currentHeadwindS = 0
    }
    previousEndS = interval.endS
    headwindTotal += interval.headwindKph * interval.durationS
    crosswindTotal += interval.crosswindKph * interval.durationS
    groundSpeedTotal += interval.groundSpeedKph * interval.durationS
    apparentSpeedTotal += interval.apparentAirSpeedKph * interval.durationS
    yawTotal += interval.yawDeg * interval.durationS
    maximumHeadwindKph = Math.max(maximumHeadwindKph, interval.headwindKph)
    maximumCrosswindKph = Math.max(maximumCrosswindKph, Math.abs(interval.crosswindKph))
  }
  const averageGroundSpeedKph = groundSpeedTotal / validDurationS
  const averageApparentAirSpeedKph = apparentSpeedTotal / validDurationS
  const coveragePct = percentage(validDurationS, input.movingTimeS)
  return {
    ...provenance,
    formulaId: 'garden-apparent-wind-v1',
    summary: {
      headwindSharePct: round((headwindTimeS / validDurationS) * 100, 1),
      headwindTimeS: round(headwindTimeS, 1),
      tailwindTimeS: round(tailwindTimeS, 1),
      longestHeadwindS: round(longestHeadwindS, 1),
      averageHeadwindKph: round(headwindTotal / validDurationS, 1),
      averageCrosswindKph: round(crosswindTotal / validDurationS, 1),
      maximumHeadwindKph: round(maximumHeadwindKph, 1),
      maximumCrosswindKph: round(maximumCrosswindKph, 1),
      averageGroundSpeedKph: round(averageGroundSpeedKph, 1),
      averageApparentAirSpeedKph: round(averageApparentAirSpeedKph, 1),
      apparentAirRatio:
        averageGroundSpeedKph > 0
          ? round(averageApparentAirSpeedKph / averageGroundSpeedKph, 3)
          : 0,
      averageYawDeg: round(yawTotal / validDurationS, 1),
      coveragePct,
    },
    coverage: { windPct: coveragePct },
  }
}

export function buildActivityEnvironment(
  input: ActivityEnvironmentInput,
): ActivityEnvironmentResult {
  if (
    !Number.isFinite(input.elapsedTimeS) ||
    input.elapsedTimeS <= 0 ||
    input.weather.activityId !== input.activityId
  )
    return { environment: null, apparentWind: null }
  const hours = validRouteHours(input.weather, input.elapsedTimeS)
  if (hours.length === 0) return { environment: null, apparentWind: null }
  const weatherAggregate = metricAggregate(hours, () => 1)
  const uv = metricAggregate(hours, hour => hour.uvIndex)
  const temperature = metricAggregate(hours, hour => hour.temperatureC)
  const cloud = metricAggregate(hours, hour => hour.cloudCover)
  const daylight = metricAggregate(hours, hour =>
    hour.daylight == null ? null : Number(hour.daylight),
  )
  const uvComplete = input.elapsedTimeS - uv.coveredS <= 1
  const elapsedSed = uvComplete ? round(uv.weightedTotal / 4_000, 3) : null
  const movingDose = uvComplete
    ? movingTelemetryDose(input, hours)
    : { totalSed: null, cumulativeSed: Array.from({ length: routeLength(input) }, () => null) }
  const doseClocks: GardenUvDoseClocks = { elapsedSed, movingTelemetrySed: movingDose.totalSed }
  const baseProvenance: GardenEstimateProvenance = {
    source: 'garden-estimate',
    formulaId: 'garden-environment-v1',
    formulaVersion: 1,
    inputVersion: 'weatherkit-route-hour-v1+strava-stream-v1',
    normalizationVersion: 1,
    computedAt: input.computedAt,
    inputAsOf: input.weather.fetchedAt ?? 0,
    temporalSamplingModel: 'weatherkit-hourly-piecewise-constant',
    spatialSamplingModel: 'route-coordinate-nearest-hour-overlap-midpoint',
  }
  const wind = windIntervals(input, hours)
  const apparentWind = summarizeWind(input, wind.intervals, wind.validDurationS, baseProvenance)
  const indices = sampleIndices(input, hours, wind.intervals, 320)
  const samples = indices.map(index => {
    const elapsedS = Math.min(input.elapsedTimeS, Math.max(0, input.timeS[index]))
    const weather = conditionAt(hours, elapsedS)
    const windInterval = wind.intervals[index]
    return {
      elapsedS: round(elapsedS, 1),
      distanceKm: round(Math.max(0, input.distanceM[index]) / 1_000, 3),
      uvIndex: weather?.uvIndex ?? null,
      cumulativeSed: elapsedSed == null ? null : cumulativeSedAt(hours, elapsedS),
      cumulativeMovingTelemetrySed: movingDose.cumulativeSed[index] ?? null,
      ambientTemperatureC: weather?.temperatureC == null ? null : round(weather.temperatureC, 1),
      cloudCoverPct: weather?.cloudCover == null ? null : round(weather.cloudCover * 100, 1),
      headwindKph: windInterval == null ? null : round(windInterval.headwindKph, 1),
      crosswindKph: windInterval == null ? null : round(windInterval.crosswindKph, 1),
      apparentAirSpeedKph: windInterval == null ? null : round(windInterval.apparentAirSpeedKph, 1),
      yawDeg: windInterval == null ? null : round(windInterval.yawDeg, 1),
    } satisfies GardenEnvironmentSample
  })
  const coverage: GardenEnvironmentCoverage = {
    weatherPct: percentage(weatherAggregate.coveredS, input.elapsedTimeS),
    uvPct: percentage(uv.coveredS, input.elapsedTimeS),
    temperaturePct: percentage(temperature.coveredS, input.elapsedTimeS),
    cloudPct: percentage(cloud.coveredS, input.elapsedTimeS),
    daylightPct: percentage(daylight.coveredS, input.elapsedTimeS),
  }
  const environment: GardenEnvironmentEstimate = {
    ...baseProvenance,
    formulaId: 'garden-environment-v1',
    summary: {
      averageUvIndex: uv.coveredS > 0 ? round(uv.weightedTotal / uv.coveredS, 2) : null,
      peakUvIndex: uv.peak == null ? null : round(uv.peak, 1),
      uviHours: uvComplete ? round(uv.weightedTotal / 3_600, 3) : null,
      ambientSed: elapsedSed,
      averageAmbientTemperatureC:
        temperature.coveredS > 0
          ? round(temperature.weightedTotal / temperature.coveredS, 1)
          : null,
      averageCloudCoverPct:
        cloud.coveredS > 0 ? round((cloud.weightedTotal / cloud.coveredS) * 100, 1) : null,
      daylightCoveragePct: percentage(daylight.weightedTotal, input.elapsedTimeS),
      weatherCoveragePct: coverage.weatherPct,
      coveredDurationS: round(weatherAggregate.coveredS, 1),
      elapsedDurationS: input.elapsedTimeS,
    },
    doseClocks,
    coverage,
    samples,
    attribution: input.attribution,
  }
  return { environment, apparentWind }
}
