import type { WeatherRouteHour } from '../../plugins/stores/weather'
import type {
  ActivityEnvironmentInput,
  GardenApparentWindEstimate,
  GardenEstimateProvenance,
} from './model'
import { conditionAt, percentage, round, routeLength, validWindSpeedKph } from './model'

export interface WindInterval {
  startS: number
  endS: number
  durationS: number
  groundSpeedKph: number
  headwindKph: number
  crosswindKph: number
  apparentAirSpeedKph: number
  yawDeg: number
}

// WeatherKit reports wind at the 10 m anemometer height. A neutral log profile gives
// ln(z / z0) / ln(10 / z0). Riders and runners: drag near z = 1 m, suburban z0 = 0.4 m, 0.285;
// myWindsock applies a fixed 0.2866, so source wind speed remains the main difference.
// Swimmers: head and recovering arm near z = 0.2 m over open water, z0 = 0.0002 m, 0.638.
const LAND_WIND_FACTOR = Math.log(1 / 0.4) / Math.log(10 / 0.4)

const WATER_WIND_FACTOR = Math.log(0.2 / 0.0002) / Math.log(10 / 0.0002)

// Excludes stops, where the course bearing is undefined. 1 m/s keeps running and walking pace;
// open-water swims move at 0.5-0.6 m/s, so they use a lower floor.
const MINIMUM_GROUND_SPEED_MPS = 1

const MINIMUM_SWIM_SPEED_MPS = 0.3

// Headwind sections follow the 60 s centred mean of the along-track component, and continue
// across stops up to the bridge. This reproduces myWindsock's longest headwind within 4% on
// four of six captured activities; strict one-second runs were 2-3x shorter.
const HEADWIND_SMOOTHING_S = 60

const HEADWIND_BRIDGE_S = 120

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

export const windIntervals = (
  input: ActivityEnvironmentInput,
  hours: readonly WeatherRouteHour[],
): { intervals: (WindInterval | null)[]; validDurationS: number } => {
  const length = routeLength(input)
  const intervals: (WindInterval | null)[] = Array.from({ length }, () => null)
  let validDurationS = 0
  const minimumSpeedMps = input.openWater ? MINIMUM_SWIM_SPEED_MPS : MINIMUM_GROUND_SPEED_MPS
  const windFactor = input.openWater ? WATER_WIND_FACTOR : LAND_WIND_FACTOR
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
    if (groundSpeedMps < minimumSpeedMps) continue
    const gpsDistanceM = haversineMeters(input.latlng[index - 1], input.latlng[index])
    if (gpsDistanceM > Math.max(250, distanceDeltaM * 3 + 50)) continue
    const weather = conditionAt(hours, startS + durationS / 2)
    const windSpeedKph = validWindSpeedKph(weather?.windSpeedKph)
    if (windSpeedKph == null || weather?.windDirectionDeg == null) continue
    const windSpeedMps = (windSpeedKph / 3.6) * windFactor
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

const longestHeadwindSection = (valid: readonly WindInterval[]): number => {
  let longestS = 0
  let sectionStartS: number | null = null
  let previousEndS: number | null = null
  let low = 0
  let high = 0
  let weightedHeadwind = 0
  let weightS = 0
  for (const interval of valid) {
    const middleS = (interval.startS + interval.endS) / 2
    while (
      high < valid.length &&
      (valid[high].startS + valid[high].endS) / 2 <= middleS + HEADWIND_SMOOTHING_S / 2
    ) {
      weightedHeadwind += valid[high].headwindKph * valid[high].durationS
      weightS += valid[high].durationS
      high += 1
    }
    while ((valid[low].startS + valid[low].endS) / 2 < middleS - HEADWIND_SMOOTHING_S / 2) {
      weightedHeadwind -= valid[low].headwindKph * valid[low].durationS
      weightS -= valid[low].durationS
      low += 1
    }
    if (weightedHeadwind / weightS > 0) {
      if (
        sectionStartS == null ||
        previousEndS == null ||
        interval.startS - previousEndS > HEADWIND_BRIDGE_S
      )
        sectionStartS = interval.startS
      longestS = Math.max(longestS, interval.endS - sectionStartS)
    } else sectionStartS = null
    previousEndS = interval.endS
  }
  return longestS
}

export const summarizeWind = (
  input: ActivityEnvironmentInput,
  intervals: readonly (WindInterval | null)[],
  validDurationS: number,
  provenance: GardenEstimateProvenance,
): GardenApparentWindEstimate | null => {
  if (validDurationS <= 0) return null
  let headwindTimeS = 0
  let tailwindTimeS = 0
  let headwindIntoTotal = 0
  let crosswindMagnitudeTotal = 0
  let groundSpeedTotal = 0
  let apparentSpeedTotal = 0
  let yawTotal = 0
  let maximumHeadwindKph = 0
  let maximumCrosswindKph = 0
  const valid = intervals.filter((interval): interval is WindInterval => interval != null)
  for (const interval of valid) {
    if (interval.headwindKph > 0) {
      headwindTimeS += interval.durationS
      headwindIntoTotal += interval.headwindKph * interval.durationS
    } else if (interval.headwindKph < 0) tailwindTimeS += interval.durationS
    crosswindMagnitudeTotal += Math.abs(interval.crosswindKph) * interval.durationS
    groundSpeedTotal += interval.groundSpeedKph * interval.durationS
    apparentSpeedTotal += interval.apparentAirSpeedKph * interval.durationS
    yawTotal += interval.yawDeg * interval.durationS
    maximumHeadwindKph = Math.max(maximumHeadwindKph, interval.headwindKph)
    maximumCrosswindKph = Math.max(maximumCrosswindKph, Math.abs(interval.crosswindKph))
  }
  const longestHeadwindS = longestHeadwindSection(valid)
  const averageGroundSpeedKph = groundSpeedTotal / validDurationS
  const averageApparentAirSpeedKph = apparentSpeedTotal / validDurationS
  const coveragePct = percentage(validDurationS, input.movingTimeS)
  return {
    ...provenance,
    formulaId: 'garden-apparent-wind-v3',
    summary: {
      headwindSharePct: round((headwindTimeS / validDurationS) * 100, 1),
      headwindTimeS: round(headwindTimeS, 1),
      tailwindTimeS: round(tailwindTimeS, 1),
      longestHeadwindS: round(longestHeadwindS, 1),
      averageHeadwindWhileIntoKph:
        headwindTimeS > 0 ? round(headwindIntoTotal / headwindTimeS, 1) : 0,
      averageCrosswindMagnitudeKph: round(crosswindMagnitudeTotal / validDurationS, 1),
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
