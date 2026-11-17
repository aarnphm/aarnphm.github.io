import type { WeatherRouteHour } from '../../plugins/stores/weather'
import type { ActivityEnvironmentInput } from './model'
import type { WindInterval } from './wind'
import { round, routeLength } from './model'

export const cumulativeSedAt = (hours: readonly WeatherRouteHour[], elapsedS: number): number => {
  let joules = 0
  for (const hour of hours) {
    if (hour.uvIndex == null || elapsedS <= hour.elapsedStartS) continue
    const endS = Math.min(elapsedS, hour.elapsedEndS)
    if (endS > hour.elapsedStartS) joules += hour.uvIndex * (endS - hour.elapsedStartS)
  }
  return round(joules / 4_000, 3)
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

export const movingTelemetryDose = (
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

export const sampleIndices = (
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
