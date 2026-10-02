export const WALK_POWER_METHOD = 'minetti-walking-net-metabolic-v1'
export const WALK_POWER_WINDOW_SECONDS = 30

export interface WalkPowerInput {
  inputSource: 'strava' | 'garmin'
  weight: { kg: number; date: string; source: 'garmin' }
  streams: { time: readonly number[]; distance: readonly number[]; altitude: readonly number[] }
  elapsedTimeS: number
}

export interface WalkPowerEstimate {
  source: 'garden-estimate'
  inputSource: WalkPowerInput['inputSource']
  method: typeof WALK_POWER_METHOD
  kind: 'net-metabolic'
  weight: WalkPowerInput['weight']
  windowSeconds: typeof WALK_POWER_WINDOW_SECONDS
  averageWatts: number
  points: { elapsedS: number; distanceKm: number; watts: number | null }[]
}

// Minetti et al. (2002), doi:10.1152/japplphysiol.01177.2001.
// Minimum net walking cost in J/kg/m, measured over gradients from -0.45 to 0.45.
// This estimates metabolic demand above rest; it is distinct from mechanical power.
const walkingCost = (grade: number): number =>
  280.5 * grade ** 5 -
  58.7 * grade ** 4 -
  76.8 * grade ** 3 +
  51.9 * grade ** 2 +
  19.6 * grade +
  2.5

const rounded = (value: number): number => Math.round(value * 1000) / 1000

export function buildWalkPowerEstimate(input: WalkPowerInput): WalkPowerEstimate | null {
  const { streams, weight, elapsedTimeS } = input
  const { time, distance, altitude } = streams
  if (
    !Number.isFinite(weight.kg) ||
    weight.kg <= 0 ||
    !Number.isFinite(elapsedTimeS) ||
    elapsedTimeS <= 0 ||
    time.length < 3 ||
    distance.length !== time.length ||
    altitude.length !== time.length ||
    time.some(
      (value, index) =>
        !Number.isFinite(value) ||
        value < 0 ||
        value > elapsedTimeS ||
        (index > 0 && value <= time[index - 1]),
    ) ||
    distance.some(
      (value, index) =>
        !Number.isFinite(value) || value < 0 || (index > 0 && value < distance[index - 1]),
    )
  )
    return null

  let anchor = 0
  let energyJ = 0
  let observedSeconds = 0
  let usable = 0
  const points: WalkPowerEstimate['points'] = time.map((elapsedS, index) => {
    const point: WalkPowerEstimate['points'][number] = {
      elapsedS,
      distanceKm: distance[index] / 1000,
      watts: null,
    }
    if (index === 0) return point
    const seconds = elapsedS - time[index - 1]
    const travelM = distance[index] - distance[index - 1]
    if (
      seconds > WALK_POWER_WINDOW_SECONDS ||
      travelM / seconds > 3 ||
      !Number.isFinite(altitude[index]) ||
      !Number.isFinite(altitude[index - 1])
    ) {
      anchor = index
      return point
    }
    if (travelM === 0) {
      // A recorded stop must remain zero instead of retaining earlier walking effort.
      point.watts = 0
      anchor = index
    } else {
      while (anchor < index - 1 && time[anchor] < elapsedS - WALK_POWER_WINDOW_SECONDS) anchor++
      const windowDistanceM = distance[index] - distance[anchor]
      const grade = (altitude[index] - altitude[anchor]) / windowDistanceM
      if (!Number.isFinite(grade) || Math.abs(grade) > 0.45) {
        anchor = index
        return point
      }
      const speedMps = windowDistanceM / (elapsedS - time[anchor])
      const watts = weight.kg * speedMps * walkingCost(grade)
      if (!Number.isFinite(watts) || watts < 0) {
        anchor = index
        return point
      }
      point.watts = rounded(watts)
    }
    energyJ += point.watts * seconds
    observedSeconds += seconds
    usable++
    return point
  })
  if (usable < 2 || observedSeconds === 0) return null
  return {
    source: 'garden-estimate',
    inputSource: input.inputSource,
    method: WALK_POWER_METHOD,
    kind: 'net-metabolic',
    weight: { ...weight },
    windowSeconds: WALK_POWER_WINDOW_SECONDS,
    averageWatts: rounded(energyJ / observedSeconds),
    points,
  }
}

export function walkPowerAt(
  estimate: WalkPowerEstimate | null | undefined,
  elapsedS: number,
): number | null {
  const points = estimate?.points
  if (!points?.length || !Number.isFinite(elapsedS)) return null
  let lo = 0
  let hi = points.length - 1
  if (elapsedS < points[lo].elapsedS || elapsedS > points[hi].elapsedS) return null
  while (lo < hi) {
    const middle = Math.ceil((lo + hi) / 2)
    if (points[middle].elapsedS <= elapsedS) lo = middle
    else hi = middle - 1
  }
  const before = points[lo]
  if (before.elapsedS === elapsedS) return before.watts
  const after = points[lo + 1]
  if (
    !after ||
    before.watts == null ||
    after.watts == null ||
    after.elapsedS - before.elapsedS > WALK_POWER_WINDOW_SECONDS
  )
    return null
  const fraction = (elapsedS - before.elapsedS) / (after.elapsedS - before.elapsedS)
  return before.watts + fraction * (after.watts - before.watts)
}
