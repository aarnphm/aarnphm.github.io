export interface CyclingPowerPoint {
  elapsedS: number
  distanceKm: number
  elevationM: number | null
  power30sWatts: number | null
  power5mWatts: number | null
  cumulativePowerWatts: number | null
}

export interface CyclingPowerTrace {
  source: 'wahoo' | 'strava'
  terrainSource: 'wahoo' | 'strava' | 'garmin'
  method: 'recorded-power-average-v1'
  points: CyclingPowerPoint[]
}

export const CYCLING_POWER_MAX_POINTS = 1_202

interface PowerStreams {
  time: readonly number[]
  watts: readonly (number | null)[]
  distance?: readonly (number | null)[]
  altitude?: readonly (number | null)[]
}

const finite = (value: number | null | undefined): value is number =>
  value != null && Number.isFinite(value)
const rounded = (value: number): number => Math.round(value * 1_000) / 1_000

class PowerWindow {
  private intervals: { durationS: number; watts: number }[] = []
  private head = 0
  private energyJ = 0
  private durationS = 0

  constructor(private readonly windowS: number) {}

  reset(): void {
    this.intervals.length = 0
    this.head = 0
    this.energyJ = 0
    this.durationS = 0
  }

  add(durationS: number, watts: number): void {
    this.intervals.push({ durationS, watts })
    this.energyJ += durationS * watts
    this.durationS += durationS
    while (this.durationS > this.windowS) {
      const first = this.intervals[this.head]
      const removed = Math.min(first.durationS, this.durationS - this.windowS)
      this.energyJ -= removed * first.watts
      this.durationS -= removed
      first.durationS -= removed
      if (first.durationS <= 1e-9) this.head++
    }
  }

  average(): number | null {
    return this.durationS >= this.windowS - 1e-9
      ? rounded(Math.max(0, this.energyJ) / this.windowS)
      : null
  }
}

const traceFields: readonly (
  | 'elevationM'
  | 'power30sWatts'
  | 'power5mWatts'
  | 'cumulativePowerWatts'
)[] = ['elevationM', 'power30sWatts', 'power5mWatts', 'cumulativePowerWatts']

function samplePowerPoints(points: CyclingPowerPoint[]): CyclingPowerPoint[] {
  if (points.length <= CYCLING_POWER_MAX_POINTS) return points
  const selected = new Set([0, points.length - 1])
  const bucketCount = (CYCLING_POWER_MAX_POINTS - 2) / 10
  for (let bucket = 0; bucket < bucketCount; bucket++) {
    const start = Math.floor((bucket * points.length) / bucketCount)
    const end = Math.floor(((bucket + 1) * points.length) / bucketCount)
    selected.add(start)
    selected.add(end - 1)
    for (const field of traceFields) {
      let lowest: number | null = null
      let highest: number | null = null
      for (let index = start; index < end; index++) {
        const value = points[index][field]
        if (value == null) continue
        if (lowest == null || value < (points[lowest][field] ?? Infinity)) lowest = index
        if (highest == null || value > (points[highest][field] ?? -Infinity)) highest = index
      }
      if (lowest != null) selected.add(lowest)
      if (highest != null) selected.add(highest)
    }
  }
  let previous = -1
  return Array.from(selected)
    .sort((a, b) => a - b)
    .map(index => {
      const point = { ...points[index] }
      // A discarded missing sample must still interrupt its displayed line or elevation area.
      for (const field of traceFields)
        for (let cursor = previous + 1; cursor < index; cursor++)
          if (points[cursor][field] == null) {
            point[field] = null
            break
          }
      previous = index
      return point
    })
}

export function buildCyclingPowerTrace(input: {
  source: CyclingPowerTrace['source']
  terrainSource?: CyclingPowerTrace['terrainSource']
  streams: PowerStreams | undefined
  startOffsetS: number
  elapsedTimeS: number
}): CyclingPowerTrace | null {
  const { source, streams, startOffsetS, elapsedTimeS } = input
  if (
    !streams ||
    streams.time.length < 2 ||
    streams.time.length !== streams.watts.length ||
    !Number.isFinite(startOffsetS) ||
    !Number.isFinite(elapsedTimeS) ||
    elapsedTimeS <= 0 ||
    elapsedTimeS > 48 * 3_600 ||
    streams.time.some(
      (time, index) => !Number.isFinite(time) || (index > 0 && time <= streams.time[index - 1]),
    )
  )
    return null

  const intervals = streams.time.slice(1).map((time, index) => time - streams.time[index])
  intervals.sort((a, b) => a - b)
  // FIT records normally represent one second. A skipped record contributes at most one
  // sample period, so even a two-second timestamp jump preserves its paused second.
  const nominalPeriodS = Math.min(1, intervals[Math.floor((intervals.length - 1) / 2)])
  const continuityLimitS = nominalPeriodS * 1.5
  const shortWindow = new PowerWindow(30)
  const longWindow = new PowerWindow(300)
  const points: CyclingPowerPoint[] = []
  let energyJ = 0
  let recordedS = 0
  let distanceKm = 0
  let cursorS = 0
  let precedingElevationM: number | null = null
  const reset = (): void => {
    shortWindow.reset()
    longWindow.reset()
  }
  const addPoint = (elapsedS: number, elevationM: number | null, observed: boolean): void => {
    const point: CyclingPowerPoint = {
      elapsedS,
      distanceKm,
      elevationM,
      power30sWatts: observed ? shortWindow.average() : null,
      power5mWatts: observed ? longWindow.average() : null,
      cumulativePowerWatts: recordedS > 0 ? rounded(energyJ / recordedS) : null,
    }
    if (points.at(-1)?.elapsedS === elapsedS) points[points.length - 1] = point
    else points.push(point)
  }

  for (let index = 0; index < streams.time.length; index++) {
    const rawStartS = streams.time[index] + startOffsetS
    const nextS = (streams.time[index + 1] ?? Infinity) + startOffsetS
    const deltaS = nextS - rawStartS
    const endS = Math.min(
      elapsedTimeS,
      rawStartS + (deltaS <= continuityLimitS ? deltaS : nominalPeriodS),
    )
    const startS = Math.max(0, rawStartS)
    if (startS >= elapsedTimeS) break
    if (endS <= startS) continue
    if (startS > cursorS) {
      reset()
      addPoint(cursorS, precedingElevationM, false)
    }
    const distance = streams.distance?.[index]
    if (finite(distance) && distance >= 0) distanceKm = Math.max(distanceKm, distance / 1_000)
    const elevation = streams.altitude?.[index]
    const elevationM = finite(elevation) ? elevation : null
    precedingElevationM = elevationM
    const watts = streams.watts[index]
    const observed = finite(watts) && watts >= 0
    if (!observed) reset()
    addPoint(startS, elevationM, observed)
    if (observed) {
      const durationS = endS - startS
      energyJ += watts * durationS
      recordedS += durationS
      shortWindow.add(durationS, watts)
      longWindow.add(durationS, watts)
    }
    cursorS = endS
    if (endS === elapsedTimeS) {
      const finalDistance = streams.distance?.[index + 1]
      const finalElevation = nextS === endS ? streams.altitude?.[index + 1] : elevationM
      if (nextS === endS && finite(finalDistance) && finalDistance >= 0)
        distanceKm = Math.max(distanceKm, finalDistance / 1_000)
      addPoint(endS, finite(finalElevation) ? finalElevation : null, observed)
      break
    }
  }
  if (cursorS < elapsedTimeS) {
    addPoint(cursorS, null, false)
    addPoint(elapsedTimeS, null, false)
  }
  if (points.filter(point => point.cumulativePowerWatts != null).length < 2) return null
  return {
    source,
    terrainSource: input.terrainSource ?? source,
    method: 'recorded-power-average-v1',
    points: samplePowerPoints(points),
  }
}
