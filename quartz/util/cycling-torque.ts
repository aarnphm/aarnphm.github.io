import type { WahooStreams } from '../plugins/stores/wahoo'

export interface CyclingMechanics {
  source: 'wahoo'
  method: 'power-cadence-v1'
  averageTorqueNm: number
  averageCadenceRpm: number
  coverage: number
  observedSeconds: number
}

export interface CyclingTorqueSample {
  elapsedS: number
  durationS: number
  watts: number | null
  cadenceRpm: number | null
  torqueNm: number | null
}

export interface CyclingTorquePoint extends CyclingTorqueSample {
  distanceKm: number
}

export interface CyclingTorqueCell {
  cadenceRpm: number
  torqueNm: number
  seconds: number
}

export interface CyclingTorqueTrace {
  source: 'wahoo'
  method: 'power-cadence-v1'
  summary: CyclingMechanics
  points: CyclingTorquePoint[]
  cells: CyclingTorqueCell[]
}

export const crankTorqueNm = (
  watts: number | null | undefined,
  cadenceRpm: number | null | undefined,
): number | null =>
  watts != null &&
  Number.isFinite(watts) &&
  watts >= 0 &&
  cadenceRpm != null &&
  Number.isFinite(cadenceRpm) &&
  cadenceRpm > 0
    ? (60 * watts) / (2 * Math.PI * cadenceRpm)
    : null

export function cyclingTorqueSamples(
  streams: Pick<WahooStreams, 'time' | 'watts' | 'cadence'> | undefined,
  startOffsetS: number,
  elapsedTimeS: number,
): CyclingTorqueSample[] {
  if (
    !streams ||
    !Number.isFinite(startOffsetS) ||
    !Number.isFinite(elapsedTimeS) ||
    elapsedTimeS <= 0 ||
    streams.time.length !== streams.watts.length ||
    streams.time.length !== streams.cadence.length ||
    streams.time.some((time, i) => !Number.isFinite(time) || (i > 0 && time <= streams.time[i - 1]))
  )
    return []
  return streams.time.flatMap((time, i) => {
    const elapsedS = time + startOffsetS
    // A recording pause never extends the final sample across the missing interval.
    const end = Math.min(
      elapsedTimeS,
      elapsedS + Math.min(1, (streams.time[i + 1] ?? time + 1) - time),
    )
    const start = Math.max(0, elapsedS)
    if (end <= start) return []
    const watts = streams.watts[i]
    const cadenceRpm = streams.cadence[i]
    return [
      {
        elapsedS: start,
        durationS: end - start,
        watts: watts != null && Number.isFinite(watts) && watts >= 0 ? watts : null,
        cadenceRpm:
          cadenceRpm != null && Number.isFinite(cadenceRpm) && cadenceRpm >= 0 ? cadenceRpm : null,
        torqueNm: crankTorqueNm(watts, cadenceRpm),
      },
    ]
  })
}

export function cyclingMechanics(
  samples: readonly CyclingTorqueSample[],
  startElapsedS: number,
  endElapsedS: number,
): CyclingMechanics | null {
  if (
    !Number.isFinite(startElapsedS) ||
    !Number.isFinite(endElapsedS) ||
    endElapsedS <= startElapsedS
  )
    return null
  let observedSeconds = 0
  let torque = 0
  let cadence = 0
  for (const sample of samples) {
    if (sample.elapsedS >= endElapsedS) break
    if (sample.torqueNm == null || sample.cadenceRpm == null) continue
    const seconds = Math.max(
      0,
      Math.min(endElapsedS, sample.elapsedS + sample.durationS) -
        Math.max(startElapsedS, sample.elapsedS),
    )
    observedSeconds += seconds
    torque += sample.torqueNm * seconds
    cadence += sample.cadenceRpm * seconds
  }
  if (observedSeconds === 0) return null
  return {
    source: 'wahoo',
    method: 'power-cadence-v1',
    averageTorqueNm: Math.round((torque / observedSeconds) * 100) / 100,
    averageCadenceRpm: Math.round((cadence / observedSeconds) * 10) / 10,
    coverage: Math.min(1, observedSeconds / (endElapsedS - startElapsedS)),
    observedSeconds,
  }
}

export function buildCyclingTorqueTrace(
  samples: readonly CyclingTorqueSample[],
  elapsedTimeS: number,
  route: readonly { elapsedS: number; d: number }[],
): CyclingTorqueTrace | null {
  const summary = cyclingMechanics(samples, 0, elapsedTimeS)
  if (!summary || samples.filter(sample => sample.torqueNm != null).length < 2) return null
  const stride = Math.max(1, Math.ceil(elapsedTimeS / 900))
  const buckets = new Map<number, CyclingTorqueSample[]>()
  const cells = new Map<string, CyclingTorqueCell>()
  for (const sample of samples) {
    const end = sample.elapsedS + sample.durationS
    for (let index = Math.floor(sample.elapsedS / stride); index * stride < end; index++) {
      const bucket = buckets.get(index) ?? []
      const start = Math.max(sample.elapsedS, index * stride)
      bucket.push({
        ...sample,
        elapsedS: start,
        durationS: Math.min(end, (index + 1) * stride) - start,
      })
      buckets.set(index, bucket)
    }
    if (sample.torqueNm == null || sample.cadenceRpm == null) continue
    const cadenceRpm = Math.floor(sample.cadenceRpm / 5) * 5
    const torqueNm = Math.floor(sample.torqueNm / 5) * 5
    const key = `${cadenceRpm}:${torqueNm}`
    const cell = cells.get(key) ?? { cadenceRpm, torqueNm, seconds: 0 }
    cell.seconds += sample.durationS
    cells.set(key, cell)
  }
  let routeIndex = 0
  const points: CyclingTorquePoint[] = []
  for (let index = 0; index * stride < elapsedTimeS; index++) {
    const elapsedS = index * stride
    const durationS = Math.min(stride, elapsedTimeS - elapsedS)
    const bucket = buckets.get(index) ?? []
    const mechanics = cyclingMechanics(bucket, elapsedS, elapsedS + durationS)
    const valid = mechanics != null && mechanics.coverage >= 0.8
    while (routeIndex + 1 < route.length && route[routeIndex + 1].elapsedS <= elapsedS) routeIndex++
    const left = route[routeIndex]
    const right = route[routeIndex + 1] ?? left
    const fraction =
      left && right.elapsedS > left.elapsedS
        ? Math.max(0, Math.min(1, (elapsedS - left.elapsedS) / (right.elapsedS - left.elapsedS)))
        : 0
    const observed = bucket.filter(sample => sample.torqueNm != null && sample.watts != null)
    const watts = valid
      ? observed.reduce((sum, sample) => sum + (sample.watts ?? 0) * sample.durationS, 0) /
        mechanics.observedSeconds
      : null
    points.push({
      elapsedS,
      durationS,
      distanceKm: left ? left.d + (right.d - left.d) * fraction : 0,
      watts,
      cadenceRpm: valid ? mechanics.averageCadenceRpm : null,
      torqueNm: valid ? mechanics.averageTorqueNm : null,
    })
  }
  return {
    source: 'wahoo',
    method: 'power-cadence-v1',
    summary,
    points,
    cells: [...cells.values()],
  }
}

export function cyclingTorqueDensity(cells: readonly CyclingTorqueCell[]): {
  limitNm: number
  cells: CyclingTorqueCell[]
} {
  const sorted = [...cells].sort((a, b) => a.torqueNm - b.torqueNm)
  const total = sorted.reduce((sum, cell) => sum + cell.seconds, 0)
  let cumulative = 0
  let limitNm = 60
  for (const cell of sorted) {
    cumulative += cell.seconds
    limitNm = Math.max(60, Math.ceil((cell.torqueNm + 5) / 10) * 10)
    if (cumulative >= total * 0.99) break
  }
  const display = new Map<string, CyclingTorqueCell>()
  for (const cell of cells) {
    const torqueNm = Math.min(cell.torqueNm, limitNm)
    const key = `${cell.cadenceRpm}:${torqueNm}`
    const bucket = display.get(key) ?? { cadenceRpm: cell.cadenceRpm, torqueNm, seconds: 0 }
    bucket.seconds += cell.seconds
    display.set(key, bucket)
  }
  return {
    limitNm,
    cells: [...display.values()].sort(
      (a, b) => a.cadenceRpm - b.cadenceRpm || a.torqueNm - b.torqueNm,
    ),
  }
}
