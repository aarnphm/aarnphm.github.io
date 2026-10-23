export const FRAME_BUDGET_MS = 1000 / 60

const REFRESH_PERIODS = [1000 / 120, 1000 / 90, 1000 / 60, 1000 / 30]

export const BUCKET_CAPACITY = 64

const CHART_FLOOR_MS = 4

const CHART_CEIL_MS = 250

const SMOOTHING_PASSES = 2

export interface RingBuffer {
  count: number
  cursor: number
  values: Float32Array
}

interface BucketSeries {
  count: number
  cursor: number
  jank: Uint8Array
  max: Float32Array
  mean: Float32Array
  min: Float32Array
  work: Float32Array
}

export interface OpenBucket {
  count: number
  jank: boolean
  max: number
  min: number
  start: number
  sum: number
  workCount: number
  workSum: number
}

export interface FrameSummary {
  fps: number
  p95: number
  slowRatio: number
}

export interface ChartSeries {
  jank: readonly boolean[]
  max: readonly number[]
  mean: readonly number[]
  min: readonly number[]
  work: readonly number[]
}

export const createRing = (capacity: number): RingBuffer => ({
  count: 0,
  cursor: 0,
  values: new Float32Array(capacity),
})

export const pushRing = (ring: RingBuffer, value: number): void => {
  ring.values[ring.cursor] = value
  ring.cursor = (ring.cursor + 1) % ring.values.length
  ring.count = Math.min(ring.count + 1, ring.values.length)
}

export const clearRing = (ring: RingBuffer): void => {
  ring.count = 0
  ring.cursor = 0
  ring.values.fill(0)
}

export const ringValues = (ring: RingBuffer): number[] => {
  const values: number[] = []
  const start = ring.count === ring.values.length ? ring.cursor : 0
  for (let index = 0; index < ring.count; index += 1)
    values.push(ring.values[(start + index) % ring.values.length])
  return values
}

export const createSeries = (capacity: number): BucketSeries => ({
  count: 0,
  cursor: 0,
  jank: new Uint8Array(capacity),
  max: new Float32Array(capacity),
  mean: new Float32Array(capacity),
  min: new Float32Array(capacity),
  work: new Float32Array(capacity),
})

export const pushSeries = (series: BucketSeries, bucket: OpenBucket): void => {
  const slot = series.cursor
  series.mean[slot] = bucket.sum / bucket.count
  series.min[slot] = bucket.min
  series.max[slot] = bucket.max
  series.work[slot] = bucket.workCount > 0 ? bucket.workSum / bucket.workCount : 0
  series.jank[slot] = bucket.jank ? 1 : 0
  series.cursor = (series.cursor + 1) % series.mean.length
  series.count = Math.min(series.count + 1, series.mean.length)
}

export const clearSeries = (series: BucketSeries): void => {
  series.count = 0
  series.cursor = 0
  series.jank.fill(0)
  series.max.fill(0)
  series.mean.fill(0)
  series.min.fill(0)
  series.work.fill(0)
}

export const seriesWindow = (series: BucketSeries): ChartSeries => {
  const capacity = series.mean.length
  const start = series.count === capacity ? series.cursor : 0
  const jank: boolean[] = []
  const max: number[] = []
  const mean: number[] = []
  const min: number[] = []
  const work: number[] = []
  for (let index = 0; index < series.count; index += 1) {
    const slot = (start + index) % capacity
    jank.push(series.jank[slot] === 1)
    max.push(series.max[slot])
    mean.push(series.mean[slot])
    min.push(series.min[slot])
    work.push(series.work[slot])
  }
  return { jank, max, mean, min, work }
}

export const summarizeFrameDurations = (
  values: readonly number[],
  frameBudget = FRAME_BUDGET_MS,
): FrameSummary => {
  if (values.length === 0) return { fps: 0, p95: 0, slowRatio: 0 }
  const sorted = [...values].sort((left, right) => left - right)
  const average = values.reduce((sum, value) => sum + value, 0) / values.length
  const p95Index = Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.95) - 1)
  const slow = values.reduce((count, value) => count + Number(value > frameBudget), 0)
  return {
    fps: average > 0 ? 1000 / average : 0,
    p95: sorted[p95Index],
    slowRatio: slow / values.length,
  }
}

export const estimateFrameBudget = (values: readonly number[]): number => {
  if (values.length < 12) return FRAME_BUDGET_MS
  const sorted = [...values].sort((left, right) => left - right)
  const fastest = sorted[Math.floor(sorted.length * 0.1)]
  let budget = FRAME_BUDGET_MS
  let closest = Number.POSITIVE_INFINITY
  for (const period of REFRESH_PERIODS) {
    const distance = Math.abs(period - fastest)
    if (distance >= closest) continue
    closest = distance
    budget = period
  }
  return budget
}

export const smoothWindow = (values: readonly number[], passes = SMOOTHING_PASSES): number[] => {
  let current = [...values]
  if (current.length < 3) return current
  for (let pass = 0; pass < passes; pass += 1) {
    const source = current
    current = source.map((value, index) => {
      const before = source[Math.max(0, index - 1)]
      const after = source[Math.min(source.length - 1, index + 1)]
      return (before + 2 * value + after) / 4
    })
  }
  return current
}

export const monotoneSlopes = (values: readonly number[]): number[] => {
  const length = values.length
  const slopes = Array.from<number>({ length }).fill(0)
  if (length < 2) return slopes
  const deltas = Array.from({ length: length - 1 }, (_, index) => values[index + 1] - values[index])
  slopes[0] = deltas[0]
  slopes[length - 1] = deltas[length - 2]
  for (let index = 1; index < length - 1; index += 1)
    slopes[index] =
      deltas[index - 1] * deltas[index] <= 0 ? 0 : (deltas[index - 1] + deltas[index]) / 2
  for (let index = 0; index < length - 1; index += 1) {
    if (deltas[index] === 0) {
      slopes[index] = 0
      slopes[index + 1] = 0
      continue
    }
    const alpha = slopes[index] / deltas[index]
    const beta = slopes[index + 1] / deltas[index]
    const magnitude = Math.hypot(alpha, beta)
    if (magnitude <= 3) continue
    const scale = 3 / magnitude
    slopes[index] = scale * alpha * deltas[index]
    slopes[index + 1] = scale * beta * deltas[index]
  }
  return slopes
}

export const frameChartRatio = (duration: number): number => {
  const floor = Math.log(CHART_FLOOR_MS)
  const span = Math.log(CHART_CEIL_MS) - floor
  const clamped = Math.min(CHART_CEIL_MS, Math.max(CHART_FLOOR_MS, duration))
  return (Math.log(clamped) - floor) / span
}

export const percentile95 = (ring: RingBuffer): number => {
  const values = ringValues(ring)
  if (values.length === 0) return 0
  values.sort((left, right) => left - right)
  return values[Math.min(values.length - 1, Math.ceil(values.length * 0.95) - 1)]
}
