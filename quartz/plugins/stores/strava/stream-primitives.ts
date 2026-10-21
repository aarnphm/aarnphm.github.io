import type { GarminStreams } from '../garmin'
import type { StravaStreams } from './raw-and-kinds'

export function round(value: number, dp: number): number {
  const f = 10 ** dp
  return Math.round(value * f) / f
}

export function haversineMeters(lat1: number, lng1: number, lat2: number, lng2: number): number {
  const dLat = (lat2 - lat1) * 111320
  const dLng = (lng2 - lng1) * 111320 * Math.cos((lat1 * Math.PI) / 180)
  return Math.hypot(dLat, dLng)
}

export function median(xs: number[]): number {
  if (xs.length === 0) return 0
  const s = [...xs].sort((p, q) => p - q)
  const m = Math.floor(s.length / 2)
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2
}

export function inferRouteHome(starts: [number, number][]): [number, number] | null {
  if (starts.length < 6) return null
  const seedLat = median(starts.map(p => p[0]))
  const seedLng = median(starts.map(p => p[1]))
  const near = starts.filter(p => haversineMeters(p[0], p[1], seedLat, seedLng) <= 200)
  if (near.length < 6) return null
  return [
    near.reduce((s, p) => s + p[0], 0) / near.length,
    near.reduce((s, p) => s + p[1], 0) / near.length,
  ]
}

export function cleanAltitude(alt: number[]): number[] {
  const n = alt.length
  const filled = alt.slice()
  let last = filled.find(x => x > 0.5) ?? 0
  for (let i = 0; i < n; i++) {
    if (filled[i] > 0.5) last = filled[i]
    else filled[i] = last
  }
  const w = 4
  if (n <= w * 2 + 1) return filled
  const out = filled.slice()
  for (let i = 0; i < n; i++) {
    let sum = 0
    let count = 0
    for (let j = Math.max(0, i - w); j <= Math.min(n - 1, i + w); j++) {
      sum += filled[j]
      count++
    }
    out[i] = sum / count
  }
  return out
}

function sampleIndices(lo: number, hi: number, maxPoints: number): number[] {
  if (hi < lo) return []
  const span = hi - lo + 1
  const stride = Math.max(1, Math.ceil(span / maxPoints))
  const idx: number[] = []
  for (let i = lo; i <= hi; i += stride) idx.push(i)
  if (idx[idx.length - 1] !== hi) idx.push(hi)
  return idx
}

export function sampleIndicesWithRequired(
  lo: number,
  hi: number,
  maxPoints: number,
  required: number[],
): number[] {
  if (hi < lo) return []
  const anchors = required.filter(index => index >= lo && index <= hi)
  return [...new Set([...sampleIndices(lo, hi, maxPoints), ...anchors])].sort((a, b) => a - b)
}

export function avgPos(arr: number[]): number | null {
  let sum = 0
  let count = 0
  for (const x of arr) {
    if (x > 0) {
      sum += x
      count++
    }
  }
  return count ? Math.round(sum / count) : null
}

export function maxPos(arr: number[]): number | null {
  let m = 0
  for (const x of arr) if (x > m) m = x
  return m > 0 ? Math.round(m) : null
}

export function roundPos(value: number | null | undefined): number | null {
  return value != null && Number.isFinite(value) && value > 0 ? Math.round(value) : null
}

export function hasPositive(values: number[] | undefined): boolean {
  return values?.some(value => value > 0) ?? false
}

export function positiveCount(values: number[] | undefined): number {
  return values?.filter(value => value > 0).length ?? 0
}

export const MAX_SPEED_WINDOW_S = 3
export const MAX_ACCEL_MPS2 = 3

export function zoneTimes(stream: number[], uppers: number[], countZero: boolean): number[] {
  const counts = Array.from({ length: uppers.length + 1 }, () => 0)
  for (const raw of stream) {
    if (raw <= 0 && !countZero) continue
    const v = raw > 0 ? raw : 0
    let z = uppers.length
    for (let i = 0; i < uppers.length; i++)
      if (v <= uppers[i]) {
        z = i
        break
      }
    counts[z]++
  }
  return counts
}

export function durationZoneTimes(
  stream: number[],
  uppers: number[],
  movingTimeS: number,
): number[] {
  const counts = zoneTimes(stream, uppers, false)
  const total = counts.reduce((sum, count) => sum + count, 0)
  if (total <= 0 || movingTimeS <= 0) return counts
  const scale = movingTimeS / total
  return counts.map(count => Math.round(count * scale))
}

export function powerHistogram(w: number[], countZero = true, bin = 25): number[] {
  let maxB = 0
  for (const raw of w) {
    if (raw <= 0 && !countZero) continue
    const b = Math.floor((raw > 0 ? raw : 0) / bin)
    if (b > maxB) maxB = b
  }
  const out = Array.from({ length: maxB + 1 }, () => 0)
  for (const raw of w) {
    if (raw <= 0 && !countZero) continue
    out[Math.floor((raw > 0 ? raw : 0) / bin)]++
  }
  return out
}

export function deriveHrBounds(hrmax: number): number[] {
  return [0.6, 0.7, 0.8, 0.9].map(p => Math.round(hrmax * p))
}

export function derivePowerBounds(ftp: number): number[] {
  return [0.55, 0.75, 0.9, 1.05, 1.2, 1.5].map(p => Math.round(ftp * p))
}

export function delta(
  garmin: number | null | undefined,
  strava: number | null | undefined,
): number | null {
  return garmin != null && strava != null ? Math.round(garmin - strava) : null
}

export function deltaFloat(
  garmin: number | null | undefined,
  strava: number | null | undefined,
  dp: number,
): number | null {
  return garmin != null && strava != null ? round(garmin - strava, dp) : null
}

export interface TimedMetricSample {
  elapsedS: number
  value: number
}

export interface TimedNullableMetricSample {
  elapsedS: number
  value: number | null
}

export function timedMetricAt(samples: TimedMetricSample[], elapsedS: number): number | null {
  if (samples.length === 0) return null
  if (elapsedS <= samples[0].elapsedS) return samples[0].value
  for (let index = 1; index < samples.length; index++) {
    const previous = samples[index - 1]
    const next = samples[index]
    if (elapsedS > next.elapsedS) continue
    const span = next.elapsedS - previous.elapsedS
    if (span <= 0) return next.value
    const fraction = (elapsedS - previous.elapsedS) / span
    return previous.value + (next.value - previous.value) * fraction
  }
  return samples[samples.length - 1].value
}

export const THERMAL_SAMPLE_MAX_DISTANCE_S = 90

export function timedNullableMetricAt(
  samples: TimedNullableMetricSample[],
  elapsedS: number,
  maxDistanceS: number,
): number | null {
  if (samples.length === 0) return null
  let low = 0
  let high = samples.length
  while (low < high) {
    const middle = (low + high) >>> 1
    if (samples[middle].elapsedS < elapsedS) low = middle + 1
    else high = middle
  }
  const previous = samples[low - 1]
  const next = samples[low]
  if (next?.elapsedS === elapsedS) return next.value
  if (!previous && !next) return null
  const nearest =
    previous && next
      ? elapsedS - previous.elapsedS <= next.elapsedS - elapsedS
        ? previous
        : next
      : (previous ?? next)
  if (!nearest || Math.abs(nearest.elapsedS - elapsedS) > maxDistanceS) return null
  if (
    !previous ||
    !next ||
    previous.value == null ||
    next.value == null ||
    next.elapsedS === previous.elapsedS ||
    next.elapsedS - previous.elapsedS > maxDistanceS * 2
  )
    return nearest.value
  const fraction = (elapsedS - previous.elapsedS) / (next.elapsedS - previous.elapsedS)
  return previous.value + (next.value - previous.value) * fraction
}

export interface TimedStreamAlignment {
  streams: StravaStreams | GarminStreams
  time: number[]
  distance: number[]
}

export function timedStreamAlignment(
  streams: StravaStreams | GarminStreams | undefined,
): TimedStreamAlignment | null {
  if (!streams || !('time' in streams) || !streams.time) return null
  const { time, distance } = streams
  if (time.length < 2 || distance.length !== time.length) return null
  let previousTime = -Infinity
  let previousDistance = 0
  const alignedDistance: number[] = []
  for (let index = 0; index < time.length; index++) {
    const elapsedS = time[index]
    const distanceM = distance[index]
    if (!Number.isFinite(elapsedS) || elapsedS < previousTime || !Number.isFinite(distanceM))
      return null
    previousTime = elapsedS
    previousDistance = Math.max(previousDistance, distanceM)
    alignedDistance.push(previousDistance)
  }
  return { streams, time, distance: alignedDistance }
}

export function nearestElapsedIndex(time: number[], elapsedS: number): number {
  let lo = 0
  let hi = time.length - 1
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2)
    if (time[mid] < elapsedS) lo = mid + 1
    else hi = mid
  }
  if (lo === 0) return 0
  return Math.abs(time[lo] - elapsedS) < Math.abs(time[lo - 1] - elapsedS) ? lo : lo - 1
}

export function validStreamIndex(value: number | null, length: number): value is number {
  return value != null && Number.isInteger(value) && value >= 0 && value < length
}

export function nullableRound(value: number | null, dp = 1): number | null {
  return value == null || !Number.isFinite(value) ? null : round(value, dp)
}

export function routeElapsedSeconds(
  streams: StravaStreams | GarminStreams,
  movingTimeS: number,
): number[] | null {
  const timed = timedStreamAlignment(streams)
  if (timed && timed.time.length === streams.latlng.length) return timed.time
  if (
    streams.latlng.length >= 2 &&
    Math.abs(streams.latlng.length - movingTimeS) / Math.max(1, movingTimeS) <= 0.15
  )
    return Array.from({ length: streams.latlng.length }, (_, index) => index)
  return null
}

export function localSpeedKph(time: number[], distance: number[]): number[] {
  const speedKph = Array.from({ length: time.length }, () => 0)
  const monotonicDistance: number[] = []
  let previousDistance = 0
  for (const value of distance) {
    previousDistance = Math.max(previousDistance, Number.isFinite(value) ? value : 0)
    monotonicDistance.push(previousDistance)
  }
  let windowStart = 0
  let previousMps = 0
  for (let index = 1; index < time.length; index++) {
    while (windowStart + 1 < index && time[index] - time[windowStart] > MAX_SPEED_WINDOW_S)
      windowStart++
    const elapsedS = time[index] - time[windowStart]
    const distanceM = monotonicDistance[index] - monotonicDistance[windowStart]
    const rawMps = elapsedS > 0 && distanceM > 0 ? distanceM / elapsedS : 0
    const sampleElapsedS = Math.max(0, Math.min(1, time[index] - time[index - 1]))
    previousMps = Math.min(rawMps, previousMps + MAX_ACCEL_MPS2 * sampleElapsedS)
    speedKph[index] = round(previousMps * 3.6, 1)
  }
  return speedKph
}

export function privateRouteBounds(
  latlng: [number, number][],
  home: [number, number] | null,
): [number, number] {
  let lo = 0
  let hi = latlng.length - 1
  if (home) {
    while (
      lo < latlng.length &&
      haversineMeters(latlng[lo][0], latlng[lo][1], home[0], home[1]) <= 200
    )
      lo++
    while (hi > lo && haversineMeters(latlng[hi][0], latlng[hi][1], home[0], home[1]) <= 200) hi--
  }
  return hi - lo >= 1 ? [lo, hi] : [0, latlng.length - 1]
}
