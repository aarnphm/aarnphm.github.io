import type {
  PublicSurfaceCurrentEstimate,
  SurfaceCurrentEstimate,
  SurfaceCurrentSample,
} from './types'
import { stravaId } from '../strava-id'
import { isRecord } from '../type-guards'

export const HOUR_MS = 3_600_000

export const MAX_SPEED_MPS = 10

export const MAX_SAMPLES = 512

export const numberIn = (value: unknown, min: number, max: number): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max

const nullableNumber = (value: unknown, min: number, max: number): value is number | null =>
  value === null || numberIn(value, min, max)

export const timestamp = (value: unknown): value is string =>
  typeof value === 'string' &&
  /^\d{4}-\d{2}-\d{2}T/.test(value) &&
  Number.isFinite(Date.parse(value))

export const direction = (u: number, v: number): number | null =>
  Math.hypot(u, v) <= 1e-12 ? null : ((Math.atan2(u, v) * 180) / Math.PI + 360) % 360

const hasOnlyKeys = (value: Record<string, unknown>, keys: readonly string[]): boolean =>
  Object.keys(value).every(key => keys.includes(key))

export function sourceUrlIsValid(value: unknown): value is string {
  return (
    typeof value === 'string' &&
    /^https:\/\/(?:opendap\.co-ops\.nos\.noaa\.gov\/thredds\/dodsC\/NOAA\/LOOFS\/MODELS\/\d{4}\/\d{2}\/\d{2}|www\.ncei\.noaa\.gov\/thredds\/dodsC\/model-loofs-files\/\d{4}\/\d{2})\/loofs\.t(?:00|06|12|18)z\.\d{8}\.fields\.n00[0-6]\.nc\.ascii$/.test(
      value,
    )
  )
}

function parseSample(
  value: unknown,
  startMs: number,
  durationS: number,
): SurfaceCurrentSample | null {
  if (
    !isRecord(value) ||
    !hasOnlyKeys(value, [
      'elapsedS',
      'speedMps',
      'directionDeg',
      'uMps',
      'vMps',
      'element',
      'validTime',
      'cycleTime',
      'sourceUrl',
    ]) ||
    !numberIn(value.elapsedS, 0, durationS) ||
    !nullableNumber(value.speedMps, 0, MAX_SPEED_MPS) ||
    !nullableNumber(value.directionDeg, 0, 360 - Number.EPSILON) ||
    !nullableNumber(value.uMps, -MAX_SPEED_MPS, MAX_SPEED_MPS) ||
    !nullableNumber(value.vMps, -MAX_SPEED_MPS, MAX_SPEED_MPS)
  )
    return null
  const { elapsedS, speedMps, directionDeg, uMps, vMps } = value
  if (speedMps === null) {
    if (
      [
        directionDeg,
        uMps,
        vMps,
        value.element,
        value.validTime,
        value.cycleTime,
        value.sourceUrl,
      ].some(item => item !== null)
    )
      return null
    return {
      elapsedS,
      speedMps,
      directionDeg,
      uMps,
      vMps,
      element: null,
      validTime: null,
      cycleTime: null,
      sourceUrl: null,
    }
  }
  if (
    uMps === null ||
    vMps === null ||
    !numberIn(value.element, 0, 100_000) ||
    !Number.isInteger(value.element) ||
    !timestamp(value.validTime) ||
    !timestamp(value.cycleTime) ||
    !sourceUrlIsValid(value.sourceUrl)
  )
    return null
  const expectedDirection = direction(uMps, vMps)
  const validMs = Date.parse(value.validTime),
    cycleMs = Date.parse(value.cycleTime)
  const sampleMs = startMs + elapsedS * 1_000
  if (
    Math.abs(speedMps - Math.hypot(uMps, vMps)) > 1e-6 ||
    (expectedDirection === null
      ? directionDeg !== null
      : directionDeg === null || Math.abs(expectedDirection - directionDeg) > 1e-6) ||
    validMs % HOUR_MS !== 0 ||
    cycleMs % (6 * HOUR_MS) !== 0 ||
    cycleMs < validMs ||
    cycleMs - validMs > 6 * HOUR_MS ||
    sampleMs < validMs ||
    sampleMs > validMs + HOUR_MS
  )
    return null
  return {
    elapsedS,
    speedMps,
    directionDeg,
    uMps,
    vMps,
    element: value.element,
    validTime: value.validTime,
    cycleTime: value.cycleTime,
    sourceUrl: value.sourceUrl,
  }
}

export function parsePublicSurfaceCurrentEstimate(
  value: unknown,
): PublicSurfaceCurrentEstimate | null {
  if (
    !isRecord(value) ||
    !hasOnlyKeys(value, [
      'source',
      'sourceKind',
      'formulaId',
      'formulaVersion',
      'activityId',
      'start',
      'end',
      'computedAt',
      'summary',
      'samples',
      'spatialSamplingModel',
      'temporalSamplingModel',
      'layer',
    ]) ||
    value.source !== 'noaa-loofs' ||
    value.sourceKind !== 'modeled' ||
    value.formulaId !== 'garden-surface-current-v1' ||
    value.formulaVersion !== 1 ||
    value.spatialSamplingModel !== 'containing-element' ||
    value.temporalSamplingModel !== 'hourly-linear-vector' ||
    value.layer !== 0 ||
    stravaId(value.activityId) === null ||
    !timestamp(value.start) ||
    !timestamp(value.end) ||
    !numberIn(value.computedAt, 0, Number.MAX_SAFE_INTEGER) ||
    !isRecord(value.summary) ||
    !Array.isArray(value.samples) ||
    value.samples.length < 2 ||
    value.samples.length > MAX_SAMPLES
  )
    return null
  const startMs = Date.parse(value.start),
    durationS = (Date.parse(value.end) - startMs) / 1_000
  const summary = value.summary
  if (
    durationS <= 0 ||
    durationS > 43_200 ||
    value.computedAt < startMs ||
    !hasOnlyKeys(summary, [
      'averageSpeedMps',
      'averageDirectionDeg',
      'coveragePct',
      'coveredDurationS',
      'elapsedDurationS',
    ]) ||
    !nullableNumber(summary.averageSpeedMps, 0, MAX_SPEED_MPS) ||
    !nullableNumber(summary.averageDirectionDeg, 0, 360 - Number.EPSILON) ||
    !numberIn(summary.coveragePct, 0, 100) ||
    !numberIn(summary.coveredDurationS, 0, durationS) ||
    !numberIn(summary.elapsedDurationS, durationS, durationS) ||
    Math.abs(summary.coveragePct - (summary.coveredDurationS / durationS) * 100) > 1e-4 ||
    (summary.coveredDurationS === 0
      ? summary.averageSpeedMps !== null || summary.averageDirectionDeg !== null
      : summary.averageSpeedMps === null)
  )
    return null
  const samples: SurfaceCurrentSample[] = []
  for (const raw of value.samples) {
    const sample = parseSample(raw, startMs, durationS)
    if (!sample || (samples.length > 0 && sample.elapsedS <= samples[samples.length - 1].elapsedS))
      return null
    samples.push(sample)
  }
  if (
    samples[0].elapsedS !== 0 ||
    samples[samples.length - 1].elapsedS !== durationS ||
    (summary.coveredDurationS > 0 && samples.filter(sample => sample.speedMps !== null).length < 2)
  )
    return null
  return {
    source: 'noaa-loofs',
    sourceKind: 'modeled',
    formulaId: 'garden-surface-current-v1',
    formulaVersion: 1,
    activityId: stravaId(value.activityId)!,
    start: value.start,
    end: value.end,
    computedAt: value.computedAt,
    summary: {
      averageSpeedMps: summary.averageSpeedMps,
      averageDirectionDeg: summary.averageDirectionDeg,
      coveragePct: summary.coveragePct,
      coveredDurationS: summary.coveredDurationS,
      elapsedDurationS: summary.elapsedDurationS,
    },
    samples,
    spatialSamplingModel: 'containing-element',
    temporalSamplingModel: 'hourly-linear-vector',
    layer: 0,
  }
}

export function parseSurfaceCurrentEstimate(value: unknown): SurfaceCurrentEstimate | null {
  if (
    !isRecord(value) ||
    typeof value.routeFingerprint !== 'string' ||
    value.routeFingerprint.length === 0 ||
    value.routeFingerprint.length > 256
  )
    return null
  const { routeFingerprint, ...publicValue } = value
  const parsed = parsePublicSurfaceCurrentEstimate(publicValue)
  return parsed ? { ...parsed, routeFingerprint } : null
}
