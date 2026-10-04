import type { PowerCurvePoint } from '../plugins/stores/strava'

interface PowerCurveActivitySpan {
  end: number
  activityId: number
  activityDate: string
}

export interface PowerCurveSeries {
  readonly durations: readonly number[]
  readonly watts: readonly number[]
  readonly offset: number
  readonly indices?: readonly number[]
  readonly activities: readonly PowerCurveActivitySpan[]
  readonly length: number
}

export type PowerCurveData = readonly PowerCurvePoint[] | PowerCurveSeries

export const powerCurveSecondsAt = (curve: PowerCurveData, index: number): number =>
  'watts' in curve
    ? curve.durations[curve.indices?.[index] ?? curve.offset + index]
    : curve[index].s

export const powerCurveWattsAt = (curve: PowerCurveData, index: number): number =>
  'watts' in curve ? curve.watts[index] : curve[index].w

export const powerCurvePointAt = (curve: PowerCurveData, index: number): PowerCurvePoint => {
  if (!('watts' in curve)) return curve[index]
  const point = { s: powerCurveSecondsAt(curve, index), w: curve.watts[index] }
  let low = 0
  let high = curve.activities.length
  while (low < high) {
    const middle = Math.floor((low + high) / 2)
    if (curve.activities[middle].end <= index) low = middle + 1
    else high = middle
  }
  const source = curve.activities[low]
  return source
    ? { ...point, activityId: source.activityId, activityDate: source.activityDate }
    : point
}

const encodeActivities = (curve: readonly PowerCurvePoint[]): string | undefined => {
  if (
    curve.length === 0 ||
    curve.some(
      point =>
        point.activityId == null ||
        !Number.isInteger(point.activityId) ||
        point.activityId < 0 ||
        point.activityDate == null ||
        !/^\d{4}-\d{2}-\d{2}$/.test(point.activityDate),
    )
  )
    return undefined
  const spans: string[] = []
  let start = 0
  for (let index = 1; index <= curve.length; index++) {
    const previous = curve[index - 1]
    const point = curve[index]
    if (
      point &&
      point.activityId === previous.activityId &&
      point.activityDate === previous.activityDate
    )
      continue
    spans.push(`${previous.activityId},${previous.activityDate},${index - start}`)
    start = index
  }
  return spans.join(';')
}

export const encodePowerCurves = (curves: readonly (readonly PowerCurvePoint[])[]): string => {
  const seconds = new Set<number>()
  for (const curve of curves) for (const point of curve) seconds.add(point.s)
  const durations = [...seconds].sort((left, right) => left - right)
  const positions = new Map(durations.map((seconds, index) => [seconds, index]))
  const dense =
    durations.length > 0 && durations.every((seconds, i) => seconds === durations[0] + i)
  return JSON.stringify({
    durations: dense ? `d|${durations[0]}|${durations.length}` : `s|${durations.join(',')}`,
    series: curves.map(curve => {
      const indices = curve.map(point => positions.get(point.s) ?? -1)
      const offset = indices[0] ?? 0
      const consecutive = indices.every((position, index) => position === offset + index)
      return {
        offset,
        watts: curve.map(point => point.w),
        ...(consecutive ? {} : { indices }),
        activities: encodeActivities(curve),
      }
    }),
  })
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

const isNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)

const decodeDurations = (encoded: string, maximumLength: number): number[] | null => {
  const fields = encoded.split('|')
  if (fields[0] === 'd' && fields.length === 3) {
    const start = Number(fields[1])
    const count = Number(fields[2])
    if (
      !Number.isSafeInteger(start) ||
      start <= 0 ||
      !Number.isSafeInteger(count) ||
      count <= 0 ||
      count > maximumLength
    )
      return null
    return Array.from({ length: count }, (_, index) => start + index)
  }
  if (fields[0] !== 's' || fields.length !== 2) return null
  if (fields[1] === '') return []
  const values = fields[1].split(',')
  if (values.length > maximumLength) return null
  const durations: number[] = []
  let previous = 0
  for (const raw of values) {
    const seconds = Number(raw)
    if (!Number.isSafeInteger(seconds) || seconds <= previous) return null
    durations.push(seconds)
    previous = seconds
  }
  return durations
}

const decodeActivities = (encoded: unknown, length: number): PowerCurveActivitySpan[] | null => {
  if (encoded === undefined) return []
  if (typeof encoded !== 'string' || encoded === '') return null
  const activities: PowerCurveActivitySpan[] = []
  let end = 0
  for (const span of encoded.split(';')) {
    const fields = span.split(',')
    if (fields.length !== 3) return null
    const activityId = Number(fields[0])
    const activityDate = fields[1]
    const count = Number(fields[2])
    if (
      !Number.isSafeInteger(activityId) ||
      activityId < 0 ||
      !/^\d{4}-\d{2}-\d{2}$/.test(activityDate) ||
      !Number.isSafeInteger(count) ||
      count <= 0 ||
      end + count > length
    )
      return null
    end += count
    activities.push({ end, activityId, activityDate })
  }
  return end === length ? activities : null
}

export const decodePowerCurves = (encoded: string | undefined): PowerCurveSeries[] => {
  if (!encoded) return []
  let value: unknown
  try {
    value = JSON.parse(encoded)
  } catch {
    return []
  }
  if (!isRecord(value) || typeof value.durations !== 'string' || !Array.isArray(value.series))
    return []
  let pointCount = 0
  for (const series of value.series) {
    if (!isRecord(series) || !Array.isArray(series.watts) || !series.watts.every(isNumber))
      return []
    pointCount += series.watts.length
  }
  const durations = decodeDurations(value.durations, pointCount)
  if (!durations) return []
  const curves: PowerCurveSeries[] = []
  for (const series of value.series) {
    if (
      !isRecord(series) ||
      !Array.isArray(series.watts) ||
      !series.watts.every(isNumber) ||
      typeof series.offset !== 'number' ||
      !Number.isInteger(series.offset) ||
      series.offset < 0
    )
      return []
    const indices = series.indices
    if (indices !== undefined) {
      if (
        !Array.isArray(indices) ||
        !indices.every(isNumber) ||
        indices.length !== series.watts.length ||
        indices.some(
          (position, index) =>
            !Number.isInteger(position) ||
            position < 0 ||
            position >= durations.length ||
            (index > 0 && position <= indices[index - 1]),
        )
      )
        return []
    } else if (series.offset + series.watts.length > durations.length) return []
    const activities = decodeActivities(series.activities, series.watts.length)
    if (!activities) return []
    curves.push({
      durations,
      watts: series.watts,
      offset: series.offset,
      ...(indices === undefined ? {} : { indices }),
      activities,
      length: series.watts.length,
    })
  }
  return curves
}
