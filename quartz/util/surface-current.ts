import { isRecord } from './type-guards'

const HOUR_MS = 3_600_000
const MAX_SPEED_MPS = 10
const MAX_ROUTE_GAP_S = 60
const MAX_SAMPLES = 512

export interface SurfaceCurrentSample {
  elapsedS: number
  speedMps: number | null
  directionDeg: number | null
  uMps: number | null
  vMps: number | null
  element: number | null
  validTime: string | null
  cycleTime: string | null
  sourceUrl: string | null
}

export interface SurfaceCurrentEstimate {
  source: 'noaa-loofs'
  sourceKind: 'modeled'
  formulaId: 'garden-surface-current-v1'
  formulaVersion: 1
  activityId: number
  routeFingerprint: string
  start: string
  end: string
  computedAt: number
  summary: {
    averageSpeedMps: number | null
    averageDirectionDeg: number | null
    coveragePct: number
    coveredDurationS: number
    elapsedDurationS: number
  }
  samples: SurfaceCurrentSample[]
  spatialSamplingModel: 'containing-element'
  temporalSamplingModel: 'hourly-linear-vector'
  layer: 0
}

export type PublicSurfaceCurrentEstimate = Omit<SurfaceCurrentEstimate, 'routeFingerprint'>

export interface SurfaceCurrentMesh {
  latitudes: number[]
  longitudes: number[]
  triangles: [number, number, number][]
}

export interface SurfaceCurrentField {
  validTime: string
  cycleTime: string
  sourceUrl: string
  elements: Map<number, { uMps: number; vMps: number; wet: boolean }>
}

const numberIn = (value: unknown, min: number, max: number): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max
const nullableNumber = (value: unknown, min: number, max: number): value is number | null =>
  value === null || numberIn(value, min, max)
const timestamp = (value: unknown): value is string =>
  typeof value === 'string' &&
  /^\d{4}-\d{2}-\d{2}T/.test(value) &&
  Number.isFinite(Date.parse(value))
const direction = (u: number, v: number): number | null =>
  Math.hypot(u, v) <= 1e-12 ? null : ((Math.atan2(u, v) * 180) / Math.PI + 360) % 360
const round = (value: number): number => Math.round(value * 1e6) / 1e6
const hasOnlyKeys = (value: Record<string, unknown>, keys: readonly string[]): boolean =>
  Object.keys(value).every(key => keys.includes(key))

function sourceUrlIsValid(value: unknown): value is string {
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
    !numberIn(value.activityId, 1, Number.MAX_SAFE_INTEGER) ||
    !Number.isInteger(value.activityId) ||
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
    activityId: value.activityId,
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

function asciiArrays(text: string): Map<string, string[]> {
  const delimiter = '---------------------------------------------'
  if (!text.includes(delimiter)) return new Map()
  const body = text.slice(text.indexOf(delimiter) + delimiter.length).trim()
  const output = new Map<string, string[]>()
  for (const match of body.matchAll(/(?:^|\n\n)(\w+)(?:\[\d+\])+\r?\n([\s\S]*?)(?=\n\n|$)/g)) {
    const values = match[2]
      .split('\n')
      .flatMap(line => line.replace(/^(?:\[\d+\])+,\s*/, '').split(','))
      .map(item => item.trim())
      .filter(Boolean)
    output.set(match[1], values)
  }
  return output
}

export function parseLoofsMeshAscii(text: string): SurfaceCurrentMesh | null {
  const arrays = asciiArrays(text)
  const latitudes = arrays.get('lat')?.map(Number)
  const longitudes = arrays
    .get('lon')
    ?.map(Number)
    .map(value => (value > 180 ? value - 360 : value))
  const connectivity = arrays.get('nv')?.map(Number)
  if (
    !latitudes ||
    !longitudes ||
    !connectivity ||
    latitudes.length < 3 ||
    latitudes.length > 100_000 ||
    latitudes.length !== longitudes.length ||
    connectivity.length < 3 ||
    connectivity.length > 300_000 ||
    connectivity.length % 3 !== 0 ||
    latitudes.some(value => !numberIn(value, -90, 90)) ||
    longitudes.some(value => !numberIn(value, -180, 180)) ||
    connectivity.some(value => !numberIn(value, 1, latitudes.length) || !Number.isInteger(value))
  )
    return null
  const elements = connectivity.length / 3
  const triangles: [number, number, number][] = []
  for (let index = 0; index < elements; index += 1)
    triangles.push([
      connectivity[index] - 1,
      connectivity[index + elements] - 1,
      connectivity[index + 2 * elements] - 1,
    ])
  return { latitudes, longitudes, triangles }
}

export function parseLoofsFieldAscii(
  text: string,
  firstElement: number,
  cycleTime: string,
  expectedTime: string,
  sourceUrl: string,
): SurfaceCurrentField | null {
  const arrays = asciiArrays(text)
  const rawTime = arrays.get('Times')?.[0]?.replaceAll('"', '')
  const rawTimeMs = rawTime
    ? Date.parse(rawTime.endsWith('Z') ? rawTime : `${rawTime}Z`)
    : Number.NaN
  const validTime = Number.isFinite(rawTimeMs) ? new Date(rawTimeMs).toISOString() : null
  const east = arrays.get('u')?.map(Number),
    north = arrays.get('v')?.map(Number),
    wet = arrays.get('wet_cells')?.map(Number)
  if (
    !validTime ||
    Date.parse(validTime) !== Date.parse(expectedTime) ||
    !timestamp(cycleTime) ||
    !sourceUrlIsValid(sourceUrl) ||
    !east ||
    !north ||
    !wet ||
    east.length !== north.length ||
    east.length !== wet.length ||
    !Number.isInteger(firstElement) ||
    firstElement < 0
  )
    return null
  const elements: SurfaceCurrentField['elements'] = new Map()
  for (let index = 0; index < east.length; index += 1) {
    const uMps = east[index],
      vMps = north[index]
    if (
      numberIn(uMps, -MAX_SPEED_MPS, MAX_SPEED_MPS) &&
      numberIn(vMps, -MAX_SPEED_MPS, MAX_SPEED_MPS) &&
      Math.hypot(uMps, vMps) <= MAX_SPEED_MPS &&
      (wet[index] === 0 || wet[index] === 1)
    )
      elements.set(firstElement + index, { uMps, vMps, wet: wet[index] === 1 })
  }
  return { validTime, cycleTime, sourceUrl, elements }
}

export function loofsRequestsForHour(
  validTime: string,
): { recentUrl: string; archiveUrl: string; cycleTime: string } | null {
  const validMs = Date.parse(validTime)
  if (!Number.isFinite(validMs) || validMs % HOUR_MS !== 0) return null
  const cycleMs = Math.ceil(validMs / (6 * HOUR_MS)) * 6 * HOUR_MS
  const cycleTime = new Date(cycleMs).toISOString(),
    day = cycleTime.slice(0, 10)
  const filename = `loofs.t${cycleTime.slice(11, 13)}z.${day.replaceAll('-', '')}.fields.n00${(validMs - cycleMs) / HOUR_MS + 6}.nc.ascii`
  return {
    cycleTime,
    recentUrl: `https://opendap.co-ops.nos.noaa.gov/thredds/dodsC/NOAA/LOOFS/MODELS/${day.replaceAll('-', '/')}/${filename}`,
    archiveUrl: `https://www.ncei.noaa.gov/thredds/dodsC/model-loofs-files/${day.slice(0, 7).replaceAll('-', '/')}/${filename}`,
  }
}

export function findSurfaceCurrentElement(
  mesh: SurfaceCurrentMesh,
  latitude: number,
  longitude: number,
): number | null {
  if (!numberIn(latitude, -90, 90) || !numberIn(longitude, -180, 180)) return null
  for (let element = 0; element < mesh.triangles.length; element += 1) {
    const [a, b, c] = mesh.triangles[element]
    const ay = mesh.latitudes[a],
      by = mesh.latitudes[b],
      cy = mesh.latitudes[c]
    const ax = mesh.longitudes[a],
      bx = mesh.longitudes[b],
      cx = mesh.longitudes[c]
    if (
      latitude < Math.min(ay, by, cy) ||
      latitude > Math.max(ay, by, cy) ||
      longitude < Math.min(ax, bx, cx) ||
      longitude > Math.max(ax, bx, cx)
    )
      continue
    const denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if (Math.abs(denominator) < 1e-12) continue
    const first = ((by - cy) * (longitude - cx) + (cx - bx) * (latitude - cy)) / denominator
    const second = ((cy - ay) * (longitude - cx) + (ax - cx) * (latitude - cy)) / denominator
    if (first >= -1e-8 && second >= -1e-8 && first + second <= 1 + 1e-8) return element
  }
  return null
}

const blankSample = (elapsedS: number): SurfaceCurrentSample => ({
  elapsedS,
  speedMps: null,
  directionDeg: null,
  uMps: null,
  vMps: null,
  element: null,
  validTime: null,
  cycleTime: null,
  sourceUrl: null,
})

function limitSamples(samples: SurfaceCurrentSample[]): SurfaceCurrentSample[] {
  if (samples.length <= MAX_SAMPLES) return samples
  const selected = new Set([0, samples.length - 1])
  for (let index = 1; index < samples.length; index += 1)
    if ((samples[index].speedMps === null) !== (samples[index - 1].speedMps === null)) {
      selected.add(index - 1)
      selected.add(index)
    }
  if (selected.size > MAX_SAMPLES) throw new Error('Too many GPS gaps for bounded current samples')
  const remaining = MAX_SAMPLES - selected.size
  for (let index = 0; index < remaining; index += 1)
    selected.add(Math.round(((samples.length - 1) * index) / Math.max(1, remaining - 1)))
  return [...selected].sort((left, right) => left - right).map(index => samples[index])
}

export function buildSurfaceCurrentEstimate(input: {
  activityId: number
  routeFingerprint: string
  start: string
  end: string
  computedAt: number
  timeS: readonly number[]
  latlng: readonly [number, number][]
  mesh: SurfaceCurrentMesh
  fields: readonly SurfaceCurrentField[]
}): SurfaceCurrentEstimate {
  const startMs = Date.parse(input.start),
    durationS = (Date.parse(input.end) - startMs) / 1_000
  if (!Number.isFinite(durationS) || durationS <= 0 || durationS > 43_200)
    throw new Error('Invalid surface current activity time window')
  const fields = new Map(input.fields.map(field => [Date.parse(field.validTime), field]))
  const points: { elapsedS: number; coordinate: [number, number] | null }[] = []
  for (let index = 0; index < Math.min(input.timeS.length, input.latlng.length); index += 1) {
    const elapsedS = input.timeS[index],
      coordinate = input.latlng[index]
    if (
      !numberIn(elapsedS, 0, durationS) ||
      (points.length > 0 && elapsedS <= points[points.length - 1].elapsedS)
    )
      continue
    const previous = points.at(-1)
    if (previous && elapsedS - previous.elapsedS > MAX_ROUTE_GAP_S)
      points.push({ elapsedS: (previous.elapsedS + elapsedS) / 2, coordinate: null })
    points.push({
      elapsedS,
      coordinate:
        numberIn(coordinate[0], -90, 90) && numberIn(coordinate[1], -180, 180) ? coordinate : null,
    })
  }
  if (points[0]?.elapsedS !== 0) points.unshift({ elapsedS: 0, coordinate: null })
  if (points.at(-1)?.elapsedS !== durationS) points.push({ elapsedS: durationS, coordinate: null })
  const samples = points.map(point => {
    if (!point.coordinate) return blankSample(point.elapsedS)
    const element = findSurfaceCurrentElement(input.mesh, point.coordinate[0], point.coordinate[1])
    if (element === null) return blankSample(point.elapsedS)
    const sampleMs = startMs + point.elapsedS * 1_000,
      hourMs = Math.floor(sampleMs / HOUR_MS) * HOUR_MS
    const lower = fields.get(hourMs),
      upper = fields.get(hourMs + HOUR_MS)
    const first = lower?.elements.get(element),
      second = upper?.elements.get(element)
    const weight = (sampleMs - hourMs) / HOUR_MS
    if (!lower || !first?.wet || (weight > 0 && !second?.wet)) return blankSample(point.elapsedS)
    const uMps = first.uMps + (second ? second.uMps - first.uMps : 0) * weight
    const vMps = first.vMps + (second ? second.vMps - first.vMps : 0) * weight
    const speedMps = Math.hypot(uMps, vMps)
    if (!numberIn(speedMps, 0, MAX_SPEED_MPS)) return blankSample(point.elapsedS)
    return {
      elapsedS: point.elapsedS,
      speedMps,
      directionDeg: direction(uMps, vMps),
      uMps,
      vMps,
      element,
      validTime: lower.validTime,
      cycleTime: lower.cycleTime,
      sourceUrl: lower.sourceUrl,
    }
  })
  let coveredDurationS = 0,
    speedIntegral = 0,
    eastIntegral = 0,
    northIntegral = 0
  for (let index = 1; index < samples.length; index += 1) {
    const previous = samples[index - 1],
      current = samples[index]
    if (
      previous.speedMps === null ||
      current.speedMps === null ||
      previous.uMps === null ||
      current.uMps === null ||
      previous.vMps === null ||
      current.vMps === null
    )
      continue
    const seconds = current.elapsedS - previous.elapsedS
    coveredDurationS += seconds
    speedIntegral += ((previous.speedMps + current.speedMps) / 2) * seconds
    eastIntegral += ((previous.uMps + current.uMps) / 2) * seconds
    northIntegral += ((previous.vMps + current.vMps) / 2) * seconds
  }
  return {
    source: 'noaa-loofs',
    sourceKind: 'modeled',
    formulaId: 'garden-surface-current-v1',
    formulaVersion: 1,
    activityId: input.activityId,
    routeFingerprint: input.routeFingerprint,
    start: input.start,
    end: input.end,
    computedAt: input.computedAt,
    summary: {
      averageSpeedMps: coveredDurationS > 0 ? round(speedIntegral / coveredDurationS) : null,
      averageDirectionDeg:
        coveredDurationS > 0
          ? direction(eastIntegral / coveredDurationS, northIntegral / coveredDurationS)
          : null,
      coveragePct: round((coveredDurationS / durationS) * 100),
      coveredDurationS,
      elapsedDurationS: durationS,
    },
    samples: limitSamples(samples),
    spatialSamplingModel: 'containing-element',
    temporalSamplingModel: 'hourly-linear-vector',
    layer: 0,
  }
}
