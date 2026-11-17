import type {
  SurfaceCurrentEstimate,
  SurfaceCurrentField,
  SurfaceCurrentMesh,
  SurfaceCurrentSample,
} from './surface-current/types'
import {
  direction,
  HOUR_MS,
  MAX_SAMPLES,
  MAX_SPEED_MPS,
  numberIn,
  sourceUrlIsValid,
  timestamp,
} from './surface-current/validation'

const MAX_ROUTE_GAP_S = 60

const round = (value: number): number => Math.round(value * 1e6) / 1e6

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
  activityId: string
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
export type {
  PublicSurfaceCurrentEstimate,
  SurfaceCurrentEstimate,
  SurfaceCurrentField,
  SurfaceCurrentMesh,
  SurfaceCurrentSample,
} from './surface-current/types'
export {
  parsePublicSurfaceCurrentEstimate,
  parseSurfaceCurrentEstimate,
} from './surface-current/validation'
