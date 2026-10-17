import type { MapRoutePoint } from '../../util/triathlon-map-route'
import { isRecord } from '../../util/type-guards'

type GeoCoord = [number, number]

export type StreetMetric = 'w' | 'hr' | 'cad' | 'spd'

export interface StreetMetricValues {
  w: number
  hr: number
  cad: number
  spd: number
}

export interface StreetMapActivity {
  id: number
  segments: readonly (readonly MapRoutePoint[])[]
  metrics: StreetMetricValues
}

interface ProjectedPoint {
  x: number
  y: number
}

interface TraceEdge {
  activityId: number
  startX: number
  startY: number
  dx: number
  dy: number
  length: number
  invLenSq: number
  ux: number
  uy: number
}

interface TraceIndex {
  cells: Map<number, TraceEdge[]>
  minX: number
  maxX: number
  minY: number
  maxY: number
}

interface StreetPiece {
  key: string
  start: GeoCoord
  end: GeoCoord
  startKey: number
  endKey: number
  roadClass: string
  activityIds: Set<number>
}

export interface StreetMapProperties extends StreetMetricValues {
  heat: number
  visits: number
  roadClass: string
}

export interface StreetMapFeature {
  type: 'Feature'
  properties: StreetMapProperties
  geometry: { type: 'LineString'; coordinates: GeoCoord[] }
}

export interface StreetMapFeatureCollection {
  type: 'FeatureCollection'
  features: StreetMapFeature[]
}

export type StreetMapMatcher = (roadFeatures: readonly unknown[]) => StreetMapFeatureCollection

const EARTH_RADIUS_M = 6_378_137
const GRID_M = 128
const MATCH_RADIUS_M = 16
const MATCH_RADIUS_SQ = MATCH_RADIUS_M * MATCH_RADIUS_M
const ROAD_PIECE_M = 24
const MIN_DIRECTION_M = 8
const MIN_ALIGNMENT = Math.cos((38 * Math.PI) / 180)
const KEY_OFFSET = 33_554_432
const KEY_SCALE = 67_108_864
const EXCLUDED_ROAD_CLASSES = new Set([
  'major_rail',
  'minor_rail',
  'service_rail',
  'ferry',
  'aerialway',
  'golf',
])

function project(point: GeoCoord): ProjectedPoint {
  const longitude = (point[0] * Math.PI) / 180
  const latitude = (Math.max(-85, Math.min(85, point[1])) * Math.PI) / 180
  return {
    x: EARTH_RADIUS_M * longitude,
    y: EARTH_RADIUS_M * Math.log(Math.tan(Math.PI / 4 + latitude / 2)),
  }
}

function packPair(ix: number, iy: number): number {
  return (ix + KEY_OFFSET) * KEY_SCALE + (iy + KEY_OFFSET)
}

function gridKey(x: number, y: number): number {
  return packPair(Math.floor(x / GRID_M), Math.floor(y / GRID_M))
}

function projectedPointKey(x: number, y: number): number {
  return packPair(Math.round(x / 2), Math.round(y / 2))
}

function pieceKey(startKey: number, endKey: number): string {
  return startKey < endKey ? `${startKey}|${endKey}` : `${endKey}|${startKey}`
}

function addCellEdge(cells: Map<number, TraceEdge[]>, key: number, edge: TraceEdge): void {
  const edges = cells.get(key)
  if (edges) edges.push(edge)
  else cells.set(key, [edge])
}

function addIndexedEdge(cells: Map<number, TraceEdge[]>, edge: TraceEdge): void {
  const x0 = edge.startX
  const y0 = edge.startY
  const x1 = x0 + edge.dx
  const y1 = y0 + edge.dy
  const minX = Math.floor((Math.min(x0, x1) - MATCH_RADIUS_M) / GRID_M)
  const maxX = Math.floor((Math.max(x0, x1) + MATCH_RADIUS_M) / GRID_M)
  const minY = Math.floor((Math.min(y0, y1) - MATCH_RADIUS_M) / GRID_M)
  const maxY = Math.floor((Math.max(y0, y1) + MATCH_RADIUS_M) / GRID_M)
  if (maxX - minX <= 1 || maxY - minY <= 1) {
    for (let gx = minX; gx <= maxX; gx++) {
      for (let gy = minY; gy <= maxY; gy++) {
        addCellEdge(cells, packPair(gx, gy), edge)
      }
    }
    return
  }
  const dx = edge.dx
  const dy = edge.dy
  for (let gx = minX; gx <= maxX; gx++) {
    const colMinX = gx * GRID_M - MATCH_RADIUS_M
    const colMaxX = (gx + 1) * GRID_M + MATCH_RADIUS_M
    const t0 = (colMinX - x0) / dx
    const t1 = (colMaxX - x0) / dx
    const tEnter = Math.max(0, Math.min(t0, t1))
    const tExit = Math.min(1, Math.max(t0, t1))
    if (tEnter > tExit) continue
    const ya = y0 + tEnter * dy
    const yb = y0 + tExit * dy
    const gy0 = Math.max(minY, Math.floor((Math.min(ya, yb) - MATCH_RADIUS_M) / GRID_M))
    const gy1 = Math.min(maxY, Math.floor((Math.max(ya, yb) + MATCH_RADIUS_M) / GRID_M))
    for (let gy = gy0; gy <= gy1; gy++) {
      addCellEdge(cells, packPair(gx, gy), edge)
    }
  }
}

function traceIndex(activities: readonly StreetMapActivity[]): TraceIndex {
  const cells = new Map<number, TraceEdge[]>()
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity
  for (const activity of activities) {
    for (const segment of activity.segments) {
      if (segment.length < 2) continue
      let start = project([segment[0].lng, segment[0].lat])
      for (let pointIndex = 1; pointIndex < segment.length; pointIndex++) {
        const current = segment[pointIndex]
        const end = project([current.lng, current.lat])
        const dx = end.x - start.x
        const dy = end.y - start.y
        const lenSq = dx * dx + dy * dy
        if (lenSq > 0) {
          const length = Math.sqrt(lenSq)
          const loX = Math.min(start.x, end.x) - MATCH_RADIUS_M
          const hiX = Math.max(start.x, end.x) + MATCH_RADIUS_M
          const loY = Math.min(start.y, end.y) - MATCH_RADIUS_M
          const hiY = Math.max(start.y, end.y) + MATCH_RADIUS_M
          if (loX < minX) minX = loX
          if (hiX > maxX) maxX = hiX
          if (loY < minY) minY = loY
          if (hiY > maxY) maxY = hiY
          addIndexedEdge(cells, {
            activityId: activity.id,
            startX: start.x,
            startY: start.y,
            dx,
            dy,
            length,
            invLenSq: 1 / lenSq,
            ux: dx / length,
            uy: dy / length,
          })
        }
        start = end
      }
    }
  }
  return { cells, minX, maxX, minY, maxY }
}

function pointSegmentDistanceSq(px: number, py: number, edge: TraceEdge): number {
  const fraction = Math.max(
    0,
    Math.min(1, ((px - edge.startX) * edge.dx + (py - edge.startY) * edge.dy) * edge.invLenSq),
  )
  const ex = px - (edge.startX + fraction * edge.dx)
  const ey = py - (edge.startY + fraction * edge.dy)
  return ex * ex + ey * ey
}

function matchingActivities(
  cells: ReadonlyMap<number, readonly TraceEdge[]>,
  midX: number,
  midY: number,
  pieceLength: number,
  rux: number,
  ruy: number,
): Set<number> | null {
  const candidates = cells.get(gridKey(midX, midY))
  if (!candidates || candidates.length === 0) return null
  const checkDirection = pieceLength >= MIN_DIRECTION_M
  let visits: Set<number> | null = null
  for (let i = 0; i < candidates.length; i++) {
    const edge = candidates[i]
    if (
      checkDirection &&
      edge.length >= MIN_DIRECTION_M &&
      Math.abs(rux * edge.ux + ruy * edge.uy) < MIN_ALIGNMENT
    )
      continue
    if (pointSegmentDistanceSq(midX, midY, edge) <= MATCH_RADIUS_SQ) {
      visits ??= new Set<number>()
      visits.add(edge.activityId)
    }
  }
  return visits
}

function readCoordinate(value: unknown): GeoCoord | null {
  if (!Array.isArray(value) || value.length < 2) return null
  const longitude = value[0]
  const latitude = value[1]
  if (
    typeof longitude !== 'number' ||
    typeof latitude !== 'number' ||
    !Number.isFinite(longitude) ||
    !Number.isFinite(latitude)
  )
    return null
  return [longitude, latitude]
}

function readLine(value: unknown): GeoCoord[] {
  if (!Array.isArray(value)) return []
  const line: GeoCoord[] = []
  for (const coordinate of value) {
    const point = readCoordinate(coordinate)
    if (point) line.push(point)
  }
  return line
}

function roadLines(feature: unknown): { lines: GeoCoord[][]; roadClass: string } | null {
  if (!isRecord(feature) || !isRecord(feature.geometry)) return null
  const properties = isRecord(feature.properties) ? feature.properties : {}
  const roadClass = typeof properties.class === 'string' ? properties.class : 'road'
  if (EXCLUDED_ROAD_CLASSES.has(roadClass)) return null
  const coordinates = feature.geometry.coordinates
  if (feature.geometry.type === 'LineString') {
    const line = readLine(coordinates)
    return line.length >= 2 ? { lines: [line], roadClass } : null
  }
  if (feature.geometry.type !== 'MultiLineString' || !Array.isArray(coordinates)) return null
  const lines: GeoCoord[][] = []
  for (const value of coordinates) {
    const line = readLine(value)
    if (line.length >= 2) lines.push(line)
  }
  return lines.length > 0 ? { lines, roadClass } : null
}

function interpolate(start: GeoCoord, end: GeoCoord, fraction: number): GeoCoord {
  return [start[0] + (end[0] - start[0]) * fraction, start[1] + (end[1] - start[1]) * fraction]
}

function heatBucket(visits: number, maximum: number): number {
  if (maximum <= 1) return 1
  return Math.min(7, Math.max(1, 1 + Math.round((6 * Math.log(visits)) / Math.log(maximum))))
}

function aggregateMetric(
  activityIds: ReadonlySet<number>,
  activities: ReadonlyMap<number, StreetMetricValues>,
  metric: StreetMetric,
): number {
  let total = 0
  let count = 0
  for (const activityId of activityIds) {
    const value = activities.get(activityId)?.[metric]
    if (value == null || !Number.isFinite(value) || value < 0) continue
    total += value
    count++
  }
  return count > 0 ? total / count : -1
}

function streetProperties(
  piece: StreetPiece,
  activities: ReadonlyMap<number, StreetMetricValues>,
  maximumVisits: number,
): StreetMapProperties {
  const visits = piece.activityIds.size
  return {
    heat: heatBucket(visits, maximumVisits),
    visits,
    roadClass: piece.roadClass,
    w: aggregateMetric(piece.activityIds, activities, 'w'),
    hr: aggregateMetric(piece.activityIds, activities, 'hr'),
    cad: aggregateMetric(piece.activityIds, activities, 'cad'),
    spd: aggregateMetric(piece.activityIds, activities, 'spd'),
  }
}

function sameStreetProperties(left: StreetMapProperties, right: StreetMapProperties): boolean {
  return (
    left.visits === right.visits &&
    left.roadClass === right.roadClass &&
    left.w === right.w &&
    left.hr === right.hr &&
    left.cad === right.cad &&
    left.spd === right.spd
  )
}

function streetFeatures(
  pieces: ReadonlyMap<string, StreetPiece>,
  activities: ReadonlyMap<number, StreetMetricValues>,
  maximumVisits: number,
): StreetMapFeature[] {
  if (pieces.size === 0) return []
  const properties = new Map<string, StreetMapProperties>()
  const adjacency = new Map<number, StreetPiece[]>()
  for (const [key, piece] of pieces) {
    properties.set(key, streetProperties(piece, activities, maximumVisits))
    const startConnected = adjacency.get(piece.startKey)
    if (startConnected) startConnected.push(piece)
    else adjacency.set(piece.startKey, [piece])
    const endConnected = adjacency.get(piece.endKey)
    if (endConnected) endConnected.push(piece)
    else adjacency.set(piece.endKey, [piece])
  }
  const remaining = new Map(pieces)
  const features: StreetMapFeature[] = []
  for (const key of [...pieces.keys()].sort()) {
    const seed = remaining.get(key)
    const seedProperties = properties.get(key)
    if (!seed || !seedProperties) continue
    remaining.delete(key)
    const coordinates = [seed.start, seed.end]
    let headKey = seed.startKey
    let tailKey = seed.endKey
    const extend = (atStart: boolean, initialKey: string): void => {
      let previousKey = initialKey
      while (true) {
        const endpointKey = atStart ? headKey : tailKey
        const candidates = adjacency.get(endpointKey)
        if (!candidates) return
        let matchCount = 0
        let nextPiece: StreetPiece | null = null
        for (let i = 0; i < candidates.length; i++) {
          const candidate = candidates[i]
          const candidateProperties = properties.get(candidate.key)
          if (candidateProperties && sameStreetProperties(candidateProperties, seedProperties)) {
            matchCount++
            if (candidate.key !== previousKey) nextPiece = candidate
          }
        }
        if (matchCount !== 2 || !nextPiece || !remaining.has(nextPiece.key)) return
        remaining.delete(nextPiece.key)
        const connectsAtStart = nextPiece.startKey === endpointKey
        const nextPoint = connectsAtStart ? nextPiece.end : nextPiece.start
        const nextEndpointKey = connectsAtStart ? nextPiece.endKey : nextPiece.startKey
        if (atStart) {
          coordinates.unshift(nextPoint)
          headKey = nextEndpointKey
        } else {
          coordinates.push(nextPoint)
          tailKey = nextEndpointKey
        }
        previousKey = nextPiece.key
      }
    }
    extend(false, key)
    extend(true, key)
    features.push({
      type: 'Feature',
      properties: seedProperties,
      geometry: { type: 'LineString', coordinates },
    })
  }
  return features
}

function matchedStreetFeatures(
  index: TraceIndex,
  activities: ReadonlyMap<number, StreetMetricValues>,
  maximumVisits: number,
  roadFeatures: readonly unknown[],
): StreetMapFeatureCollection {
  if (index.cells.size === 0 || roadFeatures.length === 0)
    return { type: 'FeatureCollection', features: [] }
  const pieces = new Map<string, StreetPiece>()
  const { cells, minX, maxX, minY, maxY } = index
  for (const feature of roadFeatures) {
    const road = roadLines(feature)
    if (!road) continue
    for (const line of road.lines) {
      let projectedStart = project(line[0])
      for (let pointIndex = 1; pointIndex < line.length; pointIndex++) {
        const start = line[pointIndex - 1]
        const end = line[pointIndex]
        const projectedEnd = project(end)
        const segMinX = Math.min(projectedStart.x, projectedEnd.x)
        const segMaxX = Math.max(projectedStart.x, projectedEnd.x)
        const segMinY = Math.min(projectedStart.y, projectedEnd.y)
        const segMaxY = Math.max(projectedStart.y, projectedEnd.y)
        if (segMaxX < minX || segMinX > maxX || segMaxY < minY || segMinY > maxY) {
          projectedStart = projectedEnd
          continue
        }
        const rdx = projectedEnd.x - projectedStart.x
        const rdy = projectedEnd.y - projectedStart.y
        const length = Math.hypot(rdx, rdy)
        if (length === 0) {
          projectedStart = projectedEnd
          continue
        }
        const divisions = Math.max(1, Math.ceil(length / ROAD_PIECE_M))
        const pieceLength = length / divisions
        const rux = rdx / length
        const ruy = rdy / length
        for (let division = 0; division < divisions; division++) {
          const f0 = division / divisions
          const f1 = (division + 1) / divisions
          const fm = (f0 + f1) * 0.5
          const midX = projectedStart.x + rdx * fm
          const midY = projectedStart.y + rdy * fm
          const activityIds = matchingActivities(cells, midX, midY, pieceLength, rux, ruy)
          if (!activityIds) continue
          const p0x = projectedStart.x + rdx * f0
          const p0y = projectedStart.y + rdy * f0
          const p1x = projectedStart.x + rdx * f1
          const p1y = projectedStart.y + rdy * f1
          const startKey = projectedPointKey(p0x, p0y)
          const endKey = projectedPointKey(p1x, p1y)
          const key = pieceKey(startKey, endKey)
          const existing = pieces.get(key)
          if (existing) {
            for (const activityId of activityIds) existing.activityIds.add(activityId)
          } else {
            pieces.set(key, {
              key,
              start: interpolate(start, end, f0),
              end: interpolate(start, end, f1),
              startKey,
              endKey,
              roadClass: road.roadClass,
              activityIds,
            })
          }
        }
        projectedStart = projectedEnd
      }
    }
  }
  return { type: 'FeatureCollection', features: streetFeatures(pieces, activities, maximumVisits) }
}

export function createStreetMapMatcher(
  activities: readonly StreetMapActivity[],
  maximumVisits: number,
): StreetMapMatcher {
  const index = traceIndex(activities)
  const metrics = new Map(activities.map(activity => [activity.id, activity.metrics]))
  return roadFeatures => matchedStreetFeatures(index, metrics, maximumVisits, roadFeatures)
}

export function streetMapFeatureCollection(
  roadFeatures: readonly unknown[],
  activities: readonly StreetMapActivity[],
  maximumVisits: number,
): StreetMapFeatureCollection {
  return createStreetMapMatcher(activities, maximumVisits)(roadFeatures)
}
