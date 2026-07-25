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
  start: ProjectedPoint
  end: ProjectedPoint
  length: number
}

interface StreetPiece {
  start: GeoCoord
  end: GeoCoord
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
const GRID_M = 32
const MATCH_RADIUS_M = 16
const ROAD_PIECE_M = 24
const MIN_DIRECTION_M = 8
const MIN_ALIGNMENT = Math.cos((38 * Math.PI) / 180)
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

function gridKey(x: number, y: number): string {
  return `${Math.floor(x / GRID_M)},${Math.floor(y / GRID_M)}`
}

function addIndexedEdge(index: Map<string, TraceEdge[]>, edge: TraceEdge): void {
  const minX = Math.floor((Math.min(edge.start.x, edge.end.x) - MATCH_RADIUS_M) / GRID_M)
  const maxX = Math.floor((Math.max(edge.start.x, edge.end.x) + MATCH_RADIUS_M) / GRID_M)
  const minY = Math.floor((Math.min(edge.start.y, edge.end.y) - MATCH_RADIUS_M) / GRID_M)
  const maxY = Math.floor((Math.max(edge.start.y, edge.end.y) + MATCH_RADIUS_M) / GRID_M)
  for (let x = minX; x <= maxX; x++) {
    for (let y = minY; y <= maxY; y++) {
      const key = `${x},${y}`
      const edges = index.get(key)
      if (edges) edges.push(edge)
      else index.set(key, [edge])
    }
  }
}

function traceIndex(activities: readonly StreetMapActivity[]): Map<string, TraceEdge[]> {
  const index = new Map<string, TraceEdge[]>()
  for (const activity of activities) {
    for (const segment of activity.segments) {
      for (let pointIndex = 1; pointIndex < segment.length; pointIndex++) {
        const previous = segment[pointIndex - 1]
        const current = segment[pointIndex]
        const start = project([previous.lng, previous.lat])
        const end = project([current.lng, current.lat])
        const length = Math.hypot(end.x - start.x, end.y - start.y)
        if (length === 0) continue
        addIndexedEdge(index, { activityId: activity.id, start, end, length })
      }
    }
  }
  return index
}

function pointSegmentDistance(point: ProjectedPoint, edge: TraceEdge): number {
  const dx = edge.end.x - edge.start.x
  const dy = edge.end.y - edge.start.y
  const fraction = Math.max(
    0,
    Math.min(
      1,
      ((point.x - edge.start.x) * dx + (point.y - edge.start.y) * dy) / (edge.length * edge.length),
    ),
  )
  return Math.hypot(
    point.x - (edge.start.x + fraction * dx),
    point.y - (edge.start.y + fraction * dy),
  )
}

function aligned(roadStart: ProjectedPoint, roadEnd: ProjectedPoint, edge: TraceEdge): boolean {
  if (edge.length < MIN_DIRECTION_M) return true
  const roadDx = roadEnd.x - roadStart.x
  const roadDy = roadEnd.y - roadStart.y
  const roadLength = Math.hypot(roadDx, roadDy)
  if (roadLength < MIN_DIRECTION_M) return true
  const traceDx = edge.end.x - edge.start.x
  const traceDy = edge.end.y - edge.start.y
  const alignment = Math.abs(roadDx * traceDx + roadDy * traceDy) / (roadLength * edge.length)
  return alignment >= MIN_ALIGNMENT
}

function matchingActivities(
  index: ReadonlyMap<string, readonly TraceEdge[]>,
  roadStart: ProjectedPoint,
  roadEnd: ProjectedPoint,
): Set<number> {
  const midpoint = { x: (roadStart.x + roadEnd.x) / 2, y: (roadStart.y + roadEnd.y) / 2 }
  const visits = new Set<number>()
  for (const edge of index.get(gridKey(midpoint.x, midpoint.y)) ?? []) {
    if (!aligned(roadStart, roadEnd, edge)) continue
    if (pointSegmentDistance(midpoint, edge) <= MATCH_RADIUS_M) visits.add(edge.activityId)
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

function pieceKey(start: ProjectedPoint, end: ProjectedPoint): string {
  const a = projectedPointKey(start)
  const b = projectedPointKey(end)
  return a < b ? `${a}|${b}` : `${b}|${a}`
}

function projectedPointKey(point: ProjectedPoint): string {
  return `${Math.round(point.x / 2)},${Math.round(point.y / 2)}`
}

function coordinateKey(point: GeoCoord): string {
  return projectedPointKey(project(point))
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
  const properties = new Map<string, StreetMapProperties>()
  const adjacency = new Map<string, string[]>()
  for (const [key, piece] of pieces) {
    properties.set(key, streetProperties(piece, activities, maximumVisits))
    for (const endpoint of [coordinateKey(piece.start), coordinateKey(piece.end)]) {
      const connected = adjacency.get(endpoint)
      if (connected) connected.push(key)
      else adjacency.set(endpoint, [key])
    }
  }
  const remaining = new Map(pieces)
  const features: StreetMapFeature[] = []
  for (const key of [...pieces.keys()].sort()) {
    const seed = remaining.get(key)
    const seedProperties = properties.get(key)
    if (!seed || !seedProperties) continue
    remaining.delete(key)
    const coordinates = [seed.start, seed.end]
    const extend = (atStart: boolean, initialKey: string): void => {
      let previousKey = initialKey
      while (true) {
        const endpoint = atStart ? coordinates[0] : coordinates[coordinates.length - 1]
        const connected = (adjacency.get(coordinateKey(endpoint)) ?? []).filter(candidateKey => {
          const candidateProperties = properties.get(candidateKey)
          return candidateProperties && sameStreetProperties(candidateProperties, seedProperties)
        })
        if (connected.length !== 2) return
        const nextKey = connected.find(candidateKey => candidateKey !== previousKey)
        if (!nextKey) return
        const next = remaining.get(nextKey)
        if (!next) return
        remaining.delete(nextKey)
        const nextPoint =
          coordinateKey(next.start) === coordinateKey(endpoint) ? next.end : next.start
        if (atStart) coordinates.unshift(nextPoint)
        else coordinates.push(nextPoint)
        previousKey = nextKey
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
  index: ReadonlyMap<string, readonly TraceEdge[]>,
  activities: ReadonlyMap<number, StreetMetricValues>,
  maximumVisits: number,
  roadFeatures: readonly unknown[],
): StreetMapFeatureCollection {
  const pieces = new Map<string, StreetPiece>()
  for (const feature of roadFeatures) {
    const road = roadLines(feature)
    if (!road) continue
    for (const line of road.lines) {
      for (let pointIndex = 1; pointIndex < line.length; pointIndex++) {
        const start = line[pointIndex - 1]
        const end = line[pointIndex]
        const projectedStart = project(start)
        const projectedEnd = project(end)
        const length = Math.hypot(
          projectedEnd.x - projectedStart.x,
          projectedEnd.y - projectedStart.y,
        )
        const divisions = Math.max(1, Math.ceil(length / ROAD_PIECE_M))
        for (let division = 0; division < divisions; division++) {
          const pieceStart = interpolate(start, end, division / divisions)
          const pieceEnd = interpolate(start, end, (division + 1) / divisions)
          const projectedPieceStart = project(pieceStart)
          const projectedPieceEnd = project(pieceEnd)
          const activityIds = matchingActivities(index, projectedPieceStart, projectedPieceEnd)
          if (activityIds.size === 0) continue
          const key = pieceKey(projectedPieceStart, projectedPieceEnd)
          const existing = pieces.get(key)
          if (existing) {
            for (const activityId of activityIds) existing.activityIds.add(activityId)
          } else {
            pieces.set(key, {
              start: pieceStart,
              end: pieceEnd,
              roadClass: road.roadClass,
              activityIds,
            })
          }
        }
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
