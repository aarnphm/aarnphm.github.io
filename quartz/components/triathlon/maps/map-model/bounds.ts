import type { GeoFC } from './geo'
import { isRecord } from '../../../../util/type-guards'

export const fcBounds = (fc: GeoFC): [[number, number], [number, number]] | null => {
  let minLng = Infinity
  let minLat = Infinity
  let maxLng = -Infinity
  let maxLat = -Infinity
  for (const f of fc.features) {
    if (!isRecord(f) || !isRecord(f.geometry) || !Array.isArray(f.geometry.coordinates)) continue
    for (const coordinate of f.geometry.coordinates) {
      if (
        Array.isArray(coordinate) &&
        coordinate.length >= 2 &&
        typeof coordinate[0] === 'number' &&
        typeof coordinate[1] === 'number'
      ) {
        const lng = coordinate[0]
        const lat = coordinate[1]
        if (lng < minLng) minLng = lng
        if (lng > maxLng) maxLng = lng
        if (lat < minLat) minLat = lat
        if (lat > maxLat) maxLat = lat
      }
    }
  }
  return Number.isFinite(minLng)
    ? [
        [minLng, minLat],
        [maxLng, maxLat],
      ]
    : null
}

export const DEFAULT_OVERVIEW_CENTER: [number, number] = [-79.4, 43.7]

export const DEFAULT_OVERVIEW_BOUNDS: [[number, number], [number, number]] = [
  [-80.102531, 43.249998],
  [-78.292016, 44.606476],
]

export const OVERVIEW_CLUSTER_RADIUS_DEG = 1.5

interface ActivityBoundsBox {
  minLng: number
  minLat: number
  maxLng: number
  maxLat: number
}

export const overviewBounds = (fc: GeoFC): [[number, number], [number, number]] | null => {
  const byActivity = new Map<string, ActivityBoundsBox>()
  let anonymousIndex = 0
  for (const f of fc.features) {
    if (!isRecord(f) || !isRecord(f.geometry) || !Array.isArray(f.geometry.coordinates)) continue
    const rawId = isRecord(f.properties) ? f.properties.id : undefined
    const key =
      typeof rawId === 'number' || typeof rawId === 'string'
        ? String(rawId)
        : `__anon_${anonymousIndex++}`
    let box = byActivity.get(key)
    for (const coordinate of f.geometry.coordinates) {
      if (
        Array.isArray(coordinate) &&
        coordinate.length >= 2 &&
        typeof coordinate[0] === 'number' &&
        typeof coordinate[1] === 'number' &&
        Number.isFinite(coordinate[0]) &&
        Number.isFinite(coordinate[1])
      ) {
        const lng = coordinate[0]
        const lat = coordinate[1]
        if (!box) {
          box = { minLng: lng, minLat: lat, maxLng: lng, maxLat: lat }
          byActivity.set(key, box)
        } else {
          if (lng < box.minLng) box.minLng = lng
          if (lng > box.maxLng) box.maxLng = lng
          if (lat < box.minLat) box.minLat = lat
          if (lat > box.maxLat) box.maxLat = lat
        }
      }
    }
  }
  const boxes = [...byActivity.values()]
  if (boxes.length === 0) return null
  if (boxes.length === 1) {
    const b = boxes[0]
    return [
      [b.minLng, b.minLat],
      [b.maxLng, b.maxLat],
    ]
  }
  const visited = new Uint8Array(boxes.length)
  let bestCount = -1
  let bestCenterDistSq = Infinity
  let bestBounds: [[number, number], [number, number]] | null = null
  const radius = OVERVIEW_CLUSTER_RADIUS_DEG
  for (let i = 0; i < boxes.length; i++) {
    if (visited[i]) continue
    visited[i] = 1
    const queue = [i]
    let minLng = boxes[i].minLng
    let minLat = boxes[i].minLat
    let maxLng = boxes[i].maxLng
    let maxLat = boxes[i].maxLat
    for (let head = 0; head < queue.length; head++) {
      const current = boxes[queue[head]]
      if (current.minLng < minLng) minLng = current.minLng
      if (current.maxLng > maxLng) maxLng = current.maxLng
      if (current.minLat < minLat) minLat = current.minLat
      if (current.maxLat > maxLat) maxLat = current.maxLat
      for (let j = 0; j < boxes.length; j++) {
        if (visited[j]) continue
        const other = boxes[j]
        if (
          current.minLng - radius <= other.maxLng &&
          other.minLng - radius <= current.maxLng &&
          current.minLat - radius <= other.maxLat &&
          other.minLat - radius <= current.maxLat
        ) {
          visited[j] = 1
          queue.push(j)
        }
      }
    }
    const count = queue.length
    const centerLng = (minLng + maxLng) * 0.5
    const centerLat = (minLat + maxLat) * 0.5
    const dLng = centerLng - DEFAULT_OVERVIEW_CENTER[0]
    const dLat = centerLat - DEFAULT_OVERVIEW_CENTER[1]
    const centerDistSq = dLng * dLng + dLat * dLat
    if (count > bestCount || (count === bestCount && centerDistSq < bestCenterDistSq)) {
      bestCount = count
      bestCenterDistSq = centerDistSq
      bestBounds = [
        [minLng, minLat],
        [maxLng, maxLat],
      ]
    }
  }
  return bestBounds
}
