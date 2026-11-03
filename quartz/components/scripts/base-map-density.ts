interface DensityFeature {
  type: 'Feature'
  geometry: { type: 'Point'; coordinates: [number, number] }
  properties: { count: number; radius: number }
}
interface DensityCollection {
  type: 'FeatureCollection'
  features: DensityFeature[]
}

interface Location {
  lon: number
  lat: number
}
interface Point {
  x: number
  y: number
}

const distance = (a: Point, b: Point) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2
const mean = (points: Point[]): Point => ({
  x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
  y: points.reduce((sum, point) => sum + point.y, 0) / points.length,
})
const farthest = (points: Point[], center: Point) =>
  points.reduce((best, point) => (distance(point, center) > distance(best, center) ? point : best))

// Bisecting k-means chooses k from the displayed scale, rather than forcing
// distant cities into a fixed number of groups. Each split reduces membership.
function partition(points: Point[], radius: number): Point[][] {
  const pending = [points]
  const groups: Point[][] = []
  while (pending.length) {
    const members = pending.pop()
    if (!members?.length) continue
    const center = mean(members)
    if (members.length < 3 || distance(farthest(members, center), center) <= radius ** 2) {
      groups.push(members)
      continue
    }
    let a = farthest(members, center)
    let b = farthest(members, a)
    let left: Point[] = []
    let right: Point[] = []
    for (let iteration = 0; iteration < 24; iteration++) {
      left = []
      right = []
      for (const point of members) {
        if (distance(point, a) <= distance(point, b)) left.push(point)
        else right.push(point)
      }
      if (!left.length || !right.length) break
      const nextA = mean(left)
      const nextB = mean(right)
      const shift = distance(a, nextA) + distance(b, nextB)
      a = nextA
      b = nextB
      if (shift < 0.01) break
    }
    if (!left.length || !right.length) groups.push(members)
    else pending.push(right, left)
  }
  return groups
}

export function clusterMapDensity(locations: readonly Location[], zoom: number): DensityCollection {
  const scale = 512 * 2 ** Math.max(0, Math.min(22, zoom))
  const points = locations
    .filter(point => Number.isFinite(point.lon) && Number.isFinite(point.lat))
    .map(point => {
      const sine = Math.sin((Math.max(-85.051129, Math.min(85.051129, point.lat)) * Math.PI) / 180)
      return {
        x: (((point.lon / 360 + 0.5) % 1) + 1) % 1,
        y: (0.5 - Math.log((1 + sine) / (1 - sine)) / (4 * Math.PI)) * scale,
      }
    })
    .sort((a, b) => a.x - b.x || a.y - b.y)
  // Cut the longitude circle at its largest empty gap, so dateline neighbors
  // share the same plane and their centroid stays near the dateline.
  let cut = 0
  let gap = -1
  for (let index = 0; index < points.length; index++) {
    const next = points[(index + 1) % points.length].x + (index === points.length - 1 ? 1 : 0)
    if (next - points[index].x > gap) {
      gap = next - points[index].x
      cut = next % 1
    }
  }
  for (const point of points) point.x = (point.x < cut ? point.x + 1 : point.x) * scale
  points.sort((a, b) => a.x - b.x || a.y - b.y)
  const features = partition(points, 96)
    .filter(members => members.length >= 3)
    .map((members): DensityFeature => {
      const center = mean(members)
      const spread = Math.sqrt(distance(farthest(members, center), center))
      const coordinates: [number, number] = [
        (((((center.x / scale) * 360) % 360) + 360) % 360) - 180,
        (Math.atan(Math.sinh(Math.PI * (1 - (2 * center.y) / scale))) * 180) / Math.PI,
      ]
      return {
        type: 'Feature',
        geometry: { type: 'Point', coordinates },
        properties: { count: members.length, radius: Math.max(64, spread * 1.5 + 32) },
      }
    })
  return { type: 'FeatureCollection', features }
}
