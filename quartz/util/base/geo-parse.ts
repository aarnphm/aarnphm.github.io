import type {
  BaseGeoMember,
  BaseGeoPayload,
  GazetteerFeature,
  GazetteerKind,
  SpatialHit,
  SpatialQuery,
} from './geo-types'

const EARTH_M = 6_371_000
const ACTIVE_M = 25_000
const CLUSTER_M = 100
const NEAR_POINT_M = 800
const NEAR_AREA_M = 250
const NEAR_CITY_M = 2000
const IN_PARK_M = 50
const SELF = new Set(['me', 'here', 'my location', 'current location'])
const NEAR_ME = 'Location "near me" is unavailable offline. Name a saved place or area.'
const KIND_RANK: Record<GazetteerKind, number> = { city: 0, neighbourhood: 1, park: 2 }

type BBox = [number, number, number, number]
type Operator = 'near' | 'around' | 'within' | 'in'
type Resolved =
  | { kind: 'point'; lat: number; lon: number }
  | { kind: 'feature'; feature: GazetteerFeature }
  | SpatialQuery

export function normalizeAnchorText(value: string): string {
  return value
    .normalize('NFD')
    .replace(/\u0300-\u036f/g, '')
    .toLowerCase()
    .replace(/['’]s/g, 's')
    .replace(/[|@&+,./()-]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

export function haversineMeters(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const rad = Math.PI / 180
  const dLat = (lat2 - lat1) * rad
  const dLon = (lon2 - lon1) * rad
  const a =
    Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * rad) * Math.cos(lat2 * rad) * Math.sin(dLon / 2) ** 2
  return 2 * EARTH_M * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
}

function pinned(member: BaseGeoMember): member is BaseGeoMember & { lat: number; lon: number } {
  return (
    typeof member.lat === 'number' &&
    typeof member.lon === 'number' &&
    Number.isFinite(member.lat) &&
    Number.isFinite(member.lon)
  )
}

function expandBBox(bbox: BBox, meters: number): BBox {
  if (meters === 0) return bbox
  const midLat = (bbox[0] + bbox[1]) / 2
  const dLat = meters / 111320
  const dLon = meters / (111320 * Math.cos((midLat * Math.PI) / 180))
  return [bbox[0] - dLat, bbox[1] + dLat, bbox[2] - dLon, bbox[3] + dLon]
}

function intersects(left: BBox, right: BBox): boolean {
  return left[0] <= right[1] && left[1] >= right[0] && left[2] <= right[3] && left[3] >= right[2]
}

function contains(bbox: BBox, lat: number, lon: number): boolean {
  return lat >= bbox[0] && lat <= bbox[1] && lon >= bbox[2] && lon <= bbox[3]
}

function sortHits(hits: SpatialHit[]): SpatialHit[] {
  hits.sort(
    (left, right) =>
      left.distanceM - right.distanceM ||
      (left.slug < right.slug ? -1 : left.slug > right.slug ? 1 : 0),
  )
  return hits
}

function pinBox(members: readonly BaseGeoMember[]): BBox | null {
  let minLat = Infinity
  let maxLat = -Infinity
  let minLon = Infinity
  let maxLon = -Infinity
  let found = false
  for (const member of members) {
    if (!pinned(member)) continue
    found = true
    minLat = Math.min(minLat, member.lat)
    maxLat = Math.max(maxLat, member.lat)
    minLon = Math.min(minLon, member.lon)
    maxLon = Math.max(maxLon, member.lon)
  }
  if (!found) return null
  return expandBBox([minLat, maxLat, minLon, maxLon], ACTIVE_M)
}

function activeFeatures(
  members: readonly BaseGeoMember[],
  features: readonly GazetteerFeature[],
): GazetteerFeature[] {
  const box = pinBox(members)
  if (!box) return []
  return features.filter(feature => intersects(feature.bbox, box))
}

function stringsOf(feature: GazetteerFeature): string[] {
  return [feature.name, ...feature.aliases].map(normalizeAnchorText)
}

function exactFeature(
  features: readonly GazetteerFeature[],
  needle: string,
): GazetteerFeature | null {
  const matched = features.filter(feature => stringsOf(feature).includes(needle))
  if (matched.length === 0) return null
  const best = Math.min(...matched.map(feature => KIND_RANK[feature.kind]))
  const top = matched.filter(feature => KIND_RANK[feature.kind] === best)
  return top.length === 1 ? top[0] : null
}

function prefixFeatures(features: readonly GazetteerFeature[], needle: string): GazetteerFeature[] {
  return features.filter(feature => stringsOf(feature).some(value => value.startsWith(needle)))
}

function levenshtein(left: string, right: string): number {
  if (left === right) return 0
  if (Math.abs(left.length - right.length) > 1) return 2
  const prev = Array.from({ length: right.length + 1 }, (_, index) => index)
  const curr = Array.from({ length: right.length + 1 }, () => 0)
  for (let i = 1; i <= left.length; i++) {
    curr[0] = i
    let rowMin = curr[0]
    for (let j = 1; j <= right.length; j++) {
      const cost = left[i - 1] === right[j - 1] ? 0 : 1
      const insert = curr[j - 1] + 1
      const remove = prev[j] + 1
      const replace = prev[j - 1] + cost
      const next = insert < remove ? insert : remove
      curr[j] = next < replace ? next : replace
      if (curr[j] < rowMin) rowMin = curr[j]
    }
    if (rowMin > 1) return 2
    for (let j = 0; j <= right.length; j++) prev[j] = curr[j]
  }
  return prev[right.length] ?? 2
}

function close(needle: string, value: string): boolean {
  return levenshtein(needle, value) === 1
}

function several(anchor: string, titles: string[]): SpatialQuery {
  return {
    kind: 'blocked',
    notice: `Several saved places match "${anchor}": ${titles.join(', ')}.`,
  }
}

function missing(title: string): SpatialQuery {
  return { kind: 'blocked', notice: `No coordinates saved for "${title}".` }
}

function unknown(anchor: string): SpatialQuery {
  return { kind: 'blocked', notice: `Unknown location "${anchor}".` }
}

function mean(values: number[]): number {
  let sum = 0
  for (const value of values) sum += value
  return sum / values.length
}

function clustered(members: readonly (BaseGeoMember & { lat: number; lon: number })[]): boolean {
  for (let i = 0; i < members.length; i++) {
    for (let j = i + 1; j < members.length; j++) {
      if (
        haversineMeters(members[i].lat, members[i].lon, members[j].lat, members[j].lon) > CLUSTER_M
      ) {
        return false
      }
    }
  }
  return true
}

function pointOf(anchor: string, members: readonly BaseGeoMember[]): Resolved {
  const missingMember = members.find(member => !pinned(member))
  if (missingMember) return missing(missingMember.title)
  const pins = members.filter(pinned)
  if (pins.length === 0) return unknown(anchor)
  if (pins.length > 1 && !clustered(pins))
    return several(
      anchor,
      pins.map(member => member.title),
    )
  return {
    kind: 'point',
    lat: mean(pins.map(member => member.lat)),
    lon: mean(pins.map(member => member.lon)),
  }
}

function resolveNear(
  anchor: string,
  members: readonly BaseGeoMember[],
  features: readonly GazetteerFeature[],
): Resolved {
  const needle = normalizeAnchorText(anchor)
  if (SELF.has(needle)) return { kind: 'blocked', notice: NEAR_ME }
  const exact = members.filter(member => normalizeAnchorText(member.title) === needle)
  if (exact.length > 0) return pointOf(anchor, exact)
  const named = exactFeature(features, needle)
  if (named) return { kind: 'feature', feature: named }
  if (needle.length >= 5) {
    const feats = prefixFeatures(features, needle)
    const titles = members.filter(member => normalizeAnchorText(member.title).startsWith(needle))
    if (feats.length === 1) return { kind: 'feature', feature: feats[0] }
    if (feats.length === 0 && titles.length === 1) return pointOf(anchor, titles)
    if (titles.length > 1 && feats.length !== 1)
      return several(
        anchor,
        titles.map(member => member.title),
      )
  }
  if (needle.length >= 4) {
    const fuzzyMembers = members.filter(member => close(needle, normalizeAnchorText(member.title)))
    const fuzzyFeatures = features.filter(feature =>
      stringsOf(feature).some(value => close(needle, value)),
    )
    if (fuzzyMembers.length + fuzzyFeatures.length === 1) {
      const feature = fuzzyFeatures[0]
      if (feature) return { kind: 'feature', feature }
      const member = fuzzyMembers[0]
      if (member) return pointOf(anchor, [member])
    }
  }
  return unknown(anchor)
}

function resolveIn(anchor: string, features: readonly GazetteerFeature[]): GazetteerFeature | null {
  const needle = normalizeAnchorText(anchor)
  const named = exactFeature(features, needle)
  if (named) return named
  if (needle.length >= 5) {
    const feats = prefixFeatures(features, needle)
    if (feats.length === 1) return feats[0]
  }
  if (needle.length >= 4) {
    const fuzzy = features.filter(feature => stringsOf(feature).some(value => close(needle, value)))
    if (fuzzy.length === 1) return fuzzy[0] ?? null
  }
  return null
}

function diskHits(
  members: readonly BaseGeoMember[],
  lat: number,
  lon: number,
  radius: number,
): SpatialHit[] {
  const hits: SpatialHit[] = []
  for (const member of members) {
    if (!pinned(member)) continue
    const distanceM = haversineMeters(member.lat, member.lon, lat, lon)
    if (distanceM <= radius) hits.push({ slug: member.slug, distanceM })
  }
  return sortHits(hits)
}

function areaHits(
  members: readonly BaseGeoMember[],
  feature: GazetteerFeature,
  meters: number,
): SpatialHit[] {
  const box = expandBBox(feature.bbox, meters)
  const hits: SpatialHit[] = []
  for (const member of members) {
    if (!pinned(member) || !contains(box, member.lat, member.lon)) continue
    hits.push({
      slug: member.slug,
      distanceM: haversineMeters(member.lat, member.lon, feature.centroid[0], feature.centroid[1]),
    })
  }
  return sortHits(hits)
}

function expansion(kind: GazetteerKind, operator: Operator, meters: number): number {
  if (operator === 'within') return meters
  if (operator === 'in') return kind === 'park' ? IN_PARK_M : 0
  return kind === 'city' ? NEAR_CITY_M : NEAR_AREA_M
}

function findWithin(text: string): { meters: number; anchor: string; residual: string } | null {
  const re = /\bwithin\s+(\d+(?:\.\d+)?|\.\d+)\s*(km|m)\s+of\b/gi
  let last: RegExpExecArray | null = null
  for (let match = re.exec(text); match; match = re.exec(text)) last = match
  if (!last) return null
  const amount = Number(last[1])
  const unit = (last[2] ?? '').toLowerCase()
  return {
    meters: unit === 'km' ? amount * 1000 : amount,
    residual: text.slice(0, last.index).trim(),
    anchor: text.slice(last.index + last[0].length).trim(),
  }
}

function findWord(
  text: string,
  source: string,
): { word: 'near' | 'around' | 'in'; anchor: string; residual: string } | null {
  const re = new RegExp(source, 'gi')
  let last: RegExpExecArray | null = null
  for (let match = re.exec(text); match; match = re.exec(text)) last = match
  if (!last) return null
  const word = (last[1] ?? '').toLowerCase()
  if (word !== 'near' && word !== 'around' && word !== 'in') return null
  return {
    word,
    residual: text.slice(0, last.index).trim(),
    anchor: text.slice(last.index + last[0].length).trim(),
  }
}

function placeFeatures(features: readonly GazetteerFeature[]): GazetteerFeature[] {
  return features.filter(
    feature =>
      feature.kind === 'park' || feature.kind === 'neighbourhood' || feature.kind === 'city',
  )
}

function fromResolved(
  residual: string,
  operator: Operator,
  meters: number,
  resolved: Resolved,
  members: readonly BaseGeoMember[],
): SpatialQuery {
  if (resolved.kind === 'blocked' || resolved.kind === 'none' || resolved.kind === 'spatial')
    return resolved
  if (resolved.kind === 'point') {
    const radius = operator === 'within' ? meters : NEAR_POINT_M
    return {
      kind: 'spatial',
      residual,
      hits: diskHits(members, resolved.lat, resolved.lon, radius),
      notice: '',
    }
  }
  return {
    kind: 'spatial',
    residual,
    hits: areaHits(members, resolved.feature, expansion(resolved.feature.kind, operator, meters)),
    notice: '',
  }
}

export function parseAndResolve(
  query: string,
  payload: BaseGeoPayload | null,
  features: GazetteerFeature[],
): SpatialQuery {
  const trimmed = query.trim()
  if (payload === null) return { kind: 'none', residual: trimmed }
  const within = findWithin(trimmed)
  if (within) {
    if (within.anchor.length === 0 || !(within.meters > 0) || !Number.isFinite(within.meters)) {
      return { kind: 'none', residual: trimmed }
    }
    const resolved = resolveNear(
      within.anchor,
      payload.members,
      activeFeatures(payload.members, features),
    )
    return fromResolved(within.residual, 'within', within.meters, resolved, payload.members)
  }
  const near = findWord(trimmed, '\\b(near|around)\\b')
  if (near) {
    if (near.anchor.length === 0) return { kind: 'none', residual: trimmed }
    const resolved = resolveNear(
      near.anchor,
      payload.members,
      activeFeatures(payload.members, features),
    )
    return fromResolved(near.residual, near.word, 0, resolved, payload.members)
  }
  const inside = findWord(trimmed, '\\b(in)\\b')
  if (inside) {
    if (inside.anchor.length === 0) return { kind: 'none', residual: trimmed }
    const feature = resolveIn(inside.anchor, placeFeatures(features))
    if (!feature) return { kind: 'none', residual: trimmed }
    return fromResolved(inside.residual, 'in', 0, { kind: 'feature', feature }, payload.members)
  }
  return { kind: 'none', residual: trimmed }
}
