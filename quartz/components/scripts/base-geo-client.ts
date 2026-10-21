import type { BaseGeoPayload, GazetteerFeature, SpatialQuery } from '../../util/base/geo-types'
import { bm25Rank } from '../../util/base/bm25'
import gazetteerFile from '../../util/base/gazetteer.json'
import { parseAndResolve } from '../../util/base/geo-parse'
import { isRecord } from '../../util/type-guards'

function isGazetteerFeature(value: unknown): value is GazetteerFeature {
  if (!isRecord(value)) return false
  return (
    typeof value.id === 'string' &&
    typeof value.name === 'string' &&
    (value.kind === 'park' || value.kind === 'neighbourhood' || value.kind === 'city') &&
    Array.isArray(value.aliases) &&
    Array.isArray(value.centroid) &&
    value.centroid.length === 2 &&
    Array.isArray(value.bbox) &&
    value.bbox.length === 4
  )
}

function gazetteerFeatures(): GazetteerFeature[] {
  const file: unknown = gazetteerFile
  if (!isRecord(file) || !Array.isArray(file.features)) return []
  const features: GazetteerFeature[] = []
  for (const entry of file.features) {
    if (isGazetteerFeature(entry)) features.push(entry)
  }
  return features
}

function isBaseGeoPayload(value: unknown): value is BaseGeoPayload {
  return isRecord(value) && Array.isArray(value.members)
}

export function readGeoPayload(bar: HTMLElement): BaseGeoPayload | null {
  const raw = bar.dataset.baseGeo
  if (!raw) return null
  try {
    const parsed: unknown = JSON.parse(raw)
    return isBaseGeoPayload(parsed) ? parsed : null
  } catch {
    return null
  }
}

export function resolveBaseQuery(raw: string, payload: BaseGeoPayload | null): SpatialQuery {
  return parseAndResolve(raw, payload, gazetteerFeatures())
}

export function lexicalSlugs(
  payload: BaseGeoPayload,
  residual: string,
  allowed: ReadonlySet<string> | null,
): string[] {
  const ranked = bm25Rank(payload.members, residual)
  const slugs: string[] = []
  for (const { slug } of ranked) {
    if (allowed !== null && !allowed.has(slug)) continue
    slugs.push(slug)
  }
  return slugs
}
