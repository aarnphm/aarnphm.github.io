export type GazetteerKind = 'park' | 'neighbourhood' | 'city'

export interface GazetteerFeature {
  id: string
  name: string
  aliases: string[]
  kind: GazetteerKind
  city: string
  centroid: [number, number]
  bbox: [number, number, number, number]
}

export interface GazetteerFile {
  licence: string
  features: GazetteerFeature[]
}

export interface BaseGeoFields {
  title: string
  type: string
  description: string
  address: string
}

export interface BaseGeoMember {
  slug: string
  title: string
  lat?: number
  lon?: number
  searchText: string
  fields: BaseGeoFields
}

export interface BaseGeoPayload {
  version: 1
  members: BaseGeoMember[]
}

export interface SpatialHit {
  slug: string
  distanceM: number
}

export type SpatialQuery =
  | { kind: 'none'; residual: string }
  | { kind: 'blocked'; notice: string }
  | { kind: 'spatial'; residual: string; hits: SpatialHit[]; notice: string }
