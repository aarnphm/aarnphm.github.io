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
  activityId: string
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
