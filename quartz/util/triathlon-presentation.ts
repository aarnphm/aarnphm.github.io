export type Locale = 'en' | 'fr'

export type DistanceSystem = 'metric' | 'imperial'

export type PowerSamplePolicy = 'recorded' | 'exclude-zero'

export interface TriathlonPresentation {
  locale: Locale
  distance: DistanceSystem
  powerSamples: PowerSamplePolicy
}

export const DEFAULT_TRIATHLON_PRESENTATION: TriathlonPresentation = Object.freeze({
  locale: 'en',
  distance: 'metric',
  powerSamples: 'recorded',
})
