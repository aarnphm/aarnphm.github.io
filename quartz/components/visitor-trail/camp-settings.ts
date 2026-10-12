export type CampSettings = {
  sites: number
  tents: number
  builders: number
  light: 'weather' | 'day' | 'night'
}
export const defaultCampSettings: CampSettings = {
  sites: 2,
  tents: 1,
  builders: 1,
  light: 'weather',
}
export const campLimits = {
  sites: { min: 1, max: 3, step: 1 },
  tents: { min: 0, max: 3, step: 1 },
  builders: { min: 0, max: 2, step: 1 },
}
