import type { HolidaySelection } from '../../util/visitor-holiday'
import type { TrailAnimalKind } from './animals'

export type ForestSettings = {
  density: number
  depth: number
  angle: number
  size: number
  curl: number
  branching: number
  variation: number
  dapple: number
}
export type AnimalSettings = Record<TrailAnimalKind, number>
export type CampSettings = {
  sites: number
  tents: number
  builders: number
  light: 'weather' | 'day' | 'night'
}
export type TravelerSettings = {
  walkers: number
  climbers: number
  alpinists: number
  rangers: number
}

export type SceneSettings = {
  forest: ForestSettings
  animals: AnimalSettings
  camps: CampSettings
  travelers: TravelerSettings
  holiday: HolidaySelection
  destination: number
}

// Each scene starts from these authored defaults and owns its tuning copy.
export const sceneSettings: SceneSettings = {
  forest: {
    density: 64,
    depth: 9,
    angle: 26,
    size: 240,
    curl: 4,
    branching: 85,
    variation: 70,
    dapple: 65,
  },
  animals: { cow: 3, dog: 1, cat: 1, goose: 3, bear: 2 },
  camps: { sites: 2, tents: 1, builders: 1, light: 'weather' },
  travelers: { walkers: 3, climbers: 1, alpinists: 2, rangers: 1 },
  holiday: 'auto',
  destination: 0,
}

// Tree planning reserves stable positions up to the density bound.
export const forestLimits: Record<keyof ForestSettings, { min: number; max: number }> = {
  density: { min: 8, max: 96 },
  depth: { min: 2, max: 9 },
  angle: { min: 15, max: 45 },
  size: { min: 60, max: 240 },
  curl: { min: 0, max: 40 },
  branching: { min: 0, max: 100 },
  variation: { min: 0, max: 100 },
  dapple: { min: 0, max: 100 },
}
export const alpineRouteCapacity = 6
