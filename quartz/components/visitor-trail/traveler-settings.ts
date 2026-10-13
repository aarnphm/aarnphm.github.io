export type TravelerSettings = {
  walkers: number
  climbers: number
  alpinists: number
  rangers: number
}

export const defaultTravelerSettings: TravelerSettings = {
  walkers: 3,
  climbers: 1,
  alpinists: 2,
  rangers: 1,
}
export const travelerLimits = {
  walkers: { min: 0, max: 6, step: 1 },
  climbers: { min: 0, max: 6, step: 1 },
  alpinists: { min: 0, max: 6, step: 1 },
  rangers: { min: 0, max: 3, step: 1 },
}
