// Mean record power of 98 male WorldTour riders, Figure 2 (C, D), Valenzuela et al. (2022).
export const WORLD_TOUR_POWER_SOURCE_URL = 'https://doi.org/10.1123/ijspp.2021-0263'

export const WORLD_TOUR_POWER_REFERENCE = {
  sprint: { durationS: 5, wattsPerKg: 18.1 },
  climb: { durationS: 1200, wattsPerKg: 6.0 },
} satisfies Record<'sprint' | 'climb', { durationS: 5 | 1200; wattsPerKg: number }>
