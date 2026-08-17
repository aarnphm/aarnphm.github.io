import type { PowerCurvePoint } from './strava'

export type PowerRankSeriesKey = 'six-weeks' | 'year'
export type PowerSkill = 'sprint' | 'attack' | 'climb'
export type PowerRankMassSource = 'tracking' | 'garmin' | 'apple'
export type PowerRankLevelName =
  | 'aspiring'
  | 'intermediate'
  | 'athletic'
  | 'sport'
  | 'elite'
  | 'semi-pro'
  | 'national star'
  | 'world class'

export interface PowerRankLevel {
  level: number
  name: PowerRankLevelName
  percentile: number
}

export interface PowerRankThreshold extends PowerRankLevel {
  watts: number
  wattsPerKg: number
}

export interface PowerRankEffort {
  watts: number
  wattsPerKg: number
  level: number
  levelName: PowerRankLevelName | null
  percentile: number
  nextLevel: number | null
  nextLevelName: PowerRankLevelName | null
  nextWatts: number | null
  wattsToNext: number | null
}

export interface PowerRankMass {
  kg: number
  date: string
  source: PowerRankMassSource
}

export interface PowerRankAthlete {
  sex: 'M' | 'F'
  age: number
}

export interface PowerRankReference {
  source: 'strava-profile-snapshot'
  capturedDate: string
  sex: 'M'
  ageMin: number
  ageMax: number
  massKg: number
}

export interface PowerRankInterval {
  durationS: number
  skill: PowerSkill
  thresholds: PowerRankThreshold[]
  efforts: Record<PowerRankSeriesKey, PowerRankEffort | null>
}

export interface PowerRankBlock {
  massKg: number | null
  massDate: string | null
  massSource: PowerRankMassSource | null
  cohortEligible: boolean
  reference: PowerRankReference
  intervals: PowerRankInterval[]
}

export const POWER_RANK_LEVELS: readonly PowerRankLevel[] = [
  { level: 1, name: 'aspiring', percentile: 1 },
  { level: 2, name: 'intermediate', percentile: 31 },
  { level: 3, name: 'athletic', percentile: 46 },
  { level: 4, name: 'sport', percentile: 56 },
  { level: 5, name: 'elite', percentile: 71 },
  { level: 6, name: 'semi-pro', percentile: 85 },
  { level: 7, name: 'national star', percentile: 94 },
  { level: 8, name: 'world class', percentile: 98 },
]

export const POWER_SKILL_DURATIONS: readonly number[] = [
  15, 30, 60, 120, 180, 300, 600, 900, 1_200, 1_800, 2_700, 3_600,
]

const REFERENCE: PowerRankReference = {
  source: 'strava-profile-snapshot',
  capturedDate: '2026-08-16',
  sex: 'M',
  ageMin: 24,
  ageMax: 29,
  massKg: 84.36818082,
}

const REFERENCE_WATTS: Readonly<Record<number, readonly number[]>> = {
  15: [402, 603, 688, 741, 826, 945, 1_152, 1_291],
  30: [351, 526, 600, 647, 721, 825, 1_005, 1_126],
  60: [240, 360, 411, 443, 494, 565, 688, 771],
  120: [216, 324, 370, 398, 444, 508, 619, 694],
  180: [202, 303, 345, 372, 415, 475, 579, 648],
  300: [178, 267, 305, 328, 366, 419, 510, 572],
  600: [169, 253, 288, 311, 347, 396, 483, 541],
  900: [163, 244, 279, 301, 335, 383, 467, 523],
  1_200: [161, 241, 275, 296, 330, 378, 460, 516],
  1_800: [152, 227, 259, 279, 312, 356, 434, 486],
  2_700: [140, 210, 239, 258, 288, 329, 401, 449],
  3_600: [130, 194, 221, 239, 266, 304, 371, 415],
}

const round = (value: number, digits: number): number => {
  const factor = 10 ** digits
  return Math.round(value * factor) / factor
}

const skillAt = (durationS: number): PowerSkill =>
  durationS <= 60 ? 'sprint' : durationS <= 600 ? 'attack' : 'climb'

const effortAt = (points: readonly PowerCurvePoint[], durationS: number): PowerCurvePoint | null =>
  points.find(point => point.s === durationS) ?? null

const percentileAt = (wattsPerKg: number, thresholds: readonly number[]): number => {
  if (wattsPerKg <= 0) return 0
  if (wattsPerKg < thresholds[0])
    return (wattsPerKg / thresholds[0]) * POWER_RANK_LEVELS[0].percentile
  for (let index = 1; index < thresholds.length; index++) {
    if (wattsPerKg >= thresholds[index]) continue
    const lowerWattsPerKg = thresholds[index - 1]
    const upperWattsPerKg = thresholds[index]
    const lowerPercentile = POWER_RANK_LEVELS[index - 1].percentile
    const upperPercentile = POWER_RANK_LEVELS[index].percentile
    const fraction = (wattsPerKg - lowerWattsPerKg) / (upperWattsPerKg - lowerWattsPerKg)
    return lowerPercentile + fraction * (upperPercentile - lowerPercentile)
  }
  return POWER_RANK_LEVELS[POWER_RANK_LEVELS.length - 1].percentile
}

const rankEffort = (
  point: PowerCurvePoint | null,
  massKg: number,
  thresholds: readonly PowerRankThreshold[],
  referenceWattsPerKg: readonly number[],
): PowerRankEffort | null => {
  if (!point) return null
  const wattsPerKg = point.w / massKg
  const level = referenceWattsPerKg.filter(threshold => wattsPerKg >= threshold).length
  const current = level > 0 ? POWER_RANK_LEVELS[level - 1] : null
  const next = level < POWER_RANK_LEVELS.length ? POWER_RANK_LEVELS[level] : null
  const nextThreshold = level < thresholds.length ? thresholds[level] : null
  const percentile = round(percentileAt(wattsPerKg, referenceWattsPerKg), 1)
  return {
    watts: point.w,
    wattsPerKg: round(wattsPerKg, 2),
    level,
    levelName: current?.name ?? null,
    percentile,
    nextLevel: next?.level ?? null,
    nextLevelName: next?.name ?? null,
    nextWatts: nextThreshold?.watts ?? null,
    wattsToNext: nextThreshold == null ? null : Math.max(0, nextThreshold.watts - point.w),
  }
}

export const emptyPowerRank = (): PowerRankBlock => ({
  massKg: null,
  massDate: null,
  massSource: null,
  cohortEligible: false,
  reference: REFERENCE,
  intervals: [],
})

export const buildPowerRank = (
  sixWeeks: readonly PowerCurvePoint[],
  year: readonly PowerCurvePoint[],
  mass: PowerRankMass | null,
  athlete: PowerRankAthlete,
): PowerRankBlock => {
  const cohortEligible =
    athlete.sex === REFERENCE.sex &&
    athlete.age >= REFERENCE.ageMin &&
    athlete.age <= REFERENCE.ageMax
  if (mass == null || !Number.isFinite(mass.kg) || mass.kg <= 0)
    return { ...emptyPowerRank(), cohortEligible }
  const rankedMass = {
    massKg: round(mass.kg, 2),
    massDate: mass.date,
    massSource: mass.source,
    cohortEligible,
    reference: REFERENCE,
  }
  if (!cohortEligible) return { ...rankedMass, intervals: [] }
  const intervalRows = POWER_SKILL_DURATIONS.flatMap(durationS => {
    const reference = REFERENCE_WATTS[durationS]
    if (!reference) return []
    const referenceWattsPerKg = reference.map(watts => watts / REFERENCE.massKg)
    const thresholds = POWER_RANK_LEVELS.map(
      (level, index): PowerRankThreshold => ({
        ...level,
        watts: Math.ceil(referenceWattsPerKg[index] * mass.kg),
        wattsPerKg: round(referenceWattsPerKg[index], 2),
      }),
    )
    return [
      {
        durationS,
        skill: skillAt(durationS),
        thresholds,
        efforts: {
          'six-weeks': rankEffort(
            effortAt(sixWeeks, durationS),
            mass.kg,
            thresholds,
            referenceWattsPerKg,
          ),
          year: rankEffort(effortAt(year, durationS), mass.kg, thresholds, referenceWattsPerKg),
        },
      },
    ]
  })
  return { ...rankedMass, intervals: intervalRows }
}
