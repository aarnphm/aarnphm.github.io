import type {
  TrainingPeaksCalendar,
  TrainingPeaksCalendarRange,
  TrainingPeaksCalendarStep,
  TrainingPeaksCalendarStructure,
  TrainingPeaksCalendarWorkout,
  TrainingPeaksCalendarZone,
  TrainingPeaksCalendarZoneBounds,
} from '../../../util/trainingpeaks-calendar'
import type { TriathlonFormatter } from '../runtime/formatter'
import { RUN_PACE_ZONE_NAMES } from '../../../util/run-pace-zones'
import { HR_ZONE_NAMES, KM_TO_MI, POWER_ZONE_NAMES, clock } from '../../../util/triathlon-card'

export type TrainingZoneSource = Pick<TrainingPeaksCalendar, 'zones' | 'localZones'> | null

export type StructuredWorkout = TrainingPeaksCalendarWorkout & {
  structure: TrainingPeaksCalendarStructure
}

export interface ZoneScale {
  source: 'trainingpeaks' | 'garden'
  threshold: number
  zones: TrainingPeaksCalendarZone[]
}

interface ZoneMatch {
  label: string
  short: string
  level: number
}

export const TARGET_LABEL = {
  percentOfFtp: '% FTP',
  percentOfThresholdPace: '% threshold pace',
  percentOfThresholdHr: '% threshold HR',
  rpe: 'RPE',
} as const

/**
 * TrainingPeaks plans RPE steps without a pace. Its planned IF for the RPE run of 2026-10-20
 * (0.76) fits intensity = RPE / 7 of threshold within 0.5%, so RPE 7 stands for threshold.
 * Only runs get an estimate: TrainingPeaks computes no IF for RPE rides.
 */
const RPE_THRESHOLD = 7
const RPE_MAX = 10

const estimated = (workout: StructuredWorkout): boolean =>
  workout.structure.metric === 'rpe' && workout.sport === 'run'

const zoneSetKey = (workout: StructuredWorkout): 'power' | 'speed' | 'heartRate' | null => {
  const { metric } = workout.structure
  if (metric === 'rpe') return estimated(workout) ? 'speed' : null
  return metric === 'percentOfFtp'
    ? 'power'
    : metric === 'percentOfThresholdHr'
      ? 'heartRate'
      : 'speed'
}

/** The target as a percentage of threshold, which sets bar height and zone. */
export const thresholdPercent = (workout: StructuredWorkout, value: number): number =>
  workout.structure.metric === 'rpe' ? (value / RPE_THRESHOLD) * 100 : value

const thresholdRange = (
  workout: StructuredWorkout,
  range: TrainingPeaksCalendarRange,
): TrainingPeaksCalendarRange => ({
  min: thresholdPercent(workout, range.min),
  max: range.max === null ? null : thresholdPercent(workout, range.max),
})

// The garden FTP and pace zones each belong to one sport; its heart rate zones cover both.
const gardenZones = (
  workout: StructuredWorkout,
  source: TrainingZoneSource,
): [TrainingPeaksCalendarZoneBounds, readonly string[]] | null => {
  const local = source?.localZones
  const key = zoneSetKey(workout)
  const sport = workout.sport
  const found =
    key === 'heartRate' && (sport === 'bike' || sport === 'run')
      ? ([local?.heartRate, HR_ZONE_NAMES] as const)
      : key === 'power' && sport === 'bike'
        ? ([local?.power, POWER_ZONE_NAMES] as const)
        : key === 'speed' && sport === 'run'
          ? ([local?.runSpeed, RUN_PACE_ZONE_NAMES] as const)
          : null
  return found?.[0] ? [found[0], found[1]] : null
}

// As in TrainingPeaks, the athlete default (workoutTypeId 0) covers sports without their own set.
export const zoneScaleFor = (
  workout: StructuredWorkout,
  source: TrainingZoneSource,
  text: TriathlonFormatter['text'],
): ZoneScale | null => {
  const key = zoneSetKey(workout)
  if (!key) return null
  const sets = source?.zones?.[key] ?? []
  const set =
    sets.find(item => item.workoutTypeId === workout.workoutTypeId) ??
    sets.find(item => item.workoutTypeId === 0)
  if (set) return { source: 'trainingpeaks', threshold: set.threshold, zones: set.zones }
  const garden = gardenZones(workout, source)
  if (!garden) return null
  const [{ threshold, bounds }, names] = garden
  return {
    source: 'garden',
    threshold,
    zones: [...bounds, Infinity].map((max, index) => ({
      label: text(names[index] ?? `Z${index + 1}`),
      min: index === 0 ? 0 : bounds[index - 1],
      max,
    })),
  }
}

const zoneAt = (scale: ZoneScale, percent: number): ZoneMatch => {
  const value = (percent / 100) * scale.threshold
  const found = scale.zones.findIndex(zone => value <= zone.max)
  const index = found < 0 ? scale.zones.length - 1 : found
  const label = scale.zones[index].label
  const id =
    scale.source === 'trainingpeaks'
      ? /^(?:zone|z)\s*(\d+[a-z]?|[a-z])\b/i.exec(label)?.[1]
      : String(index + 1)
  return {
    label,
    short: id ? `Z${id.toUpperCase()}` : label,
    // Any zone count spreads over the seven steps of the shared zone fill ramp.
    level: 1 + Math.round((index * 6) / Math.max(1, scale.zones.length - 1)),
  }
}

/** The fill level of a step top; RPE without a zone scale spreads 0–10 over the same ramp. */
export const stepLevel = (
  workout: StructuredWorkout,
  scale: ZoneScale | null,
  top: number | null,
): number | null => {
  if (top === null) return null
  if (scale) return zoneAt(scale, thresholdPercent(workout, top)).level
  return workout.structure.metric === 'rpe'
    ? Math.min(7, Math.max(1, Math.round((top / RPE_MAX) * 7)))
    : null
}

export const rangeValues = (range: TrainingPeaksCalendarRange): number[] =>
  range.max === null || range.max === range.min ? [range.min] : [range.min, range.max]

export const stepTop = (step: TrainingPeaksCalendarStep): number | null =>
  step.target?.max ?? step.target?.min ?? null

/** The target as TrainingPeaks writes it: "70–80%" or "RPE 3–5". */
export const targetText = (
  workout: StructuredWorkout,
  range: TrainingPeaksCalendarRange,
  formatter: TriathlonFormatter,
): string => {
  const values = rangeValues(range)
    .map(value => formatter.number(value, 0, 1))
    .join('–')
  return workout.structure.metric === 'rpe' ? `${formatter.text('RPE')} ${values}` : `${values}%`
}

const paceText = (speeds: number[], workout: StructuredWorkout, formatter: TriathlonFormatter) => {
  if (workout.sport === 'swim') return `${speeds.map(speed => clock(100 / speed)).join('–')} /100m`
  const imperial = formatter.presentation.distance === 'imperial'
  return `${speeds.map(speed => clock(1000 / speed / (imperial ? KM_TO_MI : 1))).join('–')} ${imperial ? '/mi' : '/km'}`
}

// TrainingPeaks scales threshold speed, so a higher percentage is a faster pace.
export const absoluteText = (
  workout: StructuredWorkout,
  scale: ZoneScale,
  range: TrainingPeaksCalendarRange,
  formatter: TriathlonFormatter,
): string => {
  const values = rangeValues(thresholdRange(workout, range)).map(
    value => (value / 100) * scale.threshold,
  )
  const metric = workout.structure.metric
  if (metric === 'percentOfFtp') return `${values.map(value => Math.round(value)).join('–')} W`
  if (metric === 'percentOfThresholdHr')
    return `${values.map(value => Math.round(value)).join('–')} bpm`
  if (metric !== 'rpe') return paceText(values, workout, formatter)
  // An easy RPE has no useful pace, so anything inside the first zone reads as slower than it.
  const floor = scale.zones[0].max
  if (values.every(value => value <= floor))
    return `${formatter.text('slower than')} ${paceText([floor], workout, formatter)}`
  return `≈ ${paceText(
    values.map(value => Math.max(value, floor)),
    workout,
    formatter,
  )}`
}

export const zoneRange = (
  workout: StructuredWorkout,
  scale: ZoneScale | null,
  range: TrainingPeaksCalendarRange | null,
): { short: string; label: string } | null => {
  if (!scale || !range) return null
  const percent = thresholdRange(workout, range)
  const low = zoneAt(scale, percent.min)
  const high = zoneAt(scale, percent.max ?? percent.min)
  return low.label === high.label
    ? low
    : { short: `${low.short}–${high.short}`, label: `${low.label} – ${high.label}` }
}

export const thresholdText = (
  workout: StructuredWorkout,
  scale: ZoneScale,
  formatter: TriathlonFormatter,
): string => {
  const metric = workout.structure.metric
  if (metric === 'percentOfFtp') return `FTP ${Math.round(scale.threshold)} W`
  if (metric === 'percentOfThresholdHr')
    return `${formatter.text('threshold HR')} ${Math.round(scale.threshold)} bpm`
  const pace = `${formatter.text('threshold pace')} ${paceText([scale.threshold], workout, formatter)}`
  return metric === 'rpe'
    ? `${pace} · ${formatter.text('pace estimated from RPE (RPE 7 ≈ threshold)')}`
    : pace
}

/** The source line under the step table. */
export const zoneSourceText = (
  workout: StructuredWorkout,
  scale: ZoneScale | null,
  formatter: TriathlonFormatter,
): string => {
  const { text } = formatter
  if (scale)
    return `${text(scale.source === 'garden' ? 'Garden zones' : 'TrainingPeaks zones')} · ${thresholdText(workout, scale, formatter)}`
  return text(workout.structure.metric === 'rpe' ? 'RPE scale 0–10' : 'No zones for this sport.')
}
