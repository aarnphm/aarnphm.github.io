import type { HighParkLap } from '../../../../util/high-park-loops'
import {
  binGrades,
  gradeAdjusted,
  lapBaseline,
  meanSpeed,
  signed,
  type LapBaseline,
} from '../../../../util/high-park-lap-compare'
import { LAP_BINS } from '../../../../util/high-park-lap-metrics'

export type LapSport = 'bike' | 'run'

export const clock = (seconds: number): string => {
  const rounded = Math.round(seconds)
  return `${Math.floor(rounded / 60)}:${String(rounded % 60).padStart(2, '0')}`
}
export const r1 = (value: number): number => Math.round(value * 10) / 10

/** Display units, after the triathlon pages' unit preference. */
export interface HighParkUnits {
  imperial: boolean
  speed: string
  distance: string
  climb: string
  torque: string
  temp: string
  /** km → display distance, km/h → display speed, m → display climb, N·m → torque, °C → temp. */
  km: (km: number) => number
  kmh: (kmh: number) => number
  m: (m: number) => number
  nm: (nm: number) => number
  celsius: (celsius: number) => number
}
export const highParkUnits = (imperial: boolean): HighParkUnits => ({
  imperial,
  speed: imperial ? 'mph' : 'km/h',
  distance: imperial ? 'mi' : 'km',
  climb: imperial ? 'ft' : 'm',
  torque: imperial ? 'lb·ft' : 'N·m',
  temp: imperial ? '°F' : '°C',
  km: km => (imperial ? km / 1.609344 : km),
  kmh: kmh => (imperial ? kmh / 1.609344 : kmh),
  m: m => (imperial ? m * 3.28084 : m),
  nm: nm => (imperial ? nm * 0.737562 : nm),
  celsius: celsius => (imperial ? (celsius * 9) / 5 + 32 : celsius),
})

/** A lap's speed in display units, to one decimal. */
export const lapSpeed = (lap: HighParkLap, units: HighParkUnits): number | null =>
  lap.km > 0 && lap.s > 0 ? r1(units.kmh((lap.km / lap.s) * 3600)) : null

/** Speed in display units → pace per display distance, m:ss. */
export const paceOf = (speed: number): string => (speed > 0 ? clock(3600 / speed) : '–')
// Pace differences read in seconds: slower is plus.
const paceDelta = (speed: number, reference: number): string => {
  const delta = Math.round(3600 / speed - 3600 / reference)
  if (delta === 0) return '±0'
  const size = Math.abs(delta)
  return `${delta > 0 ? '+' : '−'}${size < 60 ? `${size}s` : clock(size)}`
}

/** Per-lap profile viewBox: width, height, inset. */
export const PROFILE = { w: 300, h: 120, pad: 4 } as const
export type ProfileKey = 'w' | 'spd' | 'hr' | 'cad'
export interface ProfileField {
  key: ProfileKey
  label: string
  unit: string
  /** Stored value → display value. */
  scale: (value: number) => number
  format: (value: number) => string
  /** How far `value` sits from `reference`, signed. */
  delta: (value: number, reference: number) => string
}

/** The four profile charts, in reading order; on foot, speed reads as pace. */
export const profileFields = (units: HighParkUnits, sport: LapSport): ProfileField[] => {
  const same = (value: number) => value
  const whole = (key: ProfileKey, label: string, unit: string): ProfileField => ({
    key,
    label,
    unit,
    scale: same,
    format: value => value.toFixed(0),
    delta: (value, reference) => signed(value - reference),
  })
  const speed: ProfileField =
    sport === 'run'
      ? {
          key: 'spd',
          label: 'pace',
          unit: `/${units.distance}`,
          scale: units.kmh,
          format: paceOf,
          delta: paceDelta,
        }
      : {
          key: 'spd',
          label: 'speed',
          unit: units.speed,
          scale: units.kmh,
          format: value => value.toFixed(1),
          delta: (value, reference) => signed(value - reference, 1),
        }
  return [
    whole('w', 'power', 'W'),
    speed,
    whole('hr', 'heart rate', 'bpm'),
    whole('cad', 'cadence', sport === 'run' ? 'spm' : 'rpm'),
  ]
}

/** The y range a profile draws: 2nd–98th percentile, so one sprint bin cannot flatten the rest. */
export const profileRange = (values: number[]): [number, number] => {
  const sorted = [...values].sort((a, b) => a - b)
  const lo = sorted[Math.floor(sorted.length * 0.02)]
  const hi = sorted[Math.ceil(sorted.length * 0.98) - 1]
  return hi > lo ? [lo, hi] : [lo - 1, hi + 1]
}
export const profileY = (value: number, lo: number, hi: number): number => {
  const { h, pad } = PROFILE
  return r1(pad + (1 - Math.min(1, Math.max(0, (value - lo) / (hi - lo)))) * (h - 2 * pad))
}
export const profileX = (bin: number): number => r1((bin / (LAP_BINS - 1)) * PROFILE.w)

type Row = (number | null)[]
/** Everything one profile chart draws and reads, in display units. */
export interface ProfileModel {
  /** One row of bins per lap; empty when the lap has no profile. */
  rows: Row[]
  /**
   * The rows the baseline is built from and compared with. On foot the speed rows are
   * grade-adjusted over the mean-lap altitude; everything else is `rows`.
   */
  even: Row[]
  lo: number
  hi: number
  base: LapBaseline | null
  /** Each lap's whole-lap figure, plain and for the baseline comparison. */
  rest: (number | null)[]
  restEven: (number | null)[]
  /** The all-lap mean of `restEven`. */
  restBase: number | null
}

export const profileModel = (
  field: ProfileField,
  laps: readonly HighParkLap[],
  altitude: readonly number[] | null,
  lapKm: number,
  sport: LapSport,
): ProfileModel | null => {
  const rows = laps.map(lap =>
    (lap.profile?.[field.key] ?? []).map(v => (v === null ? null : field.scale(v))),
  )
  const known = rows.flat().filter((v): v is number => v !== null)
  if (!rows.some(row => row.length === LAP_BINS) || known.length === 0) return null
  const [lo, hi] = profileRange(known)
  const grades =
    sport === 'run' && field.key === 'spd' && altitude?.length === LAP_BINS
      ? binGrades(altitude, lapKm)
      : null
  const even = grades
    ? rows.map(row => row.map((v, i) => (v === null ? null : gradeAdjusted(v, grades[i]))))
    : rows
  const whole: Record<ProfileKey, (lap: HighParkLap) => number | null> = {
    w: lap => lap.w,
    hr: lap => lap.hr,
    cad: lap => lap.cad,
    spd: lap => (lap.km > 0 && lap.s > 0 ? field.scale((lap.km / lap.s) * 3600) : null),
  }
  const rest = laps.map(whole[field.key])
  const restEven = grades ? even.map(meanSpeed) : rest
  const seen = restEven.filter((v): v is number => v !== null)
  return {
    rows,
    even,
    lo,
    hi,
    base: lapBaseline(even),
    rest,
    restEven,
    restBase: seen.length ? seen.reduce((a, b) => a + b, 0) / seen.length : null,
  }
}
