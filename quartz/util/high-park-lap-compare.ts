/**
 * Comparing laps on the High Park page: which laps are picked and in what colour slot, the
 * all-lap baseline a pick is read against, and grade-adjusted speed for laps on foot.
 */
import { LAP_BINS } from './high-park-lap-metrics'

/** Laps that can be picked at once; one colour slot each. */
export const PICK_SLOTS = 4

export interface LapPick {
  lap: number
  /** Colour slot, 0 to PICK_SLOTS - 1; a pick keeps its slot while other picks come and go. */
  slot: number
}

/**
 * Picks in the order they were made; the first is the reference the others are read against.
 * Picking a lap again drops it. A pick past the last slot replaces the newest pick, so the
 * reference stays put.
 */
export const toggleLap = (picks: readonly LapPick[], lap: number): LapPick[] => {
  if (picks.some(pick => pick.lap === lap)) return picks.filter(pick => pick.lap !== lap)
  const kept = picks.length < PICK_SLOTS ? [...picks] : picks.slice(0, -1)
  let slot = 0
  while (kept.some(pick => pick.slot === slot)) slot++
  return [...kept, { lap, slot }]
}

/** One pass of the binomial kernel [1 2 1]/4; an end bin weighs itself and its one neighbour. */
export const smoothBins = (values: readonly number[]): number[] =>
  values.map((value, i) => {
    let sum = 2 * value
    let weight = 2
    for (const near of [values[i - 1], values[i + 1]])
      if (near !== undefined) {
        sum += near
        weight++
      }
    return sum / weight
  })

export interface LapBaseline {
  /** Mean across laps per bin, smoothed across bins. */
  mean: number[]
  /** Standard deviation across laps per bin, smoothed the same way; zeros for a single lap. */
  sd: number[]
}

/** The average lap: null unless every bin was observed by at least one lap. */
export const lapBaseline = (rows: readonly (readonly (number | null)[])[]): LapBaseline | null => {
  const mean: number[] = []
  const sd: number[] = []
  for (let bin = 0; bin < LAP_BINS; bin++) {
    const seen: number[] = []
    for (const row of rows) {
      const value = row.length === LAP_BINS ? row[bin] : null
      if (value != null) seen.push(value)
    }
    if (seen.length === 0) return null
    const m = seen.reduce((a, b) => a + b, 0) / seen.length
    mean.push(m)
    sd.push(Math.sqrt(seen.reduce((a, b) => a + (b - m) ** 2, 0) / seen.length))
  }
  return { mean: smoothBins(mean), sd: smoothBins(sd) }
}

/**
 * Grade (rise over run) at each bin of a closed loop, by central difference over the mean-lap
 * altitude; the bin before the first is the last.
 */
export const binGrades = (altitude: readonly number[], lapKm: number): number[] => {
  const n = altitude.length
  const step = (lapKm * 1000) / n
  if (!(step > 0)) return altitude.map(() => 0)
  return altitude.map((_, i) => (altitude[(i + 1) % n] - altitude[(i - 1 + n) % n]) / (2 * step))
}

/**
 * Energy cost of running on a gradient, J/kg/m: the fifth-order fit of Minetti, Moia, Roi,
 * Susta and Ferretti, "Energy cost of walking and running at extreme uphill and downhill
 * slopes", J Appl Physiol 93 (2002) 1039-1046, measured for grades of -0.45 to +0.45.
 */
export const minettiCost = (grade: number): number => {
  const i = Math.min(0.45, Math.max(-0.45, grade))
  return 155.4 * i ** 5 - 30.4 * i ** 4 - 43.3 * i ** 3 + 46.3 * i ** 2 + 19.5 * i + 3.6
}

/** Speed on the flat that costs what `speed` costs on `grade`: grade-adjusted pace as a speed. */
export const gradeAdjusted = (speed: number, grade: number): number =>
  (speed * minettiCost(grade)) / minettiCost(0)

/** Mean speed over equal-distance bins (the harmonic mean); null when no bin has a speed. */
export const meanSpeed = (speeds: readonly (number | null)[]): number | null => {
  let time = 0
  let n = 0
  for (const speed of speeds)
    if (speed != null && speed > 0) {
      time += 1 / speed
      n++
    }
  return n ? n / time : null
}

/** A difference with its sign spelled out: +14, −0.7, ±0. */
export const signed = (delta: number, digits = 0): string => {
  const text = Math.abs(delta).toFixed(digits)
  if (Number(text) === 0) return '±0'
  return `${delta > 0 ? '+' : '−'}${text}`
}
