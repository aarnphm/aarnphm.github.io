/**
 * Per-lap numbers and in-lap profiles for the High Park loops page, read from an activity's
 * distance-indexed records: the (privacy-trimmed) route samples, Wahoo/Garmin cycling dynamics,
 * the power/cadence torque trace and gear shifts. Every lookup is by activity distance, the same
 * axis the lap's segment effort uses, so no position data is needed.
 */

type Num = number | null | undefined

export interface LapSample {
  d: number
  alt?: Num
  w?: Num
  hr?: Num
  cad?: Num
  rightPowerPct?: Num
  speedKph?: Num
  resp?: Num
  coreTemperatureC?: Num
}
export interface LapMetricSource {
  route: readonly LapSample[]
  cyclingDynamics?: {
    distanceKm: readonly number[]
    leftTorqueEffectiveness: readonly Num[]
    rightTorqueEffectiveness: readonly Num[]
    leftPedalSmoothness: readonly Num[]
    rightPedalSmoothness: readonly Num[]
  } | null
  cyclingTorque?: {
    points: readonly { distanceKm: number; durationS: number; torqueNm: Num }[]
  } | null
  gearShifts?: readonly { distanceKm: number }[] | null
}

/** Bins across one lap, start line to finish line. */
export const LAP_BINS = 16

export interface LapProfile {
  w: (number | null)[]
  hr: (number | null)[]
  cad: (number | null)[]
  spd: (number | null)[]
}
export interface LapExtras {
  /** Mean torque, N·m, from the power/cadence trace. */
  nm: number | null
  /** Torque effectiveness and pedal smoothness, left and right, %. */
  teL: number | null
  teR: number | null
  psL: number | null
  psR: number | null
  /** Left leg's share of power, %. */
  balL: number | null
  shifts: number | null
  /** Metres climbed inside the lap. */
  gain: number | null
  /** Breaths per minute and core temperature, °C. */
  resp: number | null
  core: number | null
  profile: LapProfile | null
}

const finite = (value: Num): value is number => typeof value === 'number' && Number.isFinite(value)
const round = (value: number, digits = 0): number => {
  const scale = 10 ** digits
  return Math.round(value * scale) / scale
}
const mean = (values: Iterable<Num>, digits = 0, positive = true): number | null => {
  let sum = 0
  let n = 0
  for (const value of values)
    if (finite(value) && (!positive || value > 0)) {
      sum += value
      n++
    }
  return n ? round(sum / n, digits) : null
}

// Missing bins take a straight line between their neighbours; a profile with fewer than half
// its bins observed is dropped rather than invented.
const fillBins = (bins: (number | null)[], digits: number): (number | null)[] | null => {
  const known = bins.flatMap((value, i) => (value === null ? [] : [i]))
  if (known.length < bins.length / 2) return null
  return bins.map((value, i) => {
    if (value !== null) return round(value, digits)
    const before = known.findLast(k => k < i)
    const after = known.find(k => k > i)
    if (before === undefined) return round(bins[after!]!, digits)
    if (after === undefined) return round(bins[before]!, digits)
    const t = (i - before) / (after - before)
    return round(bins[before]! + (bins[after]! - bins[before]!) * t, digits)
  })
}

const profileOf = (samples: readonly LapSample[], startKm: number, endKm: number) => {
  const span = endKm - startKm
  const sums = { w: [], hr: [], cad: [], spd: [], alt: [] } as Record<
    'w' | 'hr' | 'cad' | 'spd' | 'alt',
    [number, number][]
  >
  for (const key of Object.keys(sums) as (keyof typeof sums)[])
    sums[key] = Array.from({ length: LAP_BINS }, () => [0, 0])
  for (const sample of samples) {
    const bin = Math.min(LAP_BINS - 1, Math.floor(((sample.d - startKm) / span) * LAP_BINS))
    const add = (key: keyof typeof sums, value: Num, allowZero = false) => {
      if (!finite(value) || (!allowZero && value <= 0)) return
      sums[key][bin][0] += value
      sums[key][bin][1]++
    }
    // Zero watts and zero cadence are real (coasting); heart rate and speed are not.
    add('w', sample.w, true)
    add('hr', sample.hr)
    add('cad', sample.cad, true)
    add('spd', sample.speedKph)
    add('alt', sample.alt, true)
  }
  const avg = (key: keyof typeof sums) => sums[key].map(([sum, n]) => (n ? sum / n : null))
  return {
    w: fillBins(avg('w'), 0),
    hr: fillBins(avg('hr'), 0),
    cad: fillBins(avg('cad'), 0),
    spd: fillBins(avg('spd'), 1),
    alt: fillBins(avg('alt'), 1),
  }
}

const inLap = (d: number, startKm: number, endKm: number) => d >= startKm && d < endKm

export function lapExtras(
  source: LapMetricSource,
  startKm: number | null,
  endKm: number | null,
): LapExtras & { altitude: (number | null)[] | null } {
  const empty = {
    nm: null,
    teL: null,
    teR: null,
    psL: null,
    psR: null,
    balL: null,
    shifts: null,
    gain: null,
    resp: null,
    core: null,
    profile: null,
    altitude: null,
  }
  if (startKm === null || endKm === null || !(endKm > startKm)) return empty
  const samples = source.route.filter(sample => finite(sample.d) && inLap(sample.d, startKm, endKm))
  const dynamics = source.cyclingDynamics
  const dynamicIndex: number[] = []
  if (dynamics)
    dynamics.distanceKm.forEach((d, i) => {
      if (inLap(d, startKm, endKm)) dynamicIndex.push(i)
    })
  const pick = (values: readonly Num[] | undefined) => dynamicIndex.map(i => values?.[i])
  let torque = 0
  let torqueS = 0
  for (const point of source.cyclingTorque?.points ?? [])
    if (inLap(point.distanceKm, startKm, endKm) && finite(point.torqueNm) && point.durationS > 0) {
      torque += point.torqueNm * point.durationS
      torqueS += point.durationS
    }
  let gain = 0
  let previousAlt: number | null = null
  for (const sample of samples) {
    if (!finite(sample.alt)) continue
    if (previousAlt !== null && sample.alt > previousAlt) gain += sample.alt - previousAlt
    previousAlt = sample.alt
  }
  const right = mean(samples.map(s => s.rightPowerPct))
  const profile = samples.length >= LAP_BINS / 2 ? profileOf(samples, startKm, endKm) : null
  return {
    nm: torqueS > 0 ? round(torque / torqueS, 1) : null,
    teL: mean(pick(dynamics?.leftTorqueEffectiveness)),
    teR: mean(pick(dynamics?.rightTorqueEffectiveness)),
    psL: mean(pick(dynamics?.leftPedalSmoothness)),
    psR: mean(pick(dynamics?.rightPedalSmoothness)),
    balL: right === null ? null : 100 - right,
    shifts: source.gearShifts?.length
      ? source.gearShifts.filter(shift => inLap(shift.distanceKm, startKm, endKm)).length
      : null,
    gain: previousAlt === null ? null : Math.round(gain),
    resp: mean(
      samples.map(s => s.resp),
      1,
    ),
    core: mean(
      samples.map(s => s.coreTemperatureC),
      2,
    ),
    profile:
      profile && (profile.w || profile.hr || profile.cad || profile.spd)
        ? {
            w: profile.w ?? [],
            hr: profile.hr ?? [],
            cad: profile.cad ?? [],
            spd: profile.spd ?? [],
          }
        : null,
    altitude: profile?.alt ?? null,
  }
}

/** The ride's average lap altitude by bin, in metres above its lowest bin. */
export function rideAltitude(profiles: readonly ((number | null)[] | null)[]): number[] | null {
  const rows = profiles.filter((row): row is (number | null)[] => row !== null)
  if (!rows.length) return null
  const bins = Array.from({ length: LAP_BINS }, (_, i) =>
    mean(
      rows.map(row => row[i]),
      1,
      false,
    ),
  )
  if (bins.some(value => value === null)) return null
  const floor = Math.min(...(bins as number[]))
  return (bins as number[]).map(value => round(value - floor, 1))
}
