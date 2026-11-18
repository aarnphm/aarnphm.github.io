import type {
  PowerToWeightDurationS,
  PowerToWeightEffort,
  PowerToWeightReference,
  PowerToWeightTrendPoint,
} from '../../../../../plugins/stores/analytics'
import type { PowerCurvePoint } from '../../../../../plugins/stores/strava'
import { POWER_TO_WEIGHT_DURATIONS } from '../../../../../plugins/stores/analytics'
import { niceStep, type AxisXTick } from '../../../../../util/triathlon-card'

export const W = 100

export const H = 34

export const restorePowerToWeightDurations = (
  stored: unknown,
  available: ReadonlySet<PowerToWeightDurationS>,
): Set<PowerToWeightDurationS> => {
  const selected = new Set(
    POWER_TO_WEIGHT_DURATIONS.filter(
      durationS => available.has(durationS) && Array.isArray(stored) && stored.includes(durationS),
    ),
  )
  return selected.size > 0 ? selected : new Set(available)
}

export const powerToWeightDurationLabel = (durationS: PowerToWeightDurationS): string =>
  durationS < 60 ? `${durationS}s` : `${durationS / 60}m`

export const deconflictPowerToWeightMonthTicks = (ticks: AxisXTick[]): AxisXTick[] => {
  if (ticks.length < 2 || ticks[1].pct - ticks[0].pct >= 5) return ticks
  return ticks
    .slice(1)
    .map((tick, index) => (index === 0 ? { ...tick, cls: 'tri-cax-xt--first' } : tick))
}

export const effortAt = (
  point: PowerToWeightTrendPoint,
  durationS: PowerToWeightDurationS,
): PowerToWeightEffort | null => point.efforts[durationS]

export const effortPowerPoint = (effort: PowerToWeightEffort | null): PowerCurvePoint | null =>
  effort == null
    ? null
    : {
        s: effort.durationS,
        w: effort.watts,
        activityId: effort.activityId,
        activityDate: effort.activityDate,
      }

export const durationFrom = (value: string | undefined): PowerToWeightDurationS | null =>
  POWER_TO_WEIGHT_DURATIONS.find(durationS => String(durationS) === value) ?? null

export const seriesPath = (
  points: readonly PowerToWeightTrendPoint[],
  durationS: PowerToWeightDurationS,
  y: (value: number) => number,
): string => {
  let drawing = false
  const commands: string[] = []
  for (let index = 0; index < points.length; index++) {
    const effort = effortAt(points[index], durationS)
    if (effort == null) {
      drawing = false
      continue
    }
    const x = points.length === 1 ? W / 2 : (index / (points.length - 1)) * W
    commands.push(`${drawing ? 'L' : 'M'} ${x.toFixed(2)} ${y(effort.wattsPerKg).toFixed(2)}`)
    drawing = true
  }
  return commands.join(' ')
}

const observedMax = (
  points: readonly PowerToWeightTrendPoint[],
  active: ReadonlySet<PowerToWeightDurationS>,
  references: readonly PowerToWeightReference[],
): number => {
  let maximum = 1
  for (const point of points)
    for (const durationS of active) {
      const effort = effortAt(point, durationS)
      if (effort != null) maximum = Math.max(maximum, effort.wattsPerKg)
    }
  for (const reference of references)
    if (active.has(reference.durationS)) maximum = Math.max(maximum, reference.average)
  return maximum
}

export const axisScale = (
  points: readonly PowerToWeightTrendPoint[],
  active: ReadonlySet<PowerToWeightDurationS>,
  references: readonly PowerToWeightReference[],
): { domainMax: number; step: number; values: number[] } => {
  const maximum = observedMax(points, active, references)
  const step = niceStep(maximum, 4)
  const domainMax = Math.ceil(maximum / step) * step
  const values = Array.from(
    { length: Math.round(domainMax / step) + 1 },
    (_, index) => index * step,
  )
  return { domainMax, step, values }
}
