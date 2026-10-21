import type { ActivityGearShift, StravaActivityDetail } from '../../plugins/stores/strava'
import type { ActivityGraphDomain, AxisYTick } from './chart-geometry'
import type { WattsHeatSample } from './watts-heatmap'

export type GearShiftHover = ActivityGearShift & { index: number; xPct: number }

export function gearShiftAtFraction(
  shifts: readonly ActivityGearShift[],
  maxDistanceKm: number,
  fraction: number,
): GearShiftHover | null {
  if (shifts.length === 0 || !Number.isFinite(maxDistanceKm) || maxDistanceKm <= 0) return null
  const normalized = Number.isFinite(fraction) ? Math.min(1, Math.max(0, fraction)) : 0
  const distanceKm = normalized * maxDistanceKm
  let low = 0
  let high = shifts.length
  while (low < high) {
    const middle = low + Math.floor((high - low) / 2)
    if (shifts[middle].distanceKm <= distanceKm) low = middle + 1
    else high = middle
  }
  const index = Math.min(shifts.length - 1, Math.max(0, low - 1))
  return { ...shifts[index], index, xPct: normalized * 100 }
}

export function sampledGearTicks(values: readonly number[], limit = 4): number[] {
  const unique = [...new Set(values)].sort((left, right) => left - right)
  if (unique.length <= limit) return unique
  return Array.from(
    new Set(
      Array.from(
        { length: limit },
        (_, index) => unique[Math.round((index / (limit - 1)) * (unique.length - 1))],
      ),
    ),
  )
}

export function gearY(value: number, min: number, max: number, height: number): number {
  return max > min ? height - ((value - min) / (max - min)) * (height - 1) : height / 2
}

export function gearStepPath(
  shifts: readonly ActivityGearShift[],
  read: (shift: ActivityGearShift) => number,
  y: (value: number) => number,
  maxDistanceKm: number,
  width: number,
): string {
  const x = (distanceKm: number): number =>
    (Math.min(maxDistanceKm, Math.max(0, distanceKm)) / maxDistanceKm) * width
  let path = `M 0 ${y(read(shifts[0])).toFixed(2)}`
  let previous = read(shifts[0])
  for (let index = 1; index < shifts.length; index++) {
    const current = read(shifts[index])
    const nextX = x(shifts[index].distanceKm).toFixed(2)
    path += ` L ${nextX} ${y(previous).toFixed(2)} L ${nextX} ${y(current).toFixed(2)}`
    previous = current
  }
  return `${path} L ${width} ${y(previous).toFixed(2)}`
}

interface GearPairingDuration {
  frontTeeth: number
  rearTeeth: number
  frontGearNum: number
  rearGearNum: number
  durationS: number
}

function gearPairingDurations(
  shifts: readonly ActivityGearShift[],
  workoutEndElapsedS: number,
): GearPairingDuration[] {
  if (shifts.length === 0 || !Number.isFinite(workoutEndElapsedS) || workoutEndElapsedS <= 0)
    return []
  const endElapsedS = Math.max(workoutEndElapsedS, shifts.at(-1)?.elapsedS ?? 0)
  const durations = new Map<string, GearPairingDuration>()
  for (let index = 0; index < shifts.length; index++) {
    const shift = shifts[index]
    const next = shifts[index + 1]
    const startElapsedS = index === 0 ? 0 : Math.min(endElapsedS, Math.max(0, shift.elapsedS))
    const nextElapsedS = next
      ? Math.min(endElapsedS, Math.max(startElapsedS, next.elapsedS))
      : endElapsedS
    const durationS = nextElapsedS - startElapsedS
    if (durationS <= 0) continue
    const key = `${shift.frontTeeth}:${shift.rearTeeth}`
    const existing = durations.get(key)
    if (existing) existing.durationS += durationS
    else
      durations.set(key, {
        frontTeeth: shift.frontTeeth,
        rearTeeth: shift.rearTeeth,
        frontGearNum: shift.frontGearNum,
        rearGearNum: shift.rearGearNum,
        durationS,
      })
  }
  return [...durations.values()]
}

export function dominantGearPairing(
  shifts: readonly ActivityGearShift[],
  workoutEndElapsedS: number,
): GearPairingDuration | null {
  let dominant: GearPairingDuration | null = null
  for (const pairing of gearPairingDurations(shifts, workoutEndElapsedS))
    if (!dominant || pairing.durationS > dominant.durationS) dominant = pairing
  return dominant
}

interface ShiftingWattsRow {
  key: string
  frontTeeth: number
  rearTeeth: number
  frontGearNum: number
  rearGearNum: number
  ratio: number
}

export const shiftingWattsData = (
  d: StravaActivityDetail,
  shifts: readonly ActivityGearShift[],
  graphDomain?: ActivityGraphDomain | null,
): { rows: ShiftingWattsRow[]; samples: WattsHeatSample[] } => {
  const observed: { key: string; watts: number }[] = []
  const pairings = new Map<string, ShiftingWattsRow>()
  let shiftIndex = 0
  for (const point of d.route) {
    while (shiftIndex + 1 < shifts.length && shifts[shiftIndex + 1].elapsedS <= point.elapsedS)
      shiftIndex++
    if (
      point.w <= 0 ||
      (graphDomain &&
        (point.d < graphDomain.startDistanceKm || point.d > graphDomain.endDistanceKm))
    )
      continue
    const shift = shifts[shiftIndex]
    if (!shift || shift.frontTeeth <= 0 || shift.rearTeeth <= 0) continue
    const key = `${shift.frontTeeth}:${shift.rearTeeth}`
    pairings.set(key, {
      key,
      frontTeeth: shift.frontTeeth,
      rearTeeth: shift.rearTeeth,
      frontGearNum: shift.frontGearNum,
      rearGearNum: shift.rearGearNum,
      ratio: shift.frontTeeth / shift.rearTeeth,
    })
    observed.push({ key, watts: point.w })
  }
  const rows = [...pairings.values()].sort(
    (left, right) =>
      left.ratio - right.ratio ||
      left.frontTeeth - right.frontTeeth ||
      right.rearTeeth - left.rearTeeth,
  )
  const rowIndex = new Map(rows.map((row, index) => [row.key, index]))
  return {
    rows,
    samples: observed.flatMap(sample => {
      const index = rowIndex.get(sample.key)
      if (index == null) return []
      const row = rows[index]
      return [
        {
          watts: sample.watts,
          value: rows.length === 1 ? 0.5 : index,
          label: `${row.frontTeeth}×${row.rearTeeth}`,
          numberLabel: `${row.frontGearNum}×${row.rearGearNum}`,
        },
      ]
    }),
  }
}

export const shiftingWattsTicks = (
  rows: readonly ShiftingWattsRow[],
  height: number,
  limit = 4,
): AxisYTick[] => {
  if (rows.length === 0) return []
  const indices =
    rows.length <= limit
      ? rows.map((_, index) => index)
      : Array.from(
          new Set(
            Array.from({ length: limit }, (_, index) =>
              Math.round((index / (limit - 1)) * (rows.length - 1)),
            ),
          ),
        )
  return indices.flatMap(index => {
    const row = rows[index]
    const vbY = rows.length === 1 ? height / 2 : height - (index / (rows.length - 1)) * (height - 1)
    return [
      { label: `${row.frontTeeth}×${row.rearTeeth}`, vbY, attrs: { 'data-shift-label': 'teeth' } },
      {
        label: `${row.frontGearNum}×${row.rearGearNum}`,
        vbY,
        attrs: { 'data-shift-label': 'number' },
      },
    ]
  })
}

export interface ActivityGearRatioDistributionPoint {
  ratio: number
  percentage: number
  pairings: { frontTeeth: number; rearTeeth: number }[]
}

export const activityGearRatioDistribution = (
  activity: StravaActivityDetail,
): ActivityGearRatioDistributionPoint[] => {
  if (activity.sport !== 'bike') return []
  const workoutEndElapsedS = Math.max(activity.movingTimeS, activity.route.at(-1)?.elapsedS ?? 0)
  const pairingsByRatio = new Map<number, GearPairingDuration[]>()
  let totalDurationS = 0
  for (const pairing of gearPairingDurations(activity.gearShifts, workoutEndElapsedS)) {
    if (
      !Number.isFinite(pairing.frontTeeth) ||
      !Number.isFinite(pairing.rearTeeth) ||
      pairing.frontTeeth <= 0 ||
      pairing.rearTeeth <= 0
    )
      continue
    const ratio = Number((pairing.frontTeeth / pairing.rearTeeth).toFixed(6))
    const pairings = pairingsByRatio.get(ratio) ?? []
    pairings.push(pairing)
    pairingsByRatio.set(ratio, pairings)
    totalDurationS += pairing.durationS
  }
  if (totalDurationS <= 0) return []
  return [...pairingsByRatio.entries()]
    .sort(([left], [right]) => left - right)
    .map(([ratio, pairings]) => ({
      ratio,
      percentage:
        (pairings.reduce((total, pairing) => total + pairing.durationS, 0) / totalDurationS) * 100,
      pairings: pairings
        .sort(
          (left, right) => left.frontTeeth - right.frontTeeth || right.rearTeeth - left.rearTeeth,
        )
        .map(({ frontTeeth, rearTeeth }) => ({ frontTeeth, rearTeeth })),
    }))
}

interface GearTeethTime {
  teeth: number
  gearNum: number
  durationS: number
}

interface GearComboTime {
  front: GearTeethTime
  rear: GearTeethTime
  durationS: number
}

// Time on each chainring, cog and combination inside the graph domain. Every gear seen in the
// ride keeps a row so zooming changes bar lengths without reshuffling the rows.
export const gearTeethDurations = (
  d: StravaActivityDetail,
  graphDomain?: ActivityGraphDomain | null,
): { front: GearTeethTime[]; rear: GearTeethTime[]; combos: GearComboTime[]; totalS: number } => {
  const shifts = d.gearShifts
  const endS = Math.max(d.movingTimeS, d.route.at(-1)?.elapsedS ?? 0, shifts.at(-1)?.elapsedS ?? 0)
  const elapsedAt = (distanceKm: number, fallback: number): number =>
    d.route.find(point => point.d >= distanceKm)?.elapsedS ?? fallback
  const windowStartS = graphDomain ? elapsedAt(graphDomain.startDistanceKm, 0) : 0
  const windowEndS = graphDomain ? elapsedAt(graphDomain.endDistanceKm, endS) : endS
  const front = new Map<number, GearTeethTime>()
  const rear = new Map<number, GearTeethTime>()
  const combos = new Map<string, GearComboTime>()
  let totalS = 0
  for (const [index, shift] of shifts.entries()) {
    const ring = front.get(shift.frontTeeth) ?? {
      teeth: shift.frontTeeth,
      gearNum: shift.frontGearNum,
      durationS: 0,
    }
    const cog = rear.get(shift.rearTeeth) ?? {
      teeth: shift.rearTeeth,
      gearNum: shift.rearGearNum,
      durationS: 0,
    }
    front.set(ring.teeth, ring)
    rear.set(cog.teeth, cog)
    const key = `${ring.teeth}:${cog.teeth}`
    const combo = combos.get(key) ?? { front: ring, rear: cog, durationS: 0 }
    combos.set(key, combo)
    const startS = Math.max(windowStartS, index === 0 ? 0 : shift.elapsedS)
    const stopS = Math.min(windowEndS, shifts[index + 1]?.elapsedS ?? endS)
    if (!(stopS > startS)) continue
    ring.durationS += stopS - startS
    cog.durationS += stopS - startS
    combo.durationS += stopS - startS
    totalS += stopS - startS
  }
  const byTeeth = (left: GearTeethTime, right: GearTeethTime) => right.teeth - left.teeth
  return {
    front: [...front.values()].sort(byTeeth),
    rear: [...rear.values()].sort(byTeeth),
    combos: [...combos.values()].sort(
      (left, right) =>
        right.front.teeth / right.rear.teeth - left.front.teeth / left.rear.teeth ||
        right.front.teeth - left.front.teeth,
    ),
    totalS,
  }
}
