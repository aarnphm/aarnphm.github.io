import { KM_TO_MI } from './triathlon-card'

export type TriathlonCalcInput = {
  swimKm: number
  bikeKm: number
  runKm: number
  swimPaceSec: number
  t1Sec: number
  bikeMph: number
  t2Sec: number
  runPaceSec: number
}

export type TriathlonCalcTimes = {
  swimSec: number
  t1Sec: number
  bikeSec: number
  t2Sec: number
  runSec: number
  totalSec: number
}

export type TriathlonCalcPaces = { swimPaceSec: number; bikeMph: number; runPaceSec: number }

export function parseClockSeconds(value: string): number {
  const trimmed = value.trim()
  if (!trimmed) return 0
  const rawParts = trimmed.split(':')
  if (rawParts.length === 2 || rawParts.length === 3) {
    const parts = rawParts.map(part => Number(part))
    if (parts.some(part => !Number.isFinite(part))) return 0
    if (parts.length === 3) return (parts[0] || 0) * 3600 + (parts[1] || 0) * 60 + (parts[2] || 0)
    return (parts[0] || 0) * 60 + (parts[1] || 0)
  }
  const seconds = Number(trimmed)
  return Number.isFinite(seconds) ? seconds : 0
}

export function formatDurationClock(sec: number): string {
  const t = Math.max(0, Math.round(sec))
  const h = Math.floor(t / 3600)
  const m = Math.floor((t % 3600) / 60)
  const s = t % 60
  return h > 0
    ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
    : `${m}:${String(s).padStart(2, '0')}`
}

function finiteNonnegative(value: number): number {
  return Number.isFinite(value) && value > 0 ? value : 0
}

export function computeTriathlonCalcTimes(input: TriathlonCalcInput): TriathlonCalcTimes {
  const swimKm = finiteNonnegative(input.swimKm)
  const bikeKm = finiteNonnegative(input.bikeKm)
  const runKm = finiteNonnegative(input.runKm)
  const swimPaceSec = finiteNonnegative(input.swimPaceSec)
  const t1Sec = finiteNonnegative(input.t1Sec)
  const bikeMph = finiteNonnegative(input.bikeMph)
  const t2Sec = finiteNonnegative(input.t2Sec)
  const runPaceSec = finiteNonnegative(input.runPaceSec)
  const swimSec = swimKm * 10 * swimPaceSec
  const bikeMiles = bikeKm * KM_TO_MI
  const bikeSec = bikeMph > 0 ? (bikeMiles / bikeMph) * 3600 : 0
  const runSec = runKm * KM_TO_MI * runPaceSec

  return {
    swimSec,
    t1Sec,
    bikeSec,
    t2Sec,
    runSec,
    totalSec: swimSec + t1Sec + bikeSec + t2Sec + runSec,
  }
}

export function solveTriathlonCalcTarget(
  input: TriathlonCalcInput,
  targetTotalSec: number,
): TriathlonCalcPaces | null {
  const times = computeTriathlonCalcTimes(input)
  const transitionSec = times.t1Sec + times.t2Sec
  const targetSportSec = targetTotalSec - transitionSec
  const currentSportSec = times.swimSec + times.bikeSec + times.runSec
  if (targetSportSec <= 0 || currentSportSec <= 0) return null

  const scale = targetSportSec / currentSportSec
  const swimSec = times.swimSec * scale
  const bikeSec = times.bikeSec * scale
  const runSec = times.runSec * scale
  const swimPaceSec = input.swimKm > 0 ? swimSec / (input.swimKm * 10) : input.swimPaceSec
  const bikeMiles = input.bikeKm * KM_TO_MI
  const bikeMph = bikeSec > 0 ? bikeMiles / (bikeSec / 3600) : input.bikeMph
  const runMiles = input.runKm * KM_TO_MI
  const runPaceSec = runMiles > 0 ? runSec / runMiles : input.runPaceSec

  if (![swimPaceSec, bikeMph, runPaceSec].every(value => Number.isFinite(value) && value > 0)) {
    return null
  }

  return { swimPaceSec, bikeMph, runPaceSec }
}
