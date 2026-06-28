import { KM_TO_MI, clock, type TriNodeFactory } from './triathlon-card'
import { tl } from './triathlon-i18n'

export const TRI_RACE_DISTANCES: [string, number, number, number][] = [
  ['sprint', 0.75, 20, 5],
  ['olympic', 1.5, 40, 10],
  ['70.3', 1.9, 90, 21.1],
  ['ironman', 3.8, 180, 42.2],
]

export const CALC_ANCHOR_PREFIX = 'calculator-'

export interface CalcShare {
  presetIdx: number
  mode: 'a' | 'p'
  unit: 'i' | 'm'
  swimPaceSec: number
  t1Sec: number
  bikeMph: number
  t2Sec: number
  runPaceSec: number
}

export function encodeCalcShare(s: CalcShare): string {
  return [
    s.presetIdx,
    s.mode,
    s.unit,
    Math.round(s.swimPaceSec),
    Math.round(s.t1Sec),
    Math.round(s.bikeMph * 100),
    Math.round(s.t2Sec),
    Math.round(s.runPaceSec),
  ].join('-')
}

export function decodeCalcShare(raw: string): CalcShare | null {
  let payload = raw.startsWith('#') ? raw.slice(1) : raw
  if (payload.startsWith(CALC_ANCHOR_PREFIX)) payload = payload.slice(CALC_ANCHOR_PREFIX.length)
  const parts = payload.split('-')
  if (parts.length !== 8) return null
  const presetIdx = Number(parts[0])
  const mode = parts[1] === 'p' ? 'p' : parts[1] === 'a' ? 'a' : null
  const unit = parts[2] === 'i' ? 'i' : parts[2] === 'm' ? 'm' : null
  if (mode === null || unit === null) return null
  if (!Number.isInteger(presetIdx) || presetIdx < 0 || presetIdx >= TRI_RACE_DISTANCES.length)
    return null
  const nums = parts.slice(3).map(Number)
  if (nums.some(n => !Number.isFinite(n) || n < 0)) return null
  return {
    presetIdx,
    mode,
    unit,
    swimPaceSec: nums[0],
    t1Sec: nums[1],
    bikeMph: nums[2] / 100,
    t2Sec: nums[3],
    runPaceSec: nums[4],
  }
}

export function calcShareToInput(s: CalcShare): TriathlonCalcInput {
  const [, swimKm, bikeKm, runKm] = TRI_RACE_DISTANCES[s.presetIdx] ?? TRI_RACE_DISTANCES[1]
  return {
    swimKm,
    bikeKm,
    runKm,
    swimPaceSec: s.swimPaceSec,
    t1Sec: s.t1Sec,
    bikeMph: s.bikeMph,
    t2Sec: s.t2Sec,
    runPaceSec: s.runPaceSec,
  }
}

export const buildTriathlonCalcCard = <N>(f: TriNodeFactory<N>, share: CalcShare): N => {
  const imperial = share.unit === 'i'
  const [label] = TRI_RACE_DISTANCES[share.presetIdx] ?? TRI_RACE_DISTANCES[1]
  const times = computeTriathlonCalcTimes(calcShareToInput(share))
  const card = f.el('div', 'tri-calc-card')

  const head = f.el('div', 'tri-calc-card-head')
  f.add(head, f.el('span', 'tri-calc-card-dist', label))
  const tabs = f.el('div', 'tri-calc-card-tabs')
  f.add(
    tabs,
    f.el(
      'span',
      'tri-calc-card-tab tri-calc-card-tab--on',
      share.mode === 'a' ? tl('average') : tl('projected'),
    ),
  )
  f.add(head, tabs)
  f.add(card, head)

  const bikeDisp = (imperial ? share.bikeMph : share.bikeMph / KM_TO_MI).toFixed(1)
  const runDisp = clock(imperial ? share.runPaceSec : share.runPaceSec * KM_TO_MI)
  const rows: [string, string, number][] = [
    [tl('swim'), `${clock(share.swimPaceSec)} /100m`, times.swimSec],
    ['T1', `${clock(share.t1Sec)} min`, times.t1Sec],
    [tl('bike'), `${bikeDisp} ${imperial ? 'mph' : 'km/h'}`, times.bikeSec],
    ['T2', `${clock(share.t2Sec)} min`, times.t2Sec],
    [tl('run'), `${runDisp} ${imperial ? '/mi' : '/km'}`, times.runSec],
  ]
  const table = f.el('table', 'tri-calc-card-io')
  const tbody = f.el('tbody')
  for (const [k, v, sec] of rows) {
    const tr = f.el('tr', 'tri-calc-card-row')
    f.add(
      tr,
      f.el('th', 'tri-calc-card-k', k),
      f.el('td', 'tri-calc-card-v', v),
      f.el('td', 'tri-calc-card-split', formatDurationClock(sec)),
    )
    f.add(tbody, tr)
  }
  const total = f.el('tr', 'tri-calc-card-row tri-calc-card-total')
  f.add(
    total,
    f.el('th', 'tri-calc-card-k', tl('finish')),
    f.el('td', 'tri-calc-card-v', ''),
    f.el('td', 'tri-calc-card-split', formatDurationClock(times.totalSec)),
  )
  f.add(tbody, total)
  f.add(table, tbody)
  f.add(card, table)
  return card
}

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

export type TriathlonCalcLeg = 'swim' | 'bike' | 'run'

export function solveTriathlonCalcLeg(
  input: TriathlonCalcInput,
  leg: TriathlonCalcLeg,
  legSec: number,
): Partial<TriathlonCalcPaces> | null {
  if (!(legSec > 0)) return null
  if (leg === 'swim') {
    if (!(input.swimKm > 0)) return null
    return { swimPaceSec: legSec / (input.swimKm * 10) }
  }
  if (leg === 'bike') {
    if (!(input.bikeKm > 0)) return null
    return { bikeMph: (input.bikeKm * KM_TO_MI * 3600) / legSec }
  }
  if (!(input.runKm > 0)) return null
  return { runPaceSec: legSec / (input.runKm * KM_TO_MI) }
}
