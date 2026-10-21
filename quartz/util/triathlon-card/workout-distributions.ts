import type { StravaActivityDetail, StravaZones } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { RUN_PACE_ZONE_NAMES } from '../run-pace-zones'
import { clock, HR_ZONE_NAMES, POWER_ZONE_NAMES, zoneClock, zoneRange } from './metric-text'
import { runPaceZoneRangeText } from './workout-lap-data'

export const buildRunPaceDistribution = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
): N | null => {
  const distribution = d.runPaceZones
  if (
    d.sport !== 'run' ||
    !distribution ||
    distribution.boundsSPerKm.length !== 5 ||
    distribution.zoneSeconds.length !== 6 ||
    distribution.tenKmRaceTimeS <= 0 ||
    distribution.boundsSPerKm.some(value => !Number.isFinite(value) || value <= 0) ||
    distribution.zoneSeconds.some(value => !Number.isFinite(value) || value < 0)
  )
    return null
  return buildWorkoutZoneDistribution(
    f,
    'pace',
    'tri-run-pace-distribution',
    'Run pace distribution',
    distribution.zoneSeconds,
    RUN_PACE_ZONE_NAMES,
    distribution.zoneSeconds.map((_, index) =>
      runPaceZoneRangeText(f.presentation, distribution.boundsSPerKm, index),
    ),
    `based on 10 km race time ${clock(distribution.tenKmRaceTimeS)}`,
  )
}

export const buildCyclingPowerDistribution = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  zones: StravaZones | null,
): N | null => {
  const seconds = d.powerZones
  const bounds = zones?.power
  if (
    d.sport !== 'bike' ||
    !seconds ||
    !bounds ||
    seconds.length !== 7 ||
    bounds.length !== 6 ||
    seconds.some(value => !Number.isFinite(value) || value < 0) ||
    bounds.some(
      (value, index) =>
        !Number.isFinite(value) || value <= 0 || (index > 0 && value <= bounds[index - 1]),
    )
  )
    return null
  return buildWorkoutZoneDistribution(
    f,
    'power',
    'tri-cycling-power-distribution',
    'Cycling power distribution',
    seconds,
    POWER_ZONE_NAMES,
    seconds.map((_, index) => `${index === 0 ? `≤ ${bounds[0]}` : zoneRange(bounds, index)} W`),
    zones?.ftp != null ? `based on FTP ${zones.ftp} W` : '',
  )
}

export const buildCyclingHeartRateDistribution = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  zones: StravaZones | null,
  vt1: number | null,
): N | null => {
  const seconds = d.hrZones
  const bounds = zones?.hr
  if (
    d.sport !== 'bike' ||
    !seconds ||
    !bounds?.length ||
    seconds.length !== bounds.length + 1 ||
    seconds.some(value => !Number.isFinite(value) || value < 0) ||
    bounds.some(
      (value, index) =>
        !Number.isFinite(value) || value <= 0 || (index > 0 && value <= bounds[index - 1]),
    )
  )
    return null
  return buildWorkoutZoneDistribution(
    f,
    'heart-rate',
    'tri-cycling-heart-rate-distribution',
    'Cycling heart rate distribution',
    seconds,
    HR_ZONE_NAMES,
    seconds.map((_, index) => `${zoneRange(bounds, index)} bpm`),
    vt1 != null ? `based on vt1 ${vt1} bpm` : '',
  )
}

const buildWorkoutZoneDistribution = <N>(
  f: TriNodeFactory<N>,
  kind: 'pace' | 'power' | 'heart-rate',
  className: string,
  label: string,
  secondsByZone: readonly number[],
  names: readonly string[],
  ranges: readonly string[],
  caption: string,
): N | null => {
  const total = secondsByZone.reduce((sum, seconds) => sum + seconds, 0)
  if (!(total > 0)) return null
  let majority = 0
  for (let index = 1; index < secondsByZone.length; index++)
    if (secondsByZone[index] > secondsByZone[majority]) majority = index
  const maximum = Math.max(...secondsByZone, 1)
  const wrap = f.el('section', `tri-workout-distribution ${className}`, undefined, {
    'aria-label': label,
  })
  const head = f.el('div', 'tri-workout-distribution-head')
  const summary = f.el('div', 'tri-training-zone-summary', undefined, { 'aria-live': 'polite' })
  f.add(
    summary,
    f.el(
      'strong',
      'tri-training-zone-summary-value',
      `${Math.round((secondsByZone[majority] / total) * 100)}% in zone ${majority + 1}`,
    ),
    f.el('span', 'tri-training-zone-summary-time', zoneClock(total)),
  )
  f.add(head, summary)
  const grid = f.el('div', `tri-training-zone-grid tri-training-zone-grid--${kind}`, undefined, {
    role: 'list',
    'aria-label': `${kind} zone distribution`,
    style: `--tri-zone-count:${secondsByZone.length}`,
  })
  for (let index = secondsByZone.length - 1; index >= 0; index--) {
    const seconds = secondsByZone[index]
    const percentage = (seconds / total) * 100
    const range = ranges[index]
    const name = names[index] ?? `zone ${index + 1}`
    const row = f.el(
      'div',
      `tri-training-zone-row${index === majority ? ' tri-training-zone-row--majority' : ''}`,
      undefined,
      {
        role: 'listitem',
        tabindex: '0',
        'aria-label': `Z${index + 1} ${name}, ${zoneClock(seconds)}, ${percentage.toFixed(1)}%, ${range}`,
      },
    )
    const bar = f.el('span', 'tri-training-zone-bar', undefined, { 'aria-hidden': 'true' })
    f.add(
      bar,
      f.el(
        'span',
        `tri-training-zone-fill tri-training-zone-fill--${kind} tri-training-zone-fill--${index + 1}`,
        undefined,
        { style: `--tri-zone-share:${(seconds / maximum) * 100}%` },
      ),
    )
    const visual = f.el('span', 'tri-training-zone-visual')
    f.add(
      visual,
      bar,
      f.el('span', 'tri-training-zone-pct', `${percentage.toFixed(1)}%`, { 'aria-hidden': 'true' }),
    )
    f.add(
      row,
      f.el('span', 'tri-training-zone-name', `Z${index + 1}`, { 'aria-hidden': 'true' }),
      visual,
      f.el('span', 'tri-training-zone-range', range, { 'aria-hidden': 'true' }),
    )
    f.add(grid, row)
  }
  f.add(wrap, head, grid, f.el('div', 'tri-dist-cap tri-training-zone-source', caption))
  return wrap
}
