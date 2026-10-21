import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { triText } from '../triathlon-i18n'
import { climbSourceLabel } from './analysis-presentation'
import {
  clock,
  cyclingSpeed,
  formatAltitude,
  formatElevationGain,
  formatVam,
  heartRate,
  isImperial,
  runPaceSeconds,
  scrubDist,
  shortDate,
  watts,
  wattsPerKg,
  zoneClock,
} from './metric-text'

const effortDuration = (seconds: number): string => {
  if (seconds < 60) return `${Math.round(seconds)} sec`
  if (seconds < 3600 && seconds % 60 === 0) {
    const minutes = seconds / 60
    return `${minutes} min`
  }
  if (seconds % 3600 === 0) {
    const hours = seconds / 3600
    return `${hours} hr`
  }
  if (seconds >= 3600 && seconds % 60 === 0)
    return `${Math.floor(seconds / 3600)}h${String((seconds % 3600) / 60).padStart(2, '0')}`
  return zoneClock(seconds)
}

const effortTable = <N>(
  f: TriNodeFactory<N>,
  title: string,
  kind: string,
  headers: string[],
  rows: string[][],
): N => {
  const block = f.el('div', `tri-effort-block tri-effort-block--${kind}`)
  const table = f.el('table', `tri-effort-table tri-effort-table--${kind}`, undefined, {
    'aria-label': `${title} efforts`,
  })
  const thead = f.el('thead')
  const heading = f.el('tr')
  for (const label of headers)
    f.add(heading, f.el('th', undefined, label.toLowerCase(), { scope: 'col' }))
  f.add(thead, heading)
  const tbody = f.el('tbody')
  for (const cells of rows) {
    const row = f.el('tr')
    cells.forEach((value, index) =>
      f.add(
        row,
        f.el(index === 0 ? 'th' : 'td', undefined, value, index === 0 ? { scope: 'row' } : {}),
      ),
    )
    f.add(tbody, row)
  }
  f.add(table, thead, tbody)
  const scroll = f.el('div', 'tri-effort-scroll', undefined, {
    role: 'region',
    'aria-label': `${title} efforts`,
    tabindex: '0',
  })
  f.add(scroll, table)
  const viewport = f.el('div', 'tri-effort-viewport')
  f.add(viewport, scroll)
  f.add(block, f.el('div', 'tri-zone-title tri-effort-title', title.toLowerCase()), viewport)
  return block
}

export const buildBestEfforts = <N>(f: TriNodeFactory<N>, d: StravaActivityDetail): N | null => {
  const efforts = d.bestEfforts
  if (
    (d.sport !== 'bike' && d.sport !== 'run') ||
    !efforts ||
    (d.sport === 'run' && efforts.distance.length === 0) ||
    (efforts.distance.length === 0 && efforts.power.length === 0 && efforts.climbs.length === 0)
  )
    return null

  const running = d.sport === 'run'
  const wrap = f.el('section', 'tri-efforts', undefined, {
    'aria-label': running
      ? triText(f.presentation.locale, 'Running best efforts')
      : 'Cycling best efforts',
    ...(efforts.distanceSource ? { 'data-effort-source': efforts.distanceSource } : {}),
  })
  if (efforts.distance.length > 0)
    f.add(
      wrap,
      effortTable(
        f,
        'Distance',
        'distance',
        ['Distance', 'Time', running ? 'Pace' : 'Speed', 'Heart rate', 'Elev'],
        efforts.distance.map(row => [
          row.label || scrubDist(f.presentation, row.targetDistanceM / 1000, d.sport),
          zoneClock(row.elapsedTimeS),
          running
            ? `${clock(runPaceSeconds(f.presentation, row.averageSpeedKph))}/${isImperial(f.presentation) ? 'mi' : 'km'}`
            : cyclingSpeed(f.presentation, row.averageSpeedKph),
          heartRate(row.averageHeartRate),
          row.elevationDeltaM == null ? '—' : formatAltitude(f.presentation, row.elevationDeltaM),
        ]),
      ),
    )
  if (running) {
    f.add(
      wrap,
      f.el(
        'p',
        'tri-effort-note',
        triText(
          f.presentation.locale,
          'Calculated from recorded distance and elapsed time, including pauses.',
        ),
      ),
    )
    return wrap
  }
  if (efforts.power.length > 0) {
    f.add(
      wrap,
      effortTable(
        f,
        'Power',
        'power',
        ['Time', 'Power', 'W/kg', 'Heart rate', 'Cadence', 'Torque', 'Coverage', 'Elev'],
        efforts.power.map(row => [
          effortDuration(row.durationS),
          watts(row.averageWatts),
          wattsPerKg(row.wattsPerKg),
          heartRate(row.averageHeartRate),
          row.mechanics ? `${row.mechanics.averageCadenceRpm.toFixed(0)} rpm` : '—',
          row.mechanics ? `${row.mechanics.averageTorqueNm.toFixed(1)} N·m` : '—',
          row.mechanics ? `${(row.mechanics.coverage * 100).toFixed(0)}%` : '—',
          formatAltitude(f.presentation, row.elevationDeltaM),
        ]),
      ),
    )
  }
  if (efforts.climbs.length > 0) {
    const title = climbSourceLabel(efforts.climbs.map(row => row.source))
    f.add(
      wrap,
      effortTable(
        f,
        title,
        'climbing',
        [
          title === 'Summit Segments' ? 'Segment' : 'Climb',
          'Time',
          'Distance',
          'Gain',
          'Grade',
          'Speed',
          'Heart rate',
          'Power',
          'W/kg',
          'VAM',
        ],
        efforts.climbs.map((row, index) => [
          row.name || `Climb ${index + 1}`,
          zoneClock(row.durationS),
          scrubDist(f.presentation, row.distanceM / 1000, 'bike'),
          formatElevationGain(f.presentation, row.elevationGainM),
          `${row.averageGradePct.toFixed(1)}%`,
          cyclingSpeed(f.presentation, row.averageSpeedKph),
          heartRate(row.averageHeartRate),
          watts(row.averageWatts),
          wattsPerKg(row.wattsPerKg),
          formatVam(f.presentation, row.vamMPerHour),
        ]),
      ),
    )
  }
  if (
    efforts.weightKg != null &&
    efforts.weightDate &&
    (efforts.power.length || efforts.climbs.length)
  )
    f.add(
      wrap,
      f.el(
        'p',
        'tri-effort-note',
        `W/kg from ${efforts.weightKg.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 2 })} kg Garmin weight · ${shortDate(efforts.weightDate)}`,
      ),
    )
  return wrap
}
