import type { TriathlonDayAnalytics } from '../triathlon-day-analytics'
import type { TriathlonPresentation } from '../triathlon-presentation'
import type { TriNodeFactory } from './construction-contracts'
import { resolveSleepMetrics, type SleepMetrics } from '../sleep-metrics'
import { buildGarminHealth, buildGarminRecovery } from '../triathlon-garmin-health'
import { ouraScoreStatus } from '../triathlon-health'
import { triText } from '../triathlon-i18n'
import { buildOuraHealth } from '../triathlon-oura-health'
import {
  dayAnalyticsBodyMetrics,
  dayAnalyticsDuration,
  dayAnalyticsGroup,
  dayAnalyticsHeatMetrics,
  dayAnalyticsList,
  dayAnalyticsNumber,
  dayAnalyticsRecoveryMetrics,
  dayAnalyticsSigned,
  dayAnalyticsSleepMetrics,
  dayAnalyticsTrainingMetrics,
  type DayAnalyticsMetric,
} from './day-analytics-metrics'
import {
  buildSleepRespirationChart,
  dayAnalyticsSleepSeries,
  dayAnalyticsSleepStages,
} from './sleep-charts'

export const sleepSupplementMetrics = (
  presentation: TriathlonPresentation,
  sleep: SleepMetrics | null | undefined,
): DayAnalyticsMetric[] => {
  if (!sleep) return []
  const metrics: DayAnalyticsMetric[] = []
  const garmin = sleep.garmin
  const number = (value: number, digits = 0): string =>
    dayAnalyticsNumber(presentation, value, digits)
  const detail = (label: string, value: number | null, unit = '', digits = 0): string | null =>
    value == null ? null : `${triText(presentation.locale, label)} ${number(value, digits)}${unit}`
  const garminRespiration = garmin
    ? [
        detail('respiration', garmin.averageBreathsPerMinute, ' brpm', 1),
        detail('lowest respiration', garmin.lowestBreathsPerMinute, ' brpm', 1),
        detail('highest respiration', garmin.highestBreathsPerMinute, ' brpm', 1),
      ].filter(value => value != null)
    : []
  if (sleep.averageBreathsPerMinute != null || garminRespiration.length > 0) {
    const source = sleep.respirationSource === 'oura' ? 'Oura' : 'Garmin'
    metrics.push({
      label: 'respiration',
      value:
        sleep.averageBreathsPerMinute == null
          ? '—'
          : `${number(sleep.averageBreathsPerMinute, 1)} brpm`,
      detail:
        garminRespiration.length > 0
          ? `${source === 'Oura' ? 'Oura\n' : ''}Garmin: ${garminRespiration.join('; ')}`
          : source,
    })
  }
  const oxygen =
    sleep.oxygenSaturation ??
    (garmin?.averageSpO2 != null && garmin.averageSpO2 > 0 && garmin.averageSpO2 <= 100
      ? { averagePct: garmin.averageSpO2, source: 'garmin' }
      : null)
  if (oxygen)
    metrics.push({
      label: 'Pulse Ox',
      value: `${number(oxygen.averagePct, 1)}%`,
      detail: [
        oxygen.source === 'oura' ? 'Oura · Garmin unavailable' : 'Garmin',
        garmin?.lowestSpO2 != null
          ? `${oxygen.source === 'oura' ? 'Garmin: ' : ''}${detail('lowest Pulse Ox', garmin.lowestSpO2, '%')}`
          : null,
      ]
        .filter(value => value != null)
        .join('\n'),
    })
  if (sleep.breathingDisturbanceIndex != null)
    metrics.push({
      label: 'breathing disturbances',
      value: `${number(sleep.breathingDisturbanceIndex)}/100`,
      detail:
        'Oura · breathing disturbance index from overnight oxygen drops; not an apnea diagnosis',
    })
  if (!garmin) return metrics
  const add = (label: string, value: number | null, unit = '', digits = 0): void => {
    if (value != null)
      metrics.push({ label, value: `${number(value, digits)}${unit}`, detail: 'Garmin' })
  }
  const batteryDetail = [
    'Garmin',
    detail('Body Battery at bedtime', garmin.bodyBatteryStart),
    detail('Body Battery at wake-up', garmin.bodyBatteryEnd),
  ]
    .filter(value => value != null)
    .join('\n')
  if (garmin.bodyBatteryChange != null)
    metrics.push({
      label: 'Body Battery change',
      value: dayAnalyticsSigned(presentation, garmin.bodyBatteryChange, 0),
      detail: batteryDetail,
    })
  else if (garmin.bodyBatteryEnd != null || garmin.bodyBatteryStart != null)
    metrics.push({
      label: garmin.bodyBatteryEnd != null ? 'Body Battery at wake-up' : 'Body Battery at bedtime',
      value: number(garmin.bodyBatteryEnd ?? garmin.bodyBatteryStart ?? 0),
      detail: batteryDetail,
    })
  add('sleep stress', garmin.averageStress, '', 1)
  add('restless moments', garmin.restlessMoments)
  return metrics
}

export const buildSleepMetricBar = <N>(
  f: TriNodeFactory<N>,
  date: string,
  sleep: SleepMetrics | null | undefined,
): N | null => {
  const metrics = sleepSupplementMetrics(f.presentation, sleep)
  if (metrics.length === 0) return null
  return dayAnalyticsList(f, date, 'sleep-metrics', metrics, 'tri-sleep-metrics')
}

const dayAnalyticsContributionGroup = <N>(
  f: TriNodeFactory<N>,
  title: string,
  contributions: Record<string, number | null> | null,
): N | null => {
  if (!contributions) return null
  const rows = Object.entries(contributions).filter(
    (entry): entry is [string, number] => entry[1] != null,
  )
  if (rows.length === 0) return null
  const group = f.el('section', 'tri-sleep-contrib')
  f.add(
    group,
    f.el('h4', 'tri-ana-block-title', triText(f.presentation.locale, title), {
      'data-i18n': title,
    }),
  )
  for (const [key, value] of rows) {
    const label = key.replaceAll('_', ' ')
    const status = ouraScoreStatus(value)
    const row = f.el('div', 'tri-sleep-contrib-row', undefined, {
      'data-health-tone': status?.tone ?? 'neutral',
    })
    const valueText = `${dayAnalyticsNumber(f.presentation, value)}${status ? ` · ${triText(f.presentation.locale, status.label)}` : ''}`
    const bar = f.el('div', 'tri-sleep-contrib-bar', undefined, {
      role: 'meter',
      'aria-label': triText(f.presentation.locale, label),
      'aria-valuemin': '0',
      'aria-valuemax': '100',
      'aria-valuenow': String(value),
      'aria-valuetext': valueText,
      'data-health-tone': status?.tone ?? 'neutral',
    })
    f.add(
      bar,
      f.el('div', 'tri-sleep-contrib-fill', undefined, {
        style: `width:${Math.max(0, Math.min(100, value))}%`,
      }),
    )
    f.add(
      row,
      f.el('span', 'tri-sleep-contrib-label', triText(f.presentation.locale, label), {
        'data-i18n': label,
      }),
      bar,
      f.el('span', 'tri-sleep-contrib-val', dayAnalyticsNumber(f.presentation, value), {
        title: valueText,
      }),
    )
    f.add(group, row)
  }
  return group
}

const buildDaySleepPeriod = <N>(f: TriNodeFactory<N>, summary: TriathlonDayAnalytics): N | null => {
  const sleep = summary.sleep
  const metrics = dayAnalyticsSleepMetrics(f.presentation, summary)
  const metricBar = buildSleepMetricBar(
    f,
    summary.date,
    summary.sleepMetrics ??
      resolveSleepMetrics(
        {
          avgBreath: sleep?.averageBreathsPerMinute ?? null,
          health: summary.ouraHealth ?? undefined,
        },
        null,
      ),
  )
  const respiration = buildSleepRespirationChart(f, summary.date, summary.sleepMetrics)
  const recovery = buildGarminRecovery(f, summary.garminHealth)
  const training = buildGarminHealth(f, summary.garminHealth)
  const oura = buildOuraHealth(f, summary.ouraHealth, summary.restorationBaseline)
  if (
    !sleep &&
    metrics.length === 0 &&
    !metricBar &&
    !respiration &&
    !recovery &&
    !training &&
    !oura.length
  )
    return null
  const group = f.el('div', 'tri-sleep-period')
  if (metrics.length > 0)
    f.add(
      group,
      dayAnalyticsList(
        f,
        summary.date,
        'sleep',
        metrics,
        'tri-day-analytics-list tri-day-sleep-summary',
      ),
    )
  if (metricBar) f.add(group, metricBar)
  const scoreGroups = [
    dayAnalyticsContributionGroup(f, 'sleep score', sleep?.sleepContrib ?? null),
    dayAnalyticsContributionGroup(f, 'readiness', sleep?.readinessContrib ?? null),
    recovery,
    training,
    ...oura,
  ]
  if (scoreGroups.some(section => section != null)) {
    const contributions = f.el('div', 'tri-day-sleep-contributions')
    for (const section of scoreGroups) if (section) f.add(contributions, section)
    f.add(group, contributions)
  }
  if (sleep) {
    const stages = dayAnalyticsSleepStages(f, summary.date, sleep)
    const hrv = dayAnalyticsSleepSeries(f, summary.date, 'hrv', 'hrv', sleep.hrv)
    const heartRate = dayAnalyticsSleepSeries(
      f,
      summary.date,
      'heart-rate',
      'resting heart rate',
      sleep.heartRate,
    )
    const movement = dayAnalyticsSleepStages(f, summary.date, sleep, 'movement')
    if (stages) f.add(group, stages)
    if (movement) f.add(group, movement)
    if (hrv) f.add(group, hrv)
    if (heartRate) f.add(group, heartRate)
  }
  if (respiration) f.add(group, respiration)
  return group
}

export const buildSleepViews = <N>(
  f: TriNodeFactory<N>,
  night: N | null,
  naps: N,
  count: number | undefined,
  title: N,
): N => {
  const root = f.el('div', 'tri-sleep-views', undefined, { 'data-sleep-views': '' })
  const header = f.el('div', 'tri-sleep-head')
  const controls = f.el('div', 'tri-map-tablist tri-sleep-view-switch', undefined, {
    role: 'group',
    'aria-label': triText(f.presentation.locale, 'sleep view'),
    'data-i18n-aria-label': 'sleep view',
  })
  for (const mode of ['night', 'naps']) {
    const button = f.el('button', 'tri-map-tab tri-sleep-view-button', undefined, {
      type: 'button',
      'data-sleep-view': mode,
      'aria-pressed': String(mode === 'night'),
    })
    f.add(
      button,
      f.el('span', undefined, triText(f.presentation.locale, mode), { 'data-i18n': mode }),
    )
    if (mode === 'naps' && count != null)
      f.add(button, f.el('span', 'tri-sleep-nap-count', String(count)))
    f.add(controls, button)
  }
  const nightPane = f.el('div', 'tri-sleep-pane', undefined, { 'data-sleep-pane': 'night' })
  f.add(
    nightPane,
    night ??
      f.el('p', 'tri-ana-empty', triText(f.presentation.locale, 'no sleep logged'), {
        'data-i18n': 'no sleep logged',
      }),
  )
  const napPane = f.el('div', 'tri-sleep-pane', undefined, {
    'data-sleep-pane': 'naps',
    hidden: '',
  })
  f.add(napPane, naps)
  f.add(header, title, controls)
  f.add(root, header, nightPane, napPane)
  return root
}

export const buildDaySleepAnalytics = <N>(
  f: TriNodeFactory<N>,
  summary: TriathlonDayAnalytics,
): N | null => {
  const night = buildDaySleepPeriod(f, summary)
  if (!night && !summary.naps?.length) return null
  const titleId = `tri-day-${summary.date}-sleep`
  const group = f.el(
    'section',
    'tri-day-analytics-group tri-day-analytics-group--sleep',
    undefined,
    { 'aria-labelledby': titleId },
  )
  const title = f.el(
    'h3',
    'tri-ana-block-title tri-day-analytics-group-title',
    triText(f.presentation.locale, 'sleep details'),
    { id: titleId, 'data-i18n': 'sleep details' },
  )
  const naps = f.el('div', 'tri-sleep-naps')
  if (summary.naps?.length) {
    const total = summary.naps.reduce((sum, nap) => sum + (nap.totalSleepS ?? 0), 0)
    f.add(
      naps,
      f.el(
        'p',
        'tri-ana-k tri-sleep-nap-summary',
        `${summary.naps.length} ${triText(f.presentation.locale, summary.naps.length === 1 ? 'nap' : 'naps')} · ${dayAnalyticsDuration(total)} · Oura`,
      ),
    )
    for (const [index, nap] of summary.naps.entries()) {
      const period = buildDaySleepPeriod(f, {
        ...summary,
        date: `${summary.date}-nap-${index}`,
        sleep: nap,
        recovery: null,
        sleepMetrics: null,
        garminHealth: null,
        ouraHealth: null,
        restorationBaseline: null,
      })
      const entry = f.el('section', 'tri-sleep-nap', undefined, {
        'data-nap-id': nap.id,
        'data-nap-type': nap.type,
      })
      f.add(
        entry,
        f.el('h4', 'tri-ana-block-title', `${triText(f.presentation.locale, 'nap')} ${index + 1}`),
      )
      const deltas: DayAnalyticsMetric[] = []
      const changes: [string, number | null][] = [
        ['sleep score change', nap.sleepScoreDelta],
        ['readiness change', nap.readinessScoreDelta],
      ]
      for (const [label, value] of changes)
        if (value != null)
          deltas.push({ label, value: `${value > 0 ? '+' : ''}${value}`, detail: 'Oura' })
      if (nap.reportedDay && nap.reportedDay !== summary.date)
        deltas.push({ label: 'score date', value: nap.reportedDay, detail: 'Oura' })
      if (deltas.length)
        f.add(entry, dayAnalyticsList(f, `${summary.date}-nap-${index}`, 'nap', deltas))
      if (period) f.add(entry, period)
      f.add(naps, entry)
    }
  } else {
    const message = summary.naps ? 'no naps recorded' : 'nap data unavailable'
    f.add(
      naps,
      f.el('p', 'tri-ana-empty', triText(f.presentation.locale, message), { 'data-i18n': message }),
    )
  }
  f.add(group, buildSleepViews(f, night, naps, summary.naps?.length, title))
  return group
}

export const buildDaySleepSection = <N>(
  f: TriNodeFactory<N>,
  summary: TriathlonDayAnalytics,
): N | null => {
  const sleep = buildDaySleepAnalytics(f, summary)
  if (!sleep) return null
  const section = f.el('section', 'tri-day-analytics tri-day-sleep-analytics', undefined, {
    'aria-label': triText(f.presentation.locale, 'sleep details'),
    'data-analytics-date': summary.date,
    'data-i18n-aria-label': 'sleep details',
  })
  const groups = f.el('div', 'tri-day-analytics-grid')
  f.add(groups, sleep)
  f.add(section, groups)
  return section
}

export const buildDayAnalytics = <N>(f: TriNodeFactory<N>, summary: TriathlonDayAnalytics): N => {
  const section = f.el('section', 'tri-day-analytics', undefined, {
    'aria-label': triText(f.presentation.locale, 'daily analytics'),
    'data-analytics-date': summary.date,
    'data-i18n-aria-label': 'daily analytics',
  })
  const groups = f.el('div', 'tri-day-analytics-grid')
  for (const group of [
    dayAnalyticsGroup(f, summary.date, 'body-recovery', 'body · recovery', [
      ...dayAnalyticsBodyMetrics(f.presentation, summary),
      ...dayAnalyticsRecoveryMetrics(f.presentation, summary),
    ]),
    buildDaySleepAnalytics(f, summary),
    dayAnalyticsGroup(
      f,
      summary.date,
      'state-load',
      'state · load',
      dayAnalyticsTrainingMetrics(f.presentation, summary),
    ),
    dayAnalyticsGroup(
      f,
      summary.date,
      'thermal',
      'thermal',
      dayAnalyticsHeatMetrics(f.presentation, summary),
    ),
  ])
    if (group) f.add(groups, group)
  f.add(section, groups)
  return section
}
