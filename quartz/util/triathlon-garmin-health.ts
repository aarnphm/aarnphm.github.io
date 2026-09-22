import type { GarminHealthDay, GarminHealthReading } from '../plugins/stores/garmin-health'
import type { TriNodeFactory } from './triathlon-card'
import { latestGarminReadiness, morningGarminReadiness } from '../plugins/stores/garmin-health'
import { healthScoreTable, healthTooltip, type HealthRow } from './triathlon-health'
import { triText } from './triathlon-i18n'

export interface GarminHealthMetric {
  label: string
  value: string
  detail?: string
}

const number = (value: number | null | undefined, digits = 0): string =>
  value == null ? '—' : value.toFixed(digits)
const phrase = (value: string | null | undefined): string =>
  value ? value.toLowerCase().replace(/_\d+$/, '').replaceAll('_', ' ') : '—'
const minutes = (value: number | null | undefined): string => {
  if (value == null) return '—'
  const total = Math.round(value)
  return `${Math.floor(total / 60)}h ${total % 60}m`
}
const freshness = (reading: GarminHealthReading<unknown>): string | undefined =>
  reading.status === 'error'
    ? reading.value == null
      ? 'refresh failed'
      : 'cached · refresh failed'
    : reading.status === 'unavailable'
      ? 'unavailable'
      : undefined

export function garminHealthSummary(day: GarminHealthDay | null | undefined): GarminHealthMetric[] {
  if (!day) return []
  const ready = latestGarminReadiness(day)
  const battery = day.bodyBattery.value?.samples.findLast(point => point.value != null)
  const status = day.trainingStatus.value
  return [
    { label: 'Body Battery', value: number(battery?.value), detail: freshness(day.bodyBattery) },
    {
      label: 'training readiness',
      value: number(ready?.score),
      detail: freshness(day.trainingReadiness),
    },
    {
      label: 'Recovery Time',
      value: minutes(ready?.recoveryTimeMinutes),
      detail: freshness(day.trainingReadiness),
    },
    {
      label: 'Training Status',
      value: status?.feedback ? phrase(status.feedback) : String(status?.status ?? '—'),
      detail: freshness(day.trainingStatus),
    },
    {
      label: 'Endurance Score',
      value: number(day.enduranceScore.value?.score),
      detail: freshness(day.enduranceScore),
    },
    {
      label: 'Hill Score',
      value: number(day.hillScore.value?.score),
      detail: freshness(day.hillScore),
    },
  ]
}

const wallClock = (timestamp: number, offset: number | null): string =>
  new Date(timestamp + (offset ?? 0) * 60_000).toISOString().slice(11, 16) +
  (offset == null ? ' UTC' : '')

export function buildGarminRecovery<N>(
  f: TriNodeFactory<N>,
  day: GarminHealthDay | null | undefined,
): N | null {
  if (!day) return null
  const t = (key: string): string => triText(f.presentation.locale, key)
  const ready = latestGarminReadiness(day)
  const morning = morningGarminReadiness(day)
  const battery = day.bodyBattery.value
  const latestBattery = battery?.samples.findLast(point => point.value != null)
  const range = battery?.samples.flatMap(point => (point.value == null ? [] : [point.value])) ?? []
  const source = (
    reading: GarminHealthReading<unknown>,
    lines: (string | null | undefined)[],
  ): string =>
    ['Garmin', freshness(reading) && t(freshness(reading) ?? ''), ...lines]
      .filter(Boolean)
      .join('\n')
  const observed =
    ready?.timestampLocal?.slice(11, 16) ??
    (ready?.timestamp != null ? wallClock(ready.timestamp, null) : null)
  const rows: HealthRow[] = [
    {
      label: 'Body Battery',
      value: number(latestBattery?.value),
      score: latestBattery?.value,
      detail: source(day.bodyBattery, [
        t('body battery description'),
        latestBattery &&
          `${t('observed at')} ${wallClock(latestBattery.timestamp, battery?.utcOffsetMinutes ?? null)}`,
        battery &&
          `${t('charged')} ${number(battery.charged)} · ${t('drained')} ${number(battery.drained)}`,
        range.length ? `${t('range')} ${Math.min(...range)}–${Math.max(...range)}` : null,
      ]),
    },
    {
      label: 'training readiness',
      value: number(ready?.score),
      score: ready?.score,
      detail: source(day.trainingReadiness, [
        t('training readiness description'),
        observed && `${t('observed at')} ${observed}`,
        ready?.level && phrase(ready.level),
        ready?.feedback && phrase(ready.feedback),
        morning &&
          `${t('morning readiness')} ${number(morning.score)}${morning.timestampLocal ? ` · ${morning.timestampLocal.slice(11, 16)}` : ''}`,
      ]),
    },
    {
      label: 'Recovery Time',
      value: minutes(ready?.recoveryTimeMinutes),
      detail: source(day.trainingReadiness, [
        t('recovery time description'),
        ...(day.trainingReadiness.value ?? []).map(
          row =>
            `${row.timestampLocal?.slice(11, 16) ?? (row.timestamp != null ? wallClock(row.timestamp, null) : '—')} · ${minutes(row.recoveryTimeMinutes)}`,
        ),
      ]),
    },
  ]
  for (const factor of ready?.factors ?? []) {
    const score = factor.feedback === 'NONE' ? null : factor.percent
    rows.push({
      label:
        factor.name === 'recovery time'
          ? 'recovery score'
          : factor.name === 'sleep score'
            ? 'sleep contribution'
            : factor.name === 'acute load'
              ? 'load contribution'
              : factor.name,
      value: score == null ? '—' : `${number(score)}%`,
      score,
      detail: source(day.trainingReadiness, [
        t('readiness factor description'),
        t(`${factor.name} factor description`),
        score == null ? t('unavailable') : phrase(factor.feedback),
        observed && `${t('observed at')} ${observed}`,
      ]),
    })
  }
  if (ready?.hrvWeeklyAverage != null)
    rows.push({
      label: 'weekly HRV',
      value: `${number(ready.hrvWeeklyAverage)} ms`,
      detail: source(day.trainingReadiness, [t('weekly hrv description')]),
    })
  const section = f.el('section', 'tri-sleep-contrib tri-health-recovery', undefined, {
    'data-garmin-date': day.date,
  })
  f.add(
    section,
    f.el('h4', 'tri-ana-block-title', t('recovery')),
    healthScoreTable(f, day.date, 'recovery', rows),
  )
  return section
}

export interface HealthLoadRange {
  max: number
  low: number | null
  high: number | null
  load: number | null
}

export function garminLoadRange(
  load: number | null,
  low: number | null,
  high: number | null,
  max: number,
): HealthLoadRange {
  const valid = (value: number | null): value is number =>
    value != null && Number.isFinite(value) && value >= 0
  const upper = Math.max(1, Number.isFinite(max) ? max : 1, ...[load, low, high].filter(valid))
  const rangeValid = valid(low) && valid(high) && low <= high
  return {
    max: upper,
    low: rangeValid ? (low / upper) * 100 : null,
    high: rangeValid ? (high / upper) * 100 : null,
    load: valid(load) ? (load / upper) * 100 : null,
  }
}

export function buildGarminHealth<N>(
  f: TriNodeFactory<N>,
  day: GarminHealthDay | null | undefined,
): N | null {
  if (!day) return null
  const t = (key: string): string => triText(f.presentation.locale, key)
  const root = f.el('section', 'tri-sleep-contrib tri-health-training', undefined, {
    'data-garmin-date': day.date,
  })
  const status = day.trainingStatus.value
  const summary = garminHealthSummary(day)
  const metrics: GarminHealthMetric[] = [
    {
      ...summary[3],
      detail: [
        'Garmin',
        status?.feedback && phrase(status.feedback),
        status?.date !== day.date ? status?.date : null,
        freshness(day.trainingStatus),
      ]
        .filter(Boolean)
        .join('\n'),
    },
  ]
  if (status)
    metrics.push(
      {
        label: 'acute load',
        value: number(status.acuteLoad),
        detail: `Garmin\n${t('acute load description')}\n${t('target range')} ${number(status.targetMin)}–${number(status.targetMax)}`,
      },
      {
        label: 'chronic load',
        value: number(status.chronicLoad),
        detail: `Garmin\n${t('chronic load description')}`,
      },
      {
        label: 'acute/chronic ratio',
        value: number(status.loadRatio, 2),
        detail: `Garmin\n${t('load ratio description')}\n${phrase(status.loadStatus)}`,
      },
    )
  const endurance = day.enduranceScore.value
  const hill = day.hillScore.value
  const level = endurance?.thresholds.findLast(row => row.lower <= endurance.score)
  const next = endurance?.thresholds.find(row => row.lower > endurance.score)
  metrics.push(
    {
      ...summary[4],
      detail: [
        'Garmin',
        freshness(day.enduranceScore),
        level && t(level.label),
        next && `${t('next level')} ${t(next.label)} · ${number(next.lower)}`,
      ]
        .filter(Boolean)
        .join('\n'),
    },
    {
      ...summary[5],
      detail: [
        'Garmin',
        freshness(day.hillScore),
        hill &&
          `${t('hill strength')} ${number(hill.strength)} · ${t('hill endurance')} ${number(hill.endurance)}`,
      ]
        .filter(Boolean)
        .join('\n'),
    },
  )
  f.add(root, f.el('h4', 'tri-ana-block-title', t('training')))
  const list = f.el('dl', 'tri-health-summary')
  for (const [index, metric] of metrics.entries()) {
    const id = `tri-health-${day.date}-training-${index}`
    const row = f.el('div', 'tri-day-analytics-metric tri-health-tooltip-trigger', undefined, {
      tabindex: '0',
      'aria-describedby': id,
    })
    f.add(
      row,
      f.el('dt', 'tri-day-analytics-label', t(metric.label)),
      f.el('dd', 'tri-day-analytics-value', metric.value),
      healthTooltip(f, id, metric.detail ?? 'Garmin'),
    )
    f.add(list, row)
  }
  f.add(root, list)
  const focus = status?.loadFocus
  if (focus) {
    f.add(root, f.el('h4', 'tri-ana-block-title tri-health-load-title', t('Load Focus')))
    const max =
      Math.max(1, ...focus.categories.flatMap(row => [row.load ?? 0, row.targetMax ?? 0])) * 1.1
    const table = f.el('div', 'tri-health-load-table', undefined, {
      role: 'table',
      'aria-label': t('Load Focus'),
    })
    for (const [index, category] of focus.categories.entries()) {
      const range = garminLoadRange(category.load, category.targetMin, category.targetMax, max)
      const id = `tri-health-${day.date}-load-${index}`
      const target =
        range.low == null ? '—' : `${number(category.targetMin)}–${number(category.targetMax)}`
      const text = `${t(category.name)} · ${t('load')} ${number(category.load)} · ${t('target range')} ${target}`
      const row = f.el('div', 'tri-health-load-row tri-health-tooltip-trigger', undefined, {
        role: 'row',
        tabindex: '0',
        'aria-label': text,
        'aria-describedby': id,
      })
      const plot = f.el('div', 'tri-health-load-plot', undefined, { role: 'cell' })
      const rail = f.el('div', 'tri-health-load-rail', undefined, { 'aria-hidden': 'true' })
      if (range.low != null && range.high != null)
        f.add(
          rail,
          f.el('span', 'tri-health-load-target', undefined, {
            style: `left:${range.low}%;width:${range.high - range.low}%`,
          }),
        )
      if (range.load != null) {
        f.add(
          rail,
          f.el('span', 'tri-health-load-marker', undefined, { style: `left:${range.load}%` }),
        )
        f.add(
          plot,
          f.el('span', 'tri-health-load-value', number(category.load), {
            style: `left:clamp(1.1rem, ${range.load}%, calc(100% - 1.1rem))`,
          }),
        )
      }
      f.add(plot, rail)
      f.add(
        row,
        f.el('span', 'tri-health-load-name', t(category.name), { role: 'rowheader' }),
        plot,
        f.el('span', 'tri-health-load-range', target, { role: 'cell' }),
        healthTooltip(
          f,
          id,
          `Garmin\n${t('load focus description')}\n${text}\n${phrase(focus.feedback)}${focus.date !== day.date ? `\n${t('load as of')} ${focus.date}` : ''}`,
        ),
      )
      f.add(table, row)
    }
    f.add(root, table, f.el('p', 'tri-health-load-note', t('load focus legend')))
  }
  return root
}
