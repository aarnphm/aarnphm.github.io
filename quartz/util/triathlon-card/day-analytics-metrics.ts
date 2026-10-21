import type { TriathlonDayAnalytics } from '../triathlon-day-analytics'
import type { TriathlonPresentation } from '../triathlon-presentation'
import type { TriNodeFactory } from './construction-contracts'
import { triText } from '../triathlon-i18n'
import { isImperial, LB_TO_KG } from './metric-text'

export type DayAnalyticsMetric = { label: string; value: string; detail?: string }

export const dayAnalyticsNumber = (
  presentation: TriathlonPresentation,
  value: number,
  digits = 0,
): string =>
  value.toLocaleString(presentation.locale === 'fr' ? 'fr-CA' : 'en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })

export const dayAnalyticsSigned = (
  presentation: TriathlonPresentation,
  value: number,
  digits = 1,
): string =>
  `${value > 0 ? '+' : ''}${dayAnalyticsNumber(presentation, value === 0 ? 0 : value, digits)}`

export const dayAnalyticsDuration = (seconds: number): string => {
  const total = Math.max(0, Math.round(seconds))
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const remaining = total % 60
  const parts: string[] = []
  if (hours > 0) parts.push(`${hours}h`)
  if (minutes > 0 || hours > 0) parts.push(`${minutes}m`)
  if (remaining > 0 || parts.length === 0) parts.push(`${remaining}s`)
  return parts.join(' ')
}

const dayAnalyticsClock = (presentation: TriathlonPresentation, value: string): string => {
  const match = /T(\d{2}):(\d{2})/.exec(value)
  if (!match) return value
  const hour = Number(match[1])
  const minute = match[2]
  if (presentation.locale === 'fr') return `${hour.toString().padStart(2, '0')}:${minute}`
  const suffix = hour < 12 ? 'am' : 'pm'
  const displayHour = hour % 12 || 12
  return `${displayHour}:${minute} ${suffix}`
}

const dayAnalyticsMass = (presentation: TriathlonPresentation, kilograms: number): string => {
  const imperial = isImperial(presentation)
  const value = imperial ? kilograms / LB_TO_KG : kilograms
  return `${dayAnalyticsNumber(presentation, value, 1)} ${imperial ? 'lb' : 'kg'}`
}

const dayAnalyticsTemperature = (
  presentation: TriathlonPresentation,
  celsius: number,
  signed = false,
): string => {
  const value = isImperial(presentation) ? (celsius * 9) / 5 + (signed ? 0 : 32) : celsius
  return `${signed && value > 0 ? '+' : ''}${dayAnalyticsNumber(presentation, value, 1)}${isImperial(presentation) ? '°F' : '°C'}`
}

export const dayAnalyticsList = <N>(
  f: TriNodeFactory<N>,
  date: string,
  key: string,
  metrics: DayAnalyticsMetric[],
  className = 'tri-day-analytics-list',
): N => {
  const list = f.el('dl', className)
  for (const [index, metric] of metrics.entries()) {
    const detailId = `tri-day-${date}-${key}-detail-${index}`
    const rowAttrs: Record<string, string> = { 'data-metric': metric.label }
    if (metric.detail) {
      rowAttrs.tabindex = '0'
      rowAttrs['aria-describedby'] = detailId
    }
    const row = f.el('div', 'tri-day-analytics-metric', undefined, rowAttrs)
    const value = f.el('dd', 'tri-day-analytics-value', metric.value)
    if (metric.detail)
      f.add(
        value,
        f.el('span', 'tri-day-analytics-detail', metric.detail, { id: detailId, role: 'tooltip' }),
      )
    f.add(
      row,
      f.el('dt', 'tri-day-analytics-label', triText(f.presentation.locale, metric.label), {
        'data-i18n': metric.label,
      }),
      value,
    )
    f.add(list, row)
  }
  return list
}

export const dayAnalyticsGroup = <N>(
  f: TriNodeFactory<N>,
  date: string,
  key: string,
  label: string,
  metrics: DayAnalyticsMetric[],
): N | null => {
  if (metrics.length === 0) return null
  const titleId = `tri-day-${date}-${key}`
  const group = f.el(
    'section',
    `tri-day-analytics-group tri-day-analytics-group--${key}`,
    undefined,
    { 'aria-labelledby': titleId },
  )
  f.add(
    group,
    f.el(
      'h3',
      'tri-ana-block-title tri-day-analytics-group-title',
      triText(f.presentation.locale, label),
      { id: titleId, 'data-i18n': label },
    ),
    dayAnalyticsList(f, date, key, metrics),
  )
  return group
}

export const dayAnalyticsBodyMetrics = (
  presentation: TriathlonPresentation,
  summary: TriathlonDayAnalytics,
): DayAnalyticsMetric[] => {
  const body = summary.body
  if (!body) return []
  const metrics: DayAnalyticsMetric[] = []
  if (body.kg != null)
    metrics.push({ label: 'body weight', value: dayAnalyticsMass(presentation, body.kg) })
  if (body.ffmi != null)
    metrics.push({ label: 'FFMI', value: dayAnalyticsNumber(presentation, body.ffmi, 2) })
  if (body.bmi != null)
    metrics.push({ label: 'bmi', value: dayAnalyticsNumber(presentation, body.bmi, 1) })
  if (body.bodyFatPct != null)
    metrics.push({
      label: 'body fat',
      value: `${dayAnalyticsNumber(presentation, body.bodyFatPct, 1)}%`,
    })
  if (body.bodyWaterPct != null)
    metrics.push({
      label: 'body water',
      value: `${dayAnalyticsNumber(presentation, body.bodyWaterPct, 1)}%`,
    })
  if (body.muscleMassKg != null)
    metrics.push({ label: 'muscle', value: dayAnalyticsMass(presentation, body.muscleMassKg) })
  if (body.boneMassKg != null)
    metrics.push({ label: 'bone', value: dayAnalyticsMass(presentation, body.boneMassKg) })
  return metrics
}

export const dayAnalyticsRecoveryMetrics = (
  presentation: TriathlonPresentation,
  summary: TriathlonDayAnalytics,
): DayAnalyticsMetric[] => {
  const recovery = summary.recovery
  if (!recovery) return []
  const metrics: DayAnalyticsMetric[] = []
  if (recovery.readiness != null)
    metrics.push({
      label: 'readiness',
      value: dayAnalyticsNumber(presentation, recovery.readiness),
      ...(recovery.readinessBaseline != null
        ? {
            detail: `${triText(presentation.locale, 'baseline')} ${dayAnalyticsNumber(presentation, recovery.readinessBaseline)}`,
          }
        : {}),
    })
  if (recovery.hrv != null)
    metrics.push({
      label: 'hrv',
      value: `${dayAnalyticsNumber(presentation, recovery.hrv)} ms`,
      detail: [
        recovery.hrvBaseline == null
          ? null
          : `${triText(presentation.locale, 'baseline')} ${dayAnalyticsNumber(presentation, recovery.hrvBaseline, 1)}`,
        recovery.hrvZ == null ? null : `z ${dayAnalyticsSigned(presentation, recovery.hrvZ, 2)}σ`,
      ]
        .filter(value => value != null)
        .join(' · '),
    })
  if (recovery.rhr != null)
    metrics.push({
      label: 'resting hr',
      value: `${dayAnalyticsNumber(presentation, recovery.rhr)} bpm`,
      detail: [
        recovery.rhrBaseline == null
          ? null
          : `${triText(presentation.locale, 'baseline')} ${dayAnalyticsNumber(presentation, recovery.rhrBaseline, 1)}`,
        recovery.rhrZ == null ? null : `z ${dayAnalyticsSigned(presentation, recovery.rhrZ, 2)}σ`,
      ]
        .filter(value => value != null)
        .join(' · '),
    })
  if (recovery.temperatureDeviationC != null)
    metrics.push({
      label: 'temperature deviation',
      value: dayAnalyticsTemperature(presentation, recovery.temperatureDeviationC, true),
    })
  return metrics
}

export const dayAnalyticsSleepMetrics = (
  presentation: TriathlonPresentation,
  summary: TriathlonDayAnalytics,
): DayAnalyticsMetric[] => {
  const sleep = summary.sleep
  const recovery = summary.recovery
  if (!sleep && !recovery && !summary.sleepMetrics) return []
  const metrics: DayAnalyticsMetric[] = []
  const score = sleep?.sleepScore ?? null
  if (score != null)
    metrics.push({ label: 'sleep score', value: dayAnalyticsNumber(presentation, score) })
  if (sleep?.readinessScore != null)
    metrics.push({
      label: 'readiness',
      value: dayAnalyticsNumber(presentation, sleep.readinessScore),
    })
  if (sleep?.bedtimeStart)
    metrics.push({ label: 'bedtime', value: dayAnalyticsClock(presentation, sleep.bedtimeStart) })
  if (sleep?.bedtimeEnd)
    metrics.push({ label: 'wake-up', value: dayAnalyticsClock(presentation, sleep.bedtimeEnd) })
  if (sleep?.timeInBedS != null)
    metrics.push({ label: 'time in bed', value: dayAnalyticsDuration(sleep.timeInBedS) })
  const totalSleep = sleep?.totalSleepS ?? recovery?.sleepDurationS
  if (totalSleep != null)
    metrics.push({ label: 'total sleep', value: dayAnalyticsDuration(totalSleep) })
  if (sleep?.efficiency != null)
    metrics.push({
      label: 'efficiency',
      value: `${dayAnalyticsNumber(presentation, sleep.efficiency)}%`,
    })
  if (sleep?.latencyS != null)
    metrics.push({ label: 'latency', value: dayAnalyticsDuration(sleep.latencyS) })
  if (sleep?.averageHeartRate != null)
    metrics.push({
      label: 'average hr',
      value: `${dayAnalyticsNumber(presentation, sleep.averageHeartRate, 1)} bpm`,
    })
  if (sleep?.lowestHeartRate != null)
    metrics.push({
      label: 'lowest hr',
      value: `${dayAnalyticsNumber(presentation, sleep.lowestHeartRate)} bpm`,
    })
  if (sleep?.averageHrv != null)
    metrics.push({
      label: 'average hrv',
      value: `${dayAnalyticsNumber(presentation, sleep.averageHrv)} ms`,
    })
  if (sleep?.restlessPeriods != null)
    metrics.push({
      label: 'restless periods',
      value: dayAnalyticsNumber(presentation, sleep.restlessPeriods),
      detail: 'Oura',
    })
  if (recovery?.sleepDebtS != null)
    metrics.push({ label: 'sleep debt', value: dayAnalyticsDuration(recovery.sleepDebtS) })
  if (recovery?.sleepBaselineS != null)
    metrics.push({ label: 'sleep baseline', value: dayAnalyticsDuration(recovery.sleepBaselineS) })
  if (recovery?.sleepTargetS != null)
    metrics.push({ label: 'sleep target', value: dayAnalyticsDuration(recovery.sleepTargetS) })
  return metrics
}

export const dayAnalyticsTrainingMetrics = (
  presentation: TriathlonPresentation,
  summary: TriathlonDayAnalytics,
): DayAnalyticsMetric[] => {
  const training = summary.training
  if (!training) return []
  const metrics: DayAnalyticsMetric[] = []
  if (training.vo2max)
    metrics.push({
      label: 'VO2max',
      value: `${dayAnalyticsNumber(presentation, training.vo2max.value, 1)} ml/kg/min`,
      detail:
        training.vo2max.method === 'garmin'
          ? 'Garmin'
          : training.vo2max.method === 'apple'
            ? 'Apple'
            : triText(presentation.locale, training.vo2max.method),
    })
  if (training.ctl != null)
    metrics.push({
      label: 'fitness · CTL',
      value: dayAnalyticsNumber(presentation, training.ctl, 1),
    })
  if (training.atl != null)
    metrics.push({
      label: 'fatigue · ATL',
      value: dayAnalyticsNumber(presentation, training.atl, 1),
    })
  if (training.tsb != null)
    metrics.push({ label: 'form · TSB', value: dayAnalyticsSigned(presentation, training.tsb, 1) })
  if (training.load != null)
    metrics.push({
      label: 'today load · TSS',
      value: dayAnalyticsNumber(presentation, training.load, 1),
      detail: triText(presentation.locale, 'site'),
    })
  if (training.garminTss != null)
    metrics.push({
      label: 'Garmin TSS',
      value: dayAnalyticsNumber(presentation, training.garminTss, 1),
    })
  if (training.exerciseLoad != null)
    metrics.push({
      label: 'exercise load',
      value: dayAnalyticsNumber(presentation, training.exerciseLoad, 1),
      detail:
        training.exerciseLoadSource === 'garmin'
          ? 'Garmin'
          : triText(presentation.locale, training.exerciseLoadSource ?? 'calculated'),
    })
  if (training.relativeEffort != null)
    metrics.push({
      label: 'relative effort',
      value: dayAnalyticsNumber(presentation, training.relativeEffort),
      detail: 'Strava',
    })
  return metrics
}

export const dayAnalyticsHeatMetrics = (
  presentation: TriathlonPresentation,
  summary: TriathlonDayAnalytics,
): DayAnalyticsMetric[] => {
  const heat = summary.heat
  if (!heat) return []
  const metrics: DayAnalyticsMetric[] = []
  if (heat.heatStrainIndex != null)
    metrics.push({ label: 'HSI', value: dayAnalyticsNumber(presentation, heat.heatStrainIndex, 1) })
  if (heat.temperatureC != null)
    metrics.push({
      label: heat.coreOrigin != null ? 'CORE temperature' : 'ambient temperature',
      value: dayAnalyticsTemperature(presentation, heat.temperatureC),
    })
  metrics.push(
    { label: 'observed', value: `${dayAnalyticsNumber(presentation, heat.observedMinutes)} min` },
    { label: 'hot min', value: dayAnalyticsNumber(presentation, heat.hotMinutes) },
    {
      label: 'acclimatisation',
      value: `${dayAnalyticsNumber(presentation, heat.acclimatisationPct)}%`,
    },
  )
  if (heat.dose > 0)
    metrics.push({ label: 'heat dose', value: dayAnalyticsNumber(presentation, heat.dose, 1) })
  if (heat.saunaMinutes > 0)
    metrics.push(
      { label: 'sauna min', value: dayAnalyticsNumber(presentation, heat.saunaMinutes) },
      {
        label: 'recorded sauna HTL',
        value: heat.saunaHtl == null ? '—' : dayAnalyticsNumber(presentation, heat.saunaHtl, 1),
      },
    )
  return metrics
}
