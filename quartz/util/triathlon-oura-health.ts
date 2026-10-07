import type { OuraHealthDay } from '../plugins/stores/oura'
import type { OuraRestorationBaseline } from './oura-health'
import type { TriNodeFactory } from './triathlon-card'
import {
  healthScoreTable,
  ouraScoreStatus,
  type HealthRow,
  type HealthStatus,
} from './triathlon-health'
import { triText } from './triathlon-i18n'

const duration = (seconds: number): string => {
  const minutes = Math.round(seconds / 60)
  return minutes >= 60 ? `${Math.floor(minutes / 60)}h ${minutes % 60}m` : `${minutes}m`
}
const clock = (seconds: number): string => {
  const minute = ((Math.round(seconds / 60) % 1440) + 1440) % 1440
  return `${Math.floor(minute / 60)
    .toString()
    .padStart(2, '0')}:${(minute % 60).toString().padStart(2, '0')}`
}

export function buildOuraHealth<N>(
  f: TriNodeFactory<N>,
  day: OuraHealthDay | null | undefined,
  baseline: OuraRestorationBaseline | null | undefined,
): N[] {
  if (!day) return []
  const t = (key: string): string => triText(f.presentation.locale, key)
  const groups: N[] = []
  const source = (description: string): string =>
    `Oura\n${description}${day.failedCollections?.length ? `\n${t('cached · refresh failed')}: ${day.failedCollections.join(', ')}` : ''}`
  const group = (key: string, rows: HealthRow[], durations = false): void => {
    if (!rows.length) return
    const section = f.el('section', 'tri-sleep-contrib tri-health-oura', undefined, {
      'data-oura-health': key,
    })
    const table = healthScoreTable(f, day.date, key, rows)
    f.add(section, f.el('h4', 'tri-ana-block-title', t(key)), table)
    if (durations) {
      const wrapper = f.el('div', 'tri-health-durations')
      f.add(wrapper, section)
      groups.push(wrapper)
    } else groups.push(section)
  }
  const numberRow = (
    label: string,
    value: number | null | undefined,
    description: string,
    suffix = '',
    digits = 0,
  ): HealthRow[] =>
    value == null
      ? []
      : [{ label, value: `${value.toFixed(digits)}${suffix}`, detail: source(description) }]
  const scoreRow = (
    label: string,
    value: number | null | undefined,
    description: string,
  ): HealthRow[] =>
    value == null
      ? []
      : [
          {
            label,
            value: value.toFixed(0),
            score: value,
            status: ouraScoreStatus(value),
            detail: source(description),
          },
        ]
  const stress = day.stress
  if (stress && (stress.stressS != null || stress.restoredS != null)) {
    const max = Math.max(1, stress.stressS ?? 0, stress.restoredS ?? 0, baseline?.seconds ?? 0)
    const row = (
      label: string,
      seconds: number | null,
      description: string,
      status: HealthStatus | null,
    ): HealthRow[] =>
      seconds == null
        ? []
        : [
            {
              label,
              value: duration(seconds),
              score: (seconds / max) * 100,
              status,
              detail: source(description),
            },
          ]
    group(
      'daytime stress',
      [
        ...row(
          'stressed',
          stress.stressS,
          `${t('oura stress description')}\n${stress.summary ?? ''}`,
          { tone: 'watch', label: 'stressed' },
        ),
        ...row(
          'restored',
          stress.restoredS,
          t('oura restoration description'),
          baseline && baseline.seconds > 0 && stress.restoredS != null
            ? stress.restoredS >= baseline.seconds
              ? { tone: 'good', label: 'at or above usual restoration' }
              : { tone: 'watch', label: 'below usual restoration' }
            : { tone: 'info', label: 'restored' },
        ),
        ...(baseline && stress.restoredS != null
          ? row(
              'usual restoration',
              baseline.seconds,
              `${t('restoration baseline description')} · n=${baseline.days}`,
              { tone: 'neutral', label: 'usual restoration' },
            )
          : []),
      ],
      true,
    )
  }
  const resilience = day.resilience
  if (resilience)
    group('resilience', [
      ...(resilience.level
        ? [
            {
              label: 'level',
              value: t(resilience.level),
              detail: source(t('resilience description')),
            },
          ]
        : []),
      ...scoreRow('sleep recovery', resilience.sleepRecovery, t('resilience description')),
      ...scoreRow('daytime recovery', resilience.daytimeRecovery, t('resilience description')),
      ...scoreRow('stress balance', resilience.stress, t('resilience description')),
    ])
  const activity = day.activity
  if (activity) {
    const periods = [
      { label: 'resting', seconds: activity.restS },
      { label: 'sedentary', seconds: activity.sedentaryS },
      { label: 'low activity', seconds: activity.lowS },
      { label: 'medium activity', seconds: activity.mediumS },
      { label: 'high activity', seconds: activity.highS },
      { label: 'not worn', seconds: activity.nonWearS },
    ]
    const total = periods.reduce((sum, period) => sum + (period.seconds ?? 0), 0)
    group(
      'movement',
      [
        ...numberRow('steps', activity.steps, t('movement description')),
        ...periods.flatMap((period): HealthRow[] =>
          period.seconds == null
            ? []
            : [
                {
                  label: period.label,
                  value: duration(period.seconds),
                  score: (period.seconds / Math.max(1, total)) * 100,
                  // These widths are time shares, so a short active period is not a poor score.
                  status: {
                    tone:
                      period.label === 'sedentary'
                        ? 'watch'
                        : period.label === 'resting' || period.label === 'not worn'
                          ? 'neutral'
                          : 'info',
                    label: period.label,
                  },
                  detail: source(
                    `${t('movement description')}\n${t('recorded duration')} ${duration(total)}`,
                  ),
                },
              ],
        ),
        ...numberRow('inactivity alerts', activity.inactivityAlerts, t('movement description')),
      ],
      true,
    )
    const targets: HealthRow[] = []
    if (activity.activeCalories != null && activity.targetCalories != null) {
      const percent = (activity.activeCalories / activity.targetCalories) * 100
      targets.push({
        label: 'activity goal',
        value: `${percent.toFixed(0)}%`,
        score: Math.min(100, percent),
        status: {
          tone: percent >= 100 ? 'good' : 'watch',
          label: percent >= 100 ? 'target met' : 'below target',
        },
        detail: source(
          `${activity.activeCalories.toFixed(0)} / ${activity.targetCalories.toFixed(0)} kcal\n${t('activity goal description')}`,
        ),
      })
    }
    if (activity.equivalentWalkingDistanceM != null && activity.targetDistanceM != null) {
      const percent = (activity.equivalentWalkingDistanceM / activity.targetDistanceM) * 100
      targets.push({
        label: 'walking equivalent',
        value: `${percent.toFixed(0)}%`,
        score: Math.min(100, percent),
        status: {
          tone: percent >= 100 ? 'good' : 'watch',
          label: percent >= 100 ? 'target met' : 'below target',
        },
        detail: source(
          `${(activity.equivalentWalkingDistanceM / 1000).toFixed(1)} / ${(activity.targetDistanceM / 1000).toFixed(1)} km\n${t('walking equivalent description')}`,
        ),
      })
    }
    group('activity score', [
      ...scoreRow('activity score', activity.score, t('activity score description')),
      ...Object.entries(activity.contributors ?? {}).flatMap(([key, value]) =>
        scoreRow(key.replaceAll('_', ' '), value, t('activity contributor description')),
      ),
      ...targets,
    ])
  }
  const guidance: HealthRow[] = []
  const sleep = day.sleepTime
  if (sleep?.startOffsetS != null && sleep.endOffsetS != null)
    guidance.push({
      label: 'bedtime window',
      value: `${clock(sleep.startOffsetS)}–${clock(sleep.endOffsetS)}`,
      detail: source(
        `${t('bedtime window description')}\n${t(sleep.recommendation?.replaceAll('_', ' ') ?? '')}`,
      ),
    })
  else if (sleep?.recommendation)
    guidance.push({
      label: 'bedtime guidance',
      value: t(sleep.recommendation.replaceAll('_', ' ')),
      detail: source(t(sleep.status?.replaceAll('_', ' ') ?? '')),
    })
  guidance.push(
    ...numberRow(
      'temperature trend',
      day.temperatureTrendC,
      t('temperature trend description'),
      '°C',
      2,
    ),
  )
  group('sleep guidance', guidance)
  group('heart health', [
    ...numberRow(
      'vascular age',
      day.cardiovascular?.vascularAge,
      t('vascular age description'),
      ` ${t('years')}`,
    ),
    ...numberRow(
      'pulse wave velocity',
      day.cardiovascular?.pulseWaveVelocity,
      t('pulse wave description'),
      ' m/s',
      1,
    ),
    ...numberRow('ring vo2max', day.vo2Max, t('oura vo2 description'), ' ml/kg/min', 1),
  ])
  return groups
}
