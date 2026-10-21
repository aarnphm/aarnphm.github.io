import type { StravaActivityDetail, StravaZones } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { hasHeartRateTrace, routeStreamFlags } from './activity-metric-data'
import { buildCyclingWorkoutAnalysis } from './cycling-workout'
import { buildHeartRateWorkoutAnalysis } from './heart-rate-workout'
import {
  buildRunLapSplits,
  buildRunWorkoutAnalysis,
  buildSwimWorkoutAnalysis,
} from './pace-workouts'
import {
  buildCyclingHeartRateDistribution,
  buildCyclingPowerDistribution,
  buildRunPaceDistribution,
} from './workout-distributions'

export const buildWorkoutAnalysisSection = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  embedded = false,
  zones: StravaZones | null = null,
  vt1: number | null = null,
  expandTabs = false,
): { element: N; metric: 'elapsed' | 'hr' | 'power' | 'pace' } | null => {
  const sport = {
    bike: 'Cycling',
    swim: 'Swim',
    run: 'Run',
    walk: 'Walk',
    strength: 'Strength',
    yoga: 'Recovery',
    treatment: 'Physiotherapy',
    sauna: 'Sauna',
  }[d.sport]
  const sportWorkout =
    d.sport === 'bike'
      ? buildCyclingWorkoutAnalysis(f, d)
      : d.sport === 'swim'
        ? buildSwimWorkoutAnalysis(f, d)
        : buildRunWorkoutAnalysis(f, d)
  const heartRateWorkout =
    sportWorkout == null && !routeStreamFlags(d).power ? buildHeartRateWorkoutAnalysis(f, d) : null
  const workout = sportWorkout ?? heartRateWorkout
  const splits = buildRunLapSplits(f, d)
  const pace = buildRunPaceDistribution(f, d)
  const power = buildCyclingPowerDistribution(f, d, zones)
  const heartRate = buildCyclingHeartRateDistribution(f, d, zones, vt1)
  const views = [
    { key: 'workout', label: 'workout analysis', shortLabel: 'WA', content: workout },
    { key: 'laps', label: 'lap splits', shortLabel: 'LS', content: splits },
    { key: 'pace', label: 'pace distribution', shortLabel: 'PD', content: pace },
    { key: 'power', label: 'power distribution', shortLabel: 'PD', content: power },
    { key: 'heart-rate', label: 'HR distribution', shortLabel: 'HR', content: heartRate },
  ].filter(
    (view): view is { key: string; label: string; shortLabel: string; content: N } =>
      view.content != null,
  )
  if (views.length === 0) return null
  const id = `tri-workout-analysis-${d.id}`
  const selected = views[0].key
  const metric =
    d.sport === 'sauna' && !hasHeartRateTrace(d)
      ? 'elapsed'
      : heartRateWorkout
        ? 'hr'
        : d.sport === 'bike'
          ? 'power'
          : 'pace'
  const wrap = f.el('section', 'tri-workout-analysis', undefined, {
    'aria-label': `${sport} analysis`,
    'data-sport': d.sport,
    'data-workout-analysis-metric': metric,
    'data-workout-analysis': '',
    'data-workout-analysis-view': selected,
  })
  const tabs = f.el('div', 'tri-map-tablist tri-workout-analysis-tabs', undefined, {
    role: 'tablist',
    'aria-label': `${sport} analysis view`,
  })
  const stage = f.el('div', 'tri-workout-analysis-stage')
  for (const view of views) {
    const active = view.key === selected
    const tab = f.el(
      'button',
      'tri-map-tab tri-workout-analysis-tab',
      expandTabs ? undefined : embedded ? view.shortLabel : view.label,
      {
        id: `${id}-${view.key}-tab`,
        type: 'button',
        role: 'tab',
        tabindex: active ? '0' : '-1',
        'aria-label': view.label,
        'aria-controls': `${id}-${view.key}-panel`,
        'aria-selected': String(active),
        'data-workout-analysis-tab': view.key,
        ...(expandTabs
          ? {
              style: `--tri-map-tab-shortcut-width: ${view.shortLabel.length}ch; --tri-map-tab-label-width: ${view.label.length}ch`,
            }
          : {}),
      },
    )
    if (expandTabs)
      f.add(
        tab,
        f.el('span', 'tri-map-tab-shortcut', view.shortLabel, { 'aria-hidden': 'true' }),
        f.el('span', 'tri-map-tab-label', view.label, { 'aria-hidden': 'true' }),
      )
    f.add(tabs, tab)
    const panel = f.el('div', 'tri-workout-analysis-panel', undefined, {
      id: `${id}-${view.key}-panel`,
      role: 'tabpanel',
      ...(active ? {} : { hidden: '', inert: '' }),
      'aria-hidden': String(!active),
      'aria-labelledby': `${id}-${view.key}-tab`,
      'data-workout-analysis-panel': view.key,
    })
    f.add(panel, view.content)
    f.add(stage, panel)
  }
  f.add(wrap, tabs, stage)
  return { element: wrap, metric }
}

export const buildWorkoutAnalysis = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  embedded = false,
  zones: StravaZones | null = null,
  expandTabs = false,
): N | null => buildWorkoutAnalysisSection(f, d, embedded, zones, null, expandTabs)?.element ?? null
