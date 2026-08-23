import type { TriathlonContext } from '../runtime/context'
import type { TriathlonFormatter } from '../runtime/formatter'
import {
  POWER_TO_WEIGHT_DURATIONS,
  type Analytics,
  type PowerToWeightDurationS,
} from '../../../plugins/stores/analytics'
import { mountPrimaryPanel } from './panel-mounts-primary'
import { mountSecondaryPanel } from './panel-mounts-secondary'
import { buildAbilities } from './panels/abilities'
import { buildBody, buildEffort, buildHeatAcclimatisation } from './panels/body'
import { buildDexa } from './panels/body-composition'
import { buildCardio } from './panels/cardio'
import { buildDistributions } from './panels/distributions'
import { buildFtpHypothesis } from './panels/ftp'
import { buildPmc, buildWeekly } from './panels/performance'
import { buildBestPowerCurve } from './panels/power'
import { buildPowerToWeightTrend, powerToWeightDurationLabel } from './panels/power-to-weight'
import { buildRecoveryChart, buildSleep } from './panels/recovery'
import {
  buildActions,
  buildLactateThreshold,
  buildReadiness,
  buildTrend,
} from './panels/thresholds'
import { buildVo2max } from './panels/vo2'
import { buildGauge } from './shared'
import { mountWeightSwitch } from './shared'
import { mountSleepPanel } from './sleep'

export const ANALYTICS_PANEL_ORDER = [
  'body',
  'dexa',
  'gauge',
  'recovery',
  'sleep',
  'vo2max',
  'lactate',
  'power',
  'abilities',
  'distributions',
  'cardio',
  'pmc',
  'weekly',
  'effort',
  'heat',
  'readiness',
  'trend',
  'actions',
  'ftp',
] as const

export type AnalyticsPanelKey = (typeof ANALYTICS_PANEL_ORDER)[number]

export interface AnalyticsSummaryValue {
  label: string
  value: string
}

export interface AnalyticsPanelSeries {
  label: string
  values: readonly number[]
}

export interface AnalyticsPanelContent {
  title: string
  values: readonly AnalyticsSummaryValue[]
  series?: readonly AnalyticsPanelSeries[]
  seriesDomain?: 'independent' | 'shared-zero'
}

export interface AnalyticsPanelView {
  element: HTMLElement
  mount?: () => () => void
}

export type AnalyticsPanelRender = HTMLElement | AnalyticsPanelView

type AnalyticsPanelMount = (root: HTMLElement) => () => void

const withPanelMount = (
  rendered: AnalyticsPanelRender,
  mounts: readonly AnalyticsPanelMount[],
): AnalyticsPanelView => {
  const view = rendered instanceof HTMLElement ? { element: rendered } : rendered
  return {
    element: view.element,
    mount: () => {
      const root = view.element.parentElement ?? view.element
      const cleanups = [view.mount?.(), ...mounts.map(mount => mount(root))].filter(
        (cleanup): cleanup is () => void => cleanup != null,
      )
      return () => {
        for (const cleanup of cleanups) cleanup()
      }
    },
  }
}

export interface AnalyticsPanelDefinition {
  key: AnalyticsPanelKey
  label: string
  search: string
  render: (data: Analytics, context: TriathlonContext) => AnalyticsPanelRender
  server: (data: Analytics, formatter: TriathlonFormatter) => AnalyticsPanelContent
}

const value = (input: number | null | undefined, suffix = '', digits = 0): string =>
  input == null || !Number.isFinite(input) ? '—' : `${input.toFixed(digits)}${suffix}`

const duration = (seconds: number | null | undefined): string => {
  if (seconds == null || !Number.isFinite(seconds)) return '—'
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.round((seconds % 3600) / 60)
  return hours > 0 ? `${hours}h ${minutes.toString().padStart(2, '0')}m` : `${minutes}m`
}

const finite = (values: readonly (number | null | undefined)[]): number[] => {
  const result: number[] = []
  for (const item of values) if (item != null && Number.isFinite(item)) result.push(item)
  return result
}

const powerToWeightValues = (data: Analytics, durationS: PowerToWeightDurationS): number[] =>
  finite(data.powerCurve.powerToWeight.points.map(point => point.efforts[durationS]?.wattsPerKg))

const latestPowerToWeight = (data: Analytics, durationS: PowerToWeightDurationS): number | null =>
  data.powerCurve.powerToWeight.points.at(-1)?.efforts[durationS]?.wattsPerKg ?? null

const definitions: Record<AnalyticsPanelKey, AnalyticsPanelDefinition> = {
  body: {
    key: 'body',
    label: 'body weight',
    search: 'body weight mass bmi ffmi composition goal',
    render: (data, context) => {
      const element = buildBody(data, context)
      return withPanelMount(element, [
        root => mountWeightSwitch(root, context),
        root => mountPrimaryPanel('body', root, data, context),
      ])
    },
    server: data => ({
      title: 'body weight',
      values: [
        { label: 'latest', value: value(data.body.latestKg, ' kg', 1) },
        { label: 'trend', value: value(data.body.trendKgPerWeek, ' kg/wk', 2) },
        { label: 'measurements', value: String(data.body.series.length) },
      ],
      series: [{ label: 'weight', values: data.body.series.map(point => point.kg) }],
    }),
  },
  dexa: {
    key: 'dexa',
    label: 'DEXA',
    search: 'dexa lean fat bone composition scan',
    render: (data, context) =>
      withPanelMount(buildDexa(data, context), [
        root => mountPrimaryPanel('dexa', root, data, context),
        root => mountSecondaryPanel('dexa', root, data, context),
      ]),
    server: data => ({
      title: 'DEXA · body composition',
      values: [
        { label: 'scans', value: String(data.tests.dexa.length) },
        { label: 'latest', value: data.tests.dexa.at(-1)?.date ?? '—' },
        { label: 'body fat', value: value(data.body.bodyFatPct, '%', 1) },
      ],
      series: [{ label: 'body fat', values: data.tests.dexa.map(scan => scan.bodyFat) }],
    }),
  },
  gauge: {
    key: 'gauge',
    label: 'training load',
    search: 'form ramp gauge training load injury risk acwr monotony strain',
    render: buildGauge,
    server: data => ({
      title: 'training load · injury risk',
      values: [
        { label: 'fitness', value: value(data.risk.ctl, '', 1) },
        { label: 'fatigue', value: value(data.risk.atl, '', 1) },
        { label: 'form', value: value(data.risk.tsb, '', 1) },
      ],
      series: [
        { label: 'fitness', values: data.daily.map(day => day.ctl) },
        { label: 'fatigue', values: data.daily.map(day => day.atl) },
        { label: 'form', values: data.daily.map(day => day.tsb) },
      ],
    }),
  },
  recovery: {
    key: 'recovery',
    label: 'recovery',
    search: 'recovery hrv resting heart rate readiness temperature baseline',
    render: (data, context) =>
      withPanelMount(buildRecoveryChart(data, context), [
        root => mountPrimaryPanel('recovery', root, data, context),
      ]),
    server: data => ({
      title: 'recovery · HRV · RHR',
      values: [
        { label: 'HRV', value: value(data.recovery.hrvLatest, ' ms') },
        { label: 'RHR', value: value(data.recovery.rhrLatest, ' bpm') },
        { label: 'readiness', value: value(data.recovery.readinessLatest) },
      ],
      series: [
        { label: 'HRV', values: finite(data.recovery.series.map(day => day.hrv)) },
        { label: 'RHR', values: finite(data.recovery.series.map(day => day.rhr)) },
        { label: 'readiness', values: finite(data.recovery.series.map(day => day.readiness)) },
      ],
    }),
  },
  sleep: {
    key: 'sleep',
    label: 'sleep',
    search: 'sleep debt target score night hypnogram',
    render: (data, context) =>
      withPanelMount(buildSleep(data, context), [
        root => mountPrimaryPanel('sleep', root, data, context),
        root => mountSleepPanel(root, data, context),
      ]),
    server: data => ({
      title: 'sleep · debt',
      values: [
        { label: 'latest', value: duration(data.recovery.sleepLatestS) },
        { label: 'baseline', value: duration(data.recovery.sleepBaselineS) },
        { label: 'debt', value: duration(data.recovery.sleepDebtS) },
      ],
      series: [
        { label: 'sleep duration', values: finite(data.recovery.series.map(day => day.sleepS)) },
        { label: 'sleep score', values: finite(data.recovery.series.map(day => day.sleepScore)) },
      ],
    }),
  },
  vo2max: {
    key: 'vo2max',
    label: 'VO2max',
    search: 'vo2max aerobic fitness age percentile laboratory',
    render: (data, context) =>
      withPanelMount(buildVo2max(data, context), [
        root => mountPrimaryPanel('vo2max', root, data, context),
        root => mountSecondaryPanel('vo2max', root, data, context),
      ]),
    server: data => ({
      title: 'VO2max · fitness age',
      values: [
        { label: 'VO2max', value: value(data.engine.vo2max.value, ' ml/kg/min', 1) },
        { label: 'method', value: data.engine.vo2max.method },
        { label: 'fitness age', value: value(data.engine.vo2max.fitnessAge) },
      ],
      series: [
        { label: 'VO2max', values: data.engine.vo2max.trend.map(point => point.vo2max) },
        { label: 'laboratory', values: data.tests.vo2max.map(test => test.value) },
      ],
    }),
  },
  lactate: {
    key: 'lactate',
    label: 'lactate threshold',
    search: 'lactate threshold lt2 pace heart rate projection',
    render: (data, context) =>
      withPanelMount(buildLactateThreshold(data, context), [
        root => mountSecondaryPanel('lactate', root, data, context),
      ]),
    server: data => ({
      title: 'lactate threshold · projection',
      values: [
        {
          label: 'heart rate',
          value: value(data.engine.lactateThreshold.heartRate?.value, ' bpm'),
        },
        { label: 'sports', value: String(data.engine.lactateThreshold.sports.length) },
        {
          label: 'projected',
          value: String(
            data.engine.lactateThreshold.sports.filter(sport => sport.projected != null).length,
          ),
        },
      ],
      series: data.engine.lactateThreshold.sports.map(sport => ({
        label: sport.sport,
        values: sport.points.map(point => point.value),
      })),
    }),
  },
  power: {
    key: 'power',
    label: 'power curve',
    search:
      'cycling power curve critical power cp w prime ftp watts duration best efforts power rank radar sprint attack climb w kg percentile',
    render: (data, context) => {
      const element = buildBestPowerCurve(data, context)
      const powerToWeight = buildPowerToWeightTrend(data, context)
      element.appendChild(powerToWeight.element)
      return withPanelMount({ element, mount: powerToWeight.mount }, [
        root => mountPrimaryPanel('power', root, data, context),
      ])
    },
    server: data => ({
      title: 'cycling · power · power-to-weight trend',
      values: [
        { label: 'FTP', value: value(data.powerCurve.ftp, ' W') },
        {
          label: 'eCP',
          value: value(
            (data.powerCurve.criticalPower ?? data.powerCurve.criticalPowerYear)
              ?.criticalPowerWatts ?? null,
            ' W',
          ),
        },
        {
          label: 'eW′',
          value: value(
            (data.powerCurve.criticalPower ?? data.powerCurve.criticalPowerYear) == null
              ? null
              : ((data.powerCurve.criticalPower ?? data.powerCurve.criticalPowerYear)
                  ?.wPrimeJoules ?? 0) / 1000,
            ' kJ',
            1,
          ),
        },
        ...POWER_TO_WEIGHT_DURATIONS.map(durationS => ({
          label: powerToWeightDurationLabel(durationS),
          value: value(latestPowerToWeight(data, durationS), ' W/kg', 2),
        })),
      ],
      series: POWER_TO_WEIGHT_DURATIONS.map(durationS => ({
        label: powerToWeightDurationLabel(durationS),
        values: powerToWeightValues(data, durationS),
      })),
      seriesDomain: 'shared-zero',
    }),
  },
  abilities: {
    key: 'abilities',
    label: 'abilities',
    search: 'abilities radar sprint threshold endurance cadence climb stride recovery',
    render: (data, context) =>
      withPanelMount(buildAbilities(data, context), [
        root => mountSecondaryPanel('abilities', root, data, context),
      ]),
    server: data => ({
      title: 'sport abilities',
      values:
        data.engine.abilities.sports.length > 0
          ? data.engine.abilities.sports.map(sport => ({
              label: sport.sport,
              value: sport.area == null ? '—' : `${sport.area.toFixed(0)}/100`,
            }))
          : [{ label: 'status', value: 'insufficient data' }],
      series: data.engine.abilities.sports.map(sport => ({
        label: sport.sport,
        values: finite(sport.axes.map(axis => axis.score)),
      })),
    }),
  },
  distributions: {
    key: 'distributions',
    label: 'distributions',
    search:
      'training zones heart rate power pace cadence skin temperature heat strain distributions telemetry',
    render: buildDistributions,
    server: data => ({
      title: 'training zone distributions · telemetry',
      values: [
        { label: 'activities', value: String(data.distributions.activities.length) },
        {
          label: 'heart-rate samples',
          value: String(
            data.distributions.activities.filter(activity => activity.heartRateZoneSeconds).length,
          ),
        },
        {
          label: 'power samples',
          value: String(
            data.distributions.activities.filter(activity => activity.powerZoneSeconds).length,
          ),
        },
        {
          label: 'pace samples',
          value: String(
            data.distributions.activities.filter(activity => activity.paceZoneSeconds).length,
          ),
        },
      ],
      series: [
        {
          label: 'power',
          values: finite(data.distributions.activities.map(activity => activity.averagePowerWatts)),
        },
        {
          label: 'cadence',
          values: finite(data.distributions.activities.map(activity => activity.cadence)),
        },
        {
          label: 'heat strain',
          values: finite(data.distributions.activities.map(activity => activity.heatStrainIndex)),
        },
      ],
    }),
  },
  cardio: {
    key: 'cardio',
    label: 'cardiovascular trends',
    search: 'cardio rhr hrv efficiency factor decoupling',
    render: (data, context) =>
      withPanelMount(buildCardio(data, context), [
        root => mountSecondaryPanel('cardio', root, data, context),
      ]),
    server: data => ({
      title: 'cardiovascular trends',
      values:
        data.engine.cardio.metrics.length > 0
          ? data.engine.cardio.metrics.map(metric => ({
              label: metric.label,
              value: value(metric.value, metric.unit ? ` ${metric.unit}` : '', 1),
            }))
          : [{ label: 'status', value: 'insufficient data' }],
      series: [
        { label: 'RHR', values: data.engine.cardio.rhrSeries.map(point => point.rhr) },
        { label: 'HRV', values: data.engine.cardio.hrvSeries.map(point => point.hrv) },
        { label: 'efficiency', values: data.engine.cardio.efSeries.map(point => point.ef) },
        {
          label: 'decoupling',
          values: data.engine.cardio.decouplingSeries.map(point => point.pct),
        },
      ],
    }),
  },
  pmc: {
    key: 'pmc',
    label: 'performance management',
    search: 'pmc training stress score tss fitness fatigue form ctl atl tsb projection',
    render: buildPmc,
    server: data => ({
      title: 'fitness · fatigue · form · TSS',
      values: [
        { label: 'fitness', value: value(data.risk.ctl, '', 1) },
        { label: 'fatigue', value: value(data.risk.atl, '', 1) },
        { label: 'form', value: value(data.risk.tsb, '', 1) },
        { label: 'TSS', value: value(data.daily.at(-1)?.load, '', 1) },
      ],
      series: [
        { label: 'fitness', values: data.daily.map(day => day.ctl) },
        { label: 'fatigue', values: data.daily.map(day => day.atl) },
        { label: 'form', values: data.daily.map(day => day.tsb) },
        { label: 'TSS', values: data.daily.map(day => day.load) },
      ],
    }),
  },
  weekly: {
    key: 'weekly',
    label: 'weekly volume',
    search: 'weekly volume sessions distance hours load targets',
    render: (data, context) =>
      withPanelMount(buildWeekly(data, context), [
        root => mountPrimaryPanel('weekly', root, data, context),
      ]),
    server: data => {
      const latest = data.weekly.at(-1)
      return {
        title: 'weekly volume',
        values: [
          { label: 'week', value: latest?.weekStart ?? '—' },
          { label: 'sessions', value: value(latest?.sessions) },
          { label: 'hours', value: value(latest?.hours, ' h', 1) },
        ],
        series: [
          { label: 'hours', values: data.weekly.map(week => week.hours) },
          { label: 'load', values: data.weekly.map(week => week.load) },
          { label: 'sessions', values: data.weekly.map(week => week.sessions) },
        ],
      }
    },
  },
  effort: {
    key: 'effort',
    label: 'weekly effort',
    search: 'weekly effort perceived exertion sessions targets',
    render: (data, context) =>
      withPanelMount(buildEffort(data, context), [
        root => mountPrimaryPanel('effort', root, data, context),
      ]),
    server: data => {
      const latest = data.weekly.at(-1)
      return {
        title: 'weekly effort',
        values: [
          { label: 'week', value: latest?.weekStart ?? '—' },
          { label: 'effort', value: value(latest?.effort, '', 1) },
          { label: 'sessions', value: value(latest?.effortSessions) },
        ],
        series: [{ label: 'effort', values: data.weekly.map(week => week.effort) }],
      }
    },
  },
  heat: {
    key: 'heat',
    label: 'heat acclimatisation',
    search: 'heat acclimatisation temperature weatherkit core strain hot minutes',
    render: (data, context) =>
      withPanelMount(buildHeatAcclimatisation(data, context), [
        root => mountPrimaryPanel('heat', root, data, context),
      ]),
    server: data => ({
      title: 'heat acclimatisation',
      values: [
        { label: 'status', value: data.heat.state },
        { label: 'acclimatisation', value: value(data.heat.currentPct, '%') },
        { label: 'hot minutes 14d', value: value(data.heat.heatMinutes14d, ' min') },
      ],
      series: [
        { label: 'acclimatisation', values: data.heat.series.map(day => day.acclimatisationPct) },
        { label: 'hot minutes', values: data.heat.series.map(day => day.hotMinutes) },
        { label: 'temperature', values: finite(data.heat.series.map(day => day.temperatureC)) },
      ],
    }),
  },
  readiness: {
    key: 'readiness',
    label: 'race readiness',
    search: 'race readiness sprint olympic 70.3 ironman prediction',
    render: buildReadiness,
    server: data => ({
      title: 'race readiness',
      values:
        data.races.length > 0
          ? data.races
              .slice(0, 4)
              .map(race => ({ label: race.distance, value: `${race.score.toFixed(0)}/100` }))
          : [{ label: 'status', value: 'insufficient data' }],
      series: [{ label: 'readiness', values: data.races.map(race => race.score) }],
    }),
  },
  trend: {
    key: 'trend',
    label: 'pace projection',
    search: 'pace trend projection forecast swim bike run',
    render: (data, context) =>
      withPanelMount(buildTrend(data, context), [
        root => mountSecondaryPanel('trend', root, data, context),
      ]),
    server: data => ({
      title: 'pace · projection',
      values:
        data.trends.length > 0
          ? data.trends.map(trend => ({
              label: trend.sport,
              value: trend.slopePerWeek == null ? '—' : value(trend.slopePerWeek, '/wk', 2),
            }))
          : [{ label: 'status', value: 'insufficient data' }],
      series: data.trends.map(trend => ({
        label: trend.sport,
        values: finite([trend.level, ...trend.forecast.map(point => point.value)]),
      })),
    }),
  },
  actions: {
    key: 'actions',
    label: 'training actions',
    search: 'training actions recommendations weakest sport',
    render: buildActions,
    server: data => ({
      title: 'training actions',
      values:
        data.actions.length > 0
          ? data.actions
              .slice(0, 4)
              .map(action => ({ label: action.sourceMetric, value: action.text }))
          : [{ label: 'status', value: 'no current action' }],
    }),
  },
  ftp: {
    key: 'ftp',
    label: 'FTP hypothesis',
    search:
      'ftp hypothesis vo2 gross metabolic efficiency cycling watts critical power pedal smoothness torque effectiveness',
    render: (data, context) =>
      withPanelMount(buildFtpHypothesis(data, context), [
        root => mountSecondaryPanel('ftp', root, data, context),
      ]),
    server: (data, formatter) => ({
      title: 'FTP hypothesis',
      values: data.engine.ftpHypothesis
        ? [
            { label: 'FTP', value: value(data.engine.ftpHypothesis.ftp, ' W') },
            {
              label: 'range',
              value: `${data.engine.ftpHypothesis.low.toFixed(0)}–${data.engine.ftpHypothesis.high.toFixed(0)} W`,
            },
            { label: 'confidence', value: data.engine.ftpHypothesis.conf },
            {
              label: 'VO2 source',
              value: `${data.engine.ftpHypothesis.vo2maxSport} · ${data.engine.ftpHypothesis.vo2maxSource}`,
            },
            {
              label: 'efficiency',
              value: `${data.engine.ftpHypothesis.efficiency.valuePct.toFixed(1)}% · ${data.engine.ftpHypothesis.efficiency.source}`,
            },
            {
              label: 'eCP',
              value: value(data.engine.ftpHypothesis.power.criticalPowerWatts, ' W'),
            },
            {
              label: 'modeled 60 min',
              value: value(data.engine.ftpHypothesis.power.modeled60MinuteWatts, ' W'),
            },
            {
              label: 'pedal evidence',
              value: data.engine.ftpHypothesis.pedaling
                ? `${data.engine.ftpHypothesis.pedaling.activityCount} rides · ${data.engine.ftpHypothesis.pedaling.sampleCount} samples`
                : '—',
            },
            {
              label: 'observation window',
              value: data.engine.ftpHypothesis.pedaling
                ? `${formatter.shortDate(data.engine.ftpHypothesis.pedaling.windowFrom)}–${formatter.shortDate(data.engine.ftpHypothesis.pedaling.windowTo)}`
                : '—',
            },
          ]
        : [{ label: 'status', value: 'insufficient data' }],
      series: data.engine.ftpHypothesis
        ? [
            {
              label: 'FTP range',
              values: [
                data.engine.ftpHypothesis.low,
                data.engine.ftpHypothesis.ftp,
                data.engine.ftpHypothesis.high,
              ],
            },
          ]
        : [],
    }),
  },
}

export const ANALYTICS_CATALOG: readonly AnalyticsPanelDefinition[] = ANALYTICS_PANEL_ORDER.map(
  key => definitions[key],
)

export const analyticsPanelDefinition = (key: string): AnalyticsPanelDefinition | undefined =>
  ANALYTICS_CATALOG.find(panel => panel.key === key)
