import { POWER_TO_WEIGHT_DURATIONS } from '../../../../plugins/stores/analytics'
import { sleepSupplementMetrics } from '../../../../util/triathlon-card'
import { garminHealthSummary } from '../../../../util/triathlon-garmin-health'
import { mountPrimaryPanel } from '../panel-mounts-primary'
import { mountSecondaryPanel } from '../panel-mounts-secondary'
import { buildBestEfforts } from '../panels/best-efforts'
import { buildBody } from '../panels/body'
import { buildDexa } from '../panels/body-composition'
import {
  bestPowerSeries,
  bestPowerSeriesLabel,
  buildBestPowerCurve,
  estimatedFtpForSeries,
} from '../panels/power'
import { buildPowerToWeightTrend, powerToWeightDurationLabel } from '../panels/power-to-weight'
import { buildRecoveryChart, buildSleep } from '../panels/recovery'
import { buildSwimPowerCurve } from '../panels/swim-power'
import { buildLactateThreshold, fmtLactateValue, lactateSourceText } from '../panels/thresholds'
import { buildVo2max } from '../panels/vo2'
import { buildGauge, mountWeightSwitch } from '../shared'
import { mountSleepPanel } from '../sleep'
import {
  bestEffortsServer,
  duration,
  finite,
  latestPowerToWeight,
  powerToWeightValues,
  value,
  withPanelMount,
  type AnalyticsPanelDefinition,
} from './shared'

export type PhysiologyPanelKey =
  | 'body'
  | 'dexa'
  | 'gauge'
  | 'recovery'
  | 'sleep'
  | 'vo2max'
  | 'lactate'
  | 'power'
  | 'swim-power'
  | 'best-efforts'

export const physiologyPanelDefinitions: Record<PhysiologyPanelKey, AnalyticsPanelDefinition> = {
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
    search:
      'sleep debt target score night nap hypnogram respiration breath pulse ox spo2 body battery stress garmin oura training readiness recovery time training status load focus endurance hill',
    render: (data, context) =>
      withPanelMount(buildSleep(data, context), [
        root => mountPrimaryPanel('sleep', root, data, context),
        root => mountSleepPanel(root, data, context),
      ]),
    server: (data, formatter) => {
      const latest = data.daily.findLast(day => day.sleepMetrics != null)
      const health = data.daily.find(day => day.date === data.meta.today)?.garminHealth
      return {
        title: formatter.text('sleep · debt'),
        values: [
          { label: formatter.text('latest'), value: duration(data.recovery.sleepLatestS) },
          { label: formatter.text('baseline'), value: duration(data.recovery.sleepBaselineS) },
          { label: formatter.text('debt'), value: duration(data.recovery.sleepDebtS) },
          ...(latest
            ? sleepSupplementMetrics(formatter.presentation, latest.sleepMetrics).map(metric => ({
                ...metric,
                label: formatter.text(metric.label),
              }))
            : []),
          ...garminHealthSummary(health).map(metric => ({
            ...metric,
            label: formatter.text(metric.label),
            detail: metric.detail ? formatter.text(metric.detail) : 'Garmin',
          })),
        ],
        series: [
          { label: 'sleep duration', values: finite(data.recovery.series.map(day => day.sleepS)) },
          { label: 'sleep score', values: finite(data.recovery.series.map(day => day.sleepScore)) },
        ],
      }
    },
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
    search: 'lactate threshold lt2 lthr pace heart rate critical power watts garmin running',
    render: (data, context) => buildLactateThreshold(data, context),
    server: (data, formatter) => ({
      title: 'lactate threshold',
      values: [
        {
          label:
            data.engine.lactateThreshold.heartRate?.source === 'garmin'
              ? `running heart rate · Garmin · ${data.engine.lactateThreshold.heartRate.date}`
              : 'heart rate · declared',
          value: value(data.engine.lactateThreshold.heartRate?.value, ' bpm'),
        },
        ...data.engine.lactateThreshold.sports.map(sport => ({
          label: `${sport.sport} · ${lactateSourceText(formatter, sport)}`,
          value: fmtLactateValue(formatter, sport),
        })),
      ],
    }),
  },
  power: {
    key: 'power',
    label: 'power curve',
    search:
      'cycling power curve critical power cp w prime ftp estimated eftp p20 watts duration best efforts power rank radar sprint attack climb w kg percentile',
    render: (data, context) => {
      const element = buildBestPowerCurve(data, context)
      const powerToWeight = buildPowerToWeightTrend(data, context)
      element.appendChild(powerToWeight.element)
      return withPanelMount({ element, mount: powerToWeight.mount }, [
        root => mountPrimaryPanel('power', root, data, context),
      ])
    },
    server: (data, formatter) => ({
      title: 'cycling · power · power-to-weight trend',
      values: [
        { label: 'FTP', value: value(data.powerCurve.ftp, ' W') },
        ...bestPowerSeries(data.powerCurve).map(({ key }) => ({
          label: `eFTP · ${bestPowerSeriesLabel(formatter, data.powerCurve, key)}`,
          value: value(estimatedFtpForSeries(data.powerCurve, key)?.watts, ' W'),
          detail: formatter.text(
            'eFTP = 95% of best recorded 20-minute power. Provisional: effort may be submaximal.',
          ),
        })),
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
  'swim-power': {
    key: 'swim-power',
    label: 'swim power curve',
    search:
      'swim freestyle swimming outdoor modeled relative drag power curve index pace duration best efforts six weeks year',
    render: buildSwimPowerCurve,
    server: (data, formatter) => ({
      title: 'swim · modeled drag power curve',
      values: [
        { label: 'swims', value: String(data.swimPowerCurve.activityCount) },
        {
          label: 'best 1 minute',
          value: value(
            data.swimPowerCurve.sixWeeks.find(point => point.durationS === 60)?.index,
            ' idx',
            1,
          ),
        },
        {
          label: 'best 5 minutes',
          value: value(
            data.swimPowerCurve.sixWeeks.find(point => point.durationS === 300)?.index,
            ' idx',
            1,
          ),
        },
      ],
      series: [
        {
          label: formatter.text('last 6 weeks'),
          values: data.swimPowerCurve.sixWeeks.map(point => point.index),
        },
        {
          label: `${formatter.text('all of')} ${data.swimPowerCurve.yearLabel}`,
          values: data.swimPowerCurve.year.map(point => point.index),
        },
      ],
      seriesDomain: 'shared-zero',
    }),
  },
  'best-efforts': {
    key: 'best-efforts',
    label: 'best efforts',
    search:
      'best efforts personal record pr top 10 medals 400m 1k mile 5k 10k half marathon marathon longest ride elevation climb power fastest',
    render: buildBestEfforts,
    server: bestEffortsServer,
  },
}
