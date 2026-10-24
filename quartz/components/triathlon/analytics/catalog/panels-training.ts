import { mountPrimaryPanel } from '../panel-mounts-primary'
import { mountSecondaryPanel } from '../panel-mounts-secondary'
import { buildAbilities } from '../panels/abilities'
import { buildEffort, buildHeatAcclimatisation } from '../panels/body'
import { buildCardio } from '../panels/cardio'
import { buildDistributions } from '../panels/distributions'
import { buildFtpHypothesis } from '../panels/ftp'
import { buildTrend } from '../panels/pace-trend'
import { buildPmc, buildWeekly } from '../panels/performance'
import { buildActions, buildReadiness } from '../panels/thresholds'
import { finite, value, withPanelMount, type AnalyticsPanelDefinition } from './shared'

export type TrainingPanelKey =
  | 'abilities'
  | 'distributions'
  | 'cardio'
  | 'pmc'
  | 'weekly'
  | 'effort'
  | 'heat'
  | 'readiness'
  | 'trend'
  | 'actions'
  | 'ftp'

export const trainingPanelDefinitions: Record<TrainingPanelKey, AnalyticsPanelDefinition> = {
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
    search:
      'heat acclimatisation temperature weatherkit core strain hot minutes sauna passive htl humidity cooldown',
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
        { label: 'sauna min', value: value(data.heat.saunaMinutes14d, ' min') },
        { label: 'recorded sauna HTL', value: value(data.heat.saunaHtl14d, '', 1) },
      ],
      series: [
        { label: 'acclimatisation', values: data.heat.series.map(day => day.acclimatisationPct) },
        { label: 'hot minutes', values: data.heat.series.map(day => day.hotMinutes) },
        { label: 'sauna HTL', values: finite(data.heat.series.map(day => day.saunaHtl)) },
        { label: 'temperature', values: finite(data.heat.series.map(day => day.temperatureC)) },
      ],
    }),
  },
  readiness: {
    key: 'readiness',
    label: 'race readiness',
    search: 'race readiness sprint olympic t50 70.3 t100 ironman prediction',
    render: buildReadiness,
    server: data => ({
      title: 'race readiness',
      values:
        data.races.length > 0
          ? data.races.map(race => ({
              label: race.distance,
              value: `${race.score.toFixed(0)}/100`,
            }))
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
