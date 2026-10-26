import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { surfaceCurrentChartSamples, type EnvironmentChartSample } from '../activity-environment'
import { myWindsockWindSamples, preferredActivityWind } from '../activity-wind'
import { activityRecordedPauses } from '../elapsed-axis'
import { triText } from '../triathlon-i18n'
import { myWindsockLogoLink } from '../triathlon-mywindsock'
import { buildEnvironmentChart } from './environment-chart'
import { environmentLink, environmentLogoLink, environmentTable } from './environment-evidence'
import { environmentEvidenceGroups } from './environment-evidence-groups'
import { environmentChartReadout } from './environment-readout'
import {
  ENVIRONMENT_CHART_VIEWS,
  ENVIRONMENT_CURRENT_CHART_VIEWS,
  ENVIRONMENT_PROVIDER_CHART_VIEWS,
  environmentViewHasSamples,
  environmentViewLabel,
  environmentViewShortLabel,
  type EnvironmentChartView,
  type EnvironmentScoreModel,
} from './environment-scale'
import { buildRelativeWindRose } from './relative-wind-rose'

export const buildEnvironmentAnalysis = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
): N | null => {
  const pelotan = d.analyses.native.pelotan
  const myWindsock = d.analyses.native.myWindsock
  const environment = d.analyses.derived.environment
  const openWaterSwim = d.sport === 'swim' && d.swimLocation === 'openWater'
  const surfaceCurrent = openWaterSwim ? environment?.surfaceCurrent : null
  const uvScore = d.analyses.derived.uvScore
  const wind = d.analyses.derived.apparentWind
  const route = d.analyses.native.myWindsockRoute ?? null
  if (!pelotan && !myWindsock && !environment && !wind && !route) return null
  const wrap = f.el('section', 'tri-environment', undefined, {
    'aria-label': triText(f.presentation.locale, 'environment'),
    'data-i18n-aria-label': 'environment',
    'data-environment-analysis': '',
  })
  f.add(
    wrap,
    f.el('div', 'tri-zone-title', triText(f.presentation.locale, 'environment'), {
      'data-i18n': 'environment',
    }),
  )
  const evidenceGroups = environmentEvidenceGroups(f, d, {
    pelotan,
    myWindsock,
    environment,
    openWaterSwim,
    surfaceCurrent,
    uvScore,
    wind,
  })
  f.add(wrap, environmentTable(f, evidenceGroups))
  const environmentSamples =
    d.sport === 'swim' && d.swimLocation === 'pool'
      ? (environment?.samples ?? []).map(sample => ({
          ...sample,
          windSpeedKph: null,
          headwindKph: null,
          crosswindKph: null,
          apparentAirSpeedKph: null,
          yawDeg: null,
        }))
      : (environment?.samples ?? [])
  const environmentElapsedS = environment?.summary.elapsedDurationS ?? d.elapsedTimeS
  const currentSamples = surfaceCurrentChartSamples(surfaceCurrent?.samples ?? [])
  const windSource = preferredActivityWind(d)
  const providerSamples = myWindsockWindSamples(route?.samples ?? [])
  const providerViews = ENVIRONMENT_PROVIDER_CHART_VIEWS.filter(view =>
    environmentViewHasSamples(f.presentation, providerSamples, view),
  )
  const environmentViews: EnvironmentChartView[] = [
    ...ENVIRONMENT_CHART_VIEWS,
    ...(openWaterSwim ? ENVIRONMENT_CURRENT_CHART_VIEWS : []),
    ...(route ? (['relative-wind'] satisfies EnvironmentChartView[]) : []),
    ...providerViews,
  ]
  const samplesForView = (view: EnvironmentChartView): readonly EnvironmentChartSample[] =>
    ENVIRONMENT_CURRENT_CHART_VIEWS.includes(view)
      ? currentSamples
      : view === 'wind'
        ? windSource.provider === 'mywindsock'
          ? windSource.samples
          : environmentSamples
        : ENVIRONMENT_PROVIDER_CHART_VIEWS.includes(view)
          ? providerSamples
          : view === 'relative-wind'
            ? []
            : environmentSamples
  const selected =
    environmentViews.find(view =>
      environmentViewHasSamples(f.presentation, samplesForView(view), view),
    ) ?? 'cumulative'
  const id = `tri-environment-${d.id}`
  const scoreModel: EnvironmentScoreModel | null = uvScore
    ? { coefficientSed: uvScore.coefficientSed, doseClock: uvScore.doseClock }
    : null
  const charts = f.el('div', 'tri-environment-graphs', undefined, {
    'data-environment-tabs': '',
    'data-environment-view': selected,
    'data-environment-wind-source': windSource.provider,
    'data-environment-series': JSON.stringify(environmentSamples),
    ...(openWaterSwim ? { 'data-environment-current-series': JSON.stringify(currentSamples) } : {}),
    ...(route
      ? {
          'data-environment-provider-series': JSON.stringify(providerSamples),
          'data-environment-relative-wind': JSON.stringify(route.relativeWindPct),
        }
      : {}),
    'data-environment-elapsed': `${environmentElapsedS}`,
    ...(scoreModel == null
      ? {}
      : {
          'data-environment-score-coefficient': `${scoreModel.coefficientSed}`,
          'data-environment-score-clock': scoreModel.doseClock,
          'data-environment-cumulative-mode': 'score',
        }),
  })
  const controls = f.el('div', 'tri-environment-controls')
  const tabs = f.el('div', 'tri-map-tablist tri-environment-tablist', undefined, {
    role: 'tablist',
    'aria-label': triText(f.presentation.locale, 'Environment graph view'),
    'data-i18n-aria-label': 'Environment graph view',
  })
  for (const view of environmentViews) {
    const active = view === selected
    const label = environmentViewLabel(view)
    const shortLabel = environmentViewShortLabel(view)
    const translatedLabel = triText(f.presentation.locale, label)
    const translatedShortLabel = triText(f.presentation.locale, shortLabel)
    const tab = f.el('button', 'tri-map-tab tri-environment-tab', undefined, {
      id: `${id}-${view}-tab`,
      type: 'button',
      role: 'tab',
      tabindex: active ? '0' : '-1',
      title: translatedLabel,
      'aria-label': translatedLabel,
      'aria-controls': `${id}-${view}-panel`,
      'aria-selected': String(active),
      'data-environment-tab': view,
      'data-i18n-aria-label': label,
      style: `--tri-map-tab-shortcut-width: ${translatedShortLabel.length}ch; --tri-map-tab-label-width: ${translatedLabel.length}ch`,
    })
    f.add(
      tab,
      f.el('span', 'tri-environment-tab-short tri-map-tab-shortcut', translatedShortLabel, {
        'aria-hidden': 'true',
        'data-i18n': shortLabel,
      }),
      f.el('span', 'tri-environment-tab-full tri-map-tab-label', translatedLabel, {
        'aria-hidden': 'true',
        'data-i18n': label,
      }),
    )
    f.add(tabs, tab)
  }
  f.add(controls, tabs)
  if (scoreModel != null) {
    const mode = f.el('div', 'tri-environment-mode', undefined, {
      role: 'group',
      'aria-label': triText(f.presentation.locale, 'Cumulative exposure unit'),
      'data-i18n-aria-label': 'Cumulative exposure unit',
    })
    f.add(
      mode,
      f.el('button', undefined, triText(f.presentation.locale, 'score'), {
        type: 'button',
        'aria-pressed': 'true',
        'data-environment-mode': 'score',
        'data-i18n': 'score',
      }),
      f.el('button', undefined, 'SED', {
        type: 'button',
        'aria-pressed': 'false',
        'data-environment-mode': 'sed',
      }),
    )
    f.add(controls, mode)
  }
  const readoutSample = samplesForView(selected).at(-1)
  const readout = f.el(
    'output',
    'tri-environment-readout',
    readoutSample
      ? environmentChartReadout(f.presentation, readoutSample, scoreModel, selected)
      : '—',
    { 'aria-live': 'polite', 'data-environment-readout': '' },
  )
  const stage = f.el('div', 'tri-environment-stage')
  for (const view of environmentViews) {
    const active = view === selected
    const panel = f.el('div', 'tri-environment-panel', undefined, {
      id: `${id}-${view}-panel`,
      role: 'tabpanel',
      ...(active ? {} : { hidden: '' }),
      'aria-hidden': String(!active),
      'aria-labelledby': `${id}-${view}-tab`,
      'data-environment-panel': view,
    })
    f.add(
      panel,
      view === 'relative-wind' && route
        ? buildRelativeWindRose(f, route.relativeWindPct, route.samples)
        : buildEnvironmentChart(
            f,
            samplesForView(view),
            environmentElapsedS,
            view,
            view === 'cumulative' ? scoreModel : null,
            ENVIRONMENT_CURRENT_CHART_VIEWS.includes(view) ? (surfaceCurrent ?? null) : null,
            view === 'wind'
              ? windSource.provider
              : ENVIRONMENT_PROVIDER_CHART_VIEWS.includes(view)
                ? 'mywindsock'
                : null,
            activityRecordedPauses(d),
          ),
    )
    if (view === 'wind')
      f.add(
        panel,
        f.el(
          'span',
          'tri-environment-wind-source',
          windSource.provider === 'mywindsock'
            ? 'myWindsock'
            : triText(f.presentation.locale, 'Garden wind estimate'),
          {
            title: windSource.capturedAt
              ? `Captured ${windSource.capturedAt}`
              : triText(f.presentation.locale, 'Garden wind estimate'),
            'data-i18n':
              windSource.provider === 'mywindsock' ? 'myWindsock' : 'Garden wind estimate',
          },
        ),
      )
    f.add(stage, panel)
  }
  f.add(charts, controls, readout, stage)
  f.add(wrap, charts)
  const attribution = f.el('div', 'tri-environment-attribution')
  const myWindsockActivityId = myWindsock?.activityId ?? route?.activityId
  if (myWindsockActivityId != null)
    f.add(
      attribution,
      myWindsockLogoLink(
        f,
        myWindsockActivityId,
        'tri-environment-provider-logo tri-environment-mywindsock-logo',
      ),
    )
  if (pelotan)
    f.add(
      attribution,
      environmentLogoLink(
        f,
        'tri-environment-pelotan-logo',
        '/static/triathlon/pelotan.png',
        'Pelotan',
        'https://pelotan.cc/pages/uv-load',
        'Pelotan UV Load™',
      ),
    )
  if (surfaceCurrent)
    f.add(
      attribution,
      environmentLink(
        f,
        'NOAA LOOFS',
        'https://tidesandcurrents.noaa.gov/ofs/loofs/loofs_info.html',
      ),
    )
  if (environment) {
    const apple = environment.attribution
    const legalUrl = apple?.legalPageUrl ?? 'https://weatherkit.apple.com/legal-attribution.html'
    const provenance = triText(
      f.presentation.locale,
      'Garden estimate from Apple WeatherKit. Apple Weather data modified by Garden calculations.',
    )
    if (apple) {
      const picture = f.el('picture', 'tri-environment-apple-logo')
      f.add(
        picture,
        f.el('source', undefined, undefined, {
          media: '(prefers-color-scheme: dark)',
          srcset: '/static/triathlon/apple-weather-dark.png',
        }),
        f.el('img', undefined, undefined, {
          src: '/static/triathlon/apple-weather-light.png',
          alt: apple.serviceName,
          loading: 'lazy',
        }),
      )
      const link = f.el('a', undefined, undefined, {
        href: legalUrl,
        target: '_blank',
        rel: 'noreferrer',
        title: provenance,
        'aria-label': `${apple.serviceName} attribution. ${provenance}`,
      })
      f.add(link, picture)
      f.add(attribution, link)
    } else {
      f.add(attribution, environmentLink(f, 'Apple Weather', legalUrl))
    }
  }
  if (myWindsock || route || pelotan || environment) f.add(wrap, attribution)
  return wrap
}
