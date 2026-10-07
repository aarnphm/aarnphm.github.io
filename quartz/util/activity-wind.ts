import type { StravaActivityDetail } from '../plugins/stores/strava'
import type { EnvironmentChartSample } from './activity-environment'
import type { MyWindsockRouteSample } from './mywindsock-route'

export interface ActivityWindSample extends EnvironmentChartSample {
  distanceKm: number
  headwindKph: number | null
  crosswindKph: number | null
}

export const myWindsockWindSamples = (
  samples: readonly MyWindsockRouteSample[],
): ActivityWindSample[] =>
  samples.map(sample => ({
    ...sample,
    headwindKph: sample.providerHeadwindKph,
    crosswindKph: sample.providerCrosswindKph,
  }))

export const preferredActivityWind = (
  detail: Pick<StravaActivityDetail, 'sport' | 'analyses'>,
): {
  provider: 'mywindsock' | 'garden'
  capturedAt: string | null
  samples: ActivityWindSample[]
} => {
  const route = detail.analyses.native.myWindsockRoute
  if (
    (detail.sport === 'bike' || detail.sport === 'run') &&
    route &&
    route.samples.filter(
      sample => sample.providerHeadwindKph != null && Number.isFinite(sample.providerHeadwindKph),
    ).length >= 2
  )
    return {
      provider: 'mywindsock',
      capturedAt: route.capturedAt,
      samples: myWindsockWindSamples(route.samples),
    }
  return {
    provider: 'garden',
    capturedAt: null,
    samples: detail.analyses.derived.environment?.samples ?? [],
  }
}
