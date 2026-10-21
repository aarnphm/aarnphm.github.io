import type { StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { triText } from '../triathlon-i18n'
import {
  gardenEnvironmentValue,
  providerEnvironmentValue,
  surfaceCurrentEnvironmentValue,
  type EnvironmentEvidenceGroup,
  type EnvironmentEvidenceRow,
} from './environment-evidence'
import { formatSigned } from './environment-readout'
import { dlabel, formatAltitude, formatTemperature, speedKph } from './metric-text'

/** The sources the environment section reads; derived once by buildEnvironmentAnalysis. */
export interface EnvironmentSources {
  pelotan: StravaActivityDetail['analyses']['native']['pelotan']
  myWindsock: StravaActivityDetail['analyses']['native']['myWindsock']
  environment: StravaActivityDetail['analyses']['derived']['environment']
  openWaterSwim: boolean
  surfaceCurrent:
    | NonNullable<StravaActivityDetail['analyses']['derived']['environment']>['surfaceCurrent']
    | null
    | undefined
  uvScore: StravaActivityDetail['analyses']['derived']['uvScore']
  wind: StravaActivityDetail['analyses']['derived']['apparentWind']
}

export const environmentEvidenceGroups = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  {
    pelotan,
    myWindsock,
    environment,
    openWaterSwim,
    surfaceCurrent,
    uvScore,
    wind,
  }: EnvironmentSources,
): EnvironmentEvidenceGroup[] => {
  const uvRows: EnvironmentEvidenceRow[] = [
    {
      label: pelotan?.score != null ? 'uv load™' : 'UV score',
      value:
        pelotan?.score != null
          ? providerEnvironmentValue(
              `${pelotan.score}${pelotan.rawBand ? ` · ${pelotan.rawBand}` : ''}`,
              'pelotan',
            )
          : uvScore
            ? gardenEnvironmentValue(`${uvScore.score} · ${uvScore.severity}`, uvScore)
            : null,
    },
    {
      label: 'ambient SED',
      value:
        environment?.summary.ambientSed != null
          ? gardenEnvironmentValue(`${environment.summary.ambientSed.toFixed(2)} SED`, environment)
          : null,
    },
    {
      label: 'average UVI',
      value:
        pelotan?.averageUvIndex != null
          ? providerEnvironmentValue(pelotan.averageUvIndex.toFixed(1), 'pelotan')
          : environment?.summary.averageUvIndex != null
            ? gardenEnvironmentValue(environment.summary.averageUvIndex.toFixed(1), environment)
            : null,
    },
    {
      label: 'peak UVI',
      value:
        environment?.summary.peakUvIndex != null
          ? gardenEnvironmentValue(environment.summary.peakUvIndex.toFixed(1), environment)
          : null,
    },
    {
      label: 'average ambient temperature',
      value:
        pelotan?.averageTemperatureC != null
          ? providerEnvironmentValue(
              formatTemperature(f.presentation, pelotan.averageTemperatureC),
              'pelotan',
            )
          : environment?.summary.averageAmbientTemperatureC != null
            ? gardenEnvironmentValue(
                formatTemperature(f.presentation, environment.summary.averageAmbientTemperatureC),
                environment,
              )
            : null,
    },
    {
      label: 'average cloud cover',
      value:
        pelotan?.averageCloudCoverPct != null
          ? providerEnvironmentValue(`${Math.round(pelotan.averageCloudCoverPct)}%`, 'pelotan')
          : environment?.summary.averageCloudCoverPct != null
            ? gardenEnvironmentValue(
                `${Math.round(environment.summary.averageCloudCoverPct)}%`,
                environment,
              )
            : null,
    },
    {
      label: 'covered duration',
      value: environment
        ? gardenEnvironmentValue(
            `${dlabel((environment.summary.elapsedDurationS * environment.coverage.uvPct) / 100)} · ${environment.coverage.uvPct.toFixed(1)}%`,
            environment,
          )
        : null,
    },
  ]

  const windRows: EnvironmentEvidenceRow[] = [
    {
      label: 'Weather Impact',
      value:
        myWindsock?.weatherImpactPct != null
          ? providerEnvironmentValue(
              `${myWindsock.weatherImpactPct > 0 ? '+' : ''}${myWindsock.weatherImpactPct.toFixed(1)}%`,
              'mywindsock',
            )
          : null,
    },
    openWaterSwim
      ? {
          label: 'surface current speed',
          detail: 'Surface current speed detail',
          value:
            surfaceCurrent?.summary.averageSpeedMps != null &&
            surfaceCurrent.summary.coveragePct > 0
              ? surfaceCurrentEnvironmentValue(
                  `${surfaceCurrent.summary.averageSpeedMps.toFixed(2)} m/s`,
                  surfaceCurrent,
                )
              : null,
        }
      : {
          label: 'CdA',
          detail: 'CdA detail',
          value:
            myWindsock?.cdaM2 != null
              ? providerEnvironmentValue(myWindsock.cdaM2.toFixed(3), 'mywindsock')
              : null,
        },
    openWaterSwim
      ? {
          label: 'surface current direction',
          detail: 'Surface current direction detail',
          value:
            surfaceCurrent?.summary.averageDirectionDeg != null &&
            surfaceCurrent.summary.coveragePct > 0
              ? surfaceCurrentEnvironmentValue(
                  `${surfaceCurrent.summary.averageDirectionDeg.toFixed(0)}° ${triText(f.presentation.locale, 'toward')}`,
                  surfaceCurrent,
                )
              : null,
        }
      : {
          label: 'Feels Like Elevation',
          value:
            myWindsock?.feelsLikeElevationM != null
              ? providerEnvironmentValue(
                  formatAltitude(f.presentation, myWindsock.feelsLikeElevationM),
                  'mywindsock',
                )
              : null,
        },
    openWaterSwim
      ? {
          label: 'surface current coverage',
          detail: 'Surface current coverage detail',
          value: surfaceCurrent
            ? surfaceCurrentEnvironmentValue(
                `${surfaceCurrent.summary.coveragePct.toFixed(1)}% · ${dlabel(surfaceCurrent.summary.coveredDurationS)} / ${dlabel(surfaceCurrent.summary.elapsedDurationS)}`,
                surfaceCurrent,
              )
            : null,
        }
      : {
          label: 'headwind share',
          detail: 'Headwind share detail',
          value:
            myWindsock?.headwindPct != null
              ? providerEnvironmentValue(`${myWindsock.headwindPct.toFixed(1)}%`, 'mywindsock')
              : wind
                ? gardenEnvironmentValue(`${wind.summary.headwindSharePct.toFixed(1)}%`, wind)
                : null,
        },
    {
      label: 'longest headwind',
      detail: 'Longest headwind detail',
      value:
        myWindsock?.longestHeadwindS != null
          ? providerEnvironmentValue(dlabel(myWindsock.longestHeadwindS), 'mywindsock')
          : wind
            ? gardenEnvironmentValue(dlabel(wind.summary.longestHeadwindS), wind)
            : null,
    },
    {
      label: 'ambient wind speed',
      detail: 'Ambient wind speed detail',
      value:
        !(d.sport === 'swim' && d.swimLocation === 'pool') &&
        environment?.summary.averageWindSpeedKph != null &&
        Number.isFinite(environment.summary.averageWindSpeedKph) &&
        environment.summary.averageWindSpeedKph >= 0
          ? gardenEnvironmentValue(
              speedKph(f.presentation, environment.summary.averageWindSpeedKph),
              environment,
            )
          : null,
    },
    {
      label: 'wind speed range',
      detail: 'Wind speed range detail',
      value:
        myWindsock?.headwindMinKph != null && myWindsock.headwindMaxKph != null
          ? providerEnvironmentValue(
              `${speedKph(f.presentation, myWindsock.headwindMinKph)}–${speedKph(f.presentation, myWindsock.headwindMaxKph)}`,
              'mywindsock',
            )
          : null,
    },
    {
      label: 'air speed',
      detail: 'Air speed detail',
      value:
        myWindsock?.airSpeedKph != null
          ? providerEnvironmentValue(speedKph(f.presentation, myWindsock.airSpeedKph), 'mywindsock')
          : wind && !openWaterSwim
            ? gardenEnvironmentValue(
                speedKph(f.presentation, wind.summary.averageApparentAirSpeedKph),
                wind,
              )
            : null,
    },
    {
      label: 'precipitation',
      value:
        myWindsock?.precipitationProbabilityPct != null
          ? providerEnvironmentValue(
              `${myWindsock.precipitationProbabilityPct.toFixed(0)}%${myWindsock.precipitationRateMmPerHour == null ? '' : ` · ${myWindsock.precipitationRateMmPerHour.toFixed(1)} mm/h`}`,
              'mywindsock',
            )
          : null,
    },
    {
      label: 'headwind when into it / crosswind',
      detail: 'Wind components detail',
      value: wind
        ? gardenEnvironmentValue(
            `${speedKph(f.presentation, wind.summary.averageHeadwindWhileIntoKph)} / ${speedKph(f.presentation, wind.summary.averageCrosswindMagnitudeKph)}`,
            wind,
          )
        : null,
    },
    {
      label: 'apparent-air ratio',
      labelMath: '$v_{\\mathrm{air}} / v_{\\mathrm{ground}}$',
      detail: 'Apparent-air ratio detail',
      // A swimmer's air speed carries no drag, so the ratio would only restate wind over pace.
      value:
        wind && !openWaterSwim
          ? gardenEnvironmentValue(wind.summary.apparentAirRatio.toFixed(3), wind)
          : null,
    },
    {
      label: 'yaw and wind coverage',
      value: wind
        ? gardenEnvironmentValue(
            `${formatSigned(wind.summary.averageYawDeg, '°')} · ${wind.summary.coveragePct.toFixed(1)}%`,
            wind,
          )
        : null,
    },
  ]
  return [
    { label: 'UV exposure', rows: uvRows },
    { label: 'wind and aero', rows: windRows },
  ]
}
