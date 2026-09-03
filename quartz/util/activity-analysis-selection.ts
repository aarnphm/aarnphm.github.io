import type { ActivityAnalyses } from '../plugins/stores/strava'
import type { UvSeverity } from './activity-provider-reports'

export type SelectedUvExposure =
  | {
      kind: 'pelotan-score'
      label: 'uv load™'
      score: number
      severity: UvSeverity | null
      rawBand: string | null
    }
  | {
      kind: 'garden-score'
      label: 'UV score'
      score: number
      severity: UvSeverity
      sourceToken: 'est.'
    }
  | { kind: 'garden-sed'; label: 'UV dose'; ambientSed: number; sourceToken: 'est.' }

export interface SelectedActivityAnalysisSummary {
  weatherImpact: { valuePct: number } | null
  uvExposure: SelectedUvExposure | null
}

export function selectActivityAnalysisSummary(
  analyses: ActivityAnalyses,
): SelectedActivityAnalysisSummary {
  const myWindsock = analyses.native.myWindsock
  const pelotan = analyses.native.pelotan
  const gardenScore = analyses.derived.uvScore
  const environment = analyses.derived.environment
  const uvExposure: SelectedUvExposure | null =
    pelotan?.score != null
      ? {
          kind: 'pelotan-score',
          label: 'uv load™',
          score: pelotan.score,
          severity: pelotan.severity,
          rawBand: pelotan.rawBand,
        }
      : gardenScore
        ? {
            kind: 'garden-score',
            label: 'UV score',
            score: gardenScore.score,
            severity: gardenScore.severity,
            sourceToken: 'est.',
          }
        : environment?.summary.ambientSed != null
          ? {
              kind: 'garden-sed',
              label: 'UV dose',
              ambientSed: environment.summary.ambientSed,
              sourceToken: 'est.',
            }
          : null
  return {
    weatherImpact:
      myWindsock?.weatherImpactPct == null ? null : { valuePct: myWindsock.weatherImpactPct },
    uvExposure,
  }
}
