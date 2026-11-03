import type { StravaActivityDetail, StravaPayload } from '../../plugins/stores/strava'
import { SPORT_ICON, SPORT_ORDER, type ActivityKind } from '../../plugins/stores/strava'
import { dist, dur } from '../../util/triathlon-card'
import { DEFAULT_TRIATHLON_PRESENTATION } from '../../util/triathlon-presentation'

interface SeasonSummaryProps {
  totals: StravaPayload['totals']
  strengthTotal: StravaPayload['strengthTotal']
  activities: readonly Pick<StravaActivityDetail, 'sport' | 'distanceKm' | 'movingTimeS'>[]
}

interface SummaryTotal {
  sport: ActivityKind
  count: number
  distanceKm: number | null
  movingTimeS: number
  extra?: boolean
}

const MORE_ARROW_ICON: readonly string[] = ['M5 12h14', 'M13 6l6 6-6 6']

export const SeasonSummary = ({ totals, strengthTotal, activities }: SeasonSummaryProps) => {
  const summaries: SummaryTotal[] = SPORT_ORDER.map(sport => ({
    sport,
    count: 0,
    distanceKm: 0,
    movingTimeS: 0,
    ...totals.find(total => total.sport === sport),
  }))
  if (strengthTotal.count > 0)
    summaries.push({ sport: 'strength', distanceKm: null, extra: true, ...strengthTotal })

  for (const sport of ['walk', 'hike', 'sauna', 'treatment'] satisfies ActivityKind[]) {
    const matching = activities.filter(activity => activity.sport === sport)
    if (matching.length === 0) continue
    summaries.push({
      sport,
      count: matching.length,
      distanceKm:
        sport === 'walk' || sport === 'hike'
          ? matching.reduce((sum, activity) => sum + activity.distanceKm, 0)
          : null,
      movingTimeS: matching.reduce((sum, activity) => sum + activity.movingTimeS, 0),
      extra: true,
    })
  }

  const hasExtras = summaries.some(summary => summary.extra)

  return (
    <div class="tri-foot" data-expanded={hasExtras ? 'false' : undefined}>
      {summaries.map(({ sport, count, distanceKm, movingTimeS, extra }) => {
        const label = sport === 'treatment' ? 'physiotherapy' : sport
        return (
          <span class="tri-leg" key={sport} data-tri-extra={extra ? 'true' : undefined}>
            <svg class="tri-ico tri-leg-ico" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              {SPORT_ICON[sport].map(d => (
                <path d={d} />
              ))}
            </svg>
            <span class="tri-leg-body">
              <span data-i18n={label}>{label}</span> ·{' '}
              {distanceKm != null ? (
                <span
                  class="tri-dist tri-unit-distance"
                  data-km={distanceKm}
                  data-kind={sport}
                  data-gloss="legdist"
                  tabindex={0}
                >
                  {dist(DEFAULT_TRIATHLON_PRESENTATION, distanceKm, sport)}
                </span>
              ) : (
                <span data-gloss="legtime" tabindex={0}>
                  {dur(movingTimeS)}
                </span>
              )}{' '}
              ·{' '}
              <span data-gloss="legcount" tabindex={0}>
                {count}
              </span>
            </span>
          </span>
        )
      })}
      {hasExtras && (
        <button
          class="tri-foot-more"
          type="button"
          aria-expanded="false"
          aria-label="more activities"
          data-i18n-aria-label="more activities"
        >
          <svg
            class="tri-ico tri-leg-ico tri-foot-more-ico"
            viewBox="0 0 24 24"
            fill="none"
            aria-hidden="true"
          >
            {MORE_ARROW_ICON.map(d => (
              <path d={d} />
            ))}
          </svg>
        </button>
      )}
    </div>
  )
}
