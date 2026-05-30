import type {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../../types/component'
import { emptyPayload, SPORT_ORDER, type Sport } from '../../plugins/stores/strava'
import { classNames } from '../../util/lang'
import { joinSegments, pathToRoot } from '../../util/path'
// @ts-ignore
import script from '../scripts/triathlon.inline'
import style from '../styles/triathlon.scss'

const SPORT_LABEL: Record<Sport, string> = { swim: 'swim', bike: 'bike', run: 'run' }
const SPORT_EMOJI: Record<Sport, string> = { swim: '🏊', bike: '🚴', run: '🏃' }
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const KM_TO_MI = 0.621371
const BASELINE = 4

const FT_PER_KM = 3280.84

const miles = (km: number): string => Math.round(km * KM_TO_MI).toLocaleString('en-US')
const dist = (km: number): string => {
  const mi = km * KM_TO_MI
  return mi < 1 ? `${Math.round(km * FT_PER_KM)} ft` : `${mi.toFixed(1)} mi`
}

const clock = (totalSeconds: number): string => {
  const m = Math.floor(totalSeconds / 60)
  const s = Math.round(totalSeconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

const rate = (sport: Sport, km: number, seconds: number): string => {
  const mi = km * KM_TO_MI
  if (sport === 'bike') return `${(mi / (seconds / 3600)).toFixed(1)}mph`
  return `${clock(seconds / mi)}/mi`
}

const ordinal = (d: number): string => {
  if (d % 100 >= 11 && d % 100 <= 13) return 'th'
  return { 1: 'st', 2: 'nd', 3: 'rd' }[d % 10] ?? 'th'
}

const prettyDate = (iso: string): string => {
  const [, m, d] = iso.split('-').map(Number)
  return `${MONTHS[(m ?? 1) - 1]} ${d}${ordinal(d ?? 1)}`
}

export default (() => {
  const TriathlonPage: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
    const payload = fileData.stravaPayload ?? emptyPayload()
    const peak = payload.days.reduce((m, d) => Math.max(m, d.durationS), 1)
    const profile = `https://www.strava.com/athletes/${payload.athleteId || ''}`

    return (
      <article
        class={classNames(displayClass, 'triathlon', 'main-col', 'popover-hint')}
        data-detail-path={joinSegments(pathToRoot(fileData.slug!), 'static/strava-detail.json')}
      >
        <div class="tri-head">
          <a class="tri-total" href={profile} target="_blank" rel="noopener noreferrer">
            {miles(payload.totalKm)}mi
          </a>
        </div>

        <div class="tri-strip">
          <div
            class="tri-bars"
            role="img"
            aria-label={`${payload.totalCount} sessions, bar height by duration`}
            style={`--tri-days:${payload.days.length}`}
          >
            {payload.days.map(d => {
              const rest = d.items.length === 0
              const h = rest ? BASELINE : Math.max(BASELINE, (d.durationS / peak) * 100)
              return (
                <span
                  class={rest ? 'tri-bar' : 'tri-bar tri-bar--day'}
                  style={`height:${h.toFixed(2)}%`}
                  data-ids={rest ? undefined : d.items.map(i => i.id).join(',')}
                  data-readout={
                    rest
                      ? 'Rest'
                      : d.items
                          .map(
                            i =>
                              `${SPORT_EMOJI[i.sport]}|${dist(i.distanceKm)}|${rate(i.sport, i.distanceKm, i.durationS)}`,
                          )
                          .join(';')
                  }
                  data-date={prettyDate(d.date)}
                />
              )
            })}
          </div>
          <div class="tri-info" aria-hidden="true">
            <span class="tri-date" />
            <span class="tri-readout" />
          </div>
        </div>

        <div class="tri-foot">
          {SPORT_ORDER.map(sport => {
            const t = payload.totals.find(x => x.sport === sport)
            return (
              <span class="tri-leg">
                <span class="tri-leg-emoji">{SPORT_EMOJI[sport]}</span>
                {SPORT_LABEL[sport]} · {dist(t?.distanceKm ?? 0)} · {t?.count ?? 0}
              </span>
            )
          })}
        </div>

        <div class="tri-scrim" aria-hidden="true" />
        <aside class="tri-panel" aria-hidden="true">
          <button class="tri-panel-close" type="button" aria-label="Close">
            ×
          </button>
          <div class="tri-panel-body" />
        </aside>
      </article>
    )
  }

  TriathlonPage.css = style
  TriathlonPage.afterDOMLoaded = script

  return TriathlonPage
}) satisfies QuartzComponentConstructor
