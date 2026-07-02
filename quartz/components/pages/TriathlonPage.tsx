import type {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../../types/component'
import {
  emptyPayload,
  SPORT_ICON,
  SPORT_ORDER,
  type ActivityKind,
  type Sport,
} from '../../plugins/stores/strava'
import { classNames } from '../../util/lang'
import { joinSegments, pathToRoot } from '../../util/path'
import { TRI_RACE_DISTANCES } from '../../util/triathlon-calculator'
import { dist, distCombined, dur } from '../../util/triathlon-card'
// @ts-ignore
import script from '../scripts/triathlon.inline'
import style from '../styles/triathlon.scss'
import {
  AnalyticsPanel,
  CalcPanel,
  CONVERSIONS,
  FuelLink,
  GearPanel,
  MapPanel,
  PacePanel,
  TrainingPanel,
} from './triathlon-panels'

const SPORT_LABEL: Record<Sport, string> = { swim: 'swim', bike: 'bike', run: 'run' }
const PX_PER_MIN = 2.4
const MAX_BAR = 300
const MIN_SEG = 3
const REST_SEG = 7
const GAP_PX = 2

const Icon = ({ sport, cls }: { sport: ActivityKind; cls: string }) => (
  <svg class={cls} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    {SPORT_ICON[sport].map(d => (
      <path d={d} />
    ))}
  </svg>
)

export default (() => {
  const TriathlonPage: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
    const payload = fileData.stravaPayload ?? emptyPayload()
    const profile = `https://www.strava.com/athletes/${payload.athleteId || 'aarnphm'}`
    const recentLoc = Object.values(payload.details)
      .sort((a, b) => b.date.localeCompare(a.date))
      .find(d => d.location)?.location
    const location = String(recentLoc ?? fileData.frontmatter?.['location'] ?? 'Toronto')
    const target = String(fileData.frontmatter?.['triathlon'] ?? '')
    const raceDates = new Set((fileData.tracking?.races ?? []).map(r => r.date))
    const trackByDate = new Map((fileData.tracking?.days ?? []).map(d => [d.date, d]))

    const yearStarts: { year: string; index: number }[] = []
    let lastYear = ''
    payload.days.forEach((d, i) => {
      const y = d.date.slice(0, 4)
      if (y !== lastYear) {
        yearStarts.push({ year: y, index: i })
        lastYear = y
      }
    })

    return (
      <article
        class={classNames(displayClass, 'triathlon', 'main-col', 'popover-hint')}
        data-detail-path={joinSegments(pathToRoot(fileData.slug!), 'static/strava-detail.json')}
        data-analytics-path={joinSegments(pathToRoot(fileData.slug!), 'static/analytics.json')}
        data-oura-detail-path={joinSegments(pathToRoot(fileData.slug!), 'static/oura-detail.json')}
        data-training-path={joinSegments(pathToRoot(fileData.slug!), 'static/training.json')}
        data-location={location}
      >
        <div class="tri-head">
          <a class="tri-total" href={profile} target="_blank" rel="noopener noreferrer">
            <span
              class="tri-dist"
              data-km={payload.totalKm}
              data-kind="combined"
              data-gloss="herodist"
              tabindex={0}
            >
              {distCombined(payload.totalKm)}
            </span>
          </a>
          <a class="tri-total" data-no-popover href="/">
            home
          </a>
        </div>

        <div class="tri-strip">
          <div class="tri-scroll">
            <div class="tri-track">
              <div
                class="tri-bars"
                role="img"
                aria-label={`${payload.totalCount} sessions, bar height by duration`}
              >
                {payload.days.map(d => {
                  const rest = d.items.length === 0
                  const track = trackByDate.get(d.date)
                  const restKind = (s: string): boolean => s === 'treatment' || s === 'yoga'
                  const segRaw = d.items.map(it =>
                    restKind(it.sport)
                      ? REST_SEG
                      : Math.max(MIN_SEG, (it.durationS / 60) * PX_PER_MIN),
                  )
                  const scalable = d.items.reduce(
                    (a, it, i) => (restKind(it.sport) ? a : a + segRaw[i]),
                    0,
                  )
                  const gaps = Math.max(0, d.items.length - 1) * GAP_PX
                  const scale = scalable + gaps > MAX_BAR ? (MAX_BAR - gaps) / scalable : 1
                  return (
                    <span
                      class={`tri-bar${rest ? '' : ' tri-bar--day'}${raceDates.has(d.date) ? ' tri-bar--race' : ''}`}
                      data-ids={rest ? undefined : d.items.map(i => i.id).join(',')}
                      data-date-iso={d.date}
                      data-event={track?.event ?? (track?.race ? 'race' : undefined)}
                    >
                      {rest ? (
                        <span class="tri-seg" style={`height:${REST_SEG}px`} />
                      ) : (
                        d.items.map((it, i) => (
                          <span
                            class={`tri-seg${restKind(it.sport) ? ' tri-seg--treatment' : ''}`}
                            style={`height:${(restKind(it.sport) ? segRaw[i] : segRaw[i] * scale).toFixed(1)}px`}
                          />
                        ))
                      )}
                    </span>
                  )
                })}
              </div>
              <div class="tri-axis">
                {yearStarts.map(({ year, index }) => (
                  <span class="tri-axis-year" style={`left:${index * 7}px`}>
                    {year}
                  </span>
                ))}
              </div>
            </div>
          </div>
          <aside class="tri-pop" aria-hidden="true" />
        </div>

        <div class="tri-foot">
          {SPORT_ORDER.map(sport => {
            const t = payload.totals.find(x => x.sport === sport)
            return (
              <span class="tri-leg">
                <Icon sport={sport} cls="tri-ico tri-leg-ico" />
                <span class="tri-leg-body">
                  {SPORT_LABEL[sport]} ·{' '}
                  <span
                    class="tri-dist"
                    data-km={t?.distanceKm ?? 0}
                    data-kind={sport}
                    data-gloss="legdist"
                    tabindex={0}
                  >
                    {dist(t?.distanceKm ?? 0, sport)}
                  </span>{' '}
                  ·{' '}
                  <span data-gloss="legcount" tabindex={0}>
                    {t?.count ?? 0}
                  </span>
                </span>
              </span>
            )
          })}
          {payload.strengthTotal.count > 0 && (
            <span class="tri-leg">
              <Icon sport="strength" cls="tri-ico tri-leg-ico" />
              <span class="tri-leg-body">
                strength ·{' '}
                <span data-gloss="legtime" tabindex={0}>
                  {dur(payload.strengthTotal.movingTimeS)}
                </span>{' '}
                ·{' '}
                <span data-gloss="legcount" tabindex={0}>
                  {payload.strengthTotal.count}
                </span>
              </span>
            </span>
          )}
        </div>

        <div class="tri-note">
          <div class="tri-conv">
            {CONVERSIONS.flatMap(([k, v]) => [
              <span class="tri-conv-k">{k}</span>,
              <span class="tri-conv-v">{v}</span>,
            ])}
          </div>
          <table class="tri-cheat">
            <thead>
              <tr>
                <th>
                  <button class="tri-cheat-unit" type="button">
                    km
                  </button>
                </th>
                <th>swim</th>
                <th>bike</th>
                <th>run</th>
              </tr>
            </thead>
            <tbody>
              {TRI_RACE_DISTANCES.map(([label, s, b, r]) => (
                <tr>
                  <th>
                    {label === target ? (
                      <span class="tri-cheat-target">
                        {label}
                        <span class="tri-cheat-sticker" aria-hidden="true">
                          🎯
                        </span>
                      </span>
                    ) : (
                      label
                    )}
                  </th>
                  <td data-km={s}>{s}</td>
                  <td data-km={b}>{b}</td>
                  <td data-km={r}>{r}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div class="tri-note-foot">
            <GearPanel />
            <PacePanel />
            <button class="tri-analytics-btn" type="button">
              analytics
            </button>
            <button class="tri-map-btn" type="button">
              map
            </button>
            <button class="tri-training-btn" type="button">
              training
            </button>
            <button class="tri-calc-btn" type="button">
              calculator
            </button>
            <FuelLink />
            <a
              class="tri-credit"
              href="https://rauno.me/run"
              target="_blank"
              rel="noopener noreferrer"
            >
              <i>inspired by rauno</i>
            </a>
          </div>
        </div>
        <CalcPanel />

        <AnalyticsPanel />
        <MapPanel />
        <TrainingPanel />
      </article>
    )
  }

  TriathlonPage.css = style
  TriathlonPage.afterDOMLoaded = script

  return TriathlonPage
}) satisfies QuartzComponentConstructor
