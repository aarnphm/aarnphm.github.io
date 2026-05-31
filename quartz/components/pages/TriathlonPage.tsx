import type {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from '../../types/component'
import { emptyPayload, SPORT_ICON, SPORT_ORDER, type Sport } from '../../plugins/stores/strava'
import { classNames } from '../../util/lang'
import { joinSegments, pathToRoot } from '../../util/path'
// @ts-ignore
import script from '../scripts/triathlon.inline'
import style from '../styles/triathlon.scss'

const SPORT_LABEL: Record<Sport, string> = { swim: 'swim', bike: 'bike', run: 'run' }
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const KM_TO_MI = 0.621371
const FT_PER_KM = 3280.84
const PX_PER_MIN = 2.4
const MAX_BAR = 300
const MIN_SEG = 3
const REST_SEG = 7
const GAP_PX = 2

const CONVERSIONS: [string, string][] = [
  ['pace', '/100m × 16.09 → /mi'],
  ['dist', 'km × 0.621 → mi'],
]
const TRI_DISTANCES: [string, string, string, string][] = [
  ['sprint', '0.75', '20', '5'],
  ['olympic', '1.5', '40', '10'],
  ['70.3', '1.9', '90', '21.1'],
  ['ironman', '3.8', '180', '42.2'],
]
const GEAR: [string, string[]][] = [
  [
    'soloist',
    [
      'ultegra di2',
      'shimano brakes',
      '172.5mm crank',
      'magene pedals',
      'garmin edge 150',
      'hrm 600',
      'specialized torch 2.0',
    ],
  ],
  ['running', ['hoka clifton 10', 'ciele athletic gocap']],
  ['swim', ['2xu trisuit', 'decathlon swimskin', 'speedo goggles']],
  ['wearables', ['oura ring 4', 'apple watch ultra 3']],
  ['fuel', ['mandarins', 'apple', 'banana']],
]

const miles = (km: number): string => Math.round(km * KM_TO_MI).toLocaleString('en-US')
const dist = (km: number, sport: Sport): string => {
  if (sport === 'swim') return `${Math.round(km * 1000).toLocaleString('en-US')} m`
  const mi = km * KM_TO_MI
  return mi < 1 ? `${Math.round(km * FT_PER_KM)} ft` : `${mi.toFixed(1)} mi`
}

const ordinal = (d: number): string => {
  if (d % 100 >= 11 && d % 100 <= 13) return 'th'
  return { 1: 'st', 2: 'nd', 3: 'rd' }[d % 10] ?? 'th'
}

const prettyDate = (iso: string): string => {
  const [, m, d] = iso.split('-').map(Number)
  return `${MONTHS[(m ?? 1) - 1]} ${d}${ordinal(d ?? 1)}`
}

const Icon = ({ sport, cls }: { sport: Sport; cls: string }) => (
  <svg class={cls} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    {SPORT_ICON[sport].map(d => (
      <path d={d} />
    ))}
  </svg>
)

export default (() => {
  const TriathlonPage: QuartzComponent = ({ fileData, displayClass }: QuartzComponentProps) => {
    const payload = fileData.stravaPayload ?? emptyPayload()
    const profile = `https://www.strava.com/athletes/${payload.athleteId || ''}`
    const location = String(fileData.frontmatter?.['location'] ?? 'Toronto')

    return (
      <article
        class={classNames(displayClass, 'triathlon', 'main-col', 'popover-hint')}
        data-detail-path={joinSegments(pathToRoot(fileData.slug!), 'static/strava-detail.json')}
        data-location={location}
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
              const segRaw = d.items.map(it => Math.max(MIN_SEG, (it.durationS / 60) * PX_PER_MIN))
              const gaps = Math.max(0, d.items.length - 1) * GAP_PX
              const rawTotal = segRaw.reduce((a, b) => a + b, 0)
              const scale = rawTotal + gaps > MAX_BAR ? (MAX_BAR - gaps) / rawTotal : 1
              return (
                <span
                  class={rest ? 'tri-bar' : 'tri-bar tri-bar--day'}
                  data-ids={rest ? undefined : d.items.map(i => i.id).join(',')}
                  data-date={prettyDate(d.date)}
                >
                  {rest ? (
                    <span class="tri-seg" style={`height:${REST_SEG}px`} />
                  ) : (
                    segRaw.map(s => (
                      <span class="tri-seg" style={`height:${(s * scale).toFixed(1)}px`} />
                    ))
                  )}
                </span>
              )
            })}
          </div>
          <aside class="tri-pop" aria-hidden="true" />
        </div>

        <div class="tri-foot">
          {SPORT_ORDER.map(sport => {
            const t = payload.totals.find(x => x.sport === sport)
            return (
              <span class="tri-leg">
                <Icon sport={sport} cls="tri-ico tri-leg-ico" />
                {SPORT_LABEL[sport]} · {dist(t?.distanceKm ?? 0, sport)} · {t?.count ?? 0}
              </span>
            )
          })}
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
                <th>km</th>
                <th>swim</th>
                <th>bike</th>
                <th>run</th>
              </tr>
            </thead>
            <tbody>
              {TRI_DISTANCES.map(([label, s, b, r]) => (
                <tr>
                  <th>{label}</th>
                  <td>{s}</td>
                  <td>{b}</td>
                  <td>{r}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div class="tri-note-foot">
            <div class="tri-gear-wrap">
              <button class="tri-gear-btn" type="button">
                gear
              </button>
              <div class="tri-gear" aria-hidden="true">
                {GEAR.map(([label, items]) => (
                  <div class="tri-gear-row">
                    <span class="tri-gear-k">{label}</span>
                    <span class="tri-gear-v">{items.join(' · ')}</span>
                  </div>
                ))}
              </div>
            </div>
            <button class="tri-calc-btn" type="button">
              calculator
            </button>
            <a
              class="tri-credit"
              href="https://rauno.me/craft/scroll"
              target="_blank"
              rel="noopener noreferrer"
            >
              <i>inspired by rauno</i>
            </a>
          </div>
        </div>

        <div class="tri-calc-scrim" aria-hidden="true" />
        <aside
          class="tri-calc"
          aria-hidden="true"
          role="dialog"
          aria-label="triathlon calculator"
          data-swim="1.5"
          data-bike="40"
          data-run="10"
        >
          <div class="tri-calc-bar">
            <span class="tri-calc-title">triathlon calculator</span>
            <button class="tri-calc-close" type="button" aria-label="Close">
              ×
            </button>
          </div>
          <div class="tri-calc-presets">
            {TRI_DISTANCES.map(([label, s, b, r]) => (
              <button
                class="tri-calc-preset"
                type="button"
                data-swim={s}
                data-bike={b}
                data-run={r}
              >
                {label}
              </button>
            ))}
          </div>
          <table class="tri-calc-io">
            <tbody>
              <tr>
                <th>swim</th>
                <td>
                  <input
                    class="tri-calc-in"
                    data-k="swim"
                    type="text"
                    value="2:00"
                    inputMode="numeric"
                  />
                </td>
                <td class="tri-calc-u">/100m</td>
                <td class="tri-calc-r" data-leg="swim">
                  —
                </td>
              </tr>
              <tr>
                <th>T1</th>
                <td>
                  <input
                    class="tri-calc-in"
                    data-k="t1"
                    type="text"
                    value="2:00"
                    inputMode="numeric"
                  />
                </td>
                <td class="tri-calc-u">min</td>
                <td class="tri-calc-r" data-leg="t1">
                  —
                </td>
              </tr>
              <tr>
                <th>bike</th>
                <td>
                  <input
                    class="tri-calc-in"
                    data-k="bike"
                    type="text"
                    value="18"
                    inputMode="decimal"
                  />
                </td>
                <td class="tri-calc-u">mph</td>
                <td class="tri-calc-r" data-leg="bike">
                  —
                </td>
              </tr>
              <tr>
                <th>T2</th>
                <td>
                  <input
                    class="tri-calc-in"
                    data-k="t2"
                    type="text"
                    value="1:30"
                    inputMode="numeric"
                  />
                </td>
                <td class="tri-calc-u">min</td>
                <td class="tri-calc-r" data-leg="t2">
                  —
                </td>
              </tr>
              <tr>
                <th>run</th>
                <td>
                  <input
                    class="tri-calc-in"
                    data-k="run"
                    type="text"
                    value="9:00"
                    inputMode="numeric"
                  />
                </td>
                <td class="tri-calc-u">/mi</td>
                <td class="tri-calc-r" data-leg="run">
                  —
                </td>
              </tr>
              <tr class="tri-calc-total">
                <th>finish</th>
                <td />
                <td />
                <td class="tri-calc-r" data-leg="total">
                  —
                </td>
              </tr>
            </tbody>
          </table>
        </aside>
      </article>
    )
  }

  TriathlonPage.css = style
  TriathlonPage.afterDOMLoaded = script

  return TriathlonPage
}) satisfies QuartzComponentConstructor
