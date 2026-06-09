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
const PACE_MI = [
  '5:30',
  '6:00',
  '6:30',
  '7:00',
  '7:30',
  '8:00',
  '8:30',
  '9:00',
  '9:30',
  '10:00',
  '10:30',
  '11:00',
]
const paceKm = (mi: string): string => {
  const [m = '0', s = '0'] = mi.split(':')
  const secKm = Math.round((Number(m) * 60 + Number(s)) * KM_TO_MI)
  return `${Math.floor(secKm / 60)}:${(secKm % 60).toString().padStart(2, '0')}`
}
const SWIM_100 = ['1:20', '1:30', '1:40', '1:50', '2:00', '2:10', '2:20', '2:30']
const swimMi = (p: string): string => {
  const [m = '0', s = '0'] = p.split(':')
  const secMi = Math.round((Number(m) * 60 + Number(s)) * 16.0934)
  return `${Math.floor(secMi / 60)}:${(secMi % 60).toString().padStart(2, '0')}`
}
const runKmh = (mi: string): string => {
  const [m = '0', s = '0'] = mi.split(':')
  const minPerMi = Number(m) + Number(s) / 60
  return minPerMi > 0 ? (60 / minPerMi / KM_TO_MI).toFixed(1) : '0'
}
const swimKmh = (p: string): string => {
  const [m = '0', s = '0'] = p.split(':')
  const sec = Number(m) * 60 + Number(s)
  return sec > 0 ? ((100 / sec) * 3.6).toFixed(1) : '0'
}
const BIKE_KMH = [25, 28, 30, 32, 35, 38, 40, 45]
const kmhToMph = (kmh: number): string => (kmh * KM_TO_MI).toFixed(1)
const clockFromSec = (sec: number): string => {
  const s = Math.round(sec)
  return `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`
}
const bikePaceKm = (kmh: number): string => clockFromSec(3600 / kmh)
const bikePaceMi = (kmh: number): string => clockFromSec(3600 / (kmh * KM_TO_MI))

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
    'Cervélo Soloist',
    [
      'Size 56 - 170mm, 7.39 kgs',
      'Cervélo All-Carbon, Tapered Soloist Fork',
      'Handlebar: Cervélo HB13 Carbon, 31.8mm clamp',
      'Handlebar Sizing: Size 56 - 40cm',
      'Stem: Cervélo ST36 Alloy',
      'Stem Sizing: Size 56 - 100mm',
      'Seatpost: Cervélo SP27 Carbon',
      'Saddle: Prologo Nago R4 PAS Tirox Lightweight',
      'Bottom Bracket: FSA, T47 BBright for 24mm spindle',
      'Headset: FSA IS2 1-1/4, 45° x 45° / 1-1/2, 36° x 45°',
      'Cervélo Aero Thru Axle Front, M12x1.5mm, 127mm length',
      'Cervélo Aero Thru Axle Rear, M12x1.5mm, 170.5mm length',
      'Front Wheel: Reserve 42TA, DT Swiss 350, 12x100mm, 24H, centerlock, tubeless compatible',
      'Rear Wheel: Reserve 49TA, DT Swiss 350, 12x142mm, HG freehub 24H, centerlock, tubeless compatible',
      'Tires: Vittoria Corsa N.EXT TLR G2.0 700x29c',
      'Shifter/Break: Shimano Ultegra, R8170',
      'Crankset: Shimano Ultegra, R8100, 52/36T',
      'Chain: Shimano M8100',
      'Cassette: Shimano Ultegra, R8100, 11-34T, 12-Speed',
      'Front/Rear Derailleur: Shimano Ultegra, R8150',
      'Brake Rotors: Shimano CL800 Centerlock',
      'Powermeter: Magene P715 S Pedal',
      'Bike Computer: Garmin Edge 1050',
      'HR monitor: Garmin HRM 600',
      'Shoes: SPECIALIZED TORCH 2.0',
    ],
  ],
  ['running', ['HOKA Clifton 10', 'Ciele Athletic Gocap']],
  ['swim', ['2XU trisuit', 'Decathlon swimskin', 'Speedo goggles']],
  ['wearables', ['Oura Ring 4', 'Apple Watch Ultra 3']],
  ['fuel', ['mandarins', 'apple', 'banana', 'Precision Fueld & Hydration', 'Maurten']],
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
    const target = String(fileData.frontmatter?.['triathlon'] ?? '')
    const raceDates = new Set((fileData.tracking?.races ?? []).map(r => r.date))

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
        data-location={location}
      >
        <div class="tri-head">
          <a class="tri-total" href={profile} target="_blank" rel="noopener noreferrer">
            {miles(payload.totalKm)}mi
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
                  const segRaw = d.items.map(it =>
                    Math.max(MIN_SEG, (it.durationS / 60) * PX_PER_MIN),
                  )
                  const gaps = Math.max(0, d.items.length - 1) * GAP_PX
                  const rawTotal = segRaw.reduce((a, b) => a + b, 0)
                  const scale = rawTotal + gaps > MAX_BAR ? (MAX_BAR - gaps) / rawTotal : 1
                  return (
                    <span
                      class={`tri-bar${rest ? '' : ' tri-bar--day'}${raceDates.has(d.date) ? ' tri-bar--race' : ''}`}
                      data-ids={rest ? undefined : d.items.map(i => i.id).join(',')}
                      data-date={prettyDate(d.date)}
                      data-date-iso={d.date}
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
              {TRI_DISTANCES.map(([label, s, b, r]) => (
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
            <div class="tri-gear-wrap">
              <button class="tri-gear-btn" type="button">
                gear
              </button>
              <div class="tri-gear" aria-hidden="true">
                {GEAR.map(([label, items]) => (
                  <div class="tri-gear-row">
                    <span class="tri-gear-k">{label}</span>
                    <span class="tri-gear-v">
                      {items.map(it => (
                        <span class="tri-gear-li">· {it}</span>
                      ))}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            <div class="tri-pace-wrap">
              <button class="tri-pace-btn" type="button">
                pace
              </button>
              <div class="tri-pace" aria-hidden="true">
                <span class="tri-pace-sec">run</span>
                <div class="tri-pace-row tri-pace-head">
                  <span>/mi</span>
                  <span>/km</span>
                  <button class="tri-pace-unit" type="button">
                    km/h
                  </button>
                </div>
                {PACE_MI.map(mi => {
                  const k = runKmh(mi)
                  return (
                    <div class="tri-pace-row">
                      <span class="tri-pace-mi">{mi}</span>
                      <span class="tri-pace-km">{paceKm(mi)}</span>
                      <span class="tri-pace-spd" data-kph={k} data-mph={kmhToMph(Number(k))}>
                        {k}
                      </span>
                    </div>
                  )
                })}
                <span class="tri-pace-sec">swim</span>
                <div class="tri-pace-row tri-pace-head">
                  <span>/100m</span>
                  <span>/mi</span>
                  <button class="tri-pace-unit" type="button">
                    km/h
                  </button>
                </div>
                {SWIM_100.map(p => {
                  const k = swimKmh(p)
                  return (
                    <div class="tri-pace-row">
                      <span class="tri-pace-mi">{p}</span>
                      <span class="tri-pace-km">{swimMi(p)}</span>
                      <span class="tri-pace-spd" data-kph={k} data-mph={kmhToMph(Number(k))}>
                        {k}
                      </span>
                    </div>
                  )
                })}
                <span class="tri-pace-sec">bike</span>
                <div class="tri-pace-row tri-pace-head">
                  <span>/mi</span>
                  <span>/km</span>
                  <button class="tri-pace-unit" type="button">
                    km/h
                  </button>
                </div>
                {BIKE_KMH.map(kmh => (
                  <div class="tri-pace-row">
                    <span class="tri-pace-mi">{bikePaceMi(kmh)}</span>
                    <span class="tri-pace-km">{bikePaceKm(kmh)}</span>
                    <span class="tri-pace-spd" data-kph={kmh} data-mph={kmhToMph(kmh)}>
                      {kmh}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            <button class="tri-analytics-btn" type="button">
              analytics
            </button>
            <button class="tri-calc-btn" type="button">
              calculator
            </button>
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

        <div class="tri-analytics-scrim" aria-hidden="true" />
        <aside
          class="tri-analytics"
          aria-hidden="true"
          role="dialog"
          aria-label="triathlon analytics"
        >
          <div class="tri-ana-bar">
            <span class="tri-ana-title">analytics</span>
            <input
              class="tri-ana-search"
              type="search"
              placeholder="search (filter:bike|run|swim, sort:distance|cadence|pace)"
              aria-label="search analytics"
              autocomplete="off"
            />
            <button class="tri-ana-close" type="button" aria-label="Close">
              ×
            </button>
          </div>
          <div class="tri-ana-body">
            <div class="tri-ana-results" aria-hidden="true" />
            <div class="tri-ana-detail" aria-hidden="true" />
            <div class="tri-ana-headline" />
            <div class="tri-ana-block" data-chart="body" />
            <div class="tri-ana-block" data-chart="gauge" />
            <div class="tri-ana-block" data-chart="pmc" />
            <div class="tri-ana-block" data-chart="ctl-sport" />
            <div class="tri-ana-block" data-chart="weekly" />
            <div class="tri-ana-block" data-chart="effort" />
            <div class="tri-ana-block" data-chart="readiness" />
            <div class="tri-ana-block" data-chart="trend" />
            <div class="tri-ana-block" data-chart="actions" />
          </div>
        </aside>
      </article>
    )
  }

  TriathlonPage.css = style
  TriathlonPage.afterDOMLoaded = script

  return TriathlonPage
}) satisfies QuartzComponentConstructor
