import { KM_TO_MI } from '../../util/triathlon-card'

const NAV = [
  ['tools', 'tools'],
  ['analytics', 'analytics'],
  ['maps', 'maps'],
  ['training', 'training'],
] as const

export type TriView = (typeof NAV)[number][0]

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
const SWIM_100 = ['1:20', '1:30', '1:40', '1:50', '2:00', '2:10', '2:20', '2:30']
const BIKE_KMH = [25, 28, 30, 32, 35, 38, 40, 45]

export const TRI_DISTANCES: [string, string, string, string][] = [
  ['sprint', '0.75', '20', '5'],
  ['olympic', '1.5', '40', '10'],
  ['70.3', '1.9', '90', '21.1'],
  ['ironman', '3.8', '180', '42.2'],
]

export const CONVERSIONS: [string, string][] = [
  ['pace', '/100m × 16.09 → /mi'],
  ['dist', 'km × 0.621 → mi'],
]

export const GEAR: [string, string[]][] = [
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
      'Radar: Garmin Varia RTL515',
      'Scale: Garmin Index S2',
      'Shoes: SPECIALIZED TORCH 2.0',
      'Socks: DANISH ENDURANCE Aero Socks',
    ],
  ],
  [
    'running',
    [
      'Shoes: HOKA Clifton 10',
      'Shoes: Saucony Endorphin Elite 3',
      'Hat: Ciele Athletic Gocap',
      'Socks: Saucony Inferno Cushion Mid 3-Pack Sock',
      'Pants: Salomon SHAKEOUT CORE 5',
      'Headphones: SHOKZ OpenRun Pro 2-Bone Conduction Headphones',
      'Utilities: Zone3 Ultimate Race Number Belt',
    ],
  ],
  [
    'swim',
    [
      'Suit: 2XU Trisuit',
      'Goggles: Decathlon Anti-fog Swimming Goggles',
      'Goggles: Speedo Unisex Adult Swim Goggles Hydrospex Classic',
      'Cap: Speedo Unisex Adult Swim Cap Silicone',
      "Pants: Speedo Speedo Men's Swimsuit Endurance+",
      'Utilities: Speedo Ergo Ear Plug',
    ],
  ],
  ['wearables', ['Oura Ring 4', 'Apple Watch Ultra 3']],
  [
    'fuel',
    [
      'mandarins',
      'apple',
      'banana',
      'Precision Fuel & Hydration Chews',
      'Maurten Gels & Carb Drinks',
      'Skratch Labs Super High-Carb Hydration Powder',
    ],
  ],
]

const paceKm = (mi: string): string => {
  const [m = '0', s = '0'] = mi.split(':')
  const secKm = Math.round((Number(m) * 60 + Number(s)) * KM_TO_MI)
  return `${Math.floor(secKm / 60)}:${(secKm % 60).toString().padStart(2, '0')}`
}
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
const kmhToMph = (kmh: number): string => (kmh * KM_TO_MI).toFixed(1)
const clockFromSec = (sec: number): string => {
  const s = Math.round(sec)
  return `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`
}
const bikePaceKm = (kmh: number): string => clockFromSec(3600 / kmh)
const bikePaceMi = (kmh: number): string => clockFromSec(3600 / (kmh * KM_TO_MI))

export const TriathlonSubnav = ({ active, root }: { active: TriView; root: string }) => (
  <nav class="tri-subnav" aria-label="triathlon sections">
    <a class="tri-subnav-home" href={`${root}/triathlon`}>
      ← triathlon
    </a>
    <span class="tri-subnav-links">
      {NAV.map(([slug, label]) => (
        <a
          class="tri-subnav-link"
          href={`${root}/triathlon/${slug}`}
          aria-current={slug === active ? 'page' : undefined}
        >
          {label}
        </a>
      ))}
    </span>
  </nav>
)

export const AnalyticsPanel = () => (
  <>
    <div class="tri-analytics-scrim" aria-hidden="true" />
    <aside class="tri-analytics" aria-hidden="true" role="dialog" aria-label="triathlon analytics">
      <div class="tri-ana-bar">
        <span class="tri-ana-title">analytics</span>
        <input
          class="tri-ana-search"
          type="search"
          placeholder="search (filter:bike|run|swim|walk, sort:distance|cadence|pace)"
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
        <div class="tri-ana-block" data-chart="recovery" />
        <div class="tri-ana-block" data-chart="sleep" />
        <div class="tri-ana-block" data-chart="vo2max" />
        <div class="tri-ana-block" data-chart="abilities" />
        <div class="tri-ana-block" data-chart="cardio" />
        <div class="tri-ana-block" data-chart="pmc" />
        <div class="tri-ana-block" data-chart="ctl-sport" />
        <div class="tri-ana-block" data-chart="weekly" />
        <div class="tri-ana-block" data-chart="effort" />
        <div class="tri-ana-block" data-chart="readiness" />
        <div class="tri-ana-block" data-chart="trend" />
        <div class="tri-ana-block" data-chart="actions" />
      </div>
    </aside>
  </>
)

export const MapPanel = () => (
  <>
    <div class="tri-map-scrim" aria-hidden="true" />
    <aside class="tri-map" aria-hidden="true" role="dialog" aria-label="triathlon route maps">
      <div class="tri-ana-bar tri-map-bar">
        <span class="tri-ana-title tri-map-title">map</span>
        <div class="tri-map-search-wrap">
          <input
            class="tri-ana-search tri-map-search"
            type="search"
            placeholder="search routes (filter:bike|run)"
            aria-label="search routes"
            autocomplete="off"
          />
          <div class="tri-ana-results tri-map-results" aria-hidden="true" />
        </div>
        <button class="tri-ana-close tri-map-close" type="button" aria-label="Close">
          ×
        </button>
      </div>
      <div class="tri-ana-body tri-map-body">
        <div class="tri-map-pane">
          <div class="tri-map-canvas" />
          <div class="tri-map-overlay">
            <div class="tri-map-modes" role="group" aria-label="map overlay metric">
              <button class="tri-map-mode" type="button" data-mode="heat" aria-pressed="true">
                heat
              </button>
              <button class="tri-map-mode" type="button" data-mode="w" aria-pressed="false">
                power
              </button>
              <button class="tri-map-mode" type="button" data-mode="hr" aria-pressed="false">
                hr
              </button>
              <button class="tri-map-mode" type="button" data-mode="cad" aria-pressed="false">
                cadence
              </button>
              <button class="tri-map-mode" type="button" data-mode="spd" aria-pressed="false">
                speed
              </button>
            </div>
            <div class="tri-map-legend tri-map-overlay-legend">
              <span class="tri-map-legend-lo" />
              <span class="tri-map-legend-bar" />
              <span class="tri-map-legend-hi" />
            </div>
          </div>
          <div class="tri-map-tip" aria-hidden="true" />
        </div>
        <div class="tri-ana-detail tri-map-detail tri-map-sidebar" aria-hidden="true" />
      </div>
    </aside>
  </>
)

export const TrainingPanel = () => (
  <>
    <div class="tri-training-scrim" aria-hidden="true" />
    <aside
      class="tri-training"
      aria-hidden="true"
      role="dialog"
      aria-label="triathlon training plan"
    >
      <div class="tri-ana-bar tri-training-bar">
        <span class="tri-ana-title tri-training-title">training</span>
        <div class="tri-training-search-wrap">
          <input
            class="tri-ana-search tri-training-search"
            type="search"
            placeholder="search plans (meta, distance, target)"
            aria-label="search training plans"
            autocomplete="off"
          />
          <div class="tri-ana-results tri-training-results" aria-hidden="true" />
        </div>
        <button class="tri-ana-close tri-training-close" type="button" aria-label="Close">
          ×
        </button>
      </div>
      <div class="tri-ana-body tri-training-body">
        <div class="tri-training-list">
          <div class="tri-training-plans" aria-label="training plans" />
          <div class="tri-training-tree" aria-label="plan sections" />
        </div>
        <div class="tri-ana-detail tri-training-doc" aria-hidden="true" />
      </div>
    </aside>
  </>
)

export const GearPanel = () => (
  <div class="tri-gear-wrap">
    <button class="tri-gear-btn" type="button">
      gear
    </button>
    <div class="tri-gear" aria-hidden="true">
      <div class="tri-gear-scroll">
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
  </div>
)

export const PacePanel = () => (
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
)

export const FuelLink = () => (
  <a
    class="tri-fuel-btn internal"
    href="/thoughts/pdfs/triathlon.pdf"
    target="_blank"
    rel="noopener noreferrer"
  >
    fuel plan
  </a>
)

export const FuelEmbed = () => (
  <div
    class="pdf-embed tri-fuel-embed"
    data-pdf-src="/thoughts/pdfs/triathlon.pdf"
    data-pdf-title="fuel plan"
    data-pdf-fit="page"
    tabindex={0}
  >
    <span class="pdf-embed-loading">Loading PDF</span>
  </div>
)

export const CalcPanel = () => (
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
    <div class="tri-calc-cell">
      <div class="tri-calc-presets">
        {TRI_DISTANCES.map(([label, s, b, r]) => (
          <button class="tri-calc-preset" type="button" data-swim={s} data-bike={b} data-run={r}>
            {label}
          </button>
        ))}
      </div>
      <div class="tri-calc-source" hidden>
        <div class="tri-calc-srcs" role="tablist" aria-label="pace source">
          <button
            class="tri-calc-src tri-calc-src--on"
            type="button"
            role="tab"
            aria-selected="true"
            data-src="avg"
          >
            average
          </button>
          <button
            class="tri-calc-src"
            type="button"
            role="tab"
            aria-selected="false"
            data-src="pred"
          >
            projected
          </button>
        </div>
      </div>
      <div class="tri-calc-box">
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
      </div>
    </div>
  </aside>
)

export const ToolsPanel = () => (
  <div class="tri-tools">
    <section class="tri-tools-sec">
      <h2 class="tri-tools-h">gear</h2>
      <GearPanel />
    </section>
    <section class="tri-tools-sec">
      <h2 class="tri-tools-h">pace</h2>
      <PacePanel />
    </section>
    <section class="tri-tools-sec">
      <h2 class="tri-tools-h">fuel</h2>
      <FuelEmbed />
    </section>
    <section class="tri-tools-sec">
      <h2 class="tri-tools-h">calculator</h2>
      <CalcPanel />
    </section>
  </div>
)
