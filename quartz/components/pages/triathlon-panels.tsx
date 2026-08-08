import type { ComponentChildren } from 'preact'
import type { TriathlonTreeYear } from '../../util/triathlon-date-route'
import { ROUTE_SPORTS, SPORT_ICON } from '../../plugins/stores/strava'
import { InlineMath } from '../../util/math-text'
import { TRI_RACE_DISTANCES } from '../../util/triathlon-calculator'
import { KM_TO_MI, LAYERS_ICON } from '../../util/triathlon-card'
import {
  CERAMICSPEED_CROSS_CHAIN_RESEARCH,
  CERAMICSPEED_TEST_CADENCE_RPM,
  CERAMICSPEED_TEST_CHAINSTAY_MM,
  CERAMICSPEED_TEST_OUTPUT_WATTS,
  DEFAULT_GEAR_CASSETTE,
  GEAR_CASSETTE_PRESET_GROUPS,
  formatGearEfficiencyDeltaPercent,
  gearRatioMatrix,
  type GearCassettePreset,
} from '../../util/triathlon-gear-ratio'

export const DISPATCH_ICON =
  'M189 375Q189 338 207 306.5Q225 275 256.5 257Q288 239 325 239H675Q712 239 743.5 257Q775 275 793 306.5Q811 338 811 375V775Q811 812 793 843.5Q775 875 743.5 893Q712 911 675 911H325Q288 911 256.5 893Q225 875 207 843.5Q189 812 189 775ZM261 375V775Q261 802 279.5 820.5Q298 839 325 839H675Q702 839 720.5 820.5Q739 802 739 775V375Q739 348 720.5 329.5Q702 311 675 311H325Q298 311 279.5 329.5Q261 348 261 375ZM411 275H339V100Q339 85 349.5 74.5Q360 64 375 64Q390 64 400.5 74.5Q411 85 411 100ZM661 275H589V150Q589 135 599.5 124.5Q610 114 625 114Q640 114 650.5 124.5Q661 135 661 150ZM375 539H625A36 36 0 0 1 625 611H375A36 36 0 0 1 375 539Z'

const NAV = [
  ['tools', 'tools'],
  ['calc', 'calculator'],
  ['analytics', 'analytics'],
  ['maps', 'maps'],
  ['training', 'training'],
  ['feed', 'feed'],
  ['on', 'on'],
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
      'Tube: Pirelli P Zero TPU',
      'Tires: Pirelli P Zero Race TLR SL-R 700x28c',
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
      'Utilities: Salomon HIGH PULSE',
      'Utilities: Salomon SOFT FLASK 150ml/5oz 28',
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
      'cherry tart juice',
      'Nuun Hydration',
      'Precision Fuel & Hydration Chews, Gels, Carb Drinks',
      'Xact Energy Bars & Gels',
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

export const TriathlonSubnav = ({ active, root }: { active?: TriView; root: string }) => (
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
          data-i18n={label}
        >
          {label}
        </a>
      ))}
    </span>
  </nav>
)

export const FeedPanel = ({ title = 'feed' }: { title?: string }) => (
  <section class="tri-feed" aria-label="activity feed" tabindex={-1} data-keyboard-scroll-scope>
    <div class="tri-ana-bar tri-feed-bar">
      <span class="tri-ana-title" data-i18n={title === 'feed' ? 'feed' : undefined}>
        {title}
      </span>
      <div class="tri-feed-search-wrap">
        <input
          class="tri-ana-search tri-feed-search"
          type="search"
          placeholder="search (filter:bike|run|swim|walk, sort:distance|cadence|pace)"
          aria-label="search activities"
          aria-controls="tri-feed-results"
          aria-expanded="false"
          autocomplete="off"
        />
        <div id="tri-feed-results" class="tri-ana-results tri-feed-results" aria-hidden="true" />
      </div>
      <span class="tri-feed-count" aria-live="polite" />
    </div>
    <div class="tri-feed-list" role="list" aria-busy="true" data-keyboard-scroll />
  </section>
)

const treeKm = (km: number): string => `${km >= 100 ? Math.round(km) : Number(km.toFixed(1))} km`

const treeDur = (timeS: number): string => {
  const minutes = Math.round(timeS / 60)
  const h = Math.floor(minutes / 60)
  return h > 0 ? `${h}h ${(minutes % 60).toString().padStart(2, '0')}m` : `${minutes}m`
}

const TreeSum = ({ count, km, timeS }: { count: number; km: number; timeS: number }) => (
  <span class="tri-tree-sum">
    <span class="tri-tree-c tri-tree-c--n">{count}</span>
    <span class="tri-tree-c tri-tree-c--km">{treeKm(km)}</span>
    <span class="tri-tree-c tri-tree-c--t">{treeDur(timeS)}</span>
  </span>
)

export const OnTreePanel = ({
  tree,
  title = 'on',
  root,
}: {
  tree: TriathlonTreeYear[]
  title?: string
  root: string
}) => (
  <section
    class="tri-feed tri-tree"
    aria-label="training log by date"
    tabindex={-1}
    data-keyboard-scroll-scope
  >
    <div class="tri-ana-bar tri-feed-bar">
      <span class="tri-ana-title" data-i18n={title === 'on' ? 'on' : undefined}>
        {title}
      </span>
      <span class="tri-feed-count">{tree.reduce((total, year) => total + year.count, 0)}</span>
    </div>
    <div class="tri-tree-list" data-keyboard-scroll>
      {tree.map(year => (
        <section class="tri-tree-year">
          <div class="tri-tree-row tri-tree-row--year">
            <a href={`${root}/${year.slug}`}>{year.year}</a>
            <TreeSum count={year.count} km={year.km} timeS={year.timeS} />
          </div>
          {year.months.map(month => (
            <section class="tri-tree-month">
              <div class="tri-tree-row tri-tree-row--month">
                <a href={`${root}/${month.slug}`}>{`${year.year} / ${month.month}`}</a>
                <TreeSum count={month.count} km={month.km} timeS={month.timeS} />
              </div>
              <div class="tri-tree-days" role="list">
                {month.days.map(day => (
                  <a class="tri-tree-day" role="listitem" href={`${root}/${day.slug}`}>
                    <span class="tri-tree-day-d">{day.day}</span>
                    <span class="tri-tree-day-sports">{day.sports.join(' · ')}</span>
                    <TreeSum count={day.count} km={day.km} timeS={day.timeS} />
                  </a>
                ))}
              </div>
            </section>
          ))}
        </section>
      ))}
      {tree.length === 0 && <div class="tri-ana-empty">no activities</div>}
    </div>
  </section>
)

type TriPanelKind = 'analytics' | 'map' | 'training'

interface TriPanelShellProps {
  kind: TriPanelKind
  page?: boolean
  label: string
  title: string
  barClass?: string
  titleClass?: string
  bodyClass?: string
  search: ComponentChildren
  children: ComponentChildren
}

const TriPanelShell = ({
  kind,
  page,
  label,
  title,
  barClass,
  titleClass,
  bodyClass,
  search,
  children,
}: TriPanelShellProps) => {
  const rootClass = `tri-${kind}`
  return (
    <>
      <div class={`${rootClass}-scrim`} aria-hidden="true" />
      <aside
        class={`${rootClass}${page ? ` ${rootClass}--page` : ''}`}
        aria-hidden={page ? 'false' : 'true'}
        role="dialog"
        aria-label={label}
        tabindex={-1}
        data-keyboard-scroll-scope
      >
        <div class={`tri-ana-bar${barClass ? ` ${barClass}` : ''}`}>
          <span class={`tri-ana-title${titleClass ? ` ${titleClass}` : ''}`} data-i18n={title}>
            {title}
          </span>
          {search}
          <button
            class={`tri-ana-close${kind === 'analytics' ? '' : ` tri-${kind}-close`}`}
            type="button"
            aria-label="Close"
          >
            ×
          </button>
        </div>
        <div class={`tri-ana-body${bodyClass ? ` ${bodyClass}` : ''}`} data-keyboard-scroll>
          {children}
        </div>
      </aside>
    </>
  )
}

export const AnalyticsPanel = ({ page }: { page?: boolean }) => (
  <TriPanelShell
    kind="analytics"
    page={page}
    label="triathlon analytics"
    title="analytics"
    search={
      <div class="tri-analytics-search-wrap">
        <input
          class="tri-ana-search"
          type="search"
          placeholder="search (filter:bike|run|swim|walk, sort:distance|cadence|pace)"
          aria-label="search analytics"
          autocomplete="off"
        />
        <button
          class="tri-ana-compare-toggle"
          type="button"
          aria-pressed="false"
          aria-label="compare activities"
          aria-controls="tri-analytics-results"
          data-i18n-aria-label="compare activities"
        >
          <svg class="tri-ana-compare-icon" viewBox="0 0 1000 1000" aria-hidden="true">
            <path d={DISPATCH_ICON} />
          </svg>
        </button>
      </div>
    }
  >
    <div id="tri-analytics-results" class="tri-ana-results" aria-hidden="true" />
    <div id="tri-analytics-detail" class="tri-ana-detail" aria-hidden="true" />
    <div class="tri-ana-block" data-chart="body" />
    <div class="tri-ana-block" data-chart="dexa" />
    <div class="tri-ana-block" data-chart="gauge" />
    <div class="tri-ana-block" data-chart="recovery" />
    <div class="tri-ana-block" data-chart="sleep" />
    <div class="tri-ana-block" data-chart="vo2max" />
    <div class="tri-ana-block" data-chart="lactate" />
    <div class="tri-ana-block" data-chart="power" />
    <div class="tri-ana-block" data-chart="abilities" />
    <div class="tri-ana-block" data-chart="cardio" />
    <div class="tri-ana-block" data-chart="pmc" />
    <div class="tri-ana-block" data-chart="weekly" />
    <div class="tri-ana-block" data-chart="effort" />
    <div class="tri-ana-block" data-chart="heat" />
    <div class="tri-ana-block" data-chart="readiness" />
    <div class="tri-ana-block" data-chart="trend" />
    <div class="tri-ana-block" data-chart="actions" />
    <div class="tri-ana-block" data-chart="ftp" />
  </TriPanelShell>
)

export const MapPanel = ({ page }: { page?: boolean }) => (
  <TriPanelShell
    kind="map"
    page={page}
    label="triathlon route maps"
    title="map"
    barClass="tri-map-bar"
    titleClass="tri-map-title"
    bodyClass="tri-map-body"
    search={
      <div class="tri-map-search-wrap">
        <input
          class="tri-ana-search tri-map-search"
          type="search"
          placeholder="search (filter:bike|run|swim|walk, sort:distance|pace|cadence)"
          aria-label="search routes"
          autocomplete="off"
        />
        <div class="tri-ana-results tri-map-results" aria-hidden="true" />
      </div>
    }
  >
    <div class="tri-map-pane">
      <div class="tri-map-canvas" />
      <div class="tri-map-overlay">
        <div class="tri-map-modes" role="group" aria-label="map overlay metric">
          <button
            class="tri-map-mode"
            type="button"
            data-mode="heat"
            aria-pressed="true"
            data-i18n="heat"
          >
            heat
          </button>
          <button
            class="tri-map-mode"
            type="button"
            data-mode="w"
            aria-pressed="false"
            data-i18n="power"
          >
            power
          </button>
          <button
            class="tri-map-mode"
            type="button"
            data-mode="hr"
            aria-pressed="false"
            data-i18n="hr"
          >
            hr
          </button>
          <button
            class="tri-map-mode"
            type="button"
            data-mode="cad"
            aria-pressed="false"
            data-i18n="cadence"
          >
            cadence
          </button>
          <button
            class="tri-map-mode"
            type="button"
            data-mode="spd"
            aria-pressed="false"
            data-i18n="speed"
          >
            speed
          </button>
        </div>
        <div class="tri-map-legend tri-map-overlay-legend">
          <span class="tri-map-legend-bar" />
          <span class="tri-map-legend-ends">
            <span class="tri-map-legend-lo" />
            <span class="tri-map-legend-hi" />
          </span>
        </div>
      </div>
      <div class="tri-map-side" role="group" aria-label="map controls">
        <button
          class="tri-map-side-fold"
          type="button"
          aria-expanded="true"
          aria-label="Collapse map controls"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M6 9l6 6 6-6" />
          </svg>
        </button>
        <div class="tri-map-side-body">
          {ROUTE_SPORTS.map(sport => (
            <button
              class="tri-map-sport"
              type="button"
              data-sport={sport}
              aria-pressed="true"
              aria-label={sport}
              title={sport}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                {SPORT_ICON[sport].map(d => (
                  <path d={d} />
                ))}
              </svg>
            </button>
          ))}
          <span class="tri-map-side-rule" />
          <button
            class="tri-map-style"
            type="button"
            aria-pressed="false"
            aria-label="satellite"
            title="satellite"
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              {LAYERS_ICON.map(d => (
                <path d={d} />
              ))}
            </svg>
          </button>
        </div>
      </div>
      <div class="tri-map-selection" aria-hidden="true" />
      <div class="tri-map-tip" aria-hidden="true" />
    </div>
    <div
      class="tri-ana-detail tri-map-detail tri-map-sidebar"
      aria-hidden="true"
      data-keyboard-scroll
    />
  </TriPanelShell>
)

export const TrainingPanel = ({ page }: { page?: boolean }) => (
  <TriPanelShell
    kind="training"
    page={page}
    label="triathlon training plan"
    title="training"
    barClass="tri-training-bar"
    titleClass="tri-training-title"
    bodyClass="tri-training-body"
    search={
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
    }
  >
    <div class="tri-training-list">
      <div class="tri-training-plans" aria-label="training plans" />
      <div class="tri-training-tree" aria-label="plan sections" />
    </div>
    <div class="tri-ana-detail tri-training-doc" aria-hidden="true" data-keyboard-scroll />
  </TriPanelShell>
)

const DEFAULT_GEAR_CHAINRINGS: readonly number[] = [52, 36]

const gearEfficiencyLevel = (crossChainLossWatts: number): string =>
  `${(8 + Math.min(crossChainLossWatts / 2.5, 1) * 32).toFixed(1)}%`

const GearRatioTable = ({
  chainrings,
  cassette,
}: {
  chainrings: readonly number[]
  cassette: GearCassettePreset
}) => {
  const matrix = gearRatioMatrix(chainrings, cassette.cogs)
  if (!matrix) return null
  return (
    <table class="tri-ratio-table" aria-label="gear ratio chart" data-i18n-aria-label="gear ratios">
      <caption>
        <InlineMath className="tri-math" tex="\Delta\eta" /> est. vs. ideal · CeramicSpeed{' '}
        <span class="tri-ratio-source-links" aria-label="CeramicSpeed research sources">
          {CERAMICSPEED_CROSS_CHAIN_RESEARCH.sources.map(source => (
            <a
              class="tri-ratio-source-link"
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              title={source.title}
              aria-label={`source ${source.id}: ${source.title}, opens in new tab`}
            >
              [{source.id}]
            </a>
          ))}
        </span>{' '}
        · {CERAMICSPEED_TEST_OUTPUT_WATTS} W · {CERAMICSPEED_TEST_CADENCE_RPM} rpm ·{' '}
        {CERAMICSPEED_TEST_CHAINSTAY_MM} mm chainstay
      </caption>
      <thead>
        <tr>
          <th scope="col" aria-label="chainring and cassette teeth">
            T
          </th>
          {cassette.cogs.map(cog => (
            <th scope="col">{cog}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {matrix.rows.map((row, rowIndex) => (
          <>
            <tr class={`tri-ratio-row tri-ratio-row--${rowIndex + 1}`}>
              <th scope="row" aria-label={`${row.chainring} tooth chainring`}>
                {row.chainring}
              </th>
              {row.cells.map(cell => (
                <td
                  data-ratio-chainring={row.chainring}
                  data-ratio-cog={cell.cog}
                  data-ratio-value={cell.ratio.toFixed(2)}
                  title={`${row.chainring}T ÷ ${cell.cog}T = ${cell.ratio.toFixed(2)}`}
                  style={`--tri-ratio-level:${(18 + cell.level * 52).toFixed(1)}%`}
                >
                  {cell.ratio.toFixed(2)}
                </td>
              ))}
            </tr>
            <tr class={`tri-ratio-efficiency-row tri-ratio-row--${rowIndex + 1}`}>
              <th
                scope="row"
                aria-label={`estimated CeramicSpeed drivetrain efficiency difference from ideal for ${row.chainring} tooth chainring`}
              >
                <InlineMath className="tri-math" tex="\Delta\eta" />
              </th>
              {row.cells.map((cell, cellIndex) => {
                const efficiency = cell.drivetrainEfficiency.toFixed(3)
                const efficiencyDelta = formatGearEfficiencyDeltaPercent(
                  cell.drivetrainEfficiency,
                  3,
                )
                const visibleEfficiencyDelta = formatGearEfficiencyDeltaPercent(
                  cell.drivetrainEfficiency,
                  2,
                )
                const compactEfficiencyDelta = formatGearEfficiencyDeltaPercent(
                  cell.drivetrainEfficiency,
                  1,
                )
                return (
                  <td
                    data-efficiency-chainring={row.chainring}
                    data-efficiency-cog={cell.cog}
                    data-efficiency-value={efficiency}
                    data-efficiency-delta={efficiencyDelta}
                    data-loss-watts={cell.drivetrainLossWatts.toFixed(2)}
                    data-cross-chain-loss-watts={cell.crossChainLossWatts.toFixed(2)}
                    aria-label={`${visibleEfficiencyDelta}% estimated drivetrain efficiency difference from ideal; ${cell.drivetrainEfficiency.toFixed(2)}% estimated drivetrain efficiency; ${cell.drivetrainLossWatts.toFixed(2)} watts drivetrain loss; ${cell.crossChainLossWatts.toFixed(2)} watts cross-chain loss`}
                    tabindex={rowIndex === 0 && cellIndex === 0 ? 0 : -1}
                    style={`--tri-efficiency-level:${gearEfficiencyLevel(cell.crossChainLossWatts)}`}
                  >
                    <span class="tri-ratio-efficiency-value tri-ratio-efficiency-value--full">
                      <InlineMath className="tri-math" tex={`${visibleEfficiencyDelta}\\%`} />
                    </span>
                    <span class="tri-ratio-efficiency-value tri-ratio-efficiency-value--compact">
                      <InlineMath className="tri-math" tex={`${compactEfficiencyDelta}\\%`} />
                    </span>
                  </td>
                )
              })}
            </tr>
          </>
        ))}
      </tbody>
    </table>
  )
}

const GearRatioCalculator = () => {
  const matrix = gearRatioMatrix(DEFAULT_GEAR_CHAINRINGS, DEFAULT_GEAR_CASSETTE.cogs)
  return (
    <section
      class="tri-ratio"
      aria-label="gear ratio calculator"
      data-i18n-aria-label="gear ratios"
    >
      <div class="tri-ratio-head">
        <span data-i18n="gear ratios">gear ratios</span>
        <output class="tri-ratio-range" aria-live="polite">
          {matrix ? `${matrix.minimum.toFixed(2)}–${matrix.maximum.toFixed(2)}` : ''}
        </output>
      </div>
      <div class="tri-ratio-controls">
        <fieldset class="tri-ratio-rings">
          <legend data-i18n="chainrings">chainrings</legend>
          <div class="tri-ratio-ring-inputs">
            {DEFAULT_GEAR_CHAINRINGS.map((chainring, index) => (
              <label class="tri-ratio-ring" data-ratio-ring={index + 1}>
                <input
                  class="tri-ratio-ring-input"
                  type="number"
                  value={chainring}
                  min="24"
                  max="64"
                  step="1"
                  inputMode="numeric"
                  aria-label={`chainring ${index + 1} teeth`}
                />
                <span aria-hidden="true">T</span>
              </label>
            ))}
          </div>
        </fieldset>
        <div class="tri-ratio-cassette">
          <span id="tri-ratio-cassette-label" data-i18n="cassette">
            cassette
          </span>
          <div class="tri-ratio-cassette-picker">
            <button
              class="tri-ratio-cassette-trigger"
              type="button"
              aria-labelledby="tri-ratio-cassette-label tri-ratio-cassette-value"
              aria-haspopup="listbox"
              aria-expanded="false"
              aria-controls="tri-ratio-cassette-menu"
            >
              <span id="tri-ratio-cassette-value" class="tri-ratio-cassette-value">
                {DEFAULT_GEAR_CASSETTE.label}
              </span>
              <svg
                class="tri-ratio-cassette-chevron"
                viewBox="0 0 16 16"
                fill="none"
                aria-hidden="true"
                focusable="false"
              >
                <path
                  d="m4 6 4 4 4-4"
                  stroke="currentColor"
                  stroke-width="1.4"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                />
              </svg>
            </button>
            <div
              id="tri-ratio-cassette-menu"
              class="tri-ratio-cassette-menu"
              role="listbox"
              aria-labelledby="tri-ratio-cassette-label"
              hidden
            >
              {GEAR_CASSETTE_PRESET_GROUPS.map(group => (
                <div class="tri-ratio-cassette-group" role="group" aria-label={group.label}>
                  <span class="tri-ratio-cassette-group-label" aria-hidden="true">
                    {group.label}
                  </span>
                  {group.presets.map(preset => (
                    <button
                      class="tri-ratio-cassette-option"
                      type="button"
                      role="option"
                      aria-selected={preset.id === DEFAULT_GEAR_CASSETTE.id}
                      data-cassette-id={preset.id}
                    >
                      <span class="tri-ratio-cassette-check" aria-hidden="true">
                        ✓
                      </span>
                      <span class="tri-ratio-cassette-option-value">{preset.label}</span>
                    </button>
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>
        <div class="tri-ratio-layout" role="group" aria-label="chainring count">
          <button
            class="tri-ratio-layout-btn tri-ratio-layout-btn--on"
            type="button"
            data-ratio-layout="2"
            aria-pressed="true"
          >
            2×
          </button>
          <button
            class="tri-ratio-layout-btn"
            type="button"
            data-ratio-layout="1"
            aria-pressed="false"
          >
            1×
          </button>
        </div>
      </div>
      <div class="tri-ratio-chart-scroll">
        <div class="tri-ratio-chart">
          <GearRatioTable chainrings={DEFAULT_GEAR_CHAINRINGS} cassette={DEFAULT_GEAR_CASSETTE} />
        </div>
      </div>
    </section>
  )
}

export const GearPanel = ({ page }: { page?: boolean }) => (
  <div class="tri-gear-wrap">
    {!page && (
      <button
        class="tri-gear-btn"
        type="button"
        data-i18n="gear"
        aria-expanded="false"
        aria-controls="tri-gear-panel"
      >
        gear
      </button>
    )}
    <div
      id={page ? undefined : 'tri-gear-panel'}
      class="tri-gear"
      aria-hidden={page ? 'false' : 'true'}
    >
      <div class="tri-gear-scroll">
        <GearRatioCalculator />
        {GEAR.map(([label, items]) => (
          <div class="tri-gear-row">
            <span class="tri-gear-k" data-i18n={label}>
              {label}
            </span>
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

export const PacePanel = ({ page }: { page?: boolean }) => (
  <div class="tri-pace-wrap">
    {!page && (
      <button
        class="tri-pace-btn"
        type="button"
        data-i18n="pace"
        aria-expanded="false"
        aria-controls="tri-pace-panel"
      >
        pace
      </button>
    )}
    <div
      id={page ? undefined : 'tri-pace-panel'}
      class="tri-pace"
      aria-hidden={page ? 'false' : 'true'}
    >
      <span class="tri-pace-sec" data-i18n="run">
        run
      </span>
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
      <span class="tri-pace-sec" data-i18n="swim">
        swim
      </span>
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
      <span class="tri-pace-sec" data-i18n="bike">
        bike
      </span>
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

export const CalcPanel = ({
  page,
  defaultDistance,
}: {
  page?: boolean
  defaultDistance?: unknown
}) => {
  const [defaultLabel, defaultSwim, defaultBike, defaultRun] =
    TRI_RACE_DISTANCES.find(([label]) => label === defaultDistance) ?? TRI_RACE_DISTANCES[1]

  return (
    <aside
      class={`tri-calc${page ? ' tri-calc--page' : ''}`}
      aria-hidden={page ? 'false' : 'true'}
      role="dialog"
      aria-label="triathlon calculator"
      data-swim={defaultSwim}
      data-bike={defaultBike}
      data-run={defaultRun}
      tabindex={-1}
      data-keyboard-scroll-scope
      data-keyboard-scroll
    >
      <div class="tri-calc-bar">
        <span class="tri-calc-title">triathlon calculator</span>
        <button
          class="tri-calc-copy"
          type="button"
          aria-label="Copy embed link"
          title="Copy embed link"
        >
          <svg
            class="copy-icon"
            width="16"
            height="16"
            viewBox="-4 -4 24 24"
            fill="currentColor"
            aria-hidden="true"
          >
            <use href="#github-copy" />
          </svg>
          <svg
            class="check-icon"
            width="16"
            height="16"
            viewBox="-4 -4 24 24"
            fill="currentColor"
            aria-hidden="true"
          >
            <use href="#github-check" />
          </svg>
        </button>
        <button class="tri-calc-close" type="button" aria-label="Close">
          ×
        </button>
      </div>
      <div class="tri-calc-cell">
        <div class="tri-calc-presets">
          {TRI_RACE_DISTANCES.map(([label, s, b, r]) => (
            <button
              class={`tri-calc-preset${label === defaultLabel ? ' tri-calc-preset--on' : ''}`}
              type="button"
              data-swim={s}
              data-bike={b}
              data-run={r}
            >
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
            <button
              class="tri-calc-src tri-calc-src--proj"
              type="button"
              role="tab"
              aria-selected="false"
              data-src="proj"
              data-i18n="projection"
              hidden
            >
              projection
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
                    aria-label="swim pace"
                    inputMode="numeric"
                  />
                </td>
                <td class="tri-calc-u">/100m</td>
                <td class="tri-calc-r" data-leg="swim">
                  <input
                    class="tri-calc-in tri-calc-legtime"
                    data-legtime="swim"
                    type="text"
                    value=""
                    placeholder="—"
                    aria-label="swim time"
                    inputMode="numeric"
                  />
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
                    aria-label="T1 time"
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
                    aria-label="bike speed"
                    inputMode="decimal"
                  />
                </td>
                <td class="tri-calc-u" data-u="bike">
                  mph
                </td>
                <td class="tri-calc-r" data-leg="bike">
                  <input
                    class="tri-calc-in tri-calc-legtime"
                    data-legtime="bike"
                    type="text"
                    value=""
                    placeholder="—"
                    aria-label="bike time"
                    inputMode="numeric"
                  />
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
                    aria-label="T2 time"
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
                    aria-label="run pace"
                    inputMode="numeric"
                  />
                </td>
                <td class="tri-calc-u" data-u="run">
                  /mi
                </td>
                <td class="tri-calc-r" data-leg="run">
                  <input
                    class="tri-calc-in tri-calc-legtime"
                    data-legtime="run"
                    type="text"
                    value=""
                    placeholder="—"
                    aria-label="run time"
                    inputMode="numeric"
                  />
                </td>
              </tr>
              <tr class="tri-calc-total">
                <th>finish</th>
                <td />
                <td />
                <td class="tri-calc-r tri-calc-target-cell" data-leg="total">
                  <input
                    class="tri-calc-in tri-calc-target"
                    data-k="target"
                    type="text"
                    value=""
                    placeholder="—"
                    aria-label="target finish time"
                    inputMode="numeric"
                  />
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
      <div class="tri-calc-proj" hidden>
        <div class="tri-calc-proj-zones" role="tablist" aria-label="heart rate zone" />
        <div class="tri-calc-proj-out" aria-live="polite" />
      </div>
    </aside>
  )
}

export const ToolsPanel = () => (
  <div class="tri-tools" data-keyboard-scroll>
    <section class="tri-tools-sec">
      <h2 class="tri-tools-h" data-i18n="gear">
        gear
      </h2>
      <GearPanel page />
    </section>
    <section class="tri-tools-sec">
      <h2 class="tri-tools-h" data-i18n="pace">
        pace
      </h2>
      <PacePanel page />
    </section>
  </div>
)
