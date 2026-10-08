import { parseClockSeconds } from '../../../util/duration'
import { formatDurationClock } from '../../../util/triathlon-calculator'
import { clock, KM_TO_MI } from '../../../util/triathlon-card'
import { ShortcutHint } from '../shell/ShortcutHint'

const RUN_PACES = [
  '3:30',
  '4:00',
  '4:30',
  '5:00',
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
  '11:30',
  '12:00',
  '12:30',
]
const SWIM_PACES = ['1:20', '1:30', '1:40', '1:50', '2:00', '2:10', '2:20', '2:30']
const TOUR_DE_FRANCE_2026_AVERAGE_KMH = 3197 / (73 + 56 / 60 + 26 / 3600)

interface PaceRow {
  pace: string
  convertedPace: string
  kmh: number
  reference?: string
}

interface PaceSection {
  sport: 'run' | 'swim' | 'bike'
  units: readonly [string, string]
  distances: readonly { label: string; km: number }[]
  rows: readonly PaceRow[]
}

const SECTIONS: readonly PaceSection[] = [
  {
    sport: 'run',
    units: ['/mi', '/km'],
    distances: [
      { label: '5 km', km: 5 },
      { label: '10 km', km: 10 },
      { label: '21.1 km', km: 21.0975 },
      { label: '42.2 km', km: 42.195 },
    ],
    rows: RUN_PACES.map(pace => {
      const seconds = parseClockSeconds(pace)
      return { pace, convertedPace: clock(seconds * KM_TO_MI), kmh: 3600 / seconds / KM_TO_MI }
    }),
  },
  {
    sport: 'swim',
    units: ['/100m', '/mi'],
    distances: [
      { label: '400 m', km: 0.4 },
      { label: '750 m', km: 0.75 },
      { label: '1,500 m', km: 1.5 },
      { label: '1,900 m', km: 1.9 },
      { label: '3,800 m', km: 3.8 },
    ],
    rows: SWIM_PACES.map(pace => {
      const seconds = parseClockSeconds(pace)
      return { pace, convertedPace: clock((seconds * 10) / KM_TO_MI), kmh: 360 / seconds }
    }),
  },
  {
    sport: 'bike',
    units: ['/mi', '/km'],
    distances: [
      { label: '20 km', km: 20 },
      { label: '40 km', km: 40 },
      { label: '90 km', km: 90 },
      { label: '180 km', km: 180 },
    ],
    rows: [
      25,
      28,
      30,
      32,
      35,
      38,
      40,
      TOUR_DE_FRANCE_2026_AVERAGE_KMH,
      45,
      30 / KM_TO_MI,
      35 / KM_TO_MI,
    ].map(kmh => ({
      pace: clock(3600 / (kmh * KM_TO_MI)),
      convertedPace: clock(3600 / kmh),
      kmh,
      reference: kmh === TOUR_DE_FRANCE_2026_AVERAGE_KMH ? 'TdF' : undefined,
    })),
  },
]

const PaceSpeed = ({ row }: { row: PaceRow }) => (
  <span class="tri-pace-spd">
    <span data-kph={row.kmh.toFixed(1)} data-mph={(row.kmh * KM_TO_MI).toFixed(1)}>
      {(row.kmh * KM_TO_MI).toFixed(1)}
    </span>
    {row.reference && (
      <abbr class="tri-pace-ref" title="2026 Tour de France winner average">
        {row.reference}
      </abbr>
    )}
  </span>
)

const PaceTable = ({ section }: { section: PaceSection }) => (
  <div
    class="tri-pace-table-scroll"
    role="region"
    aria-labelledby={`tri-pace-caption-${section.sport}`}
    tabindex={0}
  >
    <table class="tri-pace-table" data-pace-sport={section.sport}>
      <caption id={`tri-pace-caption-${section.sport}`} data-i18n={section.sport}>
        {section.sport}
      </caption>
      <thead>
        <tr>
          {section.units.map(unit => (
            <th scope="col">{unit}</th>
          ))}
          <th scope="col">
            <button class="tri-pace-unit" type="button">
              mph
            </button>
          </th>
          {section.distances.map(distance => (
            <th scope="col" data-pace-distance-km={distance.km}>
              {distance.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {section.rows.map(row => (
          <tr>
            <th scope="row">{row.pace}</th>
            <td>{row.convertedPace}</td>
            <td>
              <PaceSpeed row={row} />
            </td>
            {section.distances.map(distance => (
              <td class="tri-pace-split">{formatDurationClock((distance.km / row.kmh) * 3600)}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
)

export const PacePanel = ({ page }: { page?: boolean }) => (
  <div class="tri-pace-wrap">
    {!page && (
      <button
        class="tri-pace-btn tri-key-anchor"
        type="button"
        aria-expanded="false"
        aria-controls="tri-pace-panel"
      >
        <span data-i18n="pace">pace</span>
        <ShortcutHint>g p</ShortcutHint>
      </button>
    )}
    <div
      id={page ? undefined : 'tri-pace-panel'}
      class={`tri-pace${page ? ' tri-pace--page' : ''}`}
      aria-hidden={page ? 'false' : 'true'}
    >
      {page ? (
        <>
          <p class="tri-pace-note" data-i18n="pace chart assumption">
            Split times at constant pace. Excludes stops, transitions, terrain, and fatigue.
          </p>
          {SECTIONS.map(section => (
            <PaceTable section={section} />
          ))}
        </>
      ) : (
        SECTIONS.map(section => (
          <>
            <span class="tri-pace-sec" data-i18n={section.sport}>
              {section.sport}
            </span>
            <div class="tri-pace-row tri-pace-head">
              {section.units.map(unit => (
                <span>{unit}</span>
              ))}
              <button class="tri-pace-unit" type="button">
                mph
              </button>
            </div>
            {section.rows.map(row => (
              <div class="tri-pace-row">
                <span class="tri-pace-mi">{row.pace}</span>
                <span class="tri-pace-km">{row.convertedPace}</span>
                <PaceSpeed row={row} />
              </div>
            ))}
          </>
        ))
      )}
    </div>
  </div>
)
