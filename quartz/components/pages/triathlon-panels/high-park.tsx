import type { ComponentChild } from 'preact'
import type { TriathlonRenderData } from '../../triathlon/render-data'
import type { LapSport } from './high-park-laps'
import {
  buildHighParkLoops,
  type HighParkLoops,
  type HighParkSport,
  type HighParkVisit,
} from '../../../util/high-park-loops'

const clock = (seconds: number): string => {
  const rounded = Math.round(seconds)
  return `${Math.floor(rounded / 60)}:${String(rounded % 60).padStart(2, '0')}`
}
const MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
const shortDate = (iso: string): string =>
  `${iso.slice(8, 10)} ${MONTHS[Number(iso.slice(5, 7)) - 1] ?? ''}`
const dayNumber = (iso: string): number =>
  Date.UTC(Number(iso.slice(0, 4)), Number(iso.slice(5, 7)) - 1, Number(iso.slice(8, 10))) /
  86_400_000
const r1 = (value: number): number => Math.round(value * 10) / 10
// Server-rendered in imperial (the triathlon default); the client swaps text on `tri:unit`.
const Unit = ({ metric, imperial }: { metric: string; imperial: string }) => (
  <span class="tri-hp-u" data-metric={metric} data-imperial={imperial}>
    {imperial}
  </span>
)
const MI_M = 1609.344
// Heightfield for the 3D relief, baked by docs/verification/high-park-3d/bake.py.
const TERRAIN_HREF = '/static/high-park/terrain.json'
// The park's relief is ~35 m across a kilometre; the loop itself climbs ~10 m.
const RELIEF_X = 5
// Laps slower than this are stops, not efforts; they sit on the chart's floor.
const SLOW_CAP_S = 270

interface Scale {
  fast: number
  slow: number
}
const lapScale = (laps: readonly number[]): Scale => ({
  fast: Math.floor((Math.min(...laps) - 4) / 10) * 10,
  slow: Math.min(SLOW_CAP_S, Math.ceil(Math.max(...laps) / 10) * 10),
})
// 0 at the slow floor, 1 at the fastest lap.
const pace = (seconds: number, scale: Scale): number =>
  (scale.slow - Math.min(seconds, scale.slow)) / Math.max(1, scale.slow - scale.fast)

const LapStrip = ({ ride, scale }: { ride: HighParkVisit; scale: Scale }) => {
  const step = 4
  const height = 18
  const best = ride.lapsS.indexOf(ride.fastestS)
  const bar = (seconds: number, index: number): string =>
    `M${index * step + 1.5} ${height}V${r1(height - 2 - (height - 2) * pace(seconds, scale))}`
  return (
    <svg
      class="tri-hp-strip"
      width={ride.lapsS.length * step}
      height={height}
      viewBox={`0 0 ${ride.lapsS.length * step} ${height}`}
      aria-hidden="true"
    >
      <path d={ride.lapsS.map((s, i) => (i === best ? '' : bar(s, i))).join('')} />
      <path class="is-best" d={bar(ride.fastestS, best)} />
    </svg>
  )
}

const RideGlyph = ({ ride }: { ride: HighParkVisit }) =>
  ride.glyph ? (
    <svg class="tri-hp-glyph" viewBox="0 0 40 40" aria-hidden="true">
      <path class="tri-hp-glyph-route" d={ride.glyph.route} />
      {ride.glyph.park && <path class="tri-hp-glyph-park" d={ride.glyph.park} />}
    </svg>
  ) : (
    <span class="tri-hp-glyph" aria-hidden="true" />
  )

const LoopMap = ({ loops, data }: { loops: HighParkLoops; data: HighParkSport }) => {
  const frame = loops.frame
  const rides = data.activities.filter(ride => ride.trace)
  if (!frame || rides.length === 0) return null
  const best = data.fastest?.trace ? data.fastest : null
  const start = best ? /^M(-?\d+) (-?\d+)/.exec(best.trace) : null
  const pct = (value: number, origin: number, span: number) =>
    `${r1(((value - origin) / span) * 100)}%`
  return (
    <figure class="tri-hp-map" data-terrain={TERRAIN_HREF} data-exaggeration={RELIEF_X}>
      <div class="tri-hp-plate">
        <div class="tri-hp-flat" style={`aspect-ratio:${frame.width}/${frame.height}`}>
          <svg
            viewBox={`${frame.x} ${frame.y} ${frame.width} ${frame.height}`}
            role="img"
            aria-label={`The High Park main-road circuit drawn from the fastest lap of ${rides.length} rides.`}
          >
            <g class="tri-hp-traces">
              {rides.map(ride => (
                <path key={ride.id} d={ride.trace} />
              ))}
            </g>
            {best && <path class="tri-hp-trace-best" d={best.trace} />}
            {start && (
              <rect
                class="tri-hp-start"
                x={Number(start[1]) - 7}
                y={Number(start[2]) - 7}
                width="14"
                height="14"
              />
            )}
          </svg>
          {start && (
            <span
              class="tri-hp-start-label"
              style={`left:${pct(Number(start[1]), frame.x, frame.width)};top:${pct(Number(start[2]), frame.y, frame.height)}`}
            >
              start
            </span>
          )}
        </div>
        {/* The client shows the tabs once the relief has loaded (fun/high-park-terrain.ts). */}
        <div class="tri-hp-dims" role="group" aria-label="map view" style="display:none">
          <button class="tri-hp-dim" type="button" data-dim="3d" aria-pressed="true">
            3d
          </button>
          <button class="tri-hp-dim" type="button" data-dim="2d" aria-pressed="false">
            2d
          </button>
        </div>
      </div>
      {/* As wide as the flat map, which the plate fits by height, so the scale bar holds. */}
      <div
        class="tri-hp-legend"
        style={`--hp-flat:${r1((frame.width / frame.height) * 1000) / 1000}`}
      >
        <span class="tri-hp-2d">
          <span class="tri-hp-scale">
            <Unit metric="200 m" imperial="0.1 mi" />
            <span
              class="tri-hp-scale-bar"
              data-metric-w={pct(200, 0, frame.width)}
              data-imperial-w={pct(MI_M / 10, 0, frame.width)}
              style={`width:${pct(MI_M / 10, 0, frame.width)}`}
            />
          </span>
          <span class="tri-hp-north" aria-label="north is up">
            N ↑
          </span>
        </span>
      </div>
    </figure>
  )
}

type HoverPoint = [ride: number, lap: number, x: number, y: number]

const lapClockGeometry = (data: HighParkSport) => {
  const rides = [...data.activities].reverse()
  const laps = rides.flatMap(ride => ride.lapsS)
  if (laps.length === 0) return null
  const scale = lapScale(laps)
  const [width, height] = [640, 240]
  const [left, right, top, bottom] = [34, 10, 10, 22]
  const first = dayNumber(rides[0].date)
  const span = Math.max(1, dayNumber(rides.at(-1)!.date) - first)
  const x = (date: string) => r1(left + ((dayNumber(date) - first) / span) * (width - left - right))
  const y = (seconds: number) => r1(top + (1 - pace(seconds, scale)) * (height - top - bottom))
  const points: HoverPoint[] = data.activities.flatMap((ride, index) =>
    ride.lapsS.map((seconds, lap): HoverPoint => [index, lap, x(ride.date), y(seconds)]),
  )
  return { rides, laps, scale, width, height, left, right, first, span, x, y, points }
}

const LapClock = ({ data }: { data: HighParkSport }) => {
  const geometry = lapClockGeometry(data)
  if (!geometry) return null
  const { rides, laps, scale, width, height, left, right, first, span, x, y } = geometry
  const dots = { on: '', off: '' }
  for (const ride of rides)
    for (const seconds of ride.lapsS)
      dots[seconds > scale.slow ? 'off' : 'on'] += `M${x(ride.date)} ${y(seconds)}h0`
  let best = Infinity
  let steps = ''
  const records: HighParkVisit[] = []
  for (const ride of rides) {
    if (ride.source !== 'Strava segment' || ride.fastestS >= best) continue
    steps +=
      best === Infinity
        ? `M${x(ride.date)} ${y(ride.fastestS)}`
        : `H${x(ride.date)}V${y(ride.fastestS)}`
    best = ride.fastestS
    records.push(ride)
  }
  if (records.length) steps += `H${width - right}`
  const ticks: number[] = []
  for (let t = Math.ceil(scale.fast / 15) * 15; t <= scale.slow; t += 30) ticks.push(t)
  const months: { x: number; label: string }[] = []
  const [y0, m0] = [Number(rides[0].date.slice(0, 4)), Number(rides[0].date.slice(5, 7))]
  for (let m = m0; ; m++) {
    const iso = `${y0 + Math.floor((m - 1) / 12)}-${String(((m - 1) % 12) + 1).padStart(2, '0')}-01`
    if (dayNumber(iso) > first + span) break
    if (dayNumber(iso) >= first) months.push({ x: x(iso), label: MONTHS[(m - 1) % 12] })
  }
  return (
    <figure class="tri-hp-clock">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${laps.length} lap times by date; the fastest lap fell from ${clock(records[0]?.fastestS ?? 0)} to ${clock(best)}.`}
      >
        <g class="tri-hp-grid">
          {ticks.map(t => (
            <g key={t}>
              <path d={`M${left} ${y(t)}H${width - right}`} />
              <text x={left - 6} y={y(t) + 3}>
                {clock(t)}
              </text>
            </g>
          ))}
          {months.map(month => (
            <text key={month.label} class="is-month" x={month.x} y={height - 6}>
              {month.label}
            </text>
          ))}
        </g>
        <path class="tri-hp-dots" d={dots.on} />
        {dots.off && <path class="tri-hp-dots is-off" d={dots.off} />}
        <path class="tri-hp-record" d={steps} />
        {records.map(ride => (
          <rect
            key={ride.id}
            class="tri-hp-record-dot"
            x={x(ride.date) - 3}
            y={y(ride.fastestS) - 3}
            width="6"
            height="6"
          />
        ))}
        <rect class="tri-hp-hover" x="-20" y="-20" width="9" height="9" />
      </svg>
    </figure>
  )
}

const median = (values: number[]): number => {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = sorted.length >> 1
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

const Tally = ({ data }: { data: HighParkSport }) => {
  const laps = data.activities.flatMap(ride => ride.lapsS)
  const firstRide = data.activities.at(-1)
  const rows: [ComponentChild, ComponentChild][] = [
    ['loops', data.laps],
    [
      <Unit metric="total km" imperial="total mi" />,
      <Unit
        metric={String(Math.round(data.distanceKm))}
        imperial={String(Math.round((data.distanceKm * 1000) / MI_M))}
      />,
    ],
    ['rides', data.activities.length],
    ['hours', (laps.reduce((sum, s) => sum + s, 0) / 3600).toFixed(1)],
    ['median', clock(median(laps))],
  ]
  if (data.fastest)
    rows.push([
      'fastest',
      <a class="internal" href={data.fastest.href}>
        {clock(data.fastest.fastestS)} · {shortDate(data.fastest.date)}
      </a>,
    ])
  if (data.mostLaps)
    rows.push([
      'max # laps/ride',
      <a class="internal" href={data.mostLaps.href}>
        {data.mostLaps.laps} · {shortDate(data.mostLaps.date)}
      </a>,
    ])
  if (firstRide)
    rows.push([
      'first occurence',
      <a class="internal" href={firstRide.href}>
        {shortDate(firstRide.date)}
      </a>,
    ])
  return (
    <table class="tri-hp-tally">
      <tbody>
        {rows.map(([label, value], i) => (
          <tr key={i}>
            <th scope="row">{label}</th>
            <td>{value}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

// Each ride's drop-down is an empty slot: the client renders it on first open from the sport's
// lap data (`LapData`).
const RideLog = ({
  title,
  data,
  sport,
}: {
  title: string
  data: HighParkSport
  sport: LapSport
}) => {
  const scale = lapScale(data.activities.flatMap(ride => ride.lapsS))
  return (
    <section class="tri-hp-log" aria-label={`High Park ${title} rides`}>
      <h2>
        {title} log <span>{data.activities.length}</span>
      </h2>
      <ol>
        {data.activities.map((ride, index) => (
          <li key={ride.id}>
            <details>
              <summary>
                <time dateTime={ride.date}>{shortDate(ride.date)}</time>
                <RideGlyph ride={ride} />
                <span class="tri-hp-ride">{ride.name}</span>
                <span class="tri-hp-laps">
                  <LapStrip ride={ride} scale={scale} />
                </span>
                <span class="tri-hp-n" title={`${ride.laps} laps`}>
                  {ride.laps}×
                </span>
                <span class="tri-hp-best">{clock(ride.fastestS)}</span>
                <span class="tri-hp-toggle" aria-hidden="true" />
              </summary>
              <div class="tri-hp-lapslot" data-ride={index} data-sport={sport} />
            </details>
          </li>
        ))}
      </ol>
    </section>
  )
}

// Escape `<` so a ride name cannot close the script element.
const hoverJson = (data: HighParkSport, sport: LapSport): string =>
  JSON.stringify({
    sport,
    rides: data.activities.map(ride => [
      ride.date,
      ride.name,
      ride.lapDetail.map(lap => [
        lap.s,
        lap.km,
        lap.w,
        lap.hr,
        lap.cad,
        lap.nm,
        lap.teL,
        lap.teR,
        lap.psL,
        lap.psR,
        lap.balL,
        lap.resp,
        lap.core,
        lap.shifts,
        lap.gain,
        lap.profile && [lap.profile.w, lap.profile.hr, lap.profile.cad, lap.profile.spd],
      ]),
      ride.altitude,
      ride.href,
    ]),
    // Only the cycling laps have a lap clock to hover.
    points: sport === 'bike' ? (lapClockGeometry(data)?.points ?? []) : [],
  }).replace(/</g, '\\u003c')

const LapData = ({ data, sport }: { data: HighParkSport; sport: LapSport }) => (
  <script
    type="application/json"
    class="tri-hp-data"
    dangerouslySetInnerHTML={{ __html: hoverJson(data, sport) }}
  />
)

export const HighParkPanel = ({
  renderData,
  loops = renderData?.highParkLoops ?? buildHighParkLoops([]),
}: {
  renderData?: TriathlonRenderData
  loops?: HighParkLoops
}) => {
  const bike = loops.cycling
  const run = loops.running
  return (
    <div class="tri-fun tri-hp" tabindex={-1} data-keyboard-scroll-scope>
      <div class="tri-hp-head">
        <h1 class="tri-hp-title">High Park</h1>
        <button class="tri-hp-unit" type="button" aria-label="switch mph and km/h">
          mph
        </button>
      </div>
      {bike.activities.length > 0 ? (
        <>
          <div class="tri-hp-spread">
            <LoopMap loops={loops} data={bike} />
            <div class="tri-hp-side">
              <Tally data={bike} />
              <LapClock data={bike} />
            </div>
          </div>
          <RideLog title="ride" data={bike} sport="bike" />
          <LapData data={bike} sport="bike" />
        </>
      ) : (
        <p class="tri-hp-empty">
          no complete cycling loops matched in my synced activities yet.{' '}
          <a class="internal" href="/triathlon/on">
            open the activity log
          </a>
        </p>
      )}
      {run.activities.length > 0 ? (
        <>
          <RideLog title="run" data={run} sport="run" />
          <LapData data={run} sport="run" />
        </>
      ) : (
        <p class="tri-hp-empty">
          <span>running</span> no full loop on foot yet.
        </p>
      )}
    </div>
  )
}
