import type { HighParkLap, HighParkVisit } from '../../../util/high-park-loops'
import { LAP_BINS } from '../../../util/high-park-lap-metrics'
import { niceTicks } from '../../../util/triathlon-card/chart-geometry'

const clock = (seconds: number): string => {
  const rounded = Math.round(seconds)
  return `${Math.floor(rounded / 60)}:${String(rounded % 60).padStart(2, '0')}`
}
const r1 = (value: number): number => Math.round(value * 10) / 10
const pair = (left: number | null, right: number | null): string =>
  left === null && right === null ? '' : `${left ?? '–'}/${right ?? '–'}`

/** Distance and speed units, after the triathlon pages' unit preference. */
export interface HighParkUnits {
  imperial: boolean
  speed: string
  distance: string
  climb: string
  /** km → display distance, km/h → display speed, m → display climb. */
  km: (km: number) => number
  kmh: (kmh: number) => number
  m: (m: number) => number
}
export const highParkUnits = (imperial: boolean): HighParkUnits => ({
  imperial,
  speed: imperial ? 'mph' : 'km/h',
  distance: imperial ? 'mi' : 'km',
  climb: imperial ? 'ft' : 'm',
  km: km => (imperial ? km / 1.609344 : km),
  kmh: kmh => (imperial ? kmh / 1.609344 : kmh),
  m: m => (imperial ? m * 3.28084 : m),
})

const lapSpeed = (lap: HighParkLap, units: HighParkUnits): number | null =>
  lap.km > 0 && lap.s > 0 ? r1(units.kmh((lap.km / lap.s) * 3600)) : null

type Value = (lap: HighParkLap) => number | null
interface Metric {
  key: string
  label: string
  unit: string
  /** Second series, drawn dashed (the right leg). */
  values: [Value] | [Value, Value]
  format?: (value: number) => string
  /** Lower is better: the axis flips so better is always up. */
  lowerIsBetter?: boolean
}

// Lap-by-lap panels, in reading order. A panel needs two laps with a value.
const metrics = (units: HighParkUnits): Metric[] => [
  { key: 'time', label: 'time', unit: '', values: [l => l.s], format: clock, lowerIsBetter: true },
  {
    key: 'speed',
    label: 'speed',
    unit: units.speed,
    values: [l => lapSpeed(l, units)],
    format: v => v.toFixed(1),
  },
  { key: 'power', label: 'power', unit: 'W', values: [l => l.w] },
  { key: 'hr', label: 'heart rate', unit: 'bpm', values: [l => l.hr] },
  { key: 'cad', label: 'cadence', unit: 'rpm', values: [l => l.cad] },
  { key: 'torque', label: 'torque', unit: 'N·m', values: [l => l.nm], format: v => v.toFixed(1) },
  { key: 'te', label: 'torque effectiveness L/R', unit: '%', values: [l => l.teL, l => l.teR] },
  { key: 'ps', label: 'pedal smoothness L/R', unit: '%', values: [l => l.psL, l => l.psR] },
  { key: 'bal', label: 'balance, left', unit: '%', values: [l => l.balL] },
  { key: 'shifts', label: 'shifts', unit: '', values: [l => l.shifts] },
  {
    key: 'resp',
    label: 'breathing',
    unit: 'br/min',
    values: [l => l.resp],
    format: v => v.toFixed(1),
  },
  { key: 'core', label: 'core temp', unit: '°C', values: [l => l.core], format: v => v.toFixed(2) },
]

/** Lap-by-lap chart viewBox: width, height, inset. */
export const LAP_CHART = { w: 200, h: 54, pad: 6 } as const

const LapChart = ({
  metric,
  laps,
  best,
}: {
  metric: Metric
  laps: HighParkLap[]
  best: number
}) => {
  const { w: CW, h: CH, pad: CPAD } = LAP_CHART
  const series = metric.values.map(value => laps.map(value))
  const known = series.flat().filter((v): v is number => v !== null)
  if (series[0].filter(v => v !== null).length < 2) return null
  const [lo, hi] = [Math.min(...known), Math.max(...known)]
  const span = hi - lo || 1
  const x = (i: number) => r1(CPAD + (i / Math.max(1, laps.length - 1)) * (CW - 2 * CPAD))
  const y = (v: number) => {
    const t = (v - lo) / span
    return r1(CPAD + (metric.lowerIsBetter ? t : 1 - t) * (CH - 2 * CPAD))
  }
  const line = (values: (number | null)[]) => {
    let d = ''
    let open = false
    values.forEach((v, i) => {
      if (v === null) return void (open = false)
      d += `${open ? 'L' : 'M'}${x(i)} ${y(v)}`
      open = true
    })
    return d
  }
  const marks = (values: (number | null)[]) =>
    values.map((v, i) => (v === null || i === best ? '' : `M${x(i)} ${y(v)}h0`)).join('')
  const fmt = metric.format ?? ((v: number) => String(Math.round(v)))
  const mean = known.reduce((a, b) => a + b, 0) / known.length
  const bestValue = series[0][best]
  return (
    <figure class="tri-hp-lapchart" data-metric={metric.key}>
      <figcaption>
        <span>{metric.label}</span>
        <span>
          {fmt(lo)}–{fmt(hi)}
          {metric.unit && ` ${metric.unit}`}
        </span>
      </figcaption>
      <svg viewBox={`0 0 ${CW} ${CH}`} preserveAspectRatio="none" aria-hidden="true">
        <path class="tri-hp-lapchart-mean" d={`M${CPAD} ${y(mean)}H${CW - CPAD}`} />
        {series.map((values, s) => (
          <g key={s} class={s ? 'is-right' : undefined}>
            <path class="tri-hp-lapchart-line" d={line(values)} />
            <path class="tri-hp-lapchart-marks" d={marks(values)} />
          </g>
        ))}
        {bestValue !== null && (
          <path class="tri-hp-lapchart-best" d={`M${x(best)} ${y(bestValue)}h0`} />
        )}
        <path class="tri-hp-cursor" d="" />
      </svg>
    </figure>
  )
}

/** Per-lap profile viewBox: width, height, inset. */
export const PROFILE = { w: 300, h: 120, pad: 4 } as const
export type ProfileKey = 'w' | 'spd' | 'hr' | 'cad'
const profileFields = (units: HighParkUnits) =>
  [
    { key: 'w', label: 'power', unit: 'W', scale: (v: number) => v, digits: 0 },
    { key: 'spd', label: 'speed', unit: units.speed, scale: units.kmh, digits: 1 },
    { key: 'hr', label: 'heart rate', unit: 'bpm', scale: (v: number) => v, digits: 0 },
    { key: 'cad', label: 'cadence', unit: 'rpm', scale: (v: number) => v, digits: 0 },
  ] as const

/** The y range a profile draws: 2nd–98th percentile, so one sprint bin cannot flatten the rest. */
export const profileRange = (values: number[]): [number, number] => {
  const sorted = [...values].sort((a, b) => a - b)
  const lo = sorted[Math.floor(sorted.length * 0.02)]
  const hi = sorted[Math.ceil(sorted.length * 0.98) - 1]
  return hi > lo ? [lo, hi] : [lo - 1, hi + 1]
}
export const profileY = (value: number, lo: number, hi: number): number => {
  const { h, pad } = PROFILE
  return r1(pad + (1 - Math.min(1, Math.max(0, (value - lo) / (hi - lo)))) * (h - 2 * pad))
}
export const profileX = (bin: number): number => r1((bin / (LAP_BINS - 1)) * PROFILE.w)

const Profile = ({
  field,
  laps,
  best,
  altitude,
  lapKm,
  units,
}: {
  field: ReturnType<typeof profileFields>[number]
  laps: HighParkLap[]
  best: number
  altitude: number[] | null
  lapKm: number
  units: HighParkUnits
}) => {
  const { w: PW, h: PH } = PROFILE
  const rows = laps.map(lap =>
    (lap.profile?.[field.key] ?? []).map(v => (v === null ? null : field.scale(v))),
  )
  const known = rows.flat().filter((v): v is number => v !== null)
  if (rows.filter(row => row.length === LAP_BINS).length === 0 || known.length === 0) return null
  const [lo, hi] = profileRange(known)
  const ticks = niceTicks(lo, hi, 3).filter(t => t >= lo && t <= hi)
  const y = (v: number) => profileY(v, lo, hi)
  const line = (row: (number | null)[]) =>
    row.length === LAP_BINS
      ? row
          .map((v, i) => (v === null ? '' : `L${profileX(i)} ${y(v)}`))
          .join('')
          .replace(/^L/, 'M')
      : ''
  const top = altitude ? Math.max(1, ...altitude) : 1
  // The hill fills up to 60% of the height, so where it climbs reads against every line.
  const ground = altitude
    ? `M0 ${PH}` +
      altitude.map((a, i) => `L${profileX(i)} ${r1(PH - (a / top) * PH * 0.6)}`).join('') +
      `L${PW} ${PH}Z`
    : ''
  const fmt = (v: number) => v.toFixed(field.key === 'spd' && hi - lo < 10 ? 1 : 0)
  const lapLength = units.km(lapKm)
  const xTicks: [string, number, string][] = [
    ['0', 0, 'tri-cax-xt--first'],
    [(lapLength / 2).toFixed(2), 50, ''],
    [`${lapLength.toFixed(2)} ${units.distance}`, 100, 'tri-cax-xt--last'],
  ]
  return (
    <figure class="tri-hp-profile" data-metric={field.key}>
      <figcaption>
        <span>{field.label}</span>
        <span class="tri-hp-readout" data-idle={field.unit}>
          {field.unit}
        </span>
      </figcaption>
      <div class="tri-cax-frame">
        <div class="tri-cax-yax">
          {ticks.map(t => (
            <span key={t} class="tri-cax-yt" style={`top:${r1((y(t) / PH) * 100)}%`}>
              {fmt(t)}
            </span>
          ))}
        </div>
        <div class="tri-cax-stage">
          <span class="tri-cax-ax tri-cax-ax--y" style="top:0%;height:100%" />
          <span class="tri-cax-ax tri-cax-ax--x" style="top:100%" />
          <svg
            viewBox={`0 0 ${PW} ${PH}`}
            preserveAspectRatio="none"
            aria-hidden="true"
            data-lo={lo}
            data-hi={hi}
          >
            {ground && <path class="tri-hp-ground" d={ground} />}
            {ticks.map(t => (
              <path key={t} class="tri-hp-grid-line" d={`M0 ${y(t)}H${PW}`} />
            ))}
            {rows.map((row, i) =>
              i === best ? null : <path key={i} data-lap={i} d={line(row)} />,
            )}
            <path class="is-best" data-lap={best} d={line(rows[best])} />
            <path class="tri-hp-cursor" d="" />
            <path class="tri-hp-marker" d="" />
          </svg>
        </div>
        <div class="tri-cax-xax">
          {xTicks.map(([label, pct, cls]) => (
            <span key={pct} class={`tri-cax-xt ${cls}`.trim()} style={`left:${pct}%`}>
              {label}
            </span>
          ))}
        </div>
      </div>
    </figure>
  )
}

type Column = { label: string; cell: (lap: HighParkLap) => string | number | null }
const columns = (units: HighParkUnits): Column[] => [
  { label: 'time', cell: lap => clock(lap.s) },
  { label: units.speed, cell: lap => lapSpeed(lap, units) },
  { label: 'W', cell: lap => lap.w },
  { label: 'bpm', cell: lap => lap.hr },
  { label: 'rpm', cell: lap => lap.cad },
  { label: 'N·m', cell: lap => lap.nm },
  { label: 'TE L/R', cell: lap => pair(lap.teL, lap.teR) || null },
  { label: 'PS L/R', cell: lap => pair(lap.psL, lap.psR) || null },
  { label: 'bal L/R', cell: lap => (lap.balL === null ? null : pair(lap.balL, 100 - lap.balL)) },
  { label: 'shifts', cell: lap => lap.shifts },
  {
    label: `climb ${units.climb}`,
    cell: lap => (lap.gain === null ? null : Math.round(units.m(lap.gain))),
  },
  { label: 'br/min', cell: lap => lap.resp },
  { label: 'core °C', cell: lap => lap.core },
]

const LapTable = ({
  laps,
  best,
  units,
}: {
  laps: HighParkLap[]
  best: number
  units: HighParkUnits
}) => {
  const shown = columns(units).filter(column => laps.some(lap => column.cell(lap) !== null))
  return (
    <div class="tri-hp-lapgrid" tabindex={0} aria-label="laps">
      <table>
        <thead>
          <tr>
            <th scope="col">lap</th>
            {shown.map(column => (
              <th key={column.label} scope="col">
                {column.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {laps.map((lap, i) => (
            <tr key={i} data-lap={i} class={i === best ? 'is-best' : undefined}>
              <th scope="row">{i + 1}</th>
              {shown.map(column => (
                <td key={column.label}>{column.cell(lap) ?? ''}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** The ride's drop-down: a link to the activity, laps compared side by side, per lap, as a table. */
export const LapView = ({
  ride,
  index,
  imperial = true,
}: {
  ride: Pick<HighParkVisit, 'lapDetail' | 'altitude' | 'href'>
  index?: number
  imperial?: boolean
}) => {
  const laps = ride.lapDetail
  if (laps.length === 0) return null
  const units = highParkUnits(imperial)
  const best = laps.reduce((b, lap, i) => (lap.s < laps[b].s ? i : b), 0)
  const lapKm = laps.reduce((sum, lap) => sum + lap.km, 0) / laps.length
  return (
    <div class="tri-hp-lapview" data-ride={index} data-laps={laps.length}>
      <a class="internal tri-hp-open" href={ride.href}>
        open activity
      </a>
      {laps.length > 1 && (
        <section class="tri-hp-lapcharts" aria-label="lap by lap">
          <h3>lap by lap</h3>
          <div>
            {metrics(units).map(metric => (
              <LapChart key={metric.key} metric={metric} laps={laps} best={best} />
            ))}
          </div>
        </section>
      )}
      <section class="tri-hp-profiles" aria-label="per lap">
        <h3>per lap</h3>
        <div>
          {profileFields(units).map(field => (
            <Profile
              key={field.key}
              field={field}
              laps={laps}
              best={best}
              altitude={ride.altitude}
              lapKm={lapKm}
              units={units}
            />
          ))}
        </div>
      </section>
      <LapTable laps={laps} best={best} units={units} />
    </div>
  )
}
