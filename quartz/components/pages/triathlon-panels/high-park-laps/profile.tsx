import { PICK_SLOTS } from '../../../../util/high-park-lap-compare'
import { LAP_BINS } from '../../../../util/high-park-lap-metrics'
import { niceTicks } from '../../../../util/triathlon-card/chart-geometry'
import {
  clock,
  PROFILE,
  profileX,
  profileY,
  r1,
  type HighParkUnits,
  type LapSport,
  type ProfileField,
  type ProfileModel,
} from './fields'

/** A lap's line through its bins; empty when the lap has no profile. */
export const profileLine = (row: readonly (number | null)[], lo: number, hi: number): string =>
  row.length === LAP_BINS
    ? row
        .map((v, i) => (v === null ? '' : `L${profileX(i)} ${profileY(v, lo, hi)}`))
        .join('')
        .replace(/^L/, 'M')
    : ''

// A Catmull-Rom curve through the points, as cubic Béziers.
const curve = (points: [number, number][]): string => {
  const at = (i: number) => points[Math.min(points.length - 1, Math.max(0, i))]
  let d = `M${points[0][0]} ${points[0][1]}`
  for (let i = 0; i < points.length - 1; i++) {
    const [p0, p1, p2, p3] = [at(i - 1), at(i), at(i + 1), at(i + 2)]
    d +=
      `C${r1(p1[0] + (p2[0] - p0[0]) / 6)} ${r1(p1[1] + (p2[1] - p0[1]) / 6)}` +
      ` ${r1(p2[0] - (p3[0] - p1[0]) / 6)} ${r1(p2[1] - (p3[1] - p1[1]) / 6)} ${p2[0]} ${p2[1]}`
  }
  return d
}

// Pace ticks land on round paces, not round speeds.
const paceTicks = (lo: number, hi: number): [value: number, label: string][] => {
  const [fast, slow] = [3600 / hi, 3600 / lo]
  const step = [5, 10, 15, 30, 60, 120, 300].find(s => (slow - fast) / s <= 4) ?? 600
  const ticks: [number, string][] = []
  for (let t = Math.ceil(fast / step) * step; t <= slow; t += step) ticks.push([3600 / t, clock(t)])
  return ticks
}

/** Who a readout value or cursor dot speaks for: the fastest lap, a pick's slot, the hovered lap, the baseline. */
export const READ_WHO = [
  'best',
  ...Array.from({ length: PICK_SLOTS }, (_, slot) => String(slot)),
  'hover',
  'base',
] as const

export const Profile = ({
  field,
  model,
  best,
  altitude,
  lapKm,
  units,
  sport,
}: {
  field: ProfileField
  model: ProfileModel
  best: number
  altitude: number[] | null
  lapKm: number
  units: HighParkUnits
  sport: LapSport
}) => {
  const { w: PW, h: PH } = PROFILE
  const { rows, lo, hi, base } = model
  const pace = sport === 'run' && field.key === 'spd'
  const digits = field.key === 'spd' && hi - lo < 10 ? 1 : 0
  const ticks: [number, string][] = pace
    ? paceTicks(lo, hi)
    : niceTicks(lo, hi, 3)
        .filter(t => t >= lo && t <= hi)
        .map(t => [t, t.toFixed(digits)])
  const y = (v: number) => profileY(v, lo, hi)
  const top = altitude ? Math.max(1, ...altitude) : 1
  // The hill fills up to 60% of the height, so where it climbs reads against every line.
  const ground = altitude
    ? `M0 ${PH}` +
      altitude.map((a, i) => `L${profileX(i)} ${r1(PH - (a / top) * PH * 0.6)}`).join('') +
      `L${PW} ${PH}Z`
    : ''
  const edge = (sign: number): [number, number][] =>
    base!.mean.map((m, i) => [profileX(i), y(m + sign * base!.sd[i])])
  const band = base ? `${curve(edge(1))}${curve(edge(-1).reverse()).replace(/^M/, 'L')}Z` : ''
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
        <span>{field.unit}</span>
        {/* Pooled: the client shows the values it needs and writes their text in place. */}
        <span class="tri-hp-readout">
          <span class="tri-hp-at"> </span>
          {READ_WHO.map(who => (
            <span key={who} class="tri-hp-val" data-who={who} style="display:none">
              <i />
              <b> </b>
              <span> </span>
              <em> </em>
            </span>
          ))}
        </span>
      </figcaption>
      <div class="tri-cax-frame">
        <div class="tri-cax-yax">
          {ticks.map(([t, label]) => (
            <span key={label} class="tri-cax-yt" style={`top:${r1((y(t) / PH) * 100)}%`}>
              {label}
            </span>
          ))}
        </div>
        <div class="tri-cax-stage">
          <span class="tri-cax-ax tri-cax-ax--y" style="top:0%;height:100%" />
          <span class="tri-cax-ax tri-cax-ax--x" style="top:100%" />
          <svg viewBox={`0 0 ${PW} ${PH}`} preserveAspectRatio="none" aria-hidden="true">
            {ground && <path class="tri-hp-ground" d={ground} />}
            {ticks.map(([t, label]) => (
              <path key={label} class="tri-hp-grid-line" d={`M0 ${y(t)}H${PW}`} />
            ))}
            <g class="tri-hp-laplines">
              {rows.map((row, i) =>
                i === best ? null : <path key={i} d={profileLine(row, lo, hi)} />,
              )}
            </g>
            <path class="tri-hp-band" d={band} />
            <path class="tri-hp-fastest" d={profileLine(rows[best], lo, hi)} />
            <path class="tri-hp-base" d={base ? curve(edge(0)) : ''} />
            {Array.from({ length: PICK_SLOTS }, (_, slot) => (
              <path key={slot} class="tri-hp-pickline" data-who={slot} d="" />
            ))}
            <path class="tri-hp-hoverline" d="" />
            <path class="tri-hp-cursor" d="" />
          </svg>
          {/* HTML, not zero-length SVG strokes: WebKit misplaces those once `d` changes. */}
          {READ_WHO.map(who => (
            <i key={who} class="tri-hp-dot" data-who={who} style="display:none" />
          ))}
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
