import type { HighParkLap } from '../../../../util/high-park-loops'
import { clock, lapSpeed, paceOf, type HighParkUnits, type LapSport } from './fields'

const pair = (left: number | null, right: number | null): string =>
  left === null && right === null ? '' : `${left ?? '–'}/${right ?? '–'}`
const fixed = (value: number | null, digits: number): string | null =>
  value === null ? null : value.toFixed(digits)

type Column = { label: string; cell: (lap: HighParkLap) => string | number | null }
const columns = (units: HighParkUnits, sport: LapSport): Column[] => [
  { label: 'time', cell: lap => clock(lap.s) },
  sport === 'run'
    ? {
        label: `/${units.distance}`,
        cell: lap => {
          const speed = lapSpeed(lap, units)
          return speed === null ? null : paceOf(speed)
        },
      }
    : { label: units.speed, cell: lap => fixed(lapSpeed(lap, units), 1) },
  { label: 'W', cell: lap => lap.w },
  { label: 'bpm', cell: lap => lap.hr },
  { label: sport === 'run' ? 'spm' : 'rpm', cell: lap => lap.cad },
  { label: units.torque, cell: lap => fixed(lap.nm && units.nm(lap.nm), 1) },
  { label: 'TE L/R', cell: lap => pair(lap.teL, lap.teR) || null },
  { label: 'PS L/R', cell: lap => pair(lap.psL, lap.psR) || null },
  { label: 'bal L/R', cell: lap => (lap.balL === null ? null : pair(lap.balL, 100 - lap.balL)) },
  { label: 'shifts', cell: lap => lap.shifts },
  {
    label: `climb ${units.climb}`,
    cell: lap => (lap.gain === null ? null : Math.round(units.m(lap.gain))),
  },
  { label: 'br/min', cell: lap => fixed(lap.resp, 1) },
  { label: `core ${units.temp}`, cell: lap => fixed(lap.core && units.celsius(lap.core), 2) },
]

export const LapTable = ({
  laps,
  best,
  units,
  sport,
}: {
  laps: HighParkLap[]
  best: number
  units: HighParkUnits
  sport: LapSport
}) => {
  const shown = columns(units, sport).filter(column => laps.some(lap => column.cell(lap) !== null))
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
