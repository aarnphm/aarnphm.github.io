import { MathText } from '../../../util/math-text'
import {
  formatRatio,
  evalTerms,
  type MarkSpec,
  type Position,
  type ProbeSpec,
  type Series,
} from './model'

// A second cue beside colour: salmon circles and sage squares.
export const Marker = ({ slot, x, y }: { slot: number; x: number; y: number }) =>
  slot === 0 ? (
    <g transform={`translate(${x} ${y})`}>
      <circle class={`dch-marker dch-s${slot}`} r={2.75} />
    </g>
  ) : (
    <g transform={`translate(${x} ${y})`}>
      <rect class={`dch-marker dch-s${slot}`} x={-2.5} y={-2.5} width={5} height={5} />
    </g>
  )

export const Swatch = ({ slot, mode }: { slot: number; mode: 'bar' | 'line' }) => (
  <svg class="dch-swatch" viewBox="0 0 18 10" aria-hidden="true">
    {mode === 'bar' ? (
      <rect class={`dch-bar dch-s${slot}`} x={4.5} y={0.5} width={9} height={9} />
    ) : (
      <>
        <line class={`dch-line dch-l${slot}`} x1={0} x2={18} y1={5} y2={5} />
        <Marker slot={slot} x={9} y={5} />
      </>
    )}
  </svg>
)

export type ProbeState = {
  x: number
  y: number
  cx: number
  cy: number
  xs: string
  ys: string
  regime: string
}

type ReadoutsProps = {
  mode: 'bar' | 'line'
  positions: Position[]
  active: number
  xname: string
  measured: Series[]
  curve: Series | undefined
  marks: MarkSpec[]
  probe: ProbeSpec | null
  probeStates: ProbeState[]
  probeIndex: number
  fmt: (v: number) => string
  unit: string
  unitName: string | undefined
  ratio: string | undefined
}

// Per-state readouts and the data table; the script only shows or hides the precomputed states.
export const Readouts = ({
  mode,
  positions,
  active,
  xname,
  measured,
  curve,
  marks,
  probe,
  probeStates,
  probeIndex,
  fmt,
  unit,
  unitName,
  ratio,
}: ReadoutsProps) => (
  <>
    <div class="dch-readouts">
      {positions.map((p, i) => (
        <dl class="dch-readout" data-dch-state={String(i)} hidden={i !== active}>
          <div>
            <dt>{xname}</dt>
            <dd>
              <MathText text={p.tex} mathClass="dch-math" />
            </dd>
          </div>
          {measured.map((s, j) => (
            <div>
              <dt>
                <Swatch slot={s.slot} mode={mode} />
                {s.name}
              </dt>
              <dd>{p.values[j] === null ? '–' : `${fmt(p.values[j]!)}${unit}`}</dd>
            </div>
          ))}
          {ratio && measured.length === 2 ? (
            <div>
              <dt>{ratio}</dt>
              <dd>
                {p.values[0] && p.values[1] !== null
                  ? formatRatio(p.values[1]! / p.values[0])
                  : '–'}
              </dd>
            </div>
          ) : null}
        </dl>
      ))}
      {probe
        ? probeStates.map((s, i) => (
            <dl class="dch-readout" data-dch-probe-state={String(i)} hidden={i !== probeIndex}>
              <div>
                <dt>
                  <MathText text={probe.xname ?? 'x'} mathClass="dch-math" />
                </dt>
                <dd>{s.xs}</dd>
              </div>
              <div>
                <dt>
                  <MathText text={probe.yname ?? 'y'} mathClass="dch-math" />
                </dt>
                <dd>{s.ys}</dd>
              </div>
              <div>
                <dt>{probe.state ?? 'region'}</dt>
                <dd>{s.regime}</dd>
              </div>
            </dl>
          ))
        : null}
    </div>

    <details class="dch-details">
      <summary>table</summary>
      <div class="dch-table-wrap">
        <table class="dch-table">
          {probe && curve ? (
            <>
              <thead>
                <tr>
                  <th scope="col">point</th>
                  <th scope="col">
                    <MathText text={probe.xname ?? 'x'} mathClass="dch-math" />
                  </th>
                  <th scope="col">
                    <MathText text={probe.yname ?? 'y'} mathClass="dch-math" />
                  </th>
                </tr>
              </thead>
              <tbody>
                {marks.map(m => (
                  <tr>
                    <th scope="row">
                      <MathText text={`$${m.xlabel ?? ''}$`} mathClass="dch-math" />
                    </th>
                    <td>{m.x.toFixed(1)}</td>
                    <td>{fmt(evalTerms(curve.spec.terms!, m.x))}</td>
                  </tr>
                ))}
              </tbody>
            </>
          ) : (
            <>
              <thead>
                <tr>
                  <th scope="col">{xname}</th>
                  {measured.map(s => (
                    <th scope="col">{`${s.name}${unitName ? ` (${unitName})` : ''}`}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {positions.map(p => (
                  <tr>
                    <th scope="row">
                      <MathText text={p.tex} mathClass="dch-math" />
                    </th>
                    {p.values.map(v => (
                      <td>{v === null ? '–' : fmt(v)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </>
          )}
        </table>
      </div>
    </details>
  </>
)
