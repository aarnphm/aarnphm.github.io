import type { FunctionalComponent } from 'preact'
import { MathText } from '../../util/math-text'
import {
  LANE_H,
  LANE_X,
  LAYERS,
  STREAM_CONTEXTS,
  STREAM_HITS,
  STREAM_LINKS,
  STREAM_VIEWS,
  fmtTokens,
  markerPath,
  streamBars,
  streamContext,
  streamHit,
  streamLink,
  streamPlan,
  streamReadout,
  type Bar,
} from '../../util/mooncake-stream'
//@ts-ignore
import script from '../scripts/mooncake-stream.inline'
import style from '../styles/mooncakeFlow.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

const VIEW_H = 260
const AXIS_Y = 204
// Lane labels sit above their lanes so the drawing keeps its width on a phone.
const STREAM_LANES = [
  { kind: 'load', label: 'load prefix · pool → GPU', y: 28 },
  { kind: 'compute', label: `attention + MLP · ${LAYERS} layers`, y: 76 },
  { kind: 'stream', label: 'stream KV → decode DRAM', y: 124 },
  { kind: 'serial', label: 'serial · no overlap', y: 172 },
] as const

const MARKED_LANES = STREAM_LANES.slice(0, 3).map(lane => lane.y)
const DEFAULTS = { context: '131072', hit: '0.9', link: 'roce8' }

const PER_LAYER = [
  ['KV streamed', 'kv-layer'],
  ['prefix loaded', 'prefix-layer'],
  ['compute (est.)', 't-compute'],
  ['load', 't-load'],
  ['stream', 't-stream'],
  ['64 KiB slices', 'slices'],
] as const

const WHOLE = [
  ['KV for decode', 'kv-total'],
  ['layer-wise', 'layered'],
  ['serial', 'serial'],
  ['exposed tail', 'exposed'],
  ['bound by', 'bound'],
] as const

const Bars: FunctionalComponent<{ kind: string; bars: Bar[]; y: number }> = ({ kind, bars, y }) => (
  <g>
    {bars.map((bar, i) => (
      <rect
        class="mks-bar"
        data-mks-bar={`${kind}-${i}`}
        data-kind={kind === 'serial' ? ['load', 'compute', 'stream'][i] : kind}
        x={bar.x}
        y={y}
        width={bar.w}
        height={LANE_H}
      />
    ))}
  </g>
)

const Segments: FunctionalComponent<{
  name: string
  legend: string
  input: string
  options: readonly { value: string; label: string }[]
  checked: string
}> = ({ name, legend, input, options, checked }) => (
  <fieldset class="mck-seg-group">
    <legend>{legend}</legend>
    {options.map(option => (
      <label class="mck-seg">
        <input
          type="radio"
          name={name}
          value={option.value}
          checked={option.value === checked}
          {...{ [input]: true }}
        />
        <span>{option.label}</span>
      </label>
    ))}
  </fieldset>
)

const ReadoutTable: FunctionalComponent<{
  label: string
  rows: readonly (readonly [string, string])[]
  values: Record<string, string>
}> = ({ label, rows, values }) => (
  <table class="mck-table" aria-label={label}>
    <thead>
      <tr>
        <th scope="col">{label}</th>
        <th scope="col">value</th>
      </tr>
    </thead>
    <tbody>
      {rows.map(([name, key]) => (
        <tr>
          <th scope="row">{name}</th>
          <td data-mks-val={key}>{values[key]}</td>
        </tr>
      ))}
    </tbody>
  </table>
)

let figureCount = 0

const MooncakeLayerStreamImpl: QuartzMdxComponent<Props> = ({ caption }) => {
  const plan = streamPlan(
    streamContext(DEFAULTS.context),
    streamHit(DEFAULTS.hit),
    streamLink(DEFAULTS.link),
  )
  const values = streamReadout(plan)
  const id = `mks-${++figureCount}`

  return (
    <figure
      class="mooncake-flow mooncake-stream"
      data-mooncake-stream
      data-mks-context={DEFAULTS.context}
      data-mks-hit={DEFAULTS.hit}
      data-mks-link={DEFAULTS.link}
    >
      <div class="mck-controls">
        <Segments
          name={`${id}-context`}
          legend="context"
          input="data-mks-context-input"
          options={STREAM_CONTEXTS.map(n => ({ value: String(n), label: fmtTokens(n) }))}
          checked={DEFAULTS.context}
        />
        <Segments
          name={`${id}-hit`}
          legend="cached prefix"
          input="data-mks-hit-input"
          options={STREAM_HITS.map(h => ({ value: String(h), label: `${Math.round(h * 100)}%` }))}
          checked={DEFAULTS.hit}
        />
        <Segments
          name={`${id}-link`}
          legend="link"
          input="data-mks-link-input"
          options={STREAM_LINKS.map(link => ({ value: link.id, label: link.label }))}
          checked={DEFAULTS.link}
        />
      </div>

      {STREAM_VIEWS.map(v => {
        const bars = streamBars(plan, v.laneW)
        const right = bars.marker > LANE_X + v.laneW / 2
        return (
          <svg
            class={`mck-graph mks-graph mks-graph--${v.id}`}
            data-mks-lane-w={v.laneW}
            viewBox={`0 0 ${v.viewW} ${VIEW_H}`}
            preserveAspectRatio="xMidYMid meet"
            role="img"
            aria-label="Timeline of one prefill: per-layer prefix loads, per-layer compute and per-layer KV streams, above the same work run back to back without overlap."
          >
            {STREAM_LANES.map(lane => (
              <g>
                <text class="mck-label" x={LANE_X} y={lane.y - 7}>
                  {lane.label}
                </text>
                <rect class="mks-lane" x={LANE_X} y={lane.y} width={v.laneW} height={LANE_H} />
                <Bars kind={lane.kind} bars={bars[lane.kind]} y={lane.y} />
              </g>
            ))}
            <line class="mks-axis" x1={LANE_X} y1={AXIS_Y} x2={LANE_X + v.laneW} y2={AXIS_Y} />
            <path
              class="mks-marker"
              data-mks-marker
              d={markerPath(bars.marker, MARKED_LANES, LANE_H, AXIS_Y)}
            />
            <text class="mck-label" x={LANE_X} y={AXIS_Y + 22}>
              0
            </text>
            <text
              class="mck-label"
              data-mks-val="axis-end"
              x={LANE_X + v.laneW}
              y={AXIS_Y + 22}
              text-anchor="end"
            >
              {values['axis-end']}
            </text>
            <text
              class="mck-label"
              data-mks-val="marker"
              data-mks-marker-label
              x={bars.marker}
              y={AXIS_Y + 48}
              text-anchor={right ? 'end' : 'start'}
            >
              {values.marker}
            </text>
          </svg>
        )
      })}

      <div class="mck-readouts">
        <ReadoutTable label="per layer" rows={PER_LAYER} values={values} />
        <ReadoutTable label="whole prefill" rows={WHOLE} values={values} />
      </div>

      {caption ? (
        <figcaption class="mck-caption">
          <MathText text={caption} mathClass="mck-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const MooncakeLayerStreamComponent = MooncakeLayerStreamImpl as QuartzMdxComponent<Props>
MooncakeLayerStreamComponent.css = style
MooncakeLayerStreamComponent.afterDOMLoaded = script

export const MooncakeLayerStream = registerMdxComponent(
  'MooncakeLayerStream',
  MooncakeLayerStreamComponent,
)

export default (() => MooncakeLayerStream) satisfies (opts: undefined) => QuartzMdxComponent<Props>
