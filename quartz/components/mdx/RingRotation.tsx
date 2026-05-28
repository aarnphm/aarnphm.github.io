import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/ring-rotation.inline'
import style from '../styles/ringRotation.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; devices?: number }

const MIN_DEVICES = 2
const MAX_DEVICES = 8
const DEFAULT_DEVICES = 4
const VIEW = 360
const RING_R = 118
const CHIP_R = 162
const TOKEN_R = 16

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max)

const renderMath = (tex: string): string =>
  katex.renderToString(tex, {
    ...katexOptions,
    displayMode: false,
    output: 'html',
    macros: customMacros,
    strict: false,
    throwOnError: false,
  })

const MathFO: FunctionalComponent<{
  x: number
  y: number
  w: number
  h: number
  tex: string
  cls?: string
}> = ({ x, y, w, h, tex, cls }) => (
  <foreignObject x={x} y={y} width={w} height={h}>
    <div
      class={`rr-fo ${cls ?? ''}`.trim()}
      dangerouslySetInnerHTML={{ __html: renderMath(tex) }}
    />
  </foreignObject>
)

const polar = (angleDeg: number, radius: number) => {
  const rad = (angleDeg * Math.PI) / 180
  return { x: VIEW / 2 + radius * Math.cos(rad), y: VIEW / 2 + radius * Math.sin(rad) }
}

const deviceAngle = (i: number, p: number) => -90 + (360 * i) / p

const ringArcPath = (p: number) => {
  const start = polar(deviceAngle(0, p), RING_R)
  const segments: string[] = [`M ${start.x.toFixed(2)} ${start.y.toFixed(2)}`]
  for (let i = 1; i <= p; i++) {
    const pt = polar(deviceAngle(i % p, p), RING_R)
    segments.push(`A ${RING_R} ${RING_R} 0 0 1 ${pt.x.toFixed(2)} ${pt.y.toFixed(2)}`)
  }
  return segments.join(' ')
}

const RingRotationImpl: QuartzMdxComponent<Props> = ({ caption, devices = DEFAULT_DEVICES }) => {
  const initialP = clamp(Math.round(devices), MIN_DEVICES, MAX_DEVICES)
  const indices = Array.from({ length: initialP }, (_, i) => i)
  const ariaLabel = `Ring topology of ${initialP} devices circulating key-value slices; controls below let you step, play, reset and resize the ring.`

  return (
    <figure
      class="ring-rotation"
      data-ring-rotation
      data-devices-initial={String(initialP)}
      data-devices-min={String(MIN_DEVICES)}
      data-devices-max={String(MAX_DEVICES)}
    >
      <div class="rr-stage">
        <svg
          class="rr-graph"
          viewBox={`0 0 ${VIEW} ${VIEW}`}
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label={ariaLabel}
          data-rr-ring
        >
          <defs>
            <marker
              id="rr-arrow"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="5"
              markerHeight="5"
              orient="auto-start-reverse"
            >
              <path class="rr-arrowhead" d="M0,0 L10,5 L0,10 z" />
            </marker>
          </defs>

          <circle class="rr-ring-track" cx={VIEW / 2} cy={VIEW / 2} r={RING_R} />
          <path class="rr-ring-flow" d={ringArcPath(initialP)} marker-end="url(#rr-arrow)" />

          <g class="rr-nodes" data-rr-nodes>
            {indices.map(i => {
              const { x, y } = polar(deviceAngle(i, initialP), RING_R)
              const chip = polar(deviceAngle(i, initialP), CHIP_R)
              return (
                <g class="rr-node" data-rr-node={i}>
                  <circle class="rr-node-disc" cx={x} cy={y} r="22" />
                  <MathFO x={x - 18} y={y - 12} w={36} h={24} tex={`d_{${i}}`} cls="rr-fo--node" />
                  <rect
                    class="rr-slice-chip"
                    x={chip.x - 26}
                    y={chip.y - 13}
                    width="52"
                    height="26"
                    rx="4"
                    data-rr-chip={i}
                  />
                  <MathFO
                    x={chip.x - 26}
                    y={chip.y - 13}
                    w={52}
                    h={26}
                    tex={`k_${i},v_${i}`}
                    cls="rr-fo--chip"
                  />
                </g>
              )
            })}
          </g>

          <circle class="rr-token" cx={VIEW / 2} cy={VIEW / 2} r={TOKEN_R} data-rr-token />
          <foreignObject
            x={VIEW / 2 - TOKEN_R}
            y={VIEW / 2 - TOKEN_R / 2}
            width={TOKEN_R * 2}
            height={TOKEN_R}
            data-rr-token-text
          >
            <div
              class="rr-fo rr-fo--token"
              dangerouslySetInnerHTML={{ __html: renderMath('k_0,v_0') }}
            />
          </foreignObject>
        </svg>

        <div class="rr-side">
          <div class="rr-matrix" data-rr-matrix-wrap>
            <div class="rr-matrix-header">
              <span class="rr-matrix-title">holding matrix</span>
              <span class="rr-matrix-sub">device by slice</span>
            </div>
            <div
              class="rr-matrix-grid"
              data-rr-matrix
              style={`--rr-p: ${initialP}`}
              role="grid"
              aria-label="Per-device record of which key-value slices have been seen"
            >
              {indices.map(i =>
                indices.map(j => (
                  <span
                    class="rr-cell"
                    data-rr-cell={`${i}-${j}`}
                    data-rr-row={String(i)}
                    data-rr-col={String(j)}
                    role="gridcell"
                    aria-label={`device ${i} has slice ${j}: pending`}
                  />
                )),
              )}
            </div>
          </div>

          <dl class="rr-readout" data-rr-readout>
            <div class="rr-readout-row">
              <dt>per-device memory</dt>
              <dd>
                <MathLabel tex="\frac{L}{p} \cdot d" />
                <span class="rr-readout-eq">=</span>
                <span data-rr-mem class="rr-readout-val">
                  L/{initialP} d
                </span>
              </dd>
            </div>
            <div class="rr-readout-row">
              <dt>total communication</dt>
              <dd>
                <MathLabel tex="(p-1) \cdot \tfrac{L}{p} \cdot d" />
                <span class="rr-readout-eq">
                  <MathLabel tex="\approx" />
                </span>
                <span data-rr-comm class="rr-readout-val">
                  {initialP - 1} L/{initialP} d
                </span>
              </dd>
            </div>
            <div class="rr-readout-row">
              <dt>rounds to full coverage</dt>
              <dd>
                <MathLabel tex="p - 1" />
                <span class="rr-readout-eq">=</span>
                <span data-rr-rounds class="rr-readout-val">
                  {initialP - 1}
                </span>
              </dd>
            </div>
            <div class="rr-readout-row">
              <dt>current step</dt>
              <dd>
                <span data-rr-step class="rr-readout-val rr-readout-val--big">
                  0
                </span>
                <span class="rr-readout-eq">/</span>
                <span data-rr-step-total class="rr-readout-val">
                  {initialP - 1}
                </span>
              </dd>
            </div>
          </dl>
        </div>
      </div>

      <div class="rr-controls" role="group" aria-label="Ring rotation controls">
        <div class="rr-control rr-control--slider">
          <label class="rr-label" for="rr-device-slider">
            devices p
          </label>
          <input
            id="rr-device-slider"
            class="rr-slider"
            type="range"
            min={MIN_DEVICES}
            max={MAX_DEVICES}
            value={initialP}
            step="1"
            data-rr-devices
            aria-valuemin={MIN_DEVICES}
            aria-valuemax={MAX_DEVICES}
            aria-valuenow={initialP}
          />
          <span class="rr-slider-value" data-rr-devices-value>
            {initialP}
          </span>
        </div>

        <div class="rr-control rr-control--buttons">
          <button
            type="button"
            class="rr-btn"
            data-rr-step-btn
            aria-label="Advance one rotation step"
          >
            step
          </button>
          <button
            type="button"
            class="rr-btn rr-btn--primary"
            data-rr-play
            aria-label="Play full rotation"
          >
            <span data-rr-play-label>play</span>
          </button>
          <button type="button" class="rr-btn" data-rr-reset aria-label="Reset to initial state">
            reset
          </button>
        </div>
      </div>

      <p class="rr-intuition">
        Each device only ever holds <MathLabel tex="L/p" /> tokens of cache, but eventually sees all
        slices through circulation. Overlapping communication with compute hides the network cost
        and the result stays <strong>exact</strong>, unlike sparse or windowed approximations.
      </p>

      {caption ? (
        <figcaption class="rr-caption">
          <MathText text={caption} mathClass="rr-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const MathLabel: FunctionalComponent<{ tex: string }> = ({ tex }) => (
  <span class="rr-math" dangerouslySetInnerHTML={{ __html: renderMath(tex) }} />
)

const RingRotationComponent = RingRotationImpl as QuartzMdxComponent<Props>
RingRotationComponent.css = style
RingRotationComponent.afterDOMLoaded = script

export const RingRotation = registerMdxComponent('RingRotation', RingRotationComponent)

export default (() => RingRotation) satisfies (opts: undefined) => QuartzMdxComponent<Props>
