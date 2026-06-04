import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/razor-compression.inline'
import style from '../styles/razorCompression.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

const VIEW_W = 640
const VIEW_H = 252
const TRACK_X = 150
const TRACK_W = 460
const TRACK_H = 34
const ROW1_Y = 56
const ROW2_Y = 140
const SINK_W = 18
const COMP_CX = 380
const COMP_Y = 214
const COMP_W = 84
const COMP_H = 30

const STOPS = [2, 3, 4, 5, 8, 10]
const DEFAULT_C = 5
const P_RET = 0.15
const math = String.raw

const retained = (c: number): number => P_RET + (1 - P_RET) / c
const compression = (c: number): number => 1 / retained(c)

const tex = (t: string, d = false): string =>
  katex.renderToString(t, {
    ...katexOptions,
    displayMode: d,
    output: 'html',
    macros: customMacros,
    strict: false,
    throwOnError: false,
  })

const Fo: FunctionalComponent<{
  x: number
  y: number
  w: number
  h: number
  t: string
  cls?: string
}> = ({ x, y, w, h, t, cls }) => (
  <foreignObject x={x} y={y} width={w} height={h}>
    <div class={`rzc-fo ${cls ?? ''}`.trim()} dangerouslySetInnerHTML={{ __html: tex(t) }} />
  </foreignObject>
)

const T: FunctionalComponent<{ t: string; d?: boolean; cls?: string }> = ({ t, d, cls }) => (
  <span class={cls} dangerouslySetInnerHTML={{ __html: tex(t, d) }} />
)

const rollW0 = TRACK_W / DEFAULT_C
const dropW0 = TRACK_W - SINK_W - rollW0
const dropX0 = TRACK_X + SINK_W
const rollX0 = dropX0 + dropW0

const RazorCompressionImpl: QuartzMdxComponent<Props> = ({ caption }) => (
  <figure
    class="razor-compression"
    data-razor-compression
    data-track-x={TRACK_X}
    data-track-w={TRACK_W}
    data-sink-w={SINK_W}
    data-comp-cx={COMP_CX}
    data-comp-y={COMP_Y}
    data-row2-y={ROW2_Y}
    data-track-h={TRACK_H}
    data-stops={STOPS.join(',')}
    data-pret={P_RET}
  >
    <div class="rzc-stage">
      <svg
        class="rzc-graph"
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="Two attention heads. The retrieval head keeps its full KV cache. The non-retrieval head keeps a few sink tokens and a recent rolling window, drops the remote tokens, and folds them into a single compensation token."
      >
        <defs>
          <pattern
            id="rzc-hatch"
            width="5"
            height="5"
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <line class="rzc-hatch-line" x1="0" y1="0" x2="0" y2="5" />
          </pattern>
          <marker
            id="rzc-comp-arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto-start-reverse"
          >
            <path class="rzc-arrowhead" d="M0,0 L10,5 L0,10 z" />
          </marker>
        </defs>

        <Fo
          x={6}
          y={ROW1_Y + 2}
          w={134}
          h={TRACK_H - 4}
          t={math`\text{retrieval head}`}
          cls="rzc-fo--row"
        />
        <rect
          class="rzc-seg rzc-seg--keep"
          x={TRACK_X}
          y={ROW1_Y}
          width={TRACK_W}
          height={TRACK_H}
          rx={4}
        />
        <Fo
          x={TRACK_X}
          y={ROW1_Y}
          w={TRACK_W}
          h={TRACK_H}
          t={math`\text{full cache — every token kept}`}
          cls="rzc-fo--inseg"
        />

        <Fo
          x={6}
          y={ROW2_Y + 2}
          w={134}
          h={TRACK_H - 4}
          t={math`\text{non-retrieval head}`}
          cls="rzc-fo--row"
        />
        <rect
          class="rzc-seg rzc-seg--sink"
          x={TRACK_X}
          y={ROW2_Y}
          width={SINK_W}
          height={TRACK_H}
          rx={4}
        />
        <g data-rzc-dropped-group>
          <rect
            class="rzc-seg rzc-seg--drop"
            data-rzc-drop
            x={dropX0}
            y={ROW2_Y}
            width={dropW0}
            height={TRACK_H}
            rx={4}
          />
          <rect
            class="rzc-seg-hatch"
            data-rzc-drop-hatch
            x={dropX0}
            y={ROW2_Y}
            width={dropW0}
            height={TRACK_H}
            rx={4}
            fill="url(#rzc-hatch)"
          />
          <foreignObject
            data-rzc-drop-label
            x={dropX0}
            y={ROW2_Y + 8}
            width={dropW0}
            height={TRACK_H - 16}
          >
            <div
              class="rzc-fo rzc-fo--drop"
              dangerouslySetInnerHTML={{ __html: tex(math`\text{dropped}`) }}
            />
          </foreignObject>
        </g>
        <rect
          class="rzc-seg rzc-seg--keep"
          data-rzc-roll
          x={rollX0}
          y={ROW2_Y}
          width={rollW0}
          height={TRACK_H}
          rx={4}
        />

        <path
          class="rzc-comp-arrow"
          data-rzc-comp-arrow
          d={`M ${dropX0 + dropW0 / 2} ${ROW2_Y + TRACK_H} L ${COMP_CX} ${COMP_Y}`}
          marker-end="url(#rzc-comp-arrow)"
        />
        <g class="rzc-comp">
          <rect
            class="rzc-comp-box"
            x={COMP_CX - COMP_W / 2}
            y={COMP_Y}
            width={COMP_W}
            height={COMP_H}
            rx={4}
          />
          <Fo
            x={COMP_CX - COMP_W / 2}
            y={COMP_Y}
            w={COMP_W}
            h={COMP_H}
            t={math`\{\hat{k}, \hat{v}\}`}
            cls="rzc-fo--comp"
          />
        </g>
        <Fo
          x={COMP_CX + COMP_W / 2 + 6}
          y={COMP_Y + 4}
          w={180}
          h={COMP_H - 8}
          t={math`\text{one compensation token}`}
          cls="rzc-fo--compnote"
        />
      </svg>
    </div>

    <div class="rzc-controls">
      <div class="rzc-slider-wrap">
        <label class="rzc-slider-label" for="rzc-buffer-slider">
          <T t={math`\text{buffer } L_h = N/C`} />
        </label>
        <input
          id="rzc-buffer-slider"
          class="rzc-slider"
          type="range"
          min="0"
          max={STOPS.length - 1}
          step="1"
          value={STOPS.indexOf(DEFAULT_C)}
          data-rzc-slider
          aria-valuemin={STOPS[0]}
          aria-valuemax={STOPS[STOPS.length - 1]}
          aria-valuenow={DEFAULT_C}
          aria-valuetext={`C = ${DEFAULT_C}`}
        />
        <div class="rzc-ticks" aria-hidden="true">
          {STOPS.map(c => (
            <span
              class="rzc-tick"
              data-rzc-tick={c}
              data-rzc-active={c === DEFAULT_C ? 'true' : 'false'}
            >
              <T t={`${c}`} />
            </span>
          ))}
        </div>
      </div>

      <dl class="rzc-readout">
        <div class="rzc-readout-row">
          <dt>
            <T t={math`\text{retrieval heads}`} />
          </dt>
          <dd>
            <T t={math`p = 15\%`} />
          </dd>
        </div>
        <div class="rzc-readout-row">
          <dt>
            <T t={math`\text{window } 1/C`} />
          </dt>
          <dd data-rzc-buffer dangerouslySetInnerHTML={{ __html: tex(math`1/${DEFAULT_C}`) }} />
        </div>
        <div class="rzc-readout-row">
          <dt>
            <T t={math`\text{KV retained}`} />
          </dt>
          <dd
            data-rzc-retained
            dangerouslySetInnerHTML={{
              __html: tex(math`p + (1-p)\tfrac{1}{C} = ${retained(DEFAULT_C).toFixed(2)}`),
            }}
          />
        </div>
        <div class="rzc-readout-row rzc-readout-row--hero">
          <dt>
            <T t={math`\text{compression}`} />
          </dt>
          <dd
            data-rzc-compression
            dangerouslySetInnerHTML={{ __html: tex(`${compression(DEFAULT_C).toFixed(2)}\\times`) }}
          />
        </div>
      </dl>
    </div>

    <ul class="rzc-legend" aria-hidden="true">
      <li>
        <span class="rzc-swatch rzc-swatch--keep" /> kept (sink + rolling window)
      </li>
      <li>
        <span class="rzc-swatch rzc-swatch--drop" /> dropped, folded into{' '}
        <T t={math`\{\hat{k}, \hat{v}\}`} />
      </li>
    </ul>

    {caption ? (
      <figcaption class="rzc-caption">
        <MathText text={caption} mathClass="rzc-math" />
      </figcaption>
    ) : null}
  </figure>
)

const RazorCompressionComponent = RazorCompressionImpl as QuartzMdxComponent<Props>
RazorCompressionComponent.css = style
RazorCompressionComponent.afterDOMLoaded = script

export const RazorCompression = registerMdxComponent('RazorCompression', RazorCompressionComponent)

export default (() => RazorCompression) satisfies (opts: undefined) => QuartzMdxComponent<Props>
