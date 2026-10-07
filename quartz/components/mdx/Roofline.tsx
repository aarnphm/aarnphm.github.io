import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/roofline.inline'
import style from '../styles/roofline.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

// peak in TFLOP/s, bandwidth in TB/s. Defaults: H100 SXM, dense BF16, HBM3.
type Props = { caption?: string; peak?: number; bandwidth?: number }

const VIEW_W = 560
const VIEW_H = 262
const X0 = 56
const X1 = 540
const Y0 = 216
const Y1 = 37.3
// log10 domains: intensity 0.1 to 10k FLOP/byte, performance 0.1 to 1k TFLOP/s
const X_DEC = [-1, 4] as const
const Y_DEC = [-1, 3] as const
const BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
const INITIAL = 0
const PREFILL_TOKENS = 2048

const xOf = (i: number) => X0 + ((Math.log10(i) - X_DEC[0]) / (X_DEC[1] - X_DEC[0])) * (X1 - X0)
const yOf = (p: number) => Y0 - ((Math.log10(p) - Y_DEC[0]) / (Y_DEC[1] - Y_DEC[0])) * (Y0 - Y1)
const r1 = (n: number) => Math.round(n * 10) / 10

const fmt = (n: number) => (n < 10 ? n.toFixed(2) : n < 100 ? n.toFixed(1) : n.toFixed(0))
const pct = (n: number) => (n < 1 ? n.toFixed(2) : n < 10 ? n.toFixed(1) : n.toFixed(0))

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
  tex: string
  end?: boolean
}> = ({ x, y, w, tex, end }) => (
  <foreignObject x={x} y={y} width={w} height={22}>
    <div
      class={`rfl-fo${end ? ' rfl-fo--end' : ''}`}
      dangerouslySetInnerHTML={{ __html: renderMath(tex) }}
    />
  </foreignObject>
)

const MathLabel: FunctionalComponent<{ tex: string }> = ({ tex }) => (
  <span class="rfl-math" dangerouslySetInnerHTML={{ __html: renderMath(tex) }} />
)

const RooflineImpl: QuartzMdxComponent<Props> = ({ caption, peak = 989.5, bandwidth = 3.35 }) => {
  const ridge = peak / bandwidth
  const attained = (i: number) => Math.min(peak, bandwidth * i)
  const xr = r1(xOf(ridge))
  const yr = r1(yOf(peak))
  const roof = `M${X0} ${r1(yOf(bandwidth * 10 ** X_DEC[0]))}L${xr} ${yr}H${X1}`
  // Decode in BF16: each 2-byte weight feeds 2b FLOPs across a batch of b, so I = b FLOP/byte.
  const stops = BATCHES.map(b => {
    const p = attained(b)
    return {
      b,
      x: r1(xOf(b)),
      y: r1(yOf(p)),
      p: p === peak ? String(peak) : fmt(p),
      pct: pct((100 * p) / peak),
      bound: b < ridge ? 'memory' : 'compute',
    }
  })
  const first = stops[INITIAL]
  const xp = r1(xOf(PREFILL_TOKENS))
  const xTicks = [0.1, 1, 10, 100, 1000, 10000]
  const yTicks = [0.1, 1, 10, 100, 1000]
  const tick = (n: number) => (n >= 1000 ? `${n / 1000}k` : String(n))
  const ariaFor = (s: (typeof stops)[number]) =>
    `Roofline chart. Decode at batch ${s.b} has intensity ${s.b} FLOP/byte and is ${s.bound}-bound at ${s.p} TFLOP/s; the ridge is at ${Math.round(ridge)} FLOP/byte.`

  return (
    <figure
      class="roofline"
      data-roofline
      data-ridge={String(ridge)}
      data-stops={JSON.stringify(stops)}
    >
      <svg
        class="rfl-graph"
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={ariaFor(first)}
        data-rfl-canvas
      >
        <g class="rfl-grid">
          {yTicks.map(t => (
            <line x1={X0} x2={X1} y1={r1(yOf(t))} y2={r1(yOf(t))} />
          ))}
          {xTicks.map(t => (
            <line x1={r1(xOf(t))} x2={r1(xOf(t))} y1={16} y2={Y0} />
          ))}
        </g>
        <g class="rfl-tick" text-anchor="middle">
          {xTicks.map(t => (
            <text x={r1(xOf(t))} y={Y0 + 16}>
              {tick(t)}
            </text>
          ))}
        </g>
        <g class="rfl-tick" text-anchor="end">
          {yTicks.map(t => (
            <text x={X0 - 8} y={r1(yOf(t)) + 3.5}>
              {tick(t)}
            </text>
          ))}
        </g>
        <text class="rfl-axis" x={(X0 + X1) / 2} y={VIEW_H - 8} text-anchor="middle">
          arithmetic intensity (FLOP/byte, log)
        </text>
        <text class="rfl-axis" x={4} y={10}>
          TFLOP/s, log
        </text>

        <path class="rfl-roof" d={roof} />
        <line class="rfl-ridge" x1={xr} x2={xr} y1={yr} y2={Y0} />
        {/* Labels sit in the empty region above the slope and left of the roof. */}
        <MathFO x={r1(xOf(10)) - 144} y={r1(yOf(bandwidth * 10)) - 36} w={140} tex="P=BI" end />
        <MathFO x={xr - 158} y={6} w={150} tex={`P_{\\max}=${peak}`} end />
        <MathFO
          x={xr + 6}
          y={Y0 - 30}
          w={150}
          tex={`I_{\\mathrm{ridge}}\\approx${Math.round(ridge)}`}
        />

        <g class="rfl-trail" aria-hidden="true">
          {stops.map(s => (
            <circle cx={s.x} cy={s.y} r="2.2" />
          ))}
        </g>

        <rect
          class="rfl-prefill"
          x={xp - 4.5}
          y={r1(yOf(attained(PREFILL_TOKENS))) - 4.5}
          width="9"
          height="9"
        />
        <text
          class="rfl-mark"
          x={xp}
          y={r1(yOf(attained(PREFILL_TOKENS))) - 11}
          text-anchor="middle"
        >
          prefill, 2k tokens
        </text>

        <circle class="rfl-decode" cx={first.x} cy={first.y} r="4.5" data-rfl-point />
        <text class="rfl-mark" x={first.x + 9} y={first.y + 18} data-rfl-point-label>
          decode, batch {first.b}
        </text>
      </svg>

      <div class="rfl-controls">
        <label class="rfl-label" for="rfl-batch">
          decode batch <MathLabel tex="b" />
        </label>
        <input
          id="rfl-batch"
          class="rfl-slider"
          type="range"
          min="0"
          max={stops.length - 1}
          step="1"
          value={INITIAL}
          data-rfl-batch
          aria-valuemin={BATCHES[0]}
          aria-valuemax={BATCHES[BATCHES.length - 1]}
          aria-valuenow={first.b}
          aria-valuetext={`batch ${first.b}, ${first.bound}-bound`}
        />
        <dl class="rfl-readout" aria-live="polite">
          <div>
            <dt>
              <MathLabel tex="b" />
            </dt>
            <dd data-rfl-b>{first.b}</dd>
          </div>
          <div>
            <dt>
              <MathLabel tex="I" /> FLOP/byte
            </dt>
            <dd data-rfl-i>{first.b}</dd>
          </div>
          <div>
            <dt>
              <MathLabel tex="P" /> TFLOP/s
            </dt>
            <dd data-rfl-p>{first.p}</dd>
          </div>
          <div>
            <dt>of peak</dt>
            <dd data-rfl-pct>{first.pct}%</dd>
          </div>
          <div>
            <dt>bound</dt>
            <dd data-rfl-state>{first.bound}</dd>
          </div>
        </dl>
      </div>

      {caption ? (
        <figcaption class="rfl-caption">
          <MathText text={caption} mathClass="rfl-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const RooflineComponent = RooflineImpl as QuartzMdxComponent<Props>
RooflineComponent.css = style
RooflineComponent.afterDOMLoaded = script

export const Roofline = registerMdxComponent('Roofline', RooflineComponent)

export default (() => Roofline) satisfies (opts: undefined) => QuartzMdxComponent<Props>
