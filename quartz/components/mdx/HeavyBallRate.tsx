import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { renderFigureLabel } from '../../util/figure-label'
import { MathText } from '../../util/math-text'
import {
  HB_ALPHA,
  HB_BETA,
  HB_H,
  HB_REFERENCE,
  HB_STEPS,
  HB_W,
  HB_X0,
  HB_X1,
  HB_Y0,
  HB_Y1,
  hbRateX,
  hbRateY,
  hbTrajX,
  hbTrajY,
  hbView,
} from '../scripts/heavy-ball-model'
//@ts-ignore
import script from '../scripts/heavy-ball-rate.inline'
import style from '../styles/heavyBallRate.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; alpha?: number; beta?: number }

const renderMath = (tex: string): string =>
  renderFigureLabel(tex, m =>
    katex.renderToString(m, {
      ...katexOptions,
      displayMode: false,
      output: 'html',
      macros: customMacros,
      strict: false,
      throwOnError: false,
    }),
  )

const MathLabel: FunctionalComponent<{ tex: string }> = ({ tex }) => (
  <span class="hbr-math" dangerouslySetInnerHTML={{ __html: renderMath(tex) }} />
)

const MathFO: FunctionalComponent<{
  x: number
  y: number
  w: number
  tex: string
  end?: boolean
}> = ({ x, y, w, tex, end }) => (
  <foreignObject x={x} y={y} width={w} height={18}>
    <div
      class={`hbr-fo${end ? ' hbr-fo--end' : ''}`}
      dangerouslySetInnerHTML={{ __html: renderMath(tex) }}
    />
  </foreignObject>
)

const snap = (v: number, { min, max, step }: { min: number; max: number; step: number }) =>
  Math.min(max, Math.max(min, Math.round(v / step) * step))

const RATE_TICKS_X = [0, 1, 2, 3, 4]
const RATE_TICKS_Y = [0, 0.5, 1, 1.5, 2, 2.5, 3]
const TRAJ_TICKS_T = [0, 6, 12, 18, 24]
const TRAJ_TICKS_X = [-1, 0, 1]

let instance = 0

const HeavyBallRateImpl: QuartzMdxComponent<Props> = ({ caption, alpha = 0.5, beta = 0 }) => {
  const id = `hbr-${++instance}`
  const a0 = snap(alpha, HB_ALPHA)
  const b0 = snap(beta, HB_BETA)
  const v = hbView(a0, b0)
  const yOne = hbRateY(1)

  return (
    <figure class="heavy-ball-rate" data-heavy-ball-rate>
      <div class="hbr-controls">
        <label class="hbr-label" for={`${id}-alpha`}>
          step size <MathLabel tex="\alpha\lambda" />
        </label>
        <input
          id={`${id}-alpha`}
          class="hbr-slider"
          type="range"
          min={HB_ALPHA.min}
          max={HB_ALPHA.max}
          step={HB_ALPHA.step}
          value={a0.toFixed(2)}
          aria-valuetext={v.valuetextAlpha}
          data-hbr-alpha
        />
        <output class="hbr-chip" for={`${id}-alpha`} data-hbr-out="alpha">
          {v.alpha}
        </output>
        <label class="hbr-label" for={`${id}-beta`}>
          momentum <MathLabel tex="\beta" />
        </label>
        <input
          id={`${id}-beta`}
          class="hbr-slider"
          type="range"
          min={HB_BETA.min}
          max={HB_BETA.max}
          step={HB_BETA.step}
          value={b0.toFixed(2)}
          aria-valuetext={v.valuetextBeta}
          data-hbr-beta
        />
        <output class="hbr-chip" for={`${id}-beta`} data-hbr-out="beta">
          {v.beta}
        </output>
      </div>

      <div class="hbr-plots">
        <div class="hbr-panel">
          <p class="hbr-heading">
            Rate <MathLabel tex="\rho" /> against step size
          </p>
          <svg
            class="hbr-svg"
            viewBox={`0 0 ${HB_W} ${HB_H}`}
            role="img"
            aria-label="Convergence rate of heavy-ball momentum against step size alpha lambda, with the plain gradient-descent curve for reference."
          >
            <rect
              class="hbr-band"
              x={HB_X0}
              y={yOne}
              width={v.bandW}
              height={HB_Y0 - yOne}
              data-hbr-band
            />
            <g class="hbr-grid">
              {RATE_TICKS_Y.map(t => (
                <line x1={HB_X0} x2={HB_X1} y1={hbRateY(t)} y2={hbRateY(t)} />
              ))}
              {RATE_TICKS_X.map(t => (
                <line x1={hbRateX(t)} x2={hbRateX(t)} y1={HB_Y1} y2={HB_Y0} />
              ))}
            </g>
            <line class="hbr-one" x1={HB_X0} x2={HB_X1} y1={yOne} y2={yOne} />
            <g class="hbr-tick" text-anchor="middle">
              {RATE_TICKS_X.map(t => (
                <text x={hbRateX(t)} y={HB_Y0 + 13}>
                  {t}
                </text>
              ))}
            </g>
            <g class="hbr-tick" text-anchor="end">
              {RATE_TICKS_Y.map(t => (
                <text x={HB_X0 - 6} y={hbRateY(t) + 3.5}>
                  {t}
                </text>
              ))}
            </g>
            <MathFO x={(HB_X0 + HB_X1) / 2 - 20} y={HB_H - 16} w={40} tex="\alpha\lambda" />
            <path class="hbr-reference" d={HB_REFERENCE} />
            <MathFO x={hbRateX(3.3) - 44} y={hbRateY(2.4) - 9} w={44} tex="\beta=0" end />
            <path class="hbr-curve" d={v.curve} data-hbr-curve />
            <line
              class="hbr-guide"
              x1={v.markX}
              x2={v.markX}
              y1={HB_Y0}
              y2={v.markY}
              data-hbr-guide
            />
            <circle class="hbr-mark" cx={v.markX} cy={v.markY} r="4.5" data-hbr-mark />
          </svg>
        </div>

        <div class="hbr-panel">
          <p class="hbr-heading">
            Iterates <MathLabel tex="x_t" /> from rest at <MathLabel tex="x_0=1" />
          </p>
          <svg
            class="hbr-svg"
            viewBox={`0 0 ${HB_W} ${HB_H}`}
            role="img"
            aria-label={`First ${HB_STEPS} heavy-ball iterates for the selected step size and momentum.`}
          >
            <defs>
              <clipPath id={`${id}-clip`}>
                <rect x={HB_X0 - 4} y={HB_Y1} width={HB_X1 - HB_X0 + 8} height={HB_Y0 - HB_Y1} />
              </clipPath>
            </defs>
            <g class="hbr-grid">
              {TRAJ_TICKS_X.map(t => (
                <line x1={HB_X0} x2={HB_X1} y1={hbTrajY(t)} y2={hbTrajY(t)} />
              ))}
              {TRAJ_TICKS_T.map(t => (
                <line x1={hbTrajX(t)} x2={hbTrajX(t)} y1={HB_Y1} y2={HB_Y0} />
              ))}
            </g>
            <line class="hbr-one" x1={HB_X0} x2={HB_X1} y1={hbTrajY(0)} y2={hbTrajY(0)} />
            <g class="hbr-tick" text-anchor="middle">
              {TRAJ_TICKS_T.map(t => (
                <text x={hbTrajX(t)} y={HB_Y0 + 13}>
                  {t}
                </text>
              ))}
            </g>
            <g class="hbr-tick" text-anchor="end">
              {TRAJ_TICKS_X.map(t => (
                <text x={HB_X0 - 6} y={hbTrajY(t) + 3.5}>
                  {t}
                </text>
              ))}
            </g>
            <MathFO x={(HB_X0 + HB_X1) / 2 - 20} y={HB_H - 16} w={40} tex="t" />
            <g clip-path={`url(#${id}-clip)`}>
              <path class="hbr-traj" d={v.traj} data-hbr-traj />
              {v.points.map(p => (
                <circle class="hbr-point" cx={p.x} cy={p.y} r="2.4" data-hbr-point />
              ))}
            </g>
          </svg>
        </div>
      </div>

      <dl class="hbr-readout" aria-live="polite">
        <div>
          <dt>
            <span>
              Rate <MathLabel tex="\rho=\max|r|" />
            </span>
          </dt>
          <dd data-hbr-out="rho">{v.rho}</dd>
        </div>
        <div>
          <dt>
            <span>
              Roots of <MathLabel tex="r^2-(1+\beta-\alpha\lambda)r+\beta" />
            </span>
          </dt>
          <dd data-hbr-out="roots">{v.roots}</dd>
        </div>
        <div>
          <dt>
            <span>
              Steps until <MathLabel tex="|x_t|" /> stays below <MathLabel tex="10^{-3}" />
            </span>
          </dt>
          <dd data-hbr-out="settle">{v.settle}</dd>
        </div>
      </dl>

      {caption ? (
        <figcaption class="hbr-caption">
          <MathText text={caption} mathClass="hbr-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const HeavyBallRateComponent = HeavyBallRateImpl as QuartzMdxComponent<Props>
HeavyBallRateComponent.css = style
HeavyBallRateComponent.afterDOMLoaded = script

export const HeavyBallRate = registerMdxComponent('HeavyBallRate', HeavyBallRateComponent)

export default (() => HeavyBallRate) satisfies (opts: undefined) => QuartzMdxComponent<Props>
