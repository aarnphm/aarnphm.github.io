import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { renderFigureLabel } from '../../util/figure-label'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/gated-sae.inline'
import style from '../styles/gatedSAE.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type View = 'gated' | 'jumprelu'
type Props = { caption?: string; view?: View; theta?: number }

// The shared projection u = W_gate(x - b_dec) and the outputs match the vectors in the Gated SAE
// paper's architecture figure. The scales and magnitude biases are chosen to reproduce them.
const U = [-0.1, 1.1, 1.9, 0.2]
const SCALE = [1, 1, 1.5, 0.5]
const B_MAG = [0, -0.2, -0.45, 0.2]
const N = U.length
const THETA = { min: 0, max: 2.5, step: 0.05 } as const
const THETA_COUNT = Math.round((THETA.max - THETA.min) / THETA.step) + 1

const PI_MAG = U.map((u, i) => SCALE[i] * u + B_MAG[i])
const F_MAG = PI_MAG.map(z => Math.max(0, z))

// Each gate bias is set so feature i switches on exactly when pi_mag,i > theta:
// pi_gate > 0  <=>  u + (b_mag - theta) / e^r > 0  <=>  e^r u + b_mag > theta.
const encode = (theta: number) => {
  const bGate = B_MAG.map((b, i) => (b - theta) / SCALE[i])
  const piGate = U.map((u, i) => u + bGate[i])
  const fGate = piGate.map(p => (p > 0 ? 1 : 0))
  const fTilde = F_MAG.map((f, i) => fGate[i] * f)
  return {
    bGate,
    piGate,
    fGate,
    fTilde,
    l0: fTilde.filter(f => f > 0).length,
    l1: fTilde.reduce((a, b) => a + b, 0),
  }
}

const num = (x: number) => {
  const s = (Math.abs(x) < 0.005 ? 0 : x).toFixed(2)
  return s.startsWith('-') ? `−${s.slice(1)}` : s
}

// Plot frame for the JumpReLU view: z and the output share one scale.
const P = { w: 300, h: 260, x0: 34, x1: 290, y0: 226, y1: 12, lo: -0.6, hi: 3 }
const r1 = (n: number) => Math.round(n * 10) / 10
const px = (z: number) => r1(P.x0 + ((z - P.lo) / (P.hi - P.lo)) * (P.x1 - P.x0))
const py = (y: number) => r1(P.y0 - ((y - P.lo) / (P.hi - P.lo)) * (P.y0 - P.y1))

const STATES = Array.from({ length: THETA_COUNT }, (_, k) => {
  const theta = THETA.min + k * THETA.step
  const e = encode(theta)
  return {
    theta: theta.toFixed(2),
    bGate: e.bGate.map(num),
    piGate: e.piGate.map(num),
    fGate: e.fGate,
    fTilde: e.fTilde.map(num),
    l0: `${e.l0} of ${N}`,
    l1: num(e.l1),
    active:
      e.fTilde
        .map((f, i) => (f > 0 ? String(i + 1) : ''))
        .filter(Boolean)
        .join(', ') || 'none',
    zero: `M${px(P.lo)} ${py(0)}H${px(theta)}`,
    ramp: `M${px(theta)} ${py(theta)}L${px(P.hi)} ${py(P.hi)}`,
    tx: px(theta),
    ty: py(theta),
    pointY: e.fTilde.map(f => py(f)),
    valuetext: `theta ${theta.toFixed(2)}, ${e.l0} of ${N} features active`,
  }
})
type State = (typeof STATES)[number]

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
  <span class="gsae-math" dangerouslySetInnerHTML={{ __html: renderMath(tex) }} />
)

const MathFO: FunctionalComponent<{ x: number; y: number; w: number; h?: number; tex: string }> = ({
  x,
  y,
  w,
  h = 18,
  tex,
}) => (
  <foreignObject x={x} y={y} width={w} height={h}>
    <div class="gsae-fo" dangerouslySetInnerHTML={{ __html: renderMath(tex) }} />
  </foreignObject>
)

// Gated view geometry: one column of N cells per vector, two lanes joined at the product.
const CW = 40
const RH = 18
const COL_H = N * RH
const MAG_Y = 30
const GATE_Y = 152
const MID_Y = (MAG_Y + GATE_Y + COL_H) / 2
const U_X = 26
const SPLIT_X = 84
const SCALE_X = 104
const BMAG_X = 148
const BOX_PAD = 4
const PI_X = 222
const OP_X = 280
const OP_W = 46
const F_X = 346
const PROD_X = 420
const PROD_R = 9
const FT_X = 446
const DEC_X = 508
const DEC_W = 44
const G_W = 560
const G_H = GATE_Y + COL_H + 26
const laneMid = (y: number) => y + COL_H / 2

const Column: FunctionalComponent<{
  x: number
  y: number
  values: string[]
  kind?: string
  data?: string
  states?: (string | number)[]
}> = ({ x, y, values, kind = 'value', data, states }) => (
  <g class={`gsae-col gsae-col--${kind}`}>
    {values.map((v, i) => (
      <g>
        <rect
          class="gsae-cell"
          x={x}
          y={y + i * RH}
          width={CW}
          height={RH}
          data-gsae-on={states ? String(states[i] ? 1 : 0) : undefined}
          data-gsae-cell={data ? `${data}-${i}` : undefined}
        />
        <text
          class="gsae-num"
          x={x + CW / 2}
          y={y + i * RH + 12.5}
          text-anchor="middle"
          data-gsae-text={data ? `${data}-${i}` : undefined}
        >
          {v}
        </text>
      </g>
    ))}
  </g>
)

const OpBox: FunctionalComponent<{ y: number; tex: string }> = ({ y, tex }) => (
  <g>
    <rect class="gsae-op" x={OP_X} y={laneMid(y) - 12} width={OP_W} height={24} />
    <MathFO x={OP_X} y={laneMid(y) - 9} w={OP_W} tex={tex} />
  </g>
)

const GatedDiagram: FunctionalComponent<{ s: State }> = ({ s }) => {
  const uY = MID_Y - COL_H / 2
  const magMid = laneMid(MAG_Y)
  const gateMid = laneMid(GATE_Y)
  const blockR = BMAG_X + CW + BOX_PAD
  const wire = [
    `M${U_X + CW} ${MID_Y}H${SPLIT_X}`,
    `M${SPLIT_X} ${magMid}V${gateMid}`,
    `M${SPLIT_X} ${magMid}H${SCALE_X - BOX_PAD}`,
    `M${SPLIT_X} ${gateMid}H${BMAG_X - BOX_PAD}`,
    `M${blockR} ${magMid}H${PI_X}`,
    `M${blockR} ${gateMid}H${PI_X}`,
    `M${PI_X + CW} ${magMid}H${OP_X}`,
    `M${PI_X + CW} ${gateMid}H${OP_X}`,
    `M${OP_X + OP_W} ${magMid}H${F_X}`,
    `M${OP_X + OP_W} ${gateMid}H${F_X}`,
    `M${F_X + CW} ${magMid}H${PROD_X}V${MID_Y - PROD_R}`,
    `M${F_X + CW} ${gateMid}H${PROD_X}V${MID_Y + PROD_R}`,
    `M${PROD_X + PROD_R} ${MID_Y}H${FT_X}`,
    `M${FT_X + CW} ${MID_Y}H${DEC_X}`,
  ].join('')
  return (
    <svg
      class="gsae-svg gsae-svg--gated"
      viewBox={`0 0 ${G_W} ${G_H}`}
      role="img"
      aria-label="Gated SAE encoder on a four-feature example: the magnitude path scales, shifts and rectifies the shared projection; the gating path shifts and binarizes it; their product feeds the decoder."
    >
      <path class="gsae-wire" d={wire} />

      <MathFO x={U_X} y={uY - 20} w={CW} tex="u" />
      {U.map((_, i) => (
        <text class="gsae-index" x={U_X - 6} y={uY + i * RH + 12.5} text-anchor="end">
          {i + 1}
        </text>
      ))}
      <Column x={U_X} y={uY} values={U.map(num)} />

      <text class="gsae-lane" x={SCALE_X - BOX_PAD} y={MAG_Y - 22}>
        magnitude path
      </text>
      <rect
        class="gsae-block"
        x={SCALE_X - BOX_PAD}
        y={MAG_Y - BOX_PAD}
        width={blockR - SCALE_X + BOX_PAD}
        height={COL_H + 2 * BOX_PAD}
      />
      <MathFO x={SCALE_X} y={MAG_Y - 20} w={CW} tex="e^{r_\text{mag}}" />
      <MathFO x={BMAG_X} y={MAG_Y - 20} w={CW} tex="b_\text{mag}" />
      <Column x={SCALE_X} y={MAG_Y} values={SCALE.map(num)} kind="param" />
      <Column x={BMAG_X} y={MAG_Y} values={B_MAG.map(num)} kind="param" />
      <MathFO x={PI_X - 10} y={MAG_Y - 20} w={CW + 20} tex="\pi_\text{mag}" />
      <Column x={PI_X} y={MAG_Y} values={PI_MAG.map(num)} />
      <OpBox y={MAG_Y} tex="\operatorname{ReLU}" />
      <MathFO x={F_X} y={MAG_Y - 20} w={CW} tex="f_\text{mag}" />
      <Column x={F_X} y={MAG_Y} values={F_MAG.map(num)} />

      <text class="gsae-lane" x={BMAG_X - BOX_PAD} y={GATE_Y - 22}>
        gating path
      </text>
      <rect
        class="gsae-block"
        x={BMAG_X - BOX_PAD}
        y={GATE_Y - BOX_PAD}
        width={CW + 2 * BOX_PAD}
        height={COL_H + 2 * BOX_PAD}
      />
      <MathFO x={BMAG_X - 6} y={GATE_Y - 20} w={CW + 12} tex="b_\text{gate}" />
      <Column x={BMAG_X} y={GATE_Y} values={s.bGate} kind="param" data="bgate" />
      <MathFO x={PI_X - 10} y={GATE_Y - 20} w={CW + 20} tex="\pi_\text{gate}" />
      <Column x={PI_X} y={GATE_Y} values={s.piGate} data="pigate" />
      <OpBox y={GATE_Y} tex="\bullet>0" />
      <MathFO x={F_X} y={GATE_Y - 20} w={CW} tex="f_\text{gate}" />
      <Column
        x={F_X}
        y={GATE_Y}
        values={s.fGate.map(String)}
        kind="gate"
        data="fgate"
        states={s.fGate}
      />

      <circle class="gsae-prod" cx={PROD_X} cy={MID_Y} r={PROD_R} />
      <circle class="gsae-prod-dot" cx={PROD_X} cy={MID_Y} r={2} />
      <MathFO x={FT_X} y={MID_Y - COL_H / 2 - 20} w={CW} tex="\tilde f" />
      <Column
        x={FT_X}
        y={MID_Y - COL_H / 2}
        values={s.fTilde}
        kind="out"
        data="ftilde"
        states={s.fTilde.map(v => (v === '0.00' ? 0 : 1))}
      />
      <rect class="gsae-op" x={DEC_X} y={MID_Y - 14} width={DEC_W} height={28} />
      <MathFO x={DEC_X} y={MID_Y - 9} w={DEC_W} tex="W_\text{dec}" />
    </svg>
  )
}

const JumpPlot: FunctionalComponent<{ s: State }> = ({ s }) => {
  const ticks = [0, 1, 2, 3]
  return (
    <svg
      class="gsae-svg gsae-svg--plot"
      viewBox={`0 0 ${P.w} ${P.h}`}
      role="img"
      aria-label="JumpReLU with threshold theta: zero up to theta, then the identity. The four magnitude preactivations are plotted at their outputs."
    >
      <g class="gsae-grid">
        {ticks.map(t => (
          <line x1={P.x0} x2={P.x1} y1={py(t)} y2={py(t)} />
        ))}
        {ticks.map(t => (
          <line x1={px(t)} x2={px(t)} y1={P.y1} y2={P.y0} />
        ))}
      </g>
      <line class="gsae-axis" x1={P.x0} x2={P.x1} y1={py(0)} y2={py(0)} />
      <line class="gsae-axis" x1={px(0)} x2={px(0)} y1={P.y1} y2={P.y0} />
      <g class="gsae-tick" text-anchor="middle">
        {ticks.map(t => (
          <text x={px(t)} y={P.y0 + 13}>
            {t}
          </text>
        ))}
      </g>
      <g class="gsae-tick" text-anchor="end">
        {ticks.map(t => (
          <text x={P.x0 - 6} y={py(t) + 3.5}>
            {t}
          </text>
        ))}
      </g>
      <MathFO x={(P.x0 + P.x1) / 2 - 30} y={P.h - 17} w={60} tex="\pi_\text{mag}" />
      <line class="gsae-diag" x1={px(P.lo)} y1={py(P.lo)} x2={px(P.hi)} y2={py(P.hi)} />
      <line class="gsae-theta" x1={s.tx} x2={s.tx} y1={P.y0} y2={P.y1} data-gsae-theta-line />
      <path class="gsae-curve" d={s.zero} data-gsae-zero />
      <path class="gsae-curve" d={s.ramp} data-gsae-ramp />
      <circle class="gsae-end gsae-end--closed" cx={s.tx} cy={py(0)} r="3" data-gsae-closed />
      <circle class="gsae-end gsae-end--open" cx={s.tx} cy={s.ty} r="3" data-gsae-open />
      {PI_MAG.map((z, i) => (
        <g>
          <circle
            class="gsae-point"
            cx={px(z)}
            cy={s.pointY[i]}
            r="5"
            data-gsae-point={String(i)}
            data-gsae-on={s.fTilde[i] === '0.00' ? '0' : '1'}
          />
          <text
            class="gsae-index"
            x={px(z)}
            y={s.pointY[i] - 9}
            text-anchor="middle"
            data-gsae-point-label={String(i)}
          >
            {i + 1}
          </text>
        </g>
      ))}
    </svg>
  )
}

let instance = 0

const GatedSAEImpl: QuartzMdxComponent<Props> = ({ caption, view = 'gated', theta = 0.35 }) => {
  const id = `gsae-${++instance}`
  const v0: View = view === 'jumprelu' ? 'jumprelu' : 'gated'
  const k0 = Math.min(THETA_COUNT - 1, Math.max(0, Math.round((theta - THETA.min) / THETA.step)))
  const s = STATES[k0]

  return (
    <figure
      class="gated-sae"
      data-gated-sae
      data-gsae-view={v0}
      data-gsae-states={JSON.stringify(STATES)}
    >
      <div class="gsae-controls">
        <label class="gsae-label" for={`${id}-theta`}>
          threshold <MathLabel tex="\theta" />
        </label>
        <input
          id={`${id}-theta`}
          class="gsae-slider"
          type="range"
          min={THETA.min}
          max={THETA.max}
          step={THETA.step}
          value={s.theta}
          aria-valuetext={s.valuetext}
          data-gsae-theta
        />
        <output class="gsae-chip" for={`${id}-theta`} data-gsae-out="theta">
          {s.theta}
        </output>
      </div>
      <p class="gsae-rule">
        <MathText
          text={
            v0 === 'gated'
              ? 'Gate biases $b_\\text{gate}=(b_\\text{mag}-\\theta)\\oslash e^{r_\\text{mag}}$ give every feature the effective threshold $\\theta$.'
              : 'With tied weights, $f_\\text{gate}=\\mathbb{1}[\\pi_\\text{mag}>\\theta]$, so $\\tilde f=\\operatorname{JumpReLU}_\\theta(\\pi_\\text{mag})$.'
          }
          mathClass="gsae-math"
        />
      </p>

      <div class={`gsae-body gsae-body--${v0}`}>
        {v0 === 'gated' ? (
          // Narrow screens scroll the diagram instead of shrinking its numbers below legibility.
          <div class="gsae-scroll" tabindex={0} role="region" aria-label="Gated encoder diagram">
            <GatedDiagram s={s} />
          </div>
        ) : (
          <JumpPlot s={s} />
        )}
        {v0 === 'jumprelu' ? (
          <table class="gsae-table">
            <thead>
              <tr>
                <th scope="col">
                  <MathLabel tex="i" />
                </th>
                <th scope="col">
                  <MathLabel tex="\pi_\text{mag}" />
                </th>
                <th scope="col">
                  <MathLabel tex="f_\text{gate}" />
                </th>
                <th scope="col">
                  <MathLabel tex="\tilde f" />
                </th>
              </tr>
            </thead>
            <tbody>
              {PI_MAG.map((z, i) => (
                <tr>
                  <th scope="row">{i + 1}</th>
                  <td>{num(z)}</td>
                  <td data-gsae-text={`fgate-${i}`}>{s.fGate[i]}</td>
                  <td
                    data-gsae-text={`ftilde-${i}`}
                    data-gsae-cell={`ftilde-${i}`}
                    data-gsae-on={s.fTilde[i] === '0.00' ? '0' : '1'}
                  >
                    {s.fTilde[i]}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </div>

      <dl class="gsae-readout" aria-live="polite">
        <div>
          <dt>
            <span>
              Active features, <MathLabel tex="\|\tilde f\|_0" />
            </span>
          </dt>
          <dd data-gsae-out="l0">{s.l0}</dd>
        </div>
        <div>
          <dt>Which features</dt>
          <dd data-gsae-out="active">{s.active}</dd>
        </div>
        <div>
          <dt>
            <span>
              <MathLabel tex="\|\tilde f\|_1" />
            </span>
          </dt>
          <dd data-gsae-out="l1">{s.l1}</dd>
        </div>
      </dl>

      {caption ? (
        <figcaption class="gsae-caption">
          <MathText text={caption} mathClass="gsae-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const GatedSAEComponent = GatedSAEImpl as QuartzMdxComponent<Props>
GatedSAEComponent.css = style
GatedSAEComponent.afterDOMLoaded = script

export const GatedSAE = registerMdxComponent('GatedSAE', GatedSAEComponent)

export default (() => GatedSAE) satisfies (opts: undefined) => QuartzMdxComponent<Props>
