import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
import {
  mfaDefaultM,
  mfaDefaultR,
  mfaL,
  mfaMaxM,
  mfaMaxR,
  mfaPalette,
  mfaStateError,
} from '../../util/mfa-factor-model'
//@ts-ignore
import script from '../scripts/mfa-factor-bases.inline'
import style from '../styles/mfaFactorBases.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

const L = mfaL
const MAX_M = mfaMaxM
const MAX_R = mfaMaxR
const DEFAULT_M = mfaDefaultM
const DEFAULT_R = mfaDefaultR
const PALETTE = mfaPalette

const VIEW_W = 560
const VIEW_H = 390
const CELL = 22
const GRID = L * CELL
const LEFT_X = 64
const RIGHT_X = LEFT_X + GRID + 92
const FROW_Y = 60 + GRID + 36
const FCELL = 10
const FGRID = L * FCELL
const STRIPE_W = FGRID + 14
const STRIPE_OFFSET = LEFT_X + (RIGHT_X - LEFT_X + GRID - MAX_M * STRIPE_W) / 2

const renderMath = (tex: string, display = false): string =>
  katex.renderToString(tex, {
    ...katexOptions,
    displayMode: display,
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
      class={`mfa-fo ${cls ?? ''}`.trim()}
      dangerouslySetInnerHTML={{ __html: renderMath(tex) }}
    />
  </foreignObject>
)

const MathLabel: FunctionalComponent<{ tex: string; display?: boolean }> = ({ tex, display }) => (
  <span
    class={`mfa-math${display ? ' mfa-math--display' : ''}`}
    dangerouslySetInnerHTML={{ __html: renderMath(tex, display) }}
  />
)

const HeatGrid: FunctionalComponent<{ x: number; y: number; cell: number; attr: string }> = ({
  x,
  y,
  cell,
  attr,
}) => (
  <g>
    <rect
      class="mfa-grid-frame"
      x={x - 1}
      y={y - 1}
      width={L * cell + 2}
      height={L * cell + 2}
      rx={2}
    />
    {Array.from({ length: L }, (_, i) =>
      Array.from({ length: L }, (_, j) => (
        <rect
          class="mfa-cell"
          x={x + j * cell}
          y={y + i * cell}
          width={cell}
          height={cell}
          {...{ [attr]: `${i},${j}` }}
        />
      )),
    )}
  </g>
)

const FactorStripe: FunctionalComponent<{ idx: number }> = ({ idx }) => {
  const sx = STRIPE_OFFSET + idx * STRIPE_W
  return (
    <g class="mfa-factor" data-mfa-factor={String(idx)}>
      <rect
        class="mfa-factor-frame"
        x={sx - 1}
        y={FROW_Y - 1}
        width={FGRID + 2}
        height={FGRID + 2}
        rx={2}
      />
      {Array.from({ length: L }, (_, i) =>
        Array.from({ length: L }, (_, j) => (
          <rect
            class="mfa-factor-cell"
            x={sx + j * FCELL}
            y={FROW_Y + i * FCELL}
            width={FCELL}
            height={FCELL}
            data-mfa-factor-cell={`${idx},${i},${j}`}
          />
        )),
      )}
      <foreignObject x={sx} y={FROW_Y + FGRID + 4} width={FGRID} height={16}>
        <div class="mfa-fo mfa-fo--sm mfa-factor-label-fo" data-mfa-factor-tag={String(idx)}>
          <span
            dangerouslySetInnerHTML={{ __html: renderMath(`U_{${idx + 1}}V_{${idx + 1}}^{\\top}`) }}
          />
          <span
            class="mfa-factor-gate"
            data-mfa-factor-gate
            data-mfa-active="false"
            dangerouslySetInnerHTML={{ __html: renderMath('\\;\\text{gate}') }}
          />
        </div>
      </foreignObject>
    </g>
  )
}

type ReadoutState = { m: number; r: number; gateOn: boolean; key: string; active: boolean }

type ReadoutSpec = { labelTex: string; valueTex: (state: ReadoutState) => string }

const readoutKey = (m: number, r: number, gateOn: boolean) => `${m}-${r}-${gateOn ? 1 : 0}`

const READOUT_STATES: ReadoutState[] = []
for (let m = 1; m <= MAX_M; m++) {
  for (let r = 1; r <= MAX_R; r++) {
    for (const gateOn of [false, true]) {
      READOUT_STATES.push({
        m,
        r,
        gateOn,
        key: readoutKey(m, r, gateOn),
        active: m === DEFAULT_M && r === DEFAULT_R && !gateOn,
      })
    }
  }
}

const READOUTS: ReadoutSpec[] = [
  { labelTex: '\\text{factors used}', valueTex: state => `${state.m}\\;\\text{of}\\;${MAX_M}` },
  { labelTex: '\\text{rank per factor}', valueTex: state => `${state.r}\\le ${MAX_R}` },
  {
    labelTex: '\\text{effective rank}',
    valueTex: state => `${Math.min(state.m * state.r, L)}\\,/\\,${L}`,
  },
  {
    labelTex: '\\text{compute}',
    valueTex: state =>
      `\\mathcal{O}(L\\cdot ${state.m}\\cdot ${state.r})=\\mathcal{O}(${L * state.m * state.r})\\;\\text{vs}\\;L^2d_h`,
  },
  {
    labelTex: '\\text{cache/token}',
    valueTex: state => `${state.m * state.r}\\;\\text{scalars}\\;\\text{vs}\\;n_h d_h`,
  },
  {
    labelTex: '\\lVert A-\\hat{A}\\rVert_F',
    valueTex: state => mfaStateError(state.m, state.r, state.gateOn).toFixed(3),
  },
]

const ReadoutValue: FunctionalComponent<{ row: ReadoutSpec }> = ({ row }) => (
  <dd>
    {READOUT_STATES.map(state => (
      <span
        data-mfa-readout-state={state.key}
        data-mfa-active={state.active ? 'true' : 'false'}
        class="mfa-readout-val"
        dangerouslySetInnerHTML={{ __html: renderMath(row.valueTex(state)) }}
      />
    ))}
  </dd>
)

const Slider: FunctionalComponent<{
  id: string
  labelTex: string
  max: number
  value: number
  attr: string
  aria: string
}> = ({ id, labelTex, max, value, attr, aria }) => (
  <div class="mfa-control">
    <label class="mfa-label" for={id}>
      <MathLabel tex={labelTex} />
    </label>
    <input
      id={id}
      class="mfa-slider"
      type="range"
      min="1"
      max={max}
      step="1"
      value={value}
      {...{ [attr]: '' }}
      aria-valuemin={1}
      aria-valuemax={max}
      aria-valuenow={value}
      aria-label={aria}
    />
  </div>
)

const MFAFactorBasesImpl: QuartzMdxComponent<Props> = ({ caption }) => (
  <figure
    class="mfa-factor-bases"
    data-mfa-factor-bases
    data-mfa-l={String(L)}
    data-mfa-max-m={String(MAX_M)}
    data-mfa-max-r={String(MAX_R)}
    data-mfa-palette={PALETTE.join(',')}
  >
    <div class="mfa-stage">
      <svg
        class="mfa-graph"
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label={`Dense L by L attention matrix on the left versus a low-rank sum of ${DEFAULT_M} factor outer products on the right.`}
        data-mfa-canvas
      >
        <>
          <MathFO x={LEFT_X} y={8} w={GRID} h={20} tex="\text{dense } A" cls="mfa-fo--title" />
          <MathFO
            x={LEFT_X + GRID / 2 - 60}
            y={26}
            w={120}
            h={20}
            tex="A = QK^{\top}"
            cls="mfa-fo--sm"
          />
          <HeatGrid x={LEFT_X} y={60} cell={CELL} attr="data-mfa-dense" />
          <MathFO
            x={RIGHT_X}
            y={8}
            w={GRID}
            h={20}
            tex="\text{approximation}"
            cls="mfa-fo--title"
          />
          <MathFO
            x={RIGHT_X + GRID / 2 - 80}
            y={26}
            w={160}
            h={20}
            tex="\hat{A}=\sum_i U_i V_i^{\top}"
            cls="mfa-fo--sm"
          />
          <HeatGrid x={RIGHT_X} y={60} cell={CELL} attr="data-mfa-approx" />
          <MathFO
            x={LEFT_X + GRID / 2 - 72}
            y={FROW_Y - 24}
            w={86}
            h={20}
            tex="\text{factor bases}"
            cls="mfa-fo--factor-label"
          />
          <MathFO
            x={LEFT_X + GRID / 2 + 2}
            y={FROW_Y - 22}
            w={84}
            h={16}
            tex="U_i V_i^{\top}"
            cls="mfa-fo--sm"
          />
          {Array.from({ length: MAX_M }, (_, i) => (
            <FactorStripe idx={i} />
          ))}
        </>
      </svg>

      <aside class="mfa-side" aria-label="Factorisation readout">
        <dl class="mfa-readout" data-mfa-readout>
          {READOUTS.map(r => (
            <div class="mfa-readout-row">
              <dt>
                <MathLabel tex={r.labelTex} />
              </dt>
              <ReadoutValue row={r} />
            </div>
          ))}
        </dl>
        <div class="mfa-equation">
          <MathLabel tex="\text{score}_h \approx \sum_{i=1}^{m}(q_h U_i)(V_i^{\top}k)" display />
        </div>
      </aside>
    </div>

    <div class="mfa-controls" role="group" aria-label="Factor controls">
      <Slider
        id="mfa-m-slider"
        labelTex="\text{factors }m"
        max={MAX_M}
        value={DEFAULT_M}
        attr="data-mfa-m"
        aria="Number of factor bases"
      />
      <Slider
        id="mfa-r-slider"
        labelTex="\text{rank }r"
        max={MAX_R}
        value={DEFAULT_R}
        attr="data-mfa-r"
        aria="Rank per factor"
      />
      <div class="mfa-control mfa-control--toggle">
        <label class="mfa-checkbox">
          <input type="checkbox" data-mfa-gate aria-label="Gate each factor per token" />
          <span class="mfa-checkbox-box" aria-hidden="true" />
          <span class="mfa-checkbox-text">
            <MathLabel tex="\text{factors as gates}" />
          </span>
        </label>
      </div>
    </div>

    <p class="mfa-intuition">
      Heads of attention reuse a small library of relational patterns. Factor those into shared
      low-rank bases <MathLabel tex="U_i V_i^{\top}" />; each head becomes a sparse combination.
      With <MathLabel tex="m=L,\, r=d" /> you recover dense attention exactly. With{' '}
      <MathLabel tex="m \ll L" /> you trade reconstruction for{' '}
      <strong>
        <MathLabel tex="O(Lmr)" /> compute
      </strong>{' '}
      and a cache of{' '}
      <strong>
        <MathLabel tex="mr" /> per token
      </strong>
      .
    </p>

    {caption ? (
      <figcaption class="mfa-caption">
        <MathText text={caption} mathClass="mfa-math" />
      </figcaption>
    ) : null}
  </figure>
)

const MFAFactorBasesComponent = MFAFactorBasesImpl as QuartzMdxComponent<Props>
MFAFactorBasesComponent.css = style
MFAFactorBasesComponent.afterDOMLoaded = script

export const MFAFactorBases = registerMdxComponent('MFAFactorBases', MFAFactorBasesComponent)

export default (() => MFAFactorBases) satisfies (opts: undefined) => QuartzMdxComponent<Props>
