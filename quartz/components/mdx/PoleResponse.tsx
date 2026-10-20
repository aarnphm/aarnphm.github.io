import katex from 'katex'
import { type ComponentChildren, type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { renderFigureLabel } from '../../util/figure-label'
import { MathText } from '../../util/math-text'
import {
  type Attrs,
  type Case,
  type Comp,
  type Lock,
  type Scale,
  type TimeState,
  CASES,
  DESIGN_ZETA,
  FO_H,
  FO_W,
  K_MAX,
  LOCUS_ROWS,
  LOCUS_SCALE,
  PLANE_X,
  PLANE_Y,
  PLOT_W,
  PLOT_X,
  PLOT_Y,
  TIME_ROWS,
  TIME_SCALE,
  compPoles,
  compZeros,
  cy,
  designGain,
  designPoint,
  locusBranches,
  locusStep,
  locusView,
  planeH,
  planeViewW,
  planeW,
  plotViewW,
  sx,
  sy,
  timeView,
  tx,
  viewH,
} from '../scripts/pole-response-model'
//@ts-ignore
import script from '../scripts/pole-response.inline'
import style from '../styles/poleResponse.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = {
  caption?: string
  mode?: 'first' | 'second' | 'locus'
  /** first order: pole at s = -a */
  a?: number
  /** second order: start from a course case, or from the upper pole re + j im */
  preset?: Case
  re?: number
  im?: number
  lock?: Lock
  locks?: boolean
  cases?: boolean
  regions?: boolean
  /** root locus */
  comp?: Comp
  compensators?: boolean
  k?: number
}

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

const Label: FunctionalComponent<{
  el?: string
  tex: string
  attrs?: Attrs
  end?: boolean
  w?: number
}> = ({ el, tex, attrs, end, w = FO_W }) => (
  <foreignObject data-prs-el={el} width={w} height={FO_H} {...attrs}>
    <div
      class={`prs-fo${end ? ' prs-fo--end' : ''}`}
      dangerouslySetInnerHTML={{ __html: renderMath(tex) }}
    />
  </foreignObject>
)

const Mark: FunctionalComponent<{ cls: string; attrs: Attrs; el?: string }> = ({
  cls,
  attrs,
  el,
}) => (
  <g class={cls} data-prs-el={el} {...attrs}>
    <circle class="prs-pole" r="6" />
    <path class="prs-x" d="M-3 -3L3 3M-3 3L3 -3" />
  </g>
)

const Cross: FunctionalComponent<{ x: number; y: number }> = ({ x, y }) => (
  <path
    class="prs-open"
    d={`M${x - 4} ${y - 4}L${x + 4} ${y + 4}M${x - 4} ${y + 4}L${x + 4} ${y - 4}`}
  />
)

const ticks = (lo: number, hi: number, step: number) => {
  const out: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step)
    out.push(Math.round(v * 1e6) / 1e6)
  return out
}
const tickText = (v: number) => (v < 0 ? `−${-v}` : String(v))

const Plane: FunctionalComponent<{ s: Scale; clip: string; regions?: boolean; label: string }> = ({
  s,
  clip,
  regions,
  label,
  children,
}) => {
  const w = planeW(s)
  const h = planeH(s)
  const ox = sx(s, 0)
  const oy = sy(s, 0)
  const bottom = PLANE_Y + h
  return (
    <svg
      class="prs-plane"
      viewBox={`0 0 ${planeViewW(s)} ${viewH(s)}`}
      role="group"
      aria-label={label}
      data-prs-plane
    >
      <defs>
        <clipPath id={clip}>
          <rect x={PLANE_X} y={PLANE_Y} width={w} height={h} />
        </clipPath>
      </defs>
      {regions ? (
        <g class="prs-regions" aria-hidden="true">
          <rect
            class="prs-region--stable"
            x={PLANE_X}
            y={PLANE_Y}
            width={ox - PLANE_X}
            height={h}
          />
          <rect
            class="prs-region--unstable"
            x={ox}
            y={PLANE_Y}
            width={PLANE_X + w - ox}
            height={h}
          />
        </g>
      ) : null}
      <g class="prs-grid" aria-hidden="true">
        {ticks(s.re[0], s.re[1], 1).map(v => (
          <line x1={sx(s, v)} x2={sx(s, v)} y1={PLANE_Y} y2={bottom} />
        ))}
        {ticks(s.im[0], s.im[1], 1).map(v => (
          <line x1={PLANE_X} x2={PLANE_X + w} y1={sy(s, v)} y2={sy(s, v)} />
        ))}
      </g>
      <g class="prs-tick" aria-hidden="true">
        {ticks(s.re[0], s.re[1], 2).map(v => (
          <text x={sx(s, v)} y={bottom + 12} text-anchor="middle">
            {tickText(v)}
          </text>
        ))}
        {ticks(s.im[0], s.im[1], 2).map(v => (
          <text x={PLANE_X - 4} y={sy(s, v) + 3} text-anchor="end">
            {tickText(v)}
          </text>
        ))}
      </g>
      <line
        class={`prs-axis${regions ? ' prs-axis--region' : ''}`}
        x1={PLANE_X}
        x2={ox}
        y1={oy}
        y2={oy}
      />
      <line class="prs-axis" x1={ox} x2={PLANE_X + w} y1={oy} y2={oy} />
      <line
        class={`prs-axis${regions ? ' prs-axis--region' : ''}`}
        x1={ox}
        x2={ox}
        y1={PLANE_Y}
        y2={bottom}
      />
      {regions ? (
        <g class="prs-region-label" aria-hidden="true">
          <text x={sx(s, -4.5)} y={sy(s, 5.6)} text-anchor="middle">
            underdamped
          </text>
          <text x={sx(s, -4.5)} y={sy(s, -5.4)} text-anchor="middle">
            underdamped
          </text>
          <text x={PLANE_X + 4} y={oy - 5}>
            overdamped, critical
          </text>
          <text x={ox - 4} y={sy(s, 6.4)} text-anchor="end">
            undamped
          </text>
          <text x={(ox + PLANE_X + w) / 2} y={sy(s, 5.6)} text-anchor="middle">
            unstable
          </text>
        </g>
      ) : null}
      <rect class="prs-frame" x={PLANE_X} y={PLANE_Y} width={w} height={h} />
      <Label tex="\sigma" attrs={{ x: PLANE_X + w - FO_W - 1, y: oy - FO_H - 1 }} end />
      <Label tex="j\omega" attrs={{ x: ox - FO_W / 2, y: PLANE_Y - FO_H - 3 }} />
      <g clip-path={`url(#${clip})`}>{children}</g>
    </svg>
  )
}

const Plot: FunctionalComponent<{
  s: Scale
  clip: string
  label: string
  outside?: ComponentChildren
}> = ({ s, clip, label, outside, children }) => {
  const h = planeH(s)
  const bottom = PLOT_Y + h
  return (
    <svg
      class="prs-plot"
      viewBox={`0 0 ${plotViewW} ${viewH(s)}`}
      role="img"
      aria-label={label}
      data-prs-plot
    >
      <defs>
        <clipPath id={clip}>
          <rect x={PLOT_X} y={PLOT_Y} width={PLOT_W} height={h} />
        </clipPath>
      </defs>
      <g class="prs-grid" aria-hidden="true">
        {ticks(s.t[0], s.t[1], s.tStep).map(t => (
          <line x1={tx(s, t)} x2={tx(s, t)} y1={PLOT_Y} y2={bottom} />
        ))}
        {ticks(s.c[0], s.c[1], 0.5).map(c => (
          <line x1={PLOT_X} x2={PLOT_X + PLOT_W} y1={cy(s, c)} y2={cy(s, c)} />
        ))}
      </g>
      <rect
        class="prs-band"
        data-prs-el="band"
        x={PLOT_X}
        y={cy(s, 1.02)}
        width={PLOT_W}
        height={cy(s, 0.98) - cy(s, 1.02)}
      />
      <line class="prs-final" x1={PLOT_X} x2={PLOT_X + PLOT_W} y1={cy(s, 1)} y2={cy(s, 1)} />
      <g class="prs-tick" aria-hidden="true">
        {ticks(s.t[0], s.t[1], s.tStep).map(t => (
          <text x={tx(s, t)} y={bottom + 12} text-anchor="middle">
            {String(t)}
          </text>
        ))}
        {ticks(s.c[0], s.c[1], 0.5).map(c => (
          <text x={PLOT_X - 4} y={cy(s, c) + 3} text-anchor="end">
            {String(c)}
          </text>
        ))}
      </g>
      <rect class="prs-frame" x={PLOT_X} y={PLOT_Y} width={PLOT_W} height={h} />
      <g clip-path={`url(#${clip})`}>{children}</g>
      <Label tex="c(t)" attrs={{ x: PLOT_X - FO_W / 2 + 10, y: PLOT_Y - FO_H - 3 }} />
      <Label tex="t\,(\mathrm{s})" attrs={{ x: PLOT_X + PLOT_W - FO_W, y: bottom + 16 }} end />
      {outside}
    </svg>
  )
}

const Readout: FunctionalComponent<{
  rows: readonly string[]
  labels: Record<string, string>
  values: Record<string, string>
}> = ({ rows, labels, values }) => (
  <dl class="prs-readout" aria-live="polite">
    {rows.map(r => (
      <div>
        <dt>
          <span>
            <MathText text={labels[r]} mathClass="prs-math" />
          </span>
        </dt>
        <dd data-prs-out={r}>{values[r]}</dd>
      </div>
    ))}
  </dl>
)

const Segments: FunctionalComponent<{
  legend: string
  name: string
  input: string
  options: { id: string; label: string }[]
  value: string
}> = ({ legend, name, input, options, value }) => (
  <fieldset class="prs-segments">
    <legend class="prs-legend">
      <MathText text={legend} mathClass="prs-math" />
    </legend>
    {options.map(o => (
      <label class="prs-segment">
        <input
          type="radio"
          name={name}
          value={o.id}
          checked={o.id === value}
          {...{ [input]: '' }}
        />
        <span>
          <MathText text={o.label} mathClass="prs-math" />
        </span>
      </label>
    ))}
  </fieldset>
)

const TIME_LABELS: Record<string, string> = {
  case: 'case',
  pole: 'pole $s=-a$',
  tau: '$\\tau=1/a$',
  poles: 'poles',
  denom: 'denominator',
  zeta: '$\\zeta$',
  wn: '$\\omega_n$',
  sigmaD: '$\\sigma_d=\\zeta\\omega_n$',
  wd: '$\\omega_d=\\omega_n\\sqrt{1-\\zeta^2}$',
  tp: '$T_p=\\pi/\\omega_d$',
  os: '$\\%\\mathrm{OS}$',
}
const timeLabels = (first: boolean) => ({
  ...TIME_LABELS,
  tr: first ? '$T_r=\\ln 9/a$' : 'rise $T_r$, 10% to 90%',
  tsRule: first ? '$T_s\\approx4/a$' : '$T_s\\approx4/\\sigma_d$',
  tsBand: first ? '$T_s=-\\ln 0.02/a$' : 'settling $T_s$, 2% band',
})

const LOCKS: { id: Lock; label: string }[] = [
  { id: 'none', label: 'free' },
  { id: 'sigma', label: 'same envelope, $\\sigma_d$' },
  { id: 'omega', label: 'same frequency, $\\omega_d$' },
  { id: 'zeta', label: 'same overshoot, $\\zeta$' },
]

const COMPS: { id: Comp; label: string; loop: string }[] = [
  { id: 'none', label: 'gain $K$', loop: 'KG(s)=\\frac{K}{(s+1)(s+2)(s+10)}' },
  {
    id: 'integrator',
    label: 'integrator $K/s$',
    loop: '\\frac{K}{s}G(s)=\\frac{K}{s(s+1)(s+2)(s+10)}',
  },
  {
    id: 'pi',
    label: 'PI $K(s+0.1)/s$',
    loop: '\\frac{K(s+0.1)}{s}G(s)=\\frac{K(s+0.1)}{s(s+1)(s+2)(s+10)}',
  },
]

const LOCUS_LABELS: Record<string, string> = {
  k: '$K$',
  poles: 'closed-loop poles',
  zeta: 'dominant $\\zeta$',
  angle: 'angle of the loop at $A$',
  ess: 'step error $e(\\infty)$',
  stab: 'stability',
}

let instance = 0

const initialTime = (p: Props): TimeState => {
  if (p.mode === 'first') return { kind: 'first', p: -(p.a ?? 1) }
  if (p.re !== undefined || p.im !== undefined) {
    const re = p.re ?? -1
    const im = p.im ?? 0
    return im > 0 ? { kind: 'pair', re, im } : { kind: 'real', p1: re, p2: re }
  }
  return (CASES.find(c => c.id === p.preset) ?? CASES[2]).state
}

const TimeFigure: FunctionalComponent<Props & { id: string }> = props => {
  const { id, caption, locks, cases, regions } = props
  const s = TIME_SCALE
  const first = props.mode === 'first'
  const st = initialTime(props)
  const lock: Lock = first || st.kind !== 'pair' ? 'none' : (props.lock ?? 'none')
  const v = timeView(s, st, lock, st)
  const a = v.attrs
  const handle = (n: 0 | 1) => (
    <g
      class="prs-handle"
      data-prs-el={`h${n}`}
      data-prs-handle={String(n)}
      tabindex={0}
      role="slider"
      aria-label={n === 0 ? (first ? 'pole' : 'upper pole') : 'second real pole'}
      aria-valuetext={v.aria}
      {...a[`h${n}`]}
    >
      <circle class="prs-hit" r="14" />
      <circle class="prs-ring" r="9.5" />
      <circle class="prs-pole" r="6" />
      <path class="prs-x" d="M-3 -3L3 3M-3 3L3 -3" />
    </g>
  )
  return (
    <figure
      class="pole-response"
      data-pole-response
      data-prs-mode={first ? 'first' : 'second'}
      data-prs-state={JSON.stringify(st)}
      data-prs-lock={lock}
      data-prs-case={v.cls}
    >
      {cases || locks ? (
        <div class="prs-controls">
          {cases ? (
            <Segments
              legend="Case"
              name={`${id}-case`}
              input="data-prs-case-input"
              options={CASES.map(c => ({ id: c.id, label: c.label }))}
              value={v.cls}
            />
          ) : null}
          {locks ? (
            <Segments
              legend="Hold fixed"
              name={`${id}-lock`}
              input="data-prs-lock-input"
              options={LOCKS}
              value={lock}
            />
          ) : null}
        </div>
      ) : null}
      <div class="prs-panels">
        <Plane
          s={s}
          clip={`${id}-plane`}
          regions={regions}
          label="s-plane; drag a pole or use the arrow keys"
        >
          <line class="prs-guide" data-prs-el="guide0" {...a.guide0} />
          <line class="prs-guide" data-prs-el="guide1" {...a.guide1} />
          <line class="prs-geo" data-prs-el="radius" {...a.radius} />
          <line class="prs-proj" data-prs-el="projRe" {...a.projRe} />
          <line class="prs-proj" data-prs-el="projIm" {...a.projIm} />
          <path class="prs-arc" data-prs-el="arc" {...a.arc} />
          <Label el="lSig" tex="-\sigma_d" attrs={a.lSig} />
          <Label el="lWd" tex="j\omega_d" attrs={a.lWd} />
          <Label el="lWn" tex="\omega_n" attrs={a.lWn} />
          <Label el="lTh" tex="\theta" attrs={a.lTh} />
          <Mark cls="prs-mark prs-mark--ref" el="ref0" attrs={a.ref0} />
          <Mark cls="prs-mark prs-mark--ref" el="ref1" attrs={a.ref1} />
          <Mark cls="prs-mark" el="conj" attrs={a.conj} />
          {handle(1)}
          {handle(0)}
        </Plane>
        <Plot
          s={s}
          clip={`${id}-plot`}
          label={`Unit-step response; ${v.aria}`}
          outside={
            <>
              <Label el="lTr" tex="T_r" attrs={a.lTr} />
              <Label el="lTp" tex="T_p" attrs={a.lTp} />
              <Label el="lTs" tex="T_s" attrs={a.lTs} />
              <Label el="lTau" tex="\tau" attrs={a.lTau} />
            </>
          }
        >
          <rect class="prs-rise" data-prs-el="rise" {...a.rise} />
          <path class="prs-env" data-prs-el="env0" {...a.env0} />
          <path class="prs-env" data-prs-el="env1" {...a.env1} />
          <path class="prs-ref-curve" data-prs-el="refCurve" {...a.refCurve} />
          <line class="prs-drop" data-prs-el="slope" {...a.slope} />
          <line class="prs-drop" data-prs-el="tauH" {...a.tauH} />
          <line class="prs-drop" data-prs-el="tauV" {...a.tauV} />
          <line class="prs-drop" data-prs-el="peakH" {...a.peakH} />
          <line class="prs-drop" data-prs-el="peakV" {...a.peakV} />
          <line class="prs-drop" data-prs-el="tsV" {...a.tsV} />
          <path class="prs-curve" data-prs-el="curve" {...a.curve} />
          <circle class="prs-dot" data-prs-el="tau" r="3" {...a.tau} />
          <circle class="prs-dot" data-prs-el="peak" r="3" {...a.peak} />
          <circle class="prs-dot" data-prs-el="ts" r="3" {...a.ts} />
        </Plot>
      </div>
      <Readout
        rows={first ? TIME_ROWS.first : TIME_ROWS.second}
        labels={timeLabels(first)}
        values={v.readout}
      />
      {caption ? (
        <figcaption class="prs-caption">
          <MathText text={caption} mathClass="prs-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const LocusFigure: FunctionalComponent<Props & { id: string }> = props => {
  const { id, caption, compensators } = props
  const s = LOCUS_SCALE
  const comp: Comp = props.comp ?? 'none'
  const kDesign = designGain('none')
  const k = Math.min(K_MAX, Math.max(0, props.k ?? kDesign))
  const v = locusView(s, comp, k)
  const A = designPoint()
  const ax = sx(s, A[0])
  const ay = sy(s, A[1])
  const ox = sx(s, 0)
  const oy = sy(s, 0)
  const ray = (sign: number) => {
    const len = Math.hypot(A[0], A[1])
    const ux = A[0] / len
    const uy = A[1] / len
    const kk = Math.min(s.re[0] / ux, s.im[1] / uy)
    return { x1: ox, y1: oy, x2: sx(s, ux * kk), y2: sy(s, sign * uy * kk) }
  }
  const refStep = locusStep('none', kDesign)
  let refD = ''
  for (let i = 0; i <= 320; i++) {
    const t = s.t[0] + ((s.t[1] - s.t[0]) * i) / 320
    refD += `${i ? 'L' : 'M'}${tx(s, t)} ${cy(s, Math.max(-2, Math.min(4, refStep(t))))}`
  }
  return (
    <figure
      class="pole-response"
      data-pole-response
      data-prs-mode="locus"
      data-prs-comp={comp}
      data-prs-k={String(k)}
    >
      <div class="prs-controls">
        {compensators ? (
          <Segments
            legend="Compensator"
            name={`${id}-comp`}
            input="data-prs-comp-input"
            options={COMPS.map(c => ({ id: c.id, label: c.label }))}
            value={comp}
          />
        ) : null}
        <div class="prs-gain">
          <label class="prs-gain-label" for={`${id}-k`}>
            <MathText text="gain $K$" mathClass="prs-math" />
          </label>
          <input
            id={`${id}-k`}
            class="prs-slider"
            type="range"
            min="0"
            max={String(K_MAX)}
            step="0.1"
            value={String(Math.round(k * 10) / 10)}
            aria-valuetext={v.aria}
            data-prs-k-input
          />
          <button type="button" class="prs-button" data-prs-design>
            <MathText text={`set $K$ for $\\zeta=${DESIGN_ZETA}$`} mathClass="prs-math" />
          </button>
        </div>
      </div>
      {COMPS.map(c => (
        <p class="prs-loop" data-prs-loop={c.id}>
          <span
            class="prs-math"
            dangerouslySetInnerHTML={{ __html: renderMath(`\\displaystyle ${c.loop}`) }}
          />
        </p>
      ))}
      <div class="prs-panels">
        <Plane s={s} clip={`${id}-plane`} label="s-plane; drag along the locus to set the gain">
          <line class="prs-guide" {...ray(1)} />
          <line class="prs-guide" {...ray(-1)} />
          {COMPS.map(c => (
            <g class="prs-locus" data-prs-locus={c.id}>
              {locusBranches(c.id).map(b => (
                <path
                  class="prs-branch"
                  d={b.map((p, i) => `${i ? 'L' : 'M'}${sx(s, p.re)} ${sy(s, p.im)}`).join('')}
                />
              ))}
              {[...compPoles(c.id), ...compZeros(c.id)].map(p => (
                <line class="prs-angle" x1={sx(s, p)} y1={oy} x2={ax} y2={ay} />
              ))}
            </g>
          ))}
          {v.poles.map((p, i) => (
            <circle class="prs-closed" data-prs-cl={String(i)} r="4.5" {...p} />
          ))}
          {/* Open-loop marks sit above the closed-loop dots so a pole that starts there stays readable. */}
          {COMPS.map(c => (
            <g class="prs-locus" data-prs-locus={c.id}>
              {compPoles(c.id).map(p => (
                <Cross x={sx(s, p)} y={oy} />
              ))}
              {compZeros(c.id).map(z => (
                <circle class="prs-zero" cx={sx(s, z)} cy={oy} r="3.5" />
              ))}
            </g>
          ))}
          <rect class="prs-point" x={ax - 3} y={ay - 3} width="6" height="6" />
          <Label tex="A" attrs={{ x: ax - FO_W - 2, y: ay - FO_H - 2 }} end />
          <Label
            tex={`\\zeta=${DESIGN_ZETA}`}
            w={60}
            attrs={{ x: sx(s, -1.4) - 62, y: sy(s, 6.6) }}
            end
          />
        </Plane>
        <Plot s={s} clip={`${id}-plot`} label={`Closed-loop unit-step response; ${v.aria}`}>
          <path class="prs-ref-curve" d={refD} />
          <path class="prs-curve" data-prs-el="curve" d={v.curve} />
        </Plot>
      </div>
      <Readout rows={LOCUS_ROWS} labels={LOCUS_LABELS} values={v.readout} />
      {caption ? (
        <figcaption class="prs-caption">
          <MathText text={caption} mathClass="prs-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const PoleResponseImpl: QuartzMdxComponent<Props> = props => {
  const id = `prs-${++instance}`
  return props.mode === 'locus' ? (
    <LocusFigure {...props} id={id} />
  ) : (
    <TimeFigure {...props} id={id} />
  )
}

const PoleResponseComponent = PoleResponseImpl as QuartzMdxComponent<Props>
PoleResponseComponent.css = style
PoleResponseComponent.afterDOMLoaded = script

export const PoleResponse = registerMdxComponent('PoleResponse', PoleResponseComponent)

export default (() => PoleResponse) satisfies (opts: undefined) => QuartzMdxComponent<Props>
