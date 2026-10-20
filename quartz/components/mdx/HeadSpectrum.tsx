import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/head-spectrum.inline'
import style from '../styles/headSpectrum.scss'
import data from './qwen3-head-spectra.json'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type PromptId = 'plain' | 'comma'
type Props = { caption?: string; layer?: number; head?: number; prompt?: PromptId; k?: number }

type HeadData = (typeof data.heads)[number]

const PROMPTS: readonly { id: PromptId; label: string }[] = [
  { id: 'plain', label: 'no comma' },
  { id: 'comma', label: 'comma' },
]
const PROMPT_INDEX: Record<PromptId, number> = { plain: 0, comma: 1 }

const HEADS = [...data.heads].sort((a, b) => a.layer - b.layer || a.head - b.head)
const headId = (h: HeadData) => `${h.layer}-${h.head}`
const D_MODEL = HEADS[0].sigma.length
// The product of a 1024 x 128 and a 128 x 1024 factor has at most 128 nonzero singular values.
const D_HEAD = HEADS[0].rank

// Spectrum panel: index 1..1024 on x, log10 sigma on y. Energy panel: k = 1..128. The narrow
// layout keeps tick text legible in a phone-width frame instead of scaling the wide one down.
type Layout = {
  id: 'wide' | 'narrow'
  w: number
  h: number
  x0: number
  x1: number
  sTop: number
  sBot: number
  eTop: number
  eBot: number
  indexTicks: number[]
}
const LAYOUTS: readonly Layout[] = [
  {
    id: 'wide',
    w: 560,
    h: 318,
    x0: 58,
    x1: 548,
    sTop: 22,
    sBot: 146,
    eTop: 192,
    eBot: 300,
    indexTicks: [1, 128, 256, 512, 768, 1024],
  },
  {
    id: 'narrow',
    w: 300,
    h: 290,
    x0: 40,
    x1: 292,
    sTop: 22,
    sBot: 126,
    eTop: 172,
    eBot: 272,
    indexTicks: [1, 128, 512, 1024],
  },
]
const S_DEC = [-10, 1] as const
const SIGMA_TICKS = [0, -3, -6, -9]
const K_TICKS = [1, 32, 64, 96, 128]
const ENERGY_TICKS = [0, 0.25, 0.5, 0.75, 1]

const r1 = (v: number) => Math.round(v * 10) / 10
const scales = (g: Layout) => ({
  xS: (i: number) => g.x0 + ((i - 1) / (D_MODEL - 1)) * (g.x1 - g.x0),
  yS: (s: number) =>
    g.sBot -
    ((Math.log10(Math.max(s, 10 ** S_DEC[0])) - S_DEC[0]) / (S_DEC[1] - S_DEC[0])) *
      (g.sBot - g.sTop),
  xE: (k: number) => g.x0 + ((k - 1) / (D_HEAD - 1)) * (g.x1 - g.x0),
  yE: (e: number) => g.eBot - e * (g.eBot - g.eTop),
})
type Scales = ReturnType<typeof scales>

// The definitions in latent_projection.py: E_k over sigma^2, r_eff as exp of the entropy of p_i.
const spectrumModel = (sigma: number[]) => {
  const power = sigma.map(s => s * s)
  const total = power.reduce((a, b) => a + b, 0)
  let run = 0
  const energy = power.map(p => (run += p) / total)
  const entropy = -power.reduce((a, p) => a + (p / total) * Math.log(p / total + 1e-12), 0)
  return { energy, rEff: Math.exp(entropy) }
}

// Keeps every point where the curve turns, and one point per 4 units where it is smooth.
const sigmaPath = (sigma: number[], sc: Scales) => {
  const pts = sigma.map((s, i) => [sc.xS(i + 1), sc.yS(s)] as const)
  const kept: (readonly [number, number])[] = [pts[0]]
  for (let i = 1; i < pts.length; i++) {
    const last = kept[kept.length - 1]
    const turn =
      Math.abs(pts[i][1] - pts[i - 1][1]) > 0.25 ||
      (i + 1 < pts.length && Math.abs(pts[i + 1][1] - pts[i][1]) > 0.25)
    if (turn || pts[i][0] - last[0] >= 4 || i === pts.length - 1) kept.push(pts[i])
  }
  return 'M' + kept.map(([x, y]) => `${r1(x)},${r1(y)}`).join('L')
}

const fmtSigma = (s: number) => (s >= 0.01 ? s.toFixed(4) : s.toExponential(2))
const tokenLabel = (t: string) => t.trim() || '␣'

type Geometry = { sigmaPath: string; xS: number[]; yS: number[]; points: string[]; rEffX: number }

const MODELS = HEADS.map(h => {
  const { energy, rEff } = spectrumModel(h.sigma)
  const k90 = energy.findIndex(e => e >= 0.9) + 1
  const geometry = Object.fromEntries(
    LAYOUTS.map(g => {
      const sc = scales(g)
      const geo: Geometry = {
        sigmaPath: sigmaPath(h.sigma, sc),
        xS: Array.from({ length: D_HEAD }, (_, i) => r1(sc.xS(i + 1))),
        yS: h.sigma.slice(0, D_HEAD).map(s => r1(sc.yS(s))),
        points: energy.slice(0, D_HEAD).map((e, i) => `${r1(sc.xE(i + 1))},${r1(sc.yE(e))}`),
        rEffX: r1(sc.xE(rEff)),
      }
      return [g.id, geo]
    }),
  ) as Record<Layout['id'], Geometry>
  return { id: headId(h), head: h, energy, rEff, k90, geometry }
})
type Model = (typeof MODELS)[number]

const readout = (m: Model, k: number) => ({
  k: String(k),
  sigmaK: fmtSigma(m.head.sigma[k - 1]),
  energyK: m.energy[k - 1].toFixed(4),
})

// The inline script indexes these arrays; the model above is the only source of numbers.
const CLIENT = Object.fromEntries(
  MODELS.map(m => [
    m.id,
    {
      sigma: m.head.sigma.slice(0, D_HEAD).map(fmtSigma),
      energy: m.energy.slice(0, D_HEAD).map(e => e.toFixed(4)),
      layouts: Object.fromEntries(
        LAYOUTS.map(g => {
          const { xS, yS, points } = m.geometry[g.id]
          return [g.id, { xS, yS, points }]
        }),
      ),
      attention: m.head.attention.map(rows => rows.map(row => row.map(a => a.toFixed(4)))),
    },
  ]),
)

const SpectrumPlot = ({ g, m0, k0 }: { g: Layout; m0: Model; k0: number }) => {
  const sc = scales(g)
  const geo = m0.geometry[g.id]
  const i0 = k0 - 1
  const [xk, yk] = geo.points[i0].split(',')
  const area0 = `${r1(sc.xE(1))},${g.eBot} ${geo.points.slice(0, k0).join(' ')} ${xk},${g.eBot}`
  const rankX = r1(sc.xS(D_HEAD + 0.5))
  return (
    <svg
      class="hs-svg"
      viewBox={`0 0 ${g.w} ${g.h}`}
      style={{ aspectRatio: `${g.w} / ${g.h}` }}
      role="img"
      aria-label={`Singular values of W_OV for six heads on a log scale over index 1 to ${D_MODEL}, and the cumulative energy of the selected head over k from 1 to ${D_HEAD}.`}
      data-hs-layout={g.id}
    >
      <text class="hs-axis" x={g.x0} y={g.sTop - 9}>
        {`σᵢ of W_OV, log scale, i = 1 to ${D_MODEL}`}
      </text>
      <g class="hs-grid">
        {SIGMA_TICKS.map(t => (
          <line x1={g.x0} x2={g.x1} y1={r1(sc.yS(10 ** t))} y2={r1(sc.yS(10 ** t))} />
        ))}
        {ENERGY_TICKS.map(t => (
          <line x1={g.x0} x2={g.x1} y1={r1(sc.yE(t))} y2={r1(sc.yE(t))} />
        ))}
      </g>
      <g class="hs-tick" text-anchor="end">
        {SIGMA_TICKS.map(t => (
          <text x={g.x0 - 5} y={r1(sc.yS(10 ** t)) + 3.5}>
            10
            <tspan class="hs-sup" dy="-4">
              {t === 0 ? '0' : `−${-t}`}
            </tspan>
          </text>
        ))}
        {ENERGY_TICKS.map(t => (
          <text x={g.x0 - 5} y={r1(sc.yE(t)) + 3.5}>
            {String(t)}
          </text>
        ))}
      </g>
      <g class="hs-tick" text-anchor="middle">
        {g.indexTicks.map(t => (
          <text x={r1(sc.xS(t))} y={g.sBot + 14}>
            {String(t)}
          </text>
        ))}
        {K_TICKS.map(t => (
          <text x={r1(sc.xE(t))} y={g.eBot + 14}>
            {String(t)}
          </text>
        ))}
      </g>
      <text class="hs-axis" x={g.x0} y={g.eTop - 9}>
        {`cumulative energy Eₖ, k = 1 to ${D_HEAD}`}
      </text>

      <rect
        class="hs-kept"
        x={geo.xS[0]}
        y={g.sTop}
        width={r1(geo.xS[i0] - geo.xS[0])}
        height={g.sBot - g.sTop}
        data-hs-kept-band
      />
      <polygon class="hs-kept" points={area0} data-hs-kept-area data-hs-base={String(g.eBot)} />

      <line class="hs-rank" x1={rankX} x2={rankX} y1={g.sTop} y2={g.sBot} />
      <text class="hs-note" x={rankX + 5} y={g.sTop + 11}>
        {`rank ${D_HEAD} = d_head`}
      </text>

      {MODELS.map(m => (
        <path
          class="hs-curve"
          d={m.geometry[g.id].sigmaPath}
          data-hs-for={m.id}
          data-active={m.id === m0.id ? 'true' : 'false'}
        />
      ))}
      {MODELS.map(m => (
        <polyline
          class="hs-curve"
          points={m.geometry[g.id].points.join(' ')}
          data-hs-for={m.id}
          data-active={m.id === m0.id ? 'true' : 'false'}
        />
      ))}

      {MODELS.map(m => (
        <g class="hs-reff" data-hs-for={m.id} data-active={m.id === m0.id ? 'true' : 'false'}>
          <line x1={m.geometry[g.id].rEffX} x2={m.geometry[g.id].rEffX} y1={g.eTop} y2={g.eBot} />
          <text x={m.geometry[g.id].rEffX - 4} y={g.eBot - 6} text-anchor="end">
            r
            <tspan class="hs-sub" dy="2.5">
              eff
            </tspan>
          </text>
        </g>
      ))}

      <line
        class="hs-marker"
        x1={geo.xS[i0]}
        x2={geo.xS[i0]}
        y1={g.sTop}
        y2={g.sBot}
        data-hs-marker-s
      />
      <line class="hs-marker" x1={xk} x2={xk} y1={g.eTop} y2={g.eBot} data-hs-marker-e />
      <circle class="hs-dot" cx={geo.xS[i0]} cy={geo.yS[i0]} r="4" data-hs-dot-s />
      <circle class="hs-dot" cx={xk} cy={yk} r="4" data-hs-dot-e />
      <rect class="hs-panel" x={g.x0} y={g.sTop} width={g.x1 - g.x0} height={g.sBot - g.sTop} />
      <rect class="hs-panel" x={g.x0} y={g.eTop} width={g.x1 - g.x0} height={g.eBot - g.eTop} />
    </svg>
  )
}

const CELL = 20
const LABEL_W = 66
const LABEL_H = 58
const heatSize = (n: number) => ({ w: LABEL_W + n * CELL + 1, h: LABEL_H + n * CELL + 1 })
const RAMP_STEPS = Array.from({ length: 11 }, (_, i) => i / 10)

let instance = 0

const HeadSpectrumImpl: QuartzMdxComponent<Props> = ({
  caption,
  layer = 12,
  head = 8,
  prompt = 'comma',
  k = 32,
}) => {
  const id = `hs-${++instance}`
  const m0 = MODELS.find(m => m.head.layer === layer && m.head.head === head) ?? MODELS[0]
  const p0: PromptId = prompt in PROMPT_INDEX ? prompt : 'comma'
  const k0 = Math.min(Math.max(Math.round(k), 1), D_HEAD)
  const r0 = readout(m0, k0)

  return (
    <figure
      class="head-spectrum"
      data-head-spectrum
      data-hs-head={m0.id}
      data-hs-prompt={p0}
      data-hs-k={String(k0)}
      data-hs-states={JSON.stringify(CLIENT)}
    >
      <div class="hs-controls">
        <fieldset class="hs-segments hs-segments--heads">
          <legend class="hs-legend">Head</legend>
          {MODELS.map(m => (
            <label class="hs-segment">
              <input
                type="radio"
                name={`${id}-head`}
                value={m.id}
                checked={m.id === m0.id}
                data-hs-head-input
              />
              <span>{`L${m.head.layer} H${m.head.head}`}</span>
            </label>
          ))}
        </fieldset>
        <fieldset class="hs-segments">
          <legend class="hs-legend">Prompt</legend>
          {PROMPTS.map(p => (
            <label class="hs-segment">
              <input
                type="radio"
                name={`${id}-prompt`}
                value={p.id}
                checked={p.id === p0}
                data-hs-prompt-input
              />
              <span>{p.label}</span>
            </label>
          ))}
        </fieldset>
      </div>

      <div class="hs-plot">
        {LAYOUTS.map(g => (
          <SpectrumPlot g={g} m0={m0} k0={k0} />
        ))}
        <input
          class="hs-slider"
          type="range"
          min="1"
          max={D_HEAD}
          step="1"
          value={k0}
          aria-label="rank k"
          aria-valuetext={`k ${k0}, E_k ${r0.energyK}`}
          style={Object.fromEntries(
            LAYOUTS.flatMap(g => [
              [`--hs-x0-${g.id}`, `${(g.x0 / g.w) * 100}%`],
              [`--hs-span-${g.id}`, `${((g.x1 - g.x0) / g.w) * 100}%`],
            ]),
          )}
          data-hs-k-input
        />
      </div>

      <div class="hs-lower">
        <div class="hs-heat">
          <p class="hs-heading">Attention weights: rows are target tokens, columns source tokens</p>
          {data.prompts.map((p, pi) => {
            const n = p.tokens.length
            const { w, h } = heatSize(n)
            const pid = PROMPTS[pi].id
            const weights = m0.head.attention[pi]
            return (
              <div class="hs-heat-panel" data-hs-prompt-panel={pid}>
                <svg
                  class="hs-heat-svg"
                  viewBox={`0 0 ${w} ${h}`}
                  style={{ maxWidth: `${w * 1.05}px` }}
                  role="img"
                  aria-label={`Attention of each target token over source tokens for the prompt "${p.text}". Future tokens are masked.`}
                >
                  {p.tokens.map((t, j) => (
                    <text
                      class="hs-token"
                      transform={`translate(${LABEL_W + j * CELL + CELL / 2 + 3},${LABEL_H - 5}) rotate(-90)`}
                      data-hs-col={String(j)}
                    >
                      {tokenLabel(t)}
                    </text>
                  ))}
                  {p.tokens.map((t, i) => (
                    <text
                      class="hs-token"
                      x={LABEL_W - 5}
                      y={LABEL_H + i * CELL + CELL / 2 + 3}
                      text-anchor="end"
                      data-hs-row={String(i)}
                    >
                      {tokenLabel(t)}
                    </text>
                  ))}
                  <g
                    class="hs-cells"
                    tabindex={0}
                    role="group"
                    aria-label="Attention cells. Arrow keys move between cells."
                    data-hs-cells
                    data-hs-n={String(n)}
                  >
                    {p.tokens.flatMap((_, i) =>
                      p.tokens.map((_, j) =>
                        j > i ? (
                          <rect
                            class="hs-cell hs-cell--masked"
                            x={LABEL_W + j * CELL + 0.5}
                            y={LABEL_H + i * CELL + 0.5}
                            width={CELL - 1}
                            height={CELL - 1}
                          />
                        ) : (
                          <rect
                            class="hs-cell"
                            x={LABEL_W + j * CELL + 0.5}
                            y={LABEL_H + i * CELL + 0.5}
                            width={CELL - 1}
                            height={CELL - 1}
                            style={{ '--hs-a': weights[i][j].toFixed(4) }}
                            data-hs-cell={`${i}-${j}`}
                          />
                        ),
                      ),
                    )}
                    <rect
                      class="hs-frame"
                      x={LABEL_W - 1.5}
                      y={LABEL_H - 1.5}
                      width={n * CELL + 3}
                      height={n * CELL + 3}
                    />
                    {/* Drawn just outside the cell so the cell's own shade stays readable. */}
                    <rect
                      class="hs-active"
                      x={LABEL_W - 1}
                      y={LABEL_H + (n - 1) * CELL - 1}
                      width={CELL + 2}
                      height={CELL + 2}
                      data-hs-active
                    />
                  </g>
                </svg>
                <p class="hs-cell-readout" aria-live="polite" data-hs-cell-readout>
                  {`${tokenLabel(p.tokens[n - 1])} → ${tokenLabel(p.tokens[0])}: ${weights[n - 1][0].toFixed(4)}`}
                </p>
              </div>
            )
          })}
          <div class="hs-ramp" aria-hidden="true">
            <span>0</span>
            {RAMP_STEPS.map(a => (
              <span class="hs-swatch" style={{ '--hs-a': String(a) }} />
            ))}
            <span>1</span>
            <span class="hs-swatch hs-swatch--masked" />
            <span>masked</span>
          </div>
        </div>

        <dl class="hs-readout">
          <div>
            <dt>
              <MathText text="$k$" mathClass="hs-math" />
            </dt>
            <dd data-hs-out="k">{r0.k}</dd>
          </div>
          <div>
            <dt>
              <MathText text="$\sigma_k$" mathClass="hs-math" />
            </dt>
            <dd data-hs-out="sigmaK">{r0.sigmaK}</dd>
          </div>
          <div>
            <dt>
              <MathText text="$E_k$" mathClass="hs-math" />
            </dt>
            <dd data-hs-out="energyK">{r0.energyK}</dd>
          </div>
          {MODELS.map(m => (
            <div
              class="hs-head-rows"
              data-hs-for={m.id}
              data-active={m.id === m0.id ? 'true' : 'false'}
            >
              <dt>
                <MathText text="$\sigma_1$" mathClass="hs-math" />
              </dt>
              <dd>{fmtSigma(m.head.sigma[0])}</dd>
              <dt>
                <MathText text="$r_{\text{eff}}$" mathClass="hs-math" />
              </dt>
              <dd data-hs-reff>{m.rEff.toFixed(2)}</dd>
              <dt>
                <MathText text="$\min k : E_k \ge 0.9$" mathClass="hs-math" />
              </dt>
              <dd>{String(m.k90)}</dd>
              <dt>
                <MathText text="$W_{OV}$ shape" mathClass="hs-math" />
              </dt>
              <dd>{m.head.shape.join(' × ')}</dd>
              <dt>Rank</dt>
              <dd>{String(m.head.rank)}</dd>
              <dt>KV head read</dt>
              <dd>{`${m.head.valueHead} of ${data.numValueHeads}`}</dd>
            </div>
          ))}
        </dl>
      </div>

      {caption ? (
        <figcaption class="hs-caption">
          <MathText text={caption} mathClass="hs-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const HeadSpectrumComponent = HeadSpectrumImpl as QuartzMdxComponent<Props>
HeadSpectrumComponent.css = style
HeadSpectrumComponent.afterDOMLoaded = script

export const HeadSpectrum = registerMdxComponent('HeadSpectrum', HeadSpectrumComponent)

export default (() => HeadSpectrum) satisfies (opts: undefined) => QuartzMdxComponent<Props>
