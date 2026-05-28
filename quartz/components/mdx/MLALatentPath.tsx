import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/mla-latent-path.inline'
import style from '../styles/mlaLatentPath.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

const W = 720
const H = 360
const BW = 92
const BH = 46
const HW = 130
const HH = 34
const CX = { i: 56, l: 246, p: 470, c: 642 }
const RY = { c: 96, r: 248, kc: 58, vc: 134 }
const YM = (RY.c + RY.r) / 2

type Kind = 'input' | 'cached' | 'head' | 'concat'
type N = {
  id: string
  k: Kind
  cx: number
  cy: number
  w: number
  h: number
  tex: string
  dim?: { k: string; t: string }
}
const NODES: N[] = [
  { id: 'input', k: 'input', cx: CX.i, cy: YM, w: BW, h: BH, tex: '\\mathbf{h}_t' },
  {
    id: 'c',
    k: 'cached',
    cx: CX.l,
    cy: RY.c,
    w: BW,
    h: BH,
    tex: '\\mathbf{c}_t^{KV}',
    dim: { k: 'dc', t: 'd_c' },
  },
  {
    id: 'kr',
    k: 'cached',
    cx: CX.l,
    cy: RY.r,
    w: BW,
    h: BH,
    tex: '\\mathbf{k}_t^{R}',
    dim: { k: 'dr', t: 'd_h^R' },
  },
  {
    id: 'kc',
    k: 'head',
    cx: CX.p,
    cy: RY.kc,
    w: HW,
    h: HH,
    tex: '\\{\\mathbf{k}_{t,1}^{C},\\dots,\\mathbf{k}_{t,n_h}^{C}\\}',
    dim: { k: 'kc', t: 'n_h d_h' },
  },
  {
    id: 'vc',
    k: 'head',
    cx: CX.p,
    cy: RY.vc,
    w: HW,
    h: HH,
    tex: '\\{\\mathbf{v}_{t,1}^{C},\\dots,\\mathbf{v}_{t,n_h}^{C}\\}',
    dim: { k: 'vc', t: 'n_h d_h' },
  },
  {
    id: 'concat',
    k: 'concat',
    cx: CX.c,
    cy: YM,
    w: BW,
    h: BH,
    tex: '[\\mathbf{k}_{t,i}^{C};\\mathbf{k}_t^{R}]',
    dim: { k: 'concat', t: 'd_h+d_h^R' },
  },
]

type E = {
  fx: number
  fy: number
  tx: number
  ty: number
  lbl?: { tex: string; dx: number; dy: number; w?: number }
  cls?: string
}
const EDGES: E[] = [
  {
    fx: CX.i + BW / 2,
    fy: YM - 8,
    tx: CX.l - BW / 2,
    ty: RY.c,
    lbl: { tex: 'W^{DKV}', dx: -28, dy: -56, w: 70 },
  },
  {
    fx: CX.i + BW / 2,
    fy: YM + 8,
    tx: CX.l - BW / 2,
    ty: RY.r,
    lbl: { tex: 'W^{KR},\\,\\mathrm{RoPE}', dx: -38, dy: 40, w: 100 },
  },
  {
    fx: CX.l + BW / 2,
    fy: RY.c - 8,
    tx: CX.p - HW / 2,
    ty: RY.kc,
    lbl: { tex: 'W^{UK}', dx: -22, dy: -24 },
  },
  {
    fx: CX.l + BW / 2,
    fy: RY.c + 8,
    tx: CX.p - HW / 2,
    ty: RY.vc,
    lbl: { tex: 'W^{UV}', dx: -22, dy: 6 },
  },
  { fx: CX.p + HW / 2, fy: RY.kc, tx: CX.c - BW / 2, ty: YM - 6, cls: 'mla-edge--concat' },
  {
    fx: CX.l + BW / 2,
    fy: RY.r,
    tx: CX.c - BW / 2,
    ty: YM + 6,
    cls: 'mla-edge--concat mla-edge--rope',
  },
]

type S = {
  k: string
  tex: string
  lbl: string
  min: number
  max: number
  step: number
  def: number
  accent?: boolean
}
const SLIDERS: S[] = [
  { k: 'd', tex: 'd', lbl: 'model dim', min: 256, max: 8192, step: 256, def: 4096 },
  { k: 'nh', tex: 'n_h', lbl: 'heads', min: 4, max: 128, step: 1, def: 32 },
  { k: 'dh', tex: 'd_h', lbl: 'per-head', min: 32, max: 256, step: 16, def: 128 },
  { k: 'dc', tex: 'd_c', lbl: 'latent', min: 64, max: 1024, step: 64, def: 512, accent: true },
  { k: 'dr', tex: 'd_h^R', lbl: 'RoPE', min: 16, max: 128, step: 16, def: 64, accent: true },
]

const EQS = [
  '\\mathbf{c}_t^{KV}=W^{DKV}\\mathbf{h}_t,\\quad \\mathbf{k}_t^{C}=W^{UK}\\mathbf{c}_t^{KV},\\quad \\mathbf{v}_t^{C}=W^{UV}\\mathbf{c}_t^{KV}',
  '\\mathbf{k}_t^{R}=\\mathrm{RoPE}(W^{KR}\\mathbf{h}_t),\\quad \\mathbf{k}_{t,i}=[\\mathbf{k}_{t,i}^{C};\\mathbf{k}_t^{R}]',
  '\\mathbf{o}_{t,i}=\\sum_{j}\\operatorname{softmax}_{j}\\!\\left(\\tfrac{\\mathbf{q}_{t,i}^{\\top}\\mathbf{k}_{j,i}}{\\sqrt{d_h+d_h^R}}\\right)\\mathbf{v}_{j,i}^{C}',
]

const tex = (t: string, d = false): string =>
  katex.renderToString(t, {
    ...katexOptions,
    displayMode: d,
    output: 'html',
    macros: customMacros,
    strict: false,
    throwOnError: false,
  })

const FO: FunctionalComponent<{
  x: number
  y: number
  w: number
  h: number
  t: string
  cls?: string
  dimKey?: string
}> = ({ x, y, w, h, t, cls, dimKey }) => (
  <foreignObject x={x} y={y} width={w} height={h}>
    <div
      class={`mla-fo ${cls ?? ''}`.trim()}
      data-mla-dim={dimKey}
      dangerouslySetInnerHTML={{ __html: tex(t) }}
    />
  </foreignObject>
)

const M: FunctionalComponent<{ t: string; d?: boolean }> = ({ t, d }) => (
  <span
    class={`mla-math${d ? ' mla-math--display' : ''}`}
    dangerouslySetInnerHTML={{ __html: tex(t, d) }}
  />
)

const path = (e: E): string => {
  const mx = (e.fx + e.tx) / 2
  return `M ${e.fx} ${e.fy} C ${mx} ${e.fy}, ${mx} ${e.ty}, ${e.tx} ${e.ty}`
}

const def = Object.fromEntries(SLIDERS.map(s => [s.k, s.def])) as Record<string, number>

const MLALatentPathImpl: QuartzMdxComponent<Props> = ({ caption }) => {
  const aid = 'mla-arrow-head'
  const c = NODES.find(n => n.id === 'c')!
  const cx = c.cx - c.w / 2 - 10
  const cy = c.cy - c.h / 2 - 16
  const ch = RY.r + BH / 2 + 16 - cy

  return (
    <figure
      class="mla-latent-path"
      data-mla-latent-path
      data-mla-d={String(def.d)}
      data-mla-nh={String(def.nh)}
      data-mla-dh={String(def.dh)}
      data-mla-dc={String(def.dc)}
      data-mla-dr={String(def.dr)}
    >
      <div class="mla-stage">
        <svg
          class="mla-graph"
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label="MLA latent path: h_t projects down to a cached compressed latent c_t^KV and a cached RoPE duplicate k_t^R; per-head K and V are reconstructed on demand from c_t^KV via W^UK and W^UV, then concatenated with k_t^R."
        >
          <defs>
            <marker
              id={aid}
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path class="mla-arrowhead" d="M0,0 L10,5 L0,10 z" />
            </marker>
          </defs>

          <rect class="mla-cache-box" x={cx} y={cy} width={BW + 20} height={ch} rx={8} />
          <FO
            x={CX.l - 45}
            y={cy - 22}
            w={90}
            h={18}
            t="\mathrm{KV}\ \text{cache}"
            cls="mla-fo--cache-label"
          />

          {NODES.map(n => {
            const nx = n.cx - n.w / 2
            const ny = n.cy - n.h / 2
            return (
              <g class={`mla-node mla-node--${n.k}`} data-mla-node={n.id}>
                <rect
                  class={`mla-box mla-box--${n.k}`}
                  x={nx}
                  y={ny}
                  width={n.w}
                  height={n.h}
                  rx={n.k === 'head' ? 4 : 6}
                />
                <FO
                  x={nx}
                  y={ny}
                  w={n.w}
                  h={n.h}
                  t={n.tex}
                  cls={n.k === 'head' ? 'mla-fo--head' : 'mla-fo--node'}
                />
                {n.dim ? (
                  <FO
                    x={n.cx - 55}
                    y={ny + n.h + 4}
                    w={110}
                    h={18}
                    t={n.dim.t}
                    cls="mla-fo--dim"
                    dimKey={n.dim.k}
                  />
                ) : null}
              </g>
            )
          })}

          {EDGES.map(e => (
            <>
              <path
                class={`mla-edge ${e.cls ?? ''}`.trim()}
                d={path(e)}
                marker-end={`url(#${aid})`}
              />
              {e.lbl ? (
                <FO
                  x={(e.fx + e.tx) / 2 + e.lbl.dx}
                  y={(e.fy + e.ty) / 2 + e.lbl.dy}
                  w={e.lbl.w ?? 60}
                  h={18}
                  t={e.lbl.tex}
                  cls="mla-fo--edge"
                />
              ) : null}
            </>
          ))}
        </svg>

        <aside class="mla-side" aria-label="MLA cache readout">
          <div class="mla-card">
            <h4>cached</h4>
            <ul class="mla-cache-list">
              <li>
                <span class="mla-swatch mla-swatch--cached" aria-hidden="true" />
                <M t="\mathbf{c}_t^{KV}\in\mathbb{R}^{d_c}" />
              </li>
              <li>
                <span class="mla-swatch mla-swatch--cached" aria-hidden="true" />
                <M t="\mathbf{k}_t^{R}\in\mathbb{R}^{d_h^R}" />
              </li>
            </ul>
            <p class="mla-cache-note">
              per-head <M t="\mathbf{k}_{t,i}^{C},\mathbf{v}_{t,i}^{C}" /> are reconstructed on
              demand and never touch HBM.
            </p>
          </div>

          <dl class="mla-readout" data-mla-readout>
            <div class="mla-readout-row">
              <dt>
                MHA <M t="2 \cdot n_h \cdot d_h" />
              </dt>
              <dd data-mla-mha class="mla-readout-val">
                {2 * def.nh * def.dh}
              </dd>
            </div>
            <div class="mla-readout-row">
              <dt>
                MLA <M t="d_c + d_h^R" />
              </dt>
              <dd data-mla-mla class="mla-readout-val mla-readout-val--accent">
                {def.dc + def.dr}
              </dd>
            </div>
            <div class="mla-readout-row">
              <dt>compression</dt>
              <dd>
                <span data-mla-ratio class="mla-readout-val mla-readout-val--big">
                  {((2 * def.nh * def.dh) / (def.dc + def.dr)).toFixed(1)}x
                </span>
              </dd>
            </div>
          </dl>
        </aside>
      </div>

      <div class="mla-controls" role="group" aria-label="MLA dimensional knobs">
        {SLIDERS.map(s => (
          <div class="mla-control">
            <label class="mla-label" for={`mla-${s.k}`}>
              <M t={s.tex} /> {s.lbl}
            </label>
            <input
              id={`mla-${s.k}`}
              class="mla-slider"
              type="range"
              min={s.min}
              max={s.max}
              step={s.step}
              value={s.def}
              data-mla-input={s.k}
              aria-valuemin={s.min}
              aria-valuemax={s.max}
              aria-valuenow={s.def}
              aria-valuetext={`${s.lbl} ${s.def}`}
            />
            <span class={`mla-value${s.accent ? ' mla-value--accent' : ''}`} data-mla-value={s.k}>
              {s.def}
            </span>
          </div>
        ))}
      </div>

      <div class="mla-eqs">
        {EQS.map(e => (
          <M t={e} d />
        ))}
        <p class="mla-bet">
          <M t="d_c+d_h^R\ \ll\ 2\,n_h\,d_h" />: the joint low-rank bet. Shrink <M t="d_c" /> until
          reconstruction bends; cache size collapses with it.
        </p>
      </div>

      {caption ? (
        <figcaption class="mla-caption">
          <MathText text={caption} mathClass="mla-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const MLALatentPathComponent = MLALatentPathImpl as QuartzMdxComponent<Props>
MLALatentPathComponent.css = style
MLALatentPathComponent.afterDOMLoaded = script

export const MLALatentPath = registerMdxComponent('MLALatentPath', MLALatentPathComponent)

export default (() => MLALatentPath) satisfies (opts: undefined) => QuartzMdxComponent<Props>
