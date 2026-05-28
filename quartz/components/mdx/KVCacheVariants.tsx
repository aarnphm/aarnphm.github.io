import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
import style from '../styles/kvCacheVariants.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

const VIEW_W = 980
const VIEW_H = 332
const Q_N = 8
const BAR_W = 11
const BAR_H = 46
const PITCH = 17
const CLUSTER_W = (Q_N - 1) * PITCH + BAR_W
const MARGIN_L = 23
const ROW = { v: 92, k: 164, q: 236 } as const
const PX = [48, 244, 440, 636] as const
const DIV = [234, 430, 626] as const
const HATCH = 'kvc-hatch'
const ARROW = 'kvc-proj-head'

const LAT_CX = PX[3] + 250
const LAT_W = 26
const LAT_TOP = ROW.v
const LAT_BOT = ROW.k + BAR_H
const KV_MID = (LAT_TOP + LAT_BOT) / 2

type Panel = {
  id: string
  title: string
  sub: string
  nkv: number
  cachedKV: boolean
  latent?: boolean
  costTex: string
}

const PANELS: Panel[] = [
  { id: 'mha', title: 'MHA', sub: 'multi-head', nkv: 8, cachedKV: true, costTex: '2\\,n_h d_h' },
  { id: 'gqa', title: 'GQA', sub: 'grouped-query', nkv: 4, cachedKV: true, costTex: '2\\,n_g d_h' },
  { id: 'mqa', title: 'MQA', sub: 'multi-query', nkv: 1, cachedKV: true, costTex: '2\\,d_h' },
  {
    id: 'mla',
    title: 'MLA',
    sub: 'multi-head latent',
    nkv: 8,
    cachedKV: false,
    latent: true,
    costTex: 'd_c + d_h^R',
  },
]

const renderMath = (tex: string): string =>
  katex.renderToString(tex, {
    ...katexOptions,
    displayMode: false,
    output: 'html',
    macros: customMacros,
    strict: false,
    throwOnError: false,
  })

const qCx = (p: number, i: number): number => PX[p] + MARGIN_L + i * PITCH + BAR_W / 2
const kvCx = (p: number, g: number, nkv: number): number => {
  const r = Q_N / nkv
  return qCx(p, g * r + (r - 1) / 2)
}
const clusterCx = (p: number): number => PX[p] + MARGIN_L + CLUSTER_W / 2

const CachedBar: FunctionalComponent<{
  x: number
  y: number
  w: number
  h: number
  cls: string
  rx?: number
}> = ({ x, y, w, h, cls, rx = 2.5 }) => (
  <g>
    <rect class={`kvc-bar ${cls}`} x={x} y={y} width={w} height={h} rx={rx} />
    <rect class="kvc-hatch-fill" x={x} y={y} width={w} height={h} rx={rx} fill={`url(#${HATCH})`} />
  </g>
)

const Panel: FunctionalComponent<{ p: number; panel: Panel }> = ({ p, panel }) => {
  const { nkv, cachedKV, latent } = panel
  const r = Q_N / nkv
  const qIdx = Array.from({ length: Q_N }, (_, i) => i)
  const gIdx = Array.from({ length: nkv }, (_, g) => g)
  const cx = clusterCx(p)

  return (
    <g class="kvc-panel" data-kvc-panel={panel.id}>
      <foreignObject x={cx - 72} y={44} width={144} height={34}>
        <div class="kvc-fo kvc-fo--title">
          <span class="kvc-title-main">{panel.title}</span>
          <span class="kvc-title-sub">{panel.sub}</span>
        </div>
      </foreignObject>

      <g class="kvc-links">
        {qIdx.map(i => (
          <line
            class="kvc-link"
            x1={qCx(p, i)}
            y1={ROW.q}
            x2={kvCx(p, Math.floor(i / r), nkv)}
            y2={ROW.k + BAR_H}
          />
        ))}
        {gIdx.map(g => (
          <line
            class="kvc-link"
            x1={kvCx(p, g, nkv)}
            y1={ROW.k}
            x2={kvCx(p, g, nkv)}
            y2={ROW.v + BAR_H}
          />
        ))}
      </g>

      {gIdx.map(g => {
        const bx = kvCx(p, g, nkv) - BAR_W / 2
        return cachedKV ? (
          <>
            <CachedBar x={bx} y={ROW.v} w={BAR_W} h={BAR_H} cls="kvc-bar--cached" />
            <CachedBar x={bx} y={ROW.k} w={BAR_W} h={BAR_H} cls="kvc-bar--cached" />
          </>
        ) : (
          <>
            <rect
              class="kvc-bar kvc-bar--recompute"
              x={bx}
              y={ROW.v}
              width={BAR_W}
              height={BAR_H}
              rx={2.5}
            />
            <rect
              class="kvc-bar kvc-bar--recompute"
              x={bx}
              y={ROW.k}
              width={BAR_W}
              height={BAR_H}
              rx={2.5}
            />
          </>
        )
      })}

      {qIdx.map(i => (
        <rect
          class="kvc-bar kvc-bar--query"
          x={qCx(p, i) - BAR_W / 2}
          y={ROW.q}
          width={BAR_W}
          height={BAR_H}
          rx={2.5}
        />
      ))}

      {latent ? (
        <g class="kvc-latent">
          <line
            class="kvc-wedge"
            x1={LAT_CX - LAT_W / 2}
            y1={LAT_TOP + 14}
            x2={kvCx(p, 7, nkv) + 8}
            y2={ROW.v + 6}
          />
          <line
            class="kvc-wedge"
            x1={LAT_CX - LAT_W / 2}
            y1={LAT_BOT - 14}
            x2={kvCx(p, 7, nkv) + 8}
            y2={ROW.k + BAR_H - 6}
          />
          <line
            class="kvc-proj"
            x1={LAT_CX - LAT_W / 2}
            y1={KV_MID}
            x2={kvCx(p, 7, nkv) + 14}
            y2={KV_MID}
            marker-end={`url(#${ARROW})`}
          />
          <foreignObject x={LAT_CX - 6} y={KV_MID - 30} width={92} height={16}>
            <div class="kvc-fo kvc-fo--proj">projection</div>
          </foreignObject>
          <CachedBar
            x={LAT_CX - LAT_W / 2}
            y={LAT_TOP}
            w={LAT_W}
            h={LAT_BOT - LAT_TOP}
            cls="kvc-bar--latent"
            rx={3}
          />
          <foreignObject x={LAT_CX - 70} y={LAT_BOT + 6} width={140} height={28}>
            <div class="kvc-fo kvc-fo--latent">compressed latent KV</div>
          </foreignObject>
        </g>
      ) : null}

      <foreignObject x={cx - 78} y={290} width={156} height={30}>
        <div class="kvc-fo kvc-cost">
          <span class="kvc-cost-tag">cache / token</span>
          <span
            class="kvc-cost-math"
            dangerouslySetInnerHTML={{ __html: renderMath(panel.costTex) }}
          />
        </div>
      </foreignObject>
    </g>
  )
}

const KVCacheVariantsImpl: QuartzMdxComponent<Props> = ({ caption }) => (
  <figure class="kv-cache-variants" data-kv-cache-variants>
    <div class="kvc-legend" aria-hidden="true">
      <span class="kvc-legend-item">
        <span class="kvc-swatch kvc-swatch--query" />
        queries (recomputed each step)
      </span>
      <span class="kvc-legend-item">
        <span class="kvc-swatch kvc-swatch--cached" />
        cached during inference
      </span>
      <span class="kvc-legend-item">
        <span class="kvc-swatch kvc-swatch--recompute" />
        reconstructed on demand
      </span>
    </div>

    <div class="kvc-stage">
      <svg
        class="kvc-graph"
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="kv cache across attention variants. MHA caches every head's keys and values; GQA shares keys and values across query groups; MQA stores one shared key and value; MLA stores a compressed latent plus a RoPE key and reconstructs per-head keys and values by projection."
      >
        <defs>
          <pattern
            id={HATCH}
            width="5"
            height="5"
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <line class="kvc-hatch-line" x1="0" y1="0" x2="0" y2="5" />
          </pattern>
          <marker
            id={ARROW}
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="7"
            markerHeight="7"
            orient="auto-start-reverse"
          >
            <path class="kvc-arrowhead" d="M0,0 L10,5 L0,10 z" />
          </marker>
        </defs>

        {(['v', 'k', 'q'] as const).map(row => (
          <foreignObject x={0} y={ROW[row] + BAR_H / 2 - 9} width={44} height={18}>
            <div class="kvc-fo kvc-fo--row">
              {row === 'v' ? 'values' : row === 'k' ? 'keys' : 'queries'}
            </div>
          </foreignObject>
        ))}

        {DIV.map(x => (
          <line class="kvc-divider" x1={x} y1={46} x2={x} y2={300} />
        ))}

        {PANELS.map((panel, p) => (
          <Panel p={p} panel={panel} />
        ))}
      </svg>
    </div>

    {caption ? (
      <figcaption class="kvc-caption">
        <MathText text={caption} mathClass="kvc-math" />
      </figcaption>
    ) : null}
  </figure>
)

const KVCacheVariantsComponent = KVCacheVariantsImpl as QuartzMdxComponent<Props>
KVCacheVariantsComponent.css = style

export const KVCacheVariants = registerMdxComponent('KVCacheVariants', KVCacheVariantsComponent)

export default (() => KVCacheVariants) satisfies (opts: undefined) => QuartzMdxComponent<Props>
