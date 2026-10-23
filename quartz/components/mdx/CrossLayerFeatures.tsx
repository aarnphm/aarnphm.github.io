import { MathText } from '../../util/math-text'
//@ts-ignore
import figureScaleScript from '../scripts/figure-scale.inline'
import style from '../styles/crossLayerFeatures.scss'
import {
  STAGES,
  LC,
  STATIONS,
  SAES,
  CIRCUIT_LAYERS,
  FEATURES,
  FORMED,
  FORMING_BLOCKS,
  CIRCUIT,
  persistEdges,
  formEdges,
} from './cross-layer-features/model'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type View = 'lifecycle' | 'circuit'
type Coder = 'sae' | 'crosscoder'
type Props = { caption?: string; view?: View; coder?: Coder }

// ---------- geometry ----------

// The left gutter holds the row labels at the meta size.
const LC_X0 = 124
const LC_P = 58
const lcX = (l: number) => LC_X0 + l * LC_P
const LC_W = lcX(STATIONS - 1) + 22
const BLOCK_Y = 24
const BLOCK_H = 22
const BLOCK_W = 34
const STREAM_Y = 66
const BAR_Y = 86
const BAR_UNIT = 13
const BAR_W = 14
const MAX_S = Math.max(...LC.strength)
const VALUE_Y = BAR_Y + MAX_S * BAR_UNIT + 12
const CODER_Y = VALUE_Y + 14
const CELL = 18
const LC_H = CODER_Y + CELL + 30

const CC_X0 = 100
const CC_P = 96
const ccX = (l: number) => CC_X0 + l * CC_P
const CC_W = ccX(CIRCUIT_LAYERS - 1) + 34
const ROW_Y = [46, 94, 142]
const NODE_W = 28
const NODE_H = 18
const CC_H = 170
const rowOf = (id: number) => ROW_Y[CIRCUIT.feature(id).row]

const signed = (w: number) => (w > 0 ? `+${w}` : w < 0 ? `−${-w}` : '0')

let instance = 0

const Arrow = ({ id, soft }: { id: string; soft?: boolean }) => (
  <marker
    id={id}
    viewBox="0 0 10 10"
    refX="10"
    refY="5"
    markerWidth="6"
    markerHeight="6"
    markerUnits="userSpaceOnUse"
    orient="auto"
  >
    <path
      class={soft ? 'clf-arrowhead clf-arrowhead--soft' : 'clf-arrowhead'}
      d="M0,0 L10,5 L0,10 z"
    />
  </marker>
)

const Lifecycle = ({ arrow }: { arrow: string }) => (
  <svg
    class="clf-svg"
    data-fig-scale="--clf-u"
    viewBox={`0 0 ${LC_W} ${LC_H}`}
    preserveAspectRatio="xMidYMid meet"
    role="img"
    aria-label={`${STATIONS} residual positions, ${STAGES.length} blocks. The feature forms at blocks 2–3, persists, grows at 5, and is deleted at 7. ${LC.active.length} per-layer SAE latents versus one crosscoder latent.`}
  >
    <defs>
      <Arrow id={arrow} />
      <Arrow id={`${arrow}-soft`} soft />
    </defs>
    <text class="clf-row" x={4} y={BLOCK_Y - 8}>
      stage
    </text>
    <text class="clf-row" x={4} y={BLOCK_Y + 15}>
      block write
    </text>
    <text class="clf-row" x={4} y={STREAM_Y + 3}>
      residual
    </text>
    <text class="clf-row" x={4} y={BAR_Y + 12}>
      strength
    </text>
    <text class="clf-row" x={4} y={CODER_Y + 12}>
      latents
    </text>

    {STAGES.map((stage, b) => {
      const c = (lcX(b) + lcX(b + 1)) / 2
      const on = LC.writes[b] !== 0
      return (
        <g class="clf-block" data-clf-write={String(LC.writes[b])}>
          <text class="clf-stage" x={c} y={BLOCK_Y - 8} text-anchor="middle">
            {stage}
          </text>
          <line
            class={on ? 'clf-wire clf-wire--on' : 'clf-wire'}
            x1={c - 9}
            y1={STREAM_Y}
            x2={c - 9}
            y2={BLOCK_Y + BLOCK_H}
            marker-end={`url(#${on ? arrow : `${arrow}-soft`})`}
          />
          <line
            class={on ? 'clf-wire clf-wire--on' : 'clf-wire'}
            x1={c + 9}
            y1={BLOCK_Y + BLOCK_H}
            x2={c + 9}
            y2={STREAM_Y}
            marker-end={`url(#${on ? arrow : `${arrow}-soft`})`}
          />
          <rect
            class={on ? 'clf-blockbox clf-blockbox--on' : 'clf-blockbox'}
            x={c - BLOCK_W / 2}
            y={BLOCK_Y}
            width={BLOCK_W}
            height={BLOCK_H}
          />
          <text class="clf-value" x={c} y={BLOCK_Y + 15} text-anchor="middle">
            {signed(LC.writes[b])}
          </text>
        </g>
      )
    })}

    <line class="clf-stream" x1={lcX(0)} y1={STREAM_Y} x2={lcX(STATIONS - 1)} y2={STREAM_Y} />
    {LC.strength.map((s, l) => (
      <g>
        <circle class="clf-station" cx={lcX(l)} cy={STREAM_Y} r={3} />
        <text class="clf-axis" x={lcX(l)} y={STREAM_Y + 14} text-anchor="middle">
          {`r${l}`}
        </text>
        {s > 0 ? (
          <rect
            class="clf-bar"
            data-clf-strength={String(s)}
            x={lcX(l) - BAR_W / 2}
            y={BAR_Y}
            width={BAR_W}
            height={s * BAR_UNIT}
          />
        ) : null}
        <text class="clf-value" x={lcX(l)} y={VALUE_Y} text-anchor="middle">
          {String(s)}
        </text>
      </g>
    ))}

    <g class="clf-coder" data-clf-coder="sae">
      {LC.layers.map(({ station, s }) => {
        const x = lcX(station)
        return (
          <g>
            <rect
              class="clf-dict"
              x={x - CELL / 2 - 4}
              y={CODER_Y - 4}
              width={CELL + 8}
              height={CELL + 8}
            />
            <rect
              class={s > 0 ? 'clf-latent clf-latent--on' : 'clf-latent'}
              data-clf-sae-latent={s > 0 ? String(station) : undefined}
              x={x - CELL / 2}
              y={CODER_Y}
              width={CELL}
              height={CELL}
            />
            {s > 0 ? (
              <text class="clf-value" x={x} y={CODER_Y + 13} text-anchor="middle">
                {String(s)}
              </text>
            ) : null}
            <text class="clf-axis" x={x} y={CODER_Y + CELL + 16} text-anchor="middle">
              {`SAE ${station}`}
            </text>
          </g>
        )
      })}
    </g>

    <g class="clf-coder" data-clf-coder="crosscoder">
      <rect
        class="clf-dict"
        x={lcX(1) - CELL / 2 - 4}
        y={CODER_Y - 4}
        width={lcX(SAES) - lcX(1) + CELL + 8}
        height={CELL + 8}
      />
      <rect
        class="clf-latent clf-latent--on"
        data-clf-cc-latent
        x={lcX(LC.active[0].station) - CELL / 2}
        y={CODER_Y}
        width={lcX(LC.active[LC.active.length - 1].station) - lcX(LC.active[0].station) + CELL}
        height={CELL}
      />
      <text
        class="clf-value"
        x={(lcX(LC.active[0].station) + lcX(LC.active[LC.active.length - 1].station)) / 2}
        y={CODER_Y + 13}
        text-anchor="middle"
      >
        one latent, z = 1
      </text>
      {LC.layers.map(({ station, s }) => (
        <text
          class="clf-axis"
          data-clf-norm={String(station)}
          x={lcX(station)}
          y={CODER_Y + CELL + 16}
          text-anchor="middle"
        >
          {`‖d‖=${s}`}
        </text>
      ))}
    </g>
  </svg>
)

const nodeBox = (f: number, l: number) => ({ x: ccX(l) - NODE_W / 2, y: rowOf(f) - NODE_H / 2 })

const Circuit = ({ arrow }: { arrow: string }) => (
  <svg
    class="clf-svg"
    data-fig-scale="--clf-u"
    viewBox={`0 0 ${CC_W} ${CC_H}`}
    preserveAspectRatio="xMidYMid meet"
    role="img"
    aria-label={`Features 1 and 2 form feature 3 at L+2 and L+3. Per-layer SAEs: ${CIRCUIT.nodes.length} latents, ${CIRCUIT.edges.length} edges. Crosscoder: ${FEATURES.length} latents, ${CIRCUIT.latentEdges.length} edges.`}
  >
    <defs>
      <Arrow id={arrow} />
      <Arrow id={`${arrow}-soft`} soft />
    </defs>
    {Array.from({ length: CIRCUIT_LAYERS }, (_, l) => (
      <g>
        <text class="clf-axis" x={ccX(l)} y={14} text-anchor="middle">
          {l === 0 ? 'L' : `L+${l}`}
        </text>
        <line class="clf-grid" x1={ccX(l)} y1={22} x2={ccX(l)} y2={CC_H - 6} />
      </g>
    ))}
    {FEATURES.map(f => (
      <text class="clf-row" x={4} y={ROW_Y[f.row] + 3}>
        {`feature ${f.id}`}
      </text>
    ))}

    <g class="clf-coder" data-clf-coder="sae">
      {CIRCUIT.edges.map(e => {
        const y1 = rowOf(e.from[0])
        const dy = e.kind === 'form' ? (y1 < rowOf(e.to[0]) ? -5 : 5) : 0
        return (
          <line
            class={e.kind === 'form' ? 'clf-edge clf-edge--form' : 'clf-edge'}
            data-clf-edge={e.kind}
            x1={ccX(e.from[1]) + NODE_W / 2}
            y1={y1}
            x2={ccX(e.to[1]) - NODE_W / 2}
            y2={rowOf(e.to[0]) + dy}
            marker-end={`url(#${e.kind === 'form' ? arrow : `${arrow}-soft`})`}
          />
        )
      })}
      {CIRCUIT.nodes.map(n => {
        const { x, y } = nodeBox(n.f, n.l)
        return (
          <g>
            <rect
              class={n.f === FORMED ? 'clf-node clf-node--lit' : 'clf-node'}
              data-clf-node={n.label}
              x={x}
              y={y}
              width={NODE_W}
              height={NODE_H}
            />
            <text class="clf-value" x={ccX(n.l)} y={rowOf(n.f) + 3.5} text-anchor="middle">
              {n.label}
            </text>
          </g>
        )
      })}
    </g>

    <g class="clf-coder" data-clf-coder="crosscoder">
      {CIRCUIT.latentEdges.map(p => {
        const y1 = rowOf(p)
        const down = y1 < rowOf(FORMED)
        const b = FORMING_BLOCKS[0]
        return (
          <line
            class="clf-edge clf-edge--form"
            data-clf-edge="form"
            x1={ccX(b)}
            y1={y1 + (down ? NODE_H / 2 : -NODE_H / 2)}
            x2={ccX(b + 1) - NODE_W / 2}
            y2={rowOf(FORMED) + (down ? -5 : 5)}
            marker-end={`url(#${arrow})`}
          />
        )
      })}
      {FEATURES.map(f => {
        const x = ccX(f.first) - NODE_W / 2
        return (
          <g>
            <rect
              class={f.id === FORMED ? 'clf-node clf-node--lit' : 'clf-node'}
              data-clf-node={String(f.id)}
              x={x}
              y={ROW_Y[f.row] - NODE_H / 2}
              width={ccX(CIRCUIT_LAYERS - 1) + NODE_W / 2 - x}
              height={NODE_H}
            />
            <text class="clf-value" x={ccX(f.first)} y={ROW_Y[f.row] + 3.5} text-anchor="middle">
              {String(f.id)}
            </text>
          </g>
        )
      })}
    </g>
  </svg>
)

type Row = { label: string; sae: string; crosscoder: string }

const LIFECYCLE_ROWS: Row[] = [
  { label: 'Dictionaries', sae: `${SAES} SAEs`, crosscoder: '1 crosscoder' },
  { label: 'Latents for this feature', sae: String(LC.active.length), crosscoder: '1' },
  { label: 'Latent activations', sae: LC.active.map(l => l.s).join(' '), crosscoder: '$z = 1$' },
  {
    label: `Decoder norms, r1 to r${SAES}`,
    sae: '1 in each active SAE',
    crosscoder: LC.layers.map(l => l.s).join(' '),
  },
  {
    label: 'Sparsity penalty',
    sae: `$\\lambda(${LC.active.map(l => l.s).join('+')}) = ${LC.sum}\\lambda$`,
    crosscoder: `$\\lambda\\cdot 1\\cdot(${LC.active.map(l => l.s).join('+')}) = ${LC.sum}\\lambda$`,
  },
]

const CIRCUIT_ROWS: Row[] = [
  { label: 'Latents', sae: String(CIRCUIT.nodes.length), crosscoder: String(FEATURES.length) },
  {
    label: 'Edges',
    sae: String(CIRCUIT.edges.length),
    crosscoder: String(CIRCUIT.latentEdges.length),
  },
  { label: 'Persistence edges', sae: String(persistEdges), crosscoder: '0' },
  {
    label: 'Edges into feature 3',
    sae: String(formEdges),
    crosscoder: String(CIRCUIT.latentEdges.length),
  },
]

const CrossLayerFeaturesImpl: QuartzMdxComponent<Props> = ({
  caption,
  view = 'lifecycle',
  coder = 'sae',
}) => {
  const id = `clf-${++instance}`
  const v: View = view === 'circuit' ? 'circuit' : 'lifecycle'
  const c0: Coder = coder === 'crosscoder' ? 'crosscoder' : 'sae'
  const rows = v === 'circuit' ? CIRCUIT_ROWS : LIFECYCLE_ROWS

  return (
    <figure class="cross-layer-features" data-cross-layer-features data-clf-view={v}>
      <fieldset class="clf-segments">
        <legend class="clf-legend">Dictionary</legend>
        {(
          [
            ['sae', 'per-layer SAEs'],
            ['crosscoder', 'crosscoder'],
          ] as const
        ).map(([value, label]) => (
          <label class="clf-segment">
            <input
              type="radio"
              name={`${id}-coder`}
              value={value}
              checked={value === c0}
              data-clf-input
            />
            <span>{label}</span>
          </label>
        ))}
      </fieldset>

      <div class="clf-body">
        <div
          class="clf-scroll"
          tabindex={0}
          role="region"
          aria-label={v === 'circuit' ? 'Circuit diagram' : 'Feature lifecycle diagram'}
        >
          {v === 'circuit' ? (
            <Circuit arrow={`${id}-arrow`} />
          ) : (
            <Lifecycle arrow={`${id}-arrow`} />
          )}
        </div>
        <dl class="clf-readout">
          {rows.map(r => (
            <div>
              <dt>{r.label}</dt>
              <dd>
                <span data-clf-coder="sae">
                  <MathText text={r.sae} mathClass="clf-math" />
                </span>
                <span data-clf-coder="crosscoder">
                  <MathText text={r.crosscoder} mathClass="clf-math" />
                </span>
              </dd>
            </div>
          ))}
        </dl>
      </div>

      {caption ? (
        <figcaption class="clf-caption">
          <MathText text={caption} mathClass="clf-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const CrossLayerFeaturesComponent = CrossLayerFeaturesImpl as QuartzMdxComponent<Props>
CrossLayerFeaturesComponent.css = style
CrossLayerFeaturesComponent.afterDOMLoaded = figureScaleScript

export const CrossLayerFeatures = registerMdxComponent(
  'CrossLayerFeatures',
  CrossLayerFeaturesComponent,
)

export default (() => CrossLayerFeatures) satisfies (opts: undefined) => QuartzMdxComponent<Props>
