import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/crosscoder-masks.inline'
import style from '../styles/crosscoderMasks.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Dims = 'coder' | 'variant'
type Causality = 'acausal' | 'weak' | 'strict'
type Locality = 'global' | 'local' | 'skip'
type Props = {
  caption?: string
  dims?: Dims
  coder?: 'sae' | 'transcoder' | 'crosscoder'
  causality?: Causality
  locality?: Locality
  read?: number
}

const N = 5
const LAYERS = Array.from({ length: N }, (_, i) => i)
// Local "convolutional" crosscoders give each latent a window of K adjacent layers.
const K = 2

// A latent group shares one encoder placement: it reads `reads` and reconstructs `writes`.
type Group = { reads: number[]; writes: number[]; side: 'right' | 'left'; span: boolean }

const range = (lo: number, hi: number) => LAYERS.filter(l => l >= lo && l <= hi)

const groupsFor = (variant: string): Group[] => {
  if (variant === 'sae')
    return LAYERS.map(i => ({ reads: [i], writes: [i], side: 'right', span: false }))
  if (variant === 'transcoder')
    return LAYERS.slice(0, -1).map(i => ({
      reads: [i],
      writes: [i + 1],
      side: 'right',
      span: false,
    }))
  const [causality, locality] = variant.split('-') as [Causality, Locality]
  const used = locality === 'skip' ? LAYERS.filter(l => l % 2 === 0) : LAYERS
  if (causality === 'acausal') {
    if (locality !== 'local') return [{ reads: used, writes: used, side: 'right', span: true }]
    return LAYERS.slice(0, N - K + 1).map(i => ({
      reads: range(i, i + K - 1),
      writes: range(i, i + K - 1),
      // Alternate sides so overlapping windows do not cover each other's edges.
      side: i % 2 === 0 ? 'right' : 'left',
      span: true,
    }))
  }
  return used
    .map(i => ({
      reads: [i],
      writes: used.filter(
        j => (causality === 'weak' ? j >= i : j > i) && (locality !== 'local' || j <= i + K - 1),
      ),
      side: 'right' as const,
      span: false,
    }))
    .filter(g => g.writes.length > 0)
}

const RULES: Record<string, string> = {
  sae: 'A per-layer SAE reads one layer and reconstructs that same layer. Each layer has its own dictionary.',
  transcoder:
    'A transcoder reads one layer and predicts the next. The last layer has no transcoder.',
  'acausal-global':
    'An acausal crosscoder reads every layer and reconstructs every layer, so a prediction of an early layer can use later activations.',
  'acausal-local': `Each latent reads and reconstructs a window of $K=${K}$ adjacent layers; the windows overlap.`,
  'acausal-skip':
    'One acausal crosscoder over every other layer. The skipped layers are neither read nor reconstructed.',
  'weak-global': 'Each latent reads one layer and reconstructs that layer and every later layer.',
  'weak-local': `Each latent reads one layer and reconstructs it and the later layers inside its window of $K=${K}$.`,
  'weak-skip':
    'Weakly causal over every other layer: read one kept layer, reconstruct it and the later kept layers.',
  'strict-global':
    'Each latent reads one layer and predicts only later layers. The last layer has no latent of its own.',
  'strict-local': `Strictly causal inside a window of $K=${K}$: read one layer, predict the next. This is the transcoder mask.`,
  'strict-skip':
    'Strictly causal over every other layer: each latent jumps over a skipped layer to predict later kept layers.',
}

const CODERS = [
  ['sae', 'autoencoder'],
  ['transcoder', 'transcoder'],
  ['acausal-global', 'crosscoder'],
] as const
const CAUSALITY: readonly [Causality, string][] = [
  ['acausal', 'acausal'],
  ['weak', 'weakly causal'],
  ['strict', 'strictly causal'],
]
const LOCALITY: readonly [Locality, string][] = [
  ['global', 'global'],
  ['local', `local, K = ${K}`],
  ['skip', 'skip layer'],
]

const variantsFor = (dims: Dims) =>
  dims === 'coder'
    ? CODERS.map(([v]) => v as string)
    : CAUSALITY.flatMap(([c]) => LOCALITY.map(([l]) => `${c}-${l}`))

type State = {
  cells: string
  used: string
  lit: number[]
  groups: string
  pairs: string
  reads: string
  writes: string
}

const layerList = (ls: number[]) => (ls.length ? ls.map(l => `L${l}`).join(' ') : '–')

const stateFor = (variant: string, f: number): State => {
  const groups = groupsFor(variant)
  const allowed = LAYERS.map(i =>
    LAYERS.map(j => groups.some(g => g.reads.includes(i) && g.writes.includes(j))),
  )
  const lit = groups.flatMap((g, k) => (g.reads.includes(f) ? [k] : []))
  const written = LAYERS.filter(j => allowed[f][j])
  const pairs = allowed.flat().filter(Boolean).length
  return {
    cells: LAYERS.map(i =>
      LAYERS.map(j => (allowed[i][j] ? (i === f ? '2' : '1') : '0')).join(''),
    ).join(''),
    used: LAYERS.map(l =>
      groups.some(g => g.reads.includes(l) || g.writes.includes(l)) ? '1' : '0',
    ).join(''),
    lit,
    groups: String(groups.length),
    pairs: `${pairs} of ${N * N}`,
    reads: String(lit.length),
    writes: layerList(written),
  }
}

// ---------- geometry ----------

const PITCH = 30
const Y0 = 44
// Layer 0 sits at the bottom, as in the preview's diagrams; the stream flows upward.
const layerY = (l: number) => Y0 + (N - 1 - l) * PITCH
const LABEL_X = 4
const LEFT_BOX = 34
const STREAM_X = 96
const RIGHT_BOX = 132
const BOX_W = 30
const BOX_H = 16
const STATION_R = 3.5
const STREAM_W = RIGHT_BOX + BOX_W + 6
const GRID_X = 4
const CELL = 26
const GRID_W = GRID_X + N * PITCH + 26
// Both panels share one height, so side by side their rows line up with the stations.
const VIEW_H = layerY(0) + PITCH

type Seg = { x1: number; y1: number; x2: number; y2: number; start: boolean; end: boolean }

const boxRect = (g: Group) => {
  const ls = [...g.reads, ...g.writes]
  const hi = Math.max(...ls)
  const lo = Math.min(...ls)
  const x = g.side === 'right' ? RIGHT_BOX : LEFT_BOX
  if (!g.span) return { x, y: layerY(g.reads[0]) - BOX_H / 2, w: BOX_W, h: BOX_H }
  return { x, y: layerY(hi) - BOX_H / 2, w: BOX_W, h: layerY(lo) - layerY(hi) + BOX_H }
}

// Edges for one group: a station that is read and written gets a double-headed edge.
const edgesFor = (g: Group): Seg[] => {
  const r = boxRect(g)
  const near = g.side === 'right' ? r.x : r.x + r.w
  const dir = g.side === 'right' ? 1 : -1
  const out: Seg[] = []
  for (const l of [...new Set([...g.reads, ...g.writes])].sort()) {
    const read = g.reads.includes(l)
    const write = g.writes.includes(l)
    if (g.span || l === g.reads[0]) {
      out.push({
        x1: STREAM_X + dir * STATION_R,
        y1: layerY(l),
        x2: near,
        y2: layerY(l),
        start: write,
        end: read,
      })
      continue
    }
    // A write to another layer leaves the box at its read row and lands on that station.
    const y0 = layerY(g.reads[0])
    const dx = STREAM_X - near
    const dy = layerY(l) - y0
    const len = Math.hypot(dx, dy)
    out.push({
      x1: near,
      y1: y0,
      x2: STREAM_X - (dx / len) * STATION_R,
      y2: layerY(l) - (dy / len) * STATION_R,
      start: false,
      end: true,
    })
  }
  return out
}

let instance = 0

const CrosscoderMasksImpl: QuartzMdxComponent<Props> = ({
  caption,
  dims = 'coder',
  coder = 'crosscoder',
  causality = 'weak',
  locality = 'global',
  read = 1,
}) => {
  const id = `ccm-${++instance}`
  const d: Dims = dims === 'variant' ? 'variant' : 'coder'
  const variants = variantsFor(d)
  const c0 = CAUSALITY.some(([c]) => c === causality) ? causality : 'weak'
  const l0 = LOCALITY.some(([l]) => l === locality) ? locality : 'global'
  const v0 =
    d === 'coder'
      ? coder === 'sae' || coder === 'transcoder'
        ? coder
        : 'acausal-global'
      : `${c0}-${l0}`
  const f0 = LAYERS.includes(read) ? read : 1
  const states: Record<string, State> = {}
  for (const v of variants) for (const f of LAYERS) states[`${v}:${f}`] = stateFor(v, f)
  const s0 = states[`${v0}:${f0}`]
  const groupsByVariant = Object.fromEntries(variants.map(v => [v, groupsFor(v)]))
  const arrow = `${id}-arrow`

  const segments = (
    name: string,
    legend: string,
    options: readonly (readonly [string, string])[],
    checked: string,
  ) => (
    <fieldset class={`ccm-segments ccm-segments--${name}`}>
      <legend class="ccm-legend">{legend}</legend>
      {options.map(([value, label]) => (
        <label class="ccm-segment">
          <input
            type="radio"
            name={`${id}-${name}`}
            value={value}
            checked={value === checked}
            data-ccm-input={name}
          />
          <span>{label}</span>
        </label>
      ))}
    </fieldset>
  )

  return (
    <figure
      class="crosscoder-masks"
      data-crosscoder-masks
      data-ccm-dims={d}
      data-ccm-variant={v0}
      data-ccm-states={JSON.stringify(states)}
    >
      <div class="ccm-controls">
        {d === 'coder'
          ? segments('coder', 'Coder', CODERS, v0)
          : [
              segments('causality', 'Causality', CAUSALITY, c0),
              segments('locality', 'Locality', LOCALITY, l0),
            ]}
        {segments(
          'read',
          'Read layer',
          LAYERS.map(l => [String(l), `L${l}`] as const),
          String(f0),
        )}
      </div>

      {variants.map(v => (
        <p class="ccm-rule" data-ccm-rule={v} data-on={v === v0 ? 'true' : 'false'}>
          <MathText text={RULES[v]} mathClass="ccm-math" />
        </p>
      ))}

      <div class="ccm-body">
        <svg
          class="ccm-svg ccm-svg--stream"
          viewBox={`0 0 ${STREAM_W} ${VIEW_H}`}
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label={`A residual stream of ${N} layers with the latent groups of the selected coder; the groups that read the selected layer are lit with their reads and reconstructions.`}
        >
          <defs>
            <marker
              id={arrow}
              viewBox="0 0 10 10"
              refX="10"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              markerUnits="userSpaceOnUse"
              orient="auto-start-reverse"
            >
              <path class="ccm-arrowhead" d="M0,0 L10,5 L0,10 z" />
            </marker>
          </defs>

          <line class="ccm-stream" x1={STREAM_X} y1={layerY(N - 1)} x2={STREAM_X} y2={layerY(0)} />
          {LAYERS.map(l => (
            <g>
              <text class="ccm-axis" x={LABEL_X} y={layerY(l) + 3}>
                {`L${l}`}
              </text>
              <circle
                class="ccm-station"
                data-ccm-station={String(l)}
                data-used={s0.used[l]}
                cx={STREAM_X}
                cy={layerY(l)}
                r={STATION_R}
              />
            </g>
          ))}

          {variants.map(v => (
            <g class="ccm-variant" data-ccm-variant-group={v} data-on={v === v0 ? 'true' : 'false'}>
              {groupsByVariant[v].map((g, k) => {
                const r = boxRect(g)
                return (
                  <rect
                    class="ccm-box"
                    data-ccm-box={String(k)}
                    data-lit={v === v0 && s0.lit.includes(k) ? 'true' : 'false'}
                    x={r.x}
                    y={r.y}
                    width={r.w}
                    height={r.h}
                  />
                )
              })}
              {groupsByVariant[v].map((g, k) => (
                <g
                  class="ccm-edges"
                  data-ccm-edges={String(k)}
                  data-lit={v === v0 && s0.lit.includes(k) ? 'true' : 'false'}
                >
                  {edgesFor(g).map(e => (
                    <line
                      class="ccm-edge"
                      x1={e.x1}
                      y1={e.y1}
                      x2={e.x2}
                      y2={e.y2}
                      marker-start={e.start ? `url(#${arrow})` : undefined}
                      marker-end={e.end ? `url(#${arrow})` : undefined}
                    />
                  ))}
                </g>
              ))}
            </g>
          ))}
        </svg>
        <svg
          class="ccm-svg ccm-svg--grid"
          viewBox={`0 0 ${GRID_W} ${VIEW_H}`}
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label={`A ${N} by ${N} grid of read layer against reconstructed layer; allowed pairs are filled.`}
        >
          <text class="ccm-axis" x={GRID_X + (N * PITCH) / 2} y={12} text-anchor="middle">
            reconstructed layer
          </text>
          {LAYERS.map(j => (
            <text
              class="ccm-axis"
              x={GRID_X + j * PITCH + CELL / 2}
              y={Y0 - CELL / 2 - 5}
              text-anchor="middle"
            >
              {`L${j}`}
            </text>
          ))}
          <text class="ccm-axis" x={GRID_X + N * PITCH + 2} y={Y0 - CELL / 2 - 5}>
            read
          </text>
          {LAYERS.map(i => (
            <text class="ccm-axis" x={GRID_X + N * PITCH + 2} y={layerY(i) + 3}>
              {`L${i}`}
            </text>
          ))}
          {LAYERS.flatMap(i =>
            LAYERS.map(j => (
              <rect
                class="ccm-cell"
                data-ccm-cell={String(i * N + j)}
                data-state={s0.cells[i * N + j]}
                x={GRID_X + j * PITCH}
                y={layerY(i) - CELL / 2}
                width={CELL}
                height={CELL}
              />
            )),
          )}
        </svg>
      </div>

      <dl class="ccm-readout">
        <div>
          <dt>Latent groups</dt>
          <dd data-ccm-out="groups">{s0.groups}</dd>
        </div>
        <div>
          <dt>Read to reconstruct pairs</dt>
          <dd data-ccm-out="pairs">{s0.pairs}</dd>
        </div>
        <div>
          <dt>Groups that read the selected layer</dt>
          <dd data-ccm-out="reads">{s0.reads}</dd>
        </div>
        <div>
          <dt>Layers reconstructed from it</dt>
          <dd data-ccm-out="writes">{s0.writes}</dd>
        </div>
      </dl>

      {caption ? (
        <figcaption class="ccm-caption">
          <MathText text={caption} mathClass="ccm-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const CrosscoderMasksComponent = CrosscoderMasksImpl as QuartzMdxComponent<Props>
CrosscoderMasksComponent.css = style
CrosscoderMasksComponent.afterDOMLoaded = script

export const CrosscoderMasks = registerMdxComponent('CrosscoderMasks', CrosscoderMasksComponent)

export default (() => CrosscoderMasks) satisfies (opts: undefined) => QuartzMdxComponent<Props>
