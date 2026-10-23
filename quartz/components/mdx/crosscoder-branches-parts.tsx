import type { ComponentChildren } from 'preact'
import { NEURONS_PER_BLOCK } from '../../util/crosscoder-branches'
import { renderInlineMath } from '../../util/math-text'

// Shared panel grammar: a residual stream on the left, blocks that read from it and add back.
export const W = 200
export const H = 196
export const SX = 40
const JR = 6
const NEURON_GAP = 16
const BLOCK_H = 24

type Align = 'left' | 'center' | 'right'

export const Label = ({
  x,
  y,
  w,
  h = 16,
  tex,
  align = 'left',
}: {
  x: number
  y: number
  w: number
  h?: number
  tex: string
  align?: Align
}) => (
  <foreignObject x={x} y={y} width={w} height={h}>
    <div
      class={`cb-fo cb-fo--${align}`}
      dangerouslySetInnerHTML={{ __html: renderInlineMath(tex) }}
    />
  </foreignObject>
)

export const Junction = ({ x, y }: { x: number; y: number }) => (
  <g class="cb-junction">
    <circle cx={x} cy={y} r={JR} />
    <path d={`M${x - 3},${y} H${x + 3} M${x},${y - 3} V${y + 3}`} />
  </g>
)

export const blockWidth = (neurons: number) => neurons * NEURON_GAP + 12

// A block of neurons centred on cx; `lit` lists the neurons that carry the feature.
export const Block = ({
  cx,
  y,
  lit,
  neurons = NEURONS_PER_BLOCK,
  tex,
}: {
  cx: number
  y: number
  lit?: readonly number[]
  neurons?: number
  tex?: string
}) => {
  const w = tex ? 76 : blockWidth(neurons)
  const x = cx - w / 2
  return (
    <g>
      <rect class="cb-block" x={x} y={y} width={w} height={BLOCK_H} />
      {tex ? (
        <Label x={x} y={y + 4} w={w} tex={tex} align="center" />
      ) : (
        Array.from({ length: neurons }, (_, i) => (
          <circle
            class={lit?.includes(i) ? 'cb-neuron cb-neuron--lit' : 'cb-neuron'}
            cx={x + 6 + NEURON_GAP / 2 + i * NEURON_GAP}
            cy={y + BLOCK_H / 2}
            r={5}
          />
        ))
      )}
    </g>
  )
}

export const Stream = ({ to = H - 6 }: { to?: number }) => (
  <line class="cb-stream" x1={SX} y1={6} x2={SX} y2={to} />
)

// One block on the stream: read r at `read`, add the update back at `read + 88`.
export const OneBlock = ({ lit, neurons }: { lit: readonly number[]; neurons: number }) => {
  const cx = 120
  const read = 40
  const top = read + 18
  const write = top + BLOCK_H + 34
  return (
    <>
      <Stream />
      <path class="cb-wire" d={`M${SX},${read} H${cx} V${top}`} />
      <path class="cb-wire" d={`M${cx},${top + BLOCK_H} V${write} H${SX + JR}`} />
      <Block cx={cx} y={top} lit={lit} neurons={neurons} />
      <Junction x={SX} y={write} />
      <Label x={0} y={read - 8} w={SX - 6} tex="r_l" align="right" />
      <Label x={cx + 4} y={write - 22} w={60} tex="u_l" />
      <Label x={0} y={write + 14} w={SX - 6} tex="r_{l+1}" align="right" />
    </>
  )
}

type Pair = { l?: readonly number[]; next?: readonly number[] }

// Block l+1 reads r_{l+1} = r_l + u_l: the usual sequential drawing.
export const Sequential = ({ l, next }: Pair) => {
  const cx = 128
  const rows = [
    { read: 22, lit: l, tex: '\\text{block } l', u: 'u_l', r: 'r_{l+1}' },
    { read: 100, lit: next, tex: '\\text{block } l{+}1', u: 'u_{l+1}', r: 'r_{l+2}' },
  ]
  return (
    <>
      <Stream />
      <Label x={0} y={6} w={SX - 6} tex="r_l" align="right" />
      {rows.map(({ read, lit, tex, u, r }) => {
        const top = read + 18
        const write = top + BLOCK_H + 16
        return (
          <>
            <path class="cb-wire" d={`M${SX},${read} H${cx} V${top}`} />
            <path class="cb-wire" d={`M${cx},${top + BLOCK_H} V${write} H${SX + JR}`} />
            <Block cx={cx} y={top} lit={lit} tex={lit ? undefined : tex} />
            <Junction x={SX} y={write} />
            <Label x={cx + 4} y={write - 18} w={60} tex={u} />
            <Label x={0} y={write + 4} w={SX - 6} tex={r} align="right" />
          </>
        )
      })}
    </>
  )
}

// Both blocks read r_l. Block l+1 also receives u_l through one soft edge; when that edge is
// weak the two blocks act almost in parallel and their sum lands at r_{l+2}.
export const Parallel = ({ l, next }: Pair) => {
  const read = 22
  const cxL = 148
  const cxN = 88
  const topL = 40
  const dep = 92
  const topN = 116
  const merge = 170
  return (
    <>
      <Stream />
      <path class="cb-wire" d={`M${SX},${read} H${cxL} V${topL}`} />
      <path class="cb-wire" d={`M${cxN},${read} V${dep - JR} M${cxN},${dep + JR} V${topN}`} />
      <path class="cb-wire" d={`M${cxL},${topL + BLOCK_H} V${merge} H${SX + JR}`} />
      <path class="cb-wire" d={`M${cxN},${topN + BLOCK_H} V${merge}`} />
      <path class="cb-wire cb-wire--soft" d={`M${cxL},${dep} H${cxN + JR}`} />
      <circle class="cb-tap" cx={cxL} cy={dep} r={2.5} />
      <Junction x={cxN} y={dep} />
      <Block cx={cxL} y={topL} lit={l} tex={l ? undefined : '\\text{block } l'} />
      <Block cx={cxN} y={topN} lit={next} tex={next ? undefined : '\\text{block } l{+}1'} />
      <Junction x={SX} y={merge} />
      <circle class="cb-read" cx={SX} cy={read} r={4.5} />
      <Label x={0} y={read - 8} w={SX - 8} tex="r_l" align="right" />
      <Label x={cxL + 4} y={dep + 8} w={44} tex="u_l" />
      <Label x={cxN + 4} y={merge - 22} w={44} tex="u_{l+1}" />
      <Label x={0} y={merge + 6} w={SX - 6} tex="r_{l+2}" align="right" />
    </>
  )
}

export const Panel = ({ label, children }: { label: string; children: ComponentChildren }) => (
  <svg
    class="cb-svg"
    data-fig-scale="--cb-u"
    viewBox={`0 0 ${W} ${H}`}
    role="img"
    aria-label={label}
  >
    {children}
  </svg>
)
