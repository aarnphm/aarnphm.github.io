import { type FunctionalComponent } from 'preact'
import { renderFigureLabel } from '../../util/figure-label'
import { renderInlineMath, MathText } from '../../util/math-text'
import { BF16, E2M1, quantizeMXBlock, roundToFormat } from '../../util/minifloat'
//@ts-ignore
import script from '../scripts/mx-compute-flow.inline'
import style from '../styles/mxComputeFlow.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; pass?: string }

const BLOCK = 32
const M = 32
const K = 64
const N = 32

type Matrix = number[][]
type Axis = 'row' | 'col'

const mulberry32 = (seed: number) => () => {
  seed = (seed + 0x6d2b79f5) | 0
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}
const normal = (rand: () => number) =>
  Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand())
const bf16 = (x: number) => roundToFormat(x, BF16).value

// Synthetic BF16 tensors. Activations carry two outlier feature channels, the pattern reported for
// transformer activations; output gradients get one log-normal scale per token row.
const OUTLIER_CHANNELS = new Set([7, 40])
const tensor = (
  rows: number,
  cols: number,
  seed: number,
  entry: (r: number, c: number, z: number) => number,
) => {
  const rand = mulberry32(seed)
  return Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) => bf16(entry(r, c, normal(rand)))),
  )
}
const A = tensor(M, K, 3, (_, c, z) => z * (OUTLIER_CHANNELS.has(c) ? 12 : 1))
const W = tensor(K, N, 5, (_, __, z) => 0.2 * z)
const rowScale = (() => {
  const rand = mulberry32(9)
  return Array.from({ length: M }, () => Math.exp(normal(rand)))
})()
const E = tensor(M, N, 7, (r, _, z) => z * rowScale[r])

const transpose = (x: Matrix): Matrix => x[0].map((_, c) => x.map(row => row[c]))

/** Dequantized MXFP4 copy with blocks of 32 running along rows or along columns. */
const mxCopy = (x: Matrix, along: Axis): Matrix => {
  if (along === 'col') return transpose(mxCopy(transpose(x), 'row'))
  return x.map(row => {
    const out: number[] = []
    for (let b = 0; b < row.length; b += BLOCK)
      out.push(...quantizeMXBlock(row.slice(b, b + BLOCK), E2M1).dequantized)
    return out
  })
}
const matmul = (x: Matrix, y: Matrix): Matrix =>
  x.map(row => y[0].map((_, c) => row.reduce((s, v, k) => s + v * y[k][c], 0)))
const differing = (x: Matrix, y: Matrix) =>
  x.reduce((s, row, r) => s + row.filter((v, c) => v !== y[r][c]).length, 0)
const relRms = (approx: Matrix, ref: Matrix) => {
  let num = 0
  let den = 0
  ref.forEach((row, r) =>
    row.forEach((v, c) => {
      num += (approx[r][c] - v) ** 2
      den += v * v
    }),
  )
  return Math.sqrt(num / den)
}

// Each tensor gets two MX copies, one per reduction axis that consumes it.
const COPIES = {
  A: { byK: mxCopy(A, 'row'), byM: mxCopy(A, 'col') },
  W: { byK: mxCopy(W, 'col'), byN: mxCopy(W, 'row') },
  E: { byM: mxCopy(E, 'col'), byN: mxCopy(E, 'row') },
}

type Operand = { tex: string; shape: string; rows: number; cols: number }
type Pass = {
  id: string
  label: string
  product: string
  reduce: string
  length: number
  left: Operand
  right: Operand
  out: Operand & { name: string }
  lit: [number, number]
  ref: Matrix
  mx: Matrix
}

const pass = (
  p: Omit<Pass, 'ref' | 'mx'>,
  [x, y]: [Matrix, Matrix],
  [qx, qy]: [Matrix, Matrix],
): Pass => ({ ...p, ref: matmul(x, y), mx: matmul(qx, qy) })

const PASSES: readonly Pass[] = [
  pass(
    {
      id: 'forward',
      label: 'Forward',
      product: 'A_{i-1}W_i',
      reduce: 'K',
      length: K,
      left: { tex: 'A_{i-1}', shape: '[M,K^Q]', rows: M, cols: K },
      right: { tex: 'W_i', shape: '[K^Q,N]', rows: K, cols: N },
      out: { tex: 'A_i', shape: '[M,N]', rows: M, cols: N, name: 'activation' },
      lit: [5, 9],
    },
    [A, W],
    [COPIES.A.byK, COPIES.W.byK],
  ),
  pass(
    {
      id: 'wgrad',
      label: 'Weight gradient',
      product: 'A_{i-1}^{\\top}E_i',
      reduce: 'M',
      length: M,
      left: { tex: 'A_{i-1}^{\\top}', shape: '[K,M^Q]', rows: K, cols: M },
      right: { tex: 'E_i', shape: '[M^Q,N]', rows: M, cols: N },
      out: { tex: 'G_i', shape: '[K,N]', rows: K, cols: N, name: 'weight gradient' },
      lit: [7, 9],
    },
    [transpose(A), E],
    [transpose(COPIES.A.byM), COPIES.E.byM],
  ),
  pass(
    {
      id: 'dgrad',
      label: 'Input gradient',
      product: 'E_iW_i^{\\top}',
      reduce: 'N',
      length: N,
      left: { tex: 'E_i', shape: '[M,N^Q]', rows: M, cols: N },
      right: { tex: 'W_i^{\\top}', shape: '[N^Q,K]', rows: N, cols: K },
      out: { tex: 'E_{i-1}', shape: '[M,K]', rows: M, cols: K, name: 'input gradient' },
      lit: [5, 7],
    },
    [E, transpose(W)],
    [COPIES.E.byN, transpose(COPIES.W.byN)],
  ),
]

const COPY_ROWS: readonly {
  tensor: string
  copy: string
  along: string
  pass: string
  size: number
  differ: number
}[] = [
  {
    tensor: 'A_{i-1}',
    copy: 'A_{i-1}',
    along: 'K',
    pass: 'forward',
    size: M * K,
    differ: differing(COPIES.A.byK, COPIES.A.byM),
  },
  {
    tensor: 'A_{i-1}',
    copy: 'A_{i-1}^{\\top}',
    along: 'M',
    pass: 'wgrad',
    size: M * K,
    differ: differing(COPIES.A.byK, COPIES.A.byM),
  },
  {
    tensor: 'W_i',
    copy: 'W_i',
    along: 'K',
    pass: 'forward',
    size: K * N,
    differ: differing(COPIES.W.byK, COPIES.W.byN),
  },
  {
    tensor: 'W_i',
    copy: 'W_i^{\\top}',
    along: 'N',
    pass: 'dgrad',
    size: K * N,
    differ: differing(COPIES.W.byK, COPIES.W.byN),
  },
  {
    tensor: 'E_i',
    copy: 'E_i',
    along: 'M',
    pass: 'wgrad',
    size: M * N,
    differ: differing(COPIES.E.byM, COPIES.E.byN),
  },
  {
    tensor: 'E_i',
    copy: 'E_i',
    along: 'N',
    pass: 'dgrad',
    size: M * N,
    differ: differing(COPIES.E.byM, COPIES.E.byN),
  },
]

const P = 3
const GAP = 44
const PAD = 4
// Every pass fits the larger reduction (K) on one side and the smaller dimension on the other.
const VIEW_W = PAD * 2 + K * P + GAP + N * P
const VIEW_H = VIEW_W + 22

const math = (tex: string) => renderFigureLabel(tex, m => renderInlineMath(m))
const fmt = (x: number) => (Math.abs(x) >= 0.01 ? x.toFixed(3) : x.toExponential(2))

const Label: FunctionalComponent<{
  x: number
  y: number
  w: number
  tex: string
  align: 'start' | 'end'
}> = ({ x, y, w, tex, align }) => (
  <foreignObject x={align === 'end' ? x - w : x} y={y} width={w} height={18}>
    <div class={`mxf-fo mxf-fo--${align}`} dangerouslySetInnerHTML={{ __html: math(tex) }} />
  </foreignObject>
)

const PassPanel: FunctionalComponent<{ p: Pass }> = ({ p }) => {
  const R = p.length
  const outX = PAD + R * P + GAP
  const outY = PAD + R * P + GAP
  const [lr, lc] = p.lit
  const strip = P - 1
  const run = BLOCK * P - 1.5
  return (
    <g class="mxf-pass" data-mxf-panel={p.id}>
      {Array.from({ length: p.left.rows }, (_, r) =>
        Array.from({ length: R / BLOCK }, (_, b) => (
          <rect
            class={`mxf-block${r === lr ? ' mxf-block--lit' : ''}`}
            x={PAD + b * BLOCK * P}
            y={outY + r * P}
            width={run}
            height={strip}
          />
        )),
      )}
      {Array.from({ length: p.right.cols }, (_, c) =>
        Array.from({ length: R / BLOCK }, (_, b) => (
          <rect
            class={`mxf-block${c === lc ? ' mxf-block--lit' : ''}`}
            x={outX + c * P}
            y={PAD + b * BLOCK * P}
            width={strip}
            height={run}
          />
        )),
      )}
      <rect
        class="mxf-out"
        x={outX}
        y={outY}
        width={p.out.cols * P - 1}
        height={p.out.rows * P - 1}
      />
      <rect
        class="mxf-out-lit"
        x={outX + lc * P - 1}
        y={outY + lr * P - 1}
        width={P + 1}
        height={P + 1}
      />
      <Label
        x={PAD}
        y={outY - 40}
        w={R * P + GAP - 6}
        align="start"
        tex={`${p.left.tex}\\ \\text{MX}${p.left.shape}`}
      />
      <Label
        x={PAD}
        y={outY - 22}
        w={R * P + GAP - 6}
        align="start"
        tex={`\\text{blocks along } ${p.reduce}`}
      />
      <Label
        x={outX - 6}
        y={PAD}
        w={R * P + GAP - 12}
        align="end"
        tex={`${p.right.tex}\\ \\text{MX}${p.right.shape}`}
      />
      <Label
        x={outX - 6}
        y={PAD + 18}
        w={R * P + GAP - 12}
        align="end"
        tex={`\\text{blocks along } ${p.reduce}`}
      />
      <Label
        x={outX}
        y={outY + p.out.rows * P + 4}
        w={VIEW_W - outX}
        align="start"
        tex={`${p.out.tex}\\ \\text{BF16}${p.out.shape}`}
      />
    </g>
  )
}

const READOUT = (p: Pass): [string, string][] => {
  const [r, c] = p.lit
  const ref = p.ref[r][c]
  const mx = p.mx[r][c]
  return [
    ['Product', `$${p.product}$`],
    ['Reduction length', `$${p.reduce}=${p.length}$`],
    ['Shared scales per operand row or column', `$${p.length}/${BLOCK} = ${p.length / BLOCK}$`],
    [`Lit output from BF16 inputs`, `$${fmt(ref)}$`],
    [`Lit output from MXFP4 copies`, `$${fmt(mx)}$`],
    ['Relative RMS error over the whole output', `$${(relRms(p.mx, p.ref) * 100).toFixed(1)}\\%$`],
  ]
}

let instance = 0

const MXComputeFlowImpl: QuartzMdxComponent<Props> = ({ caption, pass: initial = 'forward' }) => {
  const id = `mxf-${++instance}`
  const p0 = PASSES.some(p => p.id === initial) ? initial : 'forward'

  return (
    <figure class="mx-compute-flow" data-mx-compute-flow data-mxf-pass={p0}>
      <fieldset class="mxf-segments">
        <legend class="mxf-legend">MatMul</legend>
        {PASSES.map(p => (
          <label class="mxf-segment">
            <input
              type="radio"
              name={`${id}-pass`}
              value={p.id}
              checked={p.id === p0}
              data-mxf-pass-input
            />
            <span>{p.label}</span>
          </label>
        ))}
      </fieldset>

      <div class="mxf-stage">
        <svg
          class="mxf-svg"
          viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
          role="img"
          aria-label="The two MX operands of the selected matrix product, drawn element by element with blocks of 32 along the reduction axis, and the BF16 output."
        >
          {PASSES.map(p => (
            <PassPanel p={p} />
          ))}
        </svg>

        <div class="mxf-side">
          {PASSES.map(p => (
            <dl class="mxf-readout" data-mxf-panel={p.id}>
              {READOUT(p).map(([label, value]) => (
                <div>
                  <dt>{label}</dt>
                  <dd>
                    <MathText text={value} mathClass="mxf-math" />
                  </dd>
                </div>
              ))}
            </dl>
          ))}
        </div>
      </div>

      <table class="mxf-copies">
        <caption>MXFP4 copies made from each BF16 tensor</caption>
        <thead>
          <tr>
            <th scope="col">BF16 source</th>
            <th scope="col">MX operand</th>
            <th scope="col">Blocks along</th>
            <th scope="col">Used by</th>
            <th scope="col">Values that differ between the two copies</th>
          </tr>
        </thead>
        <tbody>
          {COPY_ROWS.map((row, i) => (
            <tr data-mxf-use={row.pass}>
              {i % 2 === 0 ? (
                <th
                  scope="rowgroup"
                  rowSpan={2}
                  dangerouslySetInnerHTML={{ __html: math(row.tensor) }}
                />
              ) : null}
              <td dangerouslySetInnerHTML={{ __html: math(row.copy) }} />
              <td dangerouslySetInnerHTML={{ __html: math(row.along) }} />
              <td>{PASSES.find(p => p.id === row.pass)?.label}</td>
              {i % 2 === 0 ? (
                <td class="mxf-num" rowSpan={2}>
                  {`${row.differ} / ${row.size}`}
                </td>
              ) : null}
            </tr>
          ))}
        </tbody>
      </table>

      {caption ? (
        <figcaption class="mxf-caption">
          <MathText text={caption} mathClass="mxf-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const MXComputeFlowComponent = MXComputeFlowImpl as QuartzMdxComponent<Props>
MXComputeFlowComponent.css = style
MXComputeFlowComponent.afterDOMLoaded = script

export const MXComputeFlow = registerMdxComponent('MXComputeFlow', MXComputeFlowComponent)

export default (() => MXComputeFlow) satisfies (opts: undefined) => QuartzMdxComponent<Props>
