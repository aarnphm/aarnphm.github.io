import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/all-reduce-steps.inline'
import style from '../styles/allReduceSteps.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; gpus?: number }

const GPU_COUNTS = [2, 3, 4, 6] as const
const LETTERS = 'ABCDEF'

type CellState = 'own' | 'partial' | 'recv' | 'full'
type Step = {
  phase: 'start' | 'reduce-scatter' | 'all-gather'
  labels: string[]
  states: CellState[]
  senders: number[]
  sends: string
  sent: string
}

const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b))
const frac = (num: number, den: number) => {
  if (num === 0) return '0'
  const g = gcd(num, den)
  return den / g === 1 ? `${num / g}` : `${num / g}/${den / g}`
}

// Ring all-reduce on N GPUs, each holding N chunks. Reduce-scatter step t: GPU i sends its
// partial of chunk (i - t) mod N to GPU i + 1, which adds its own. After N - 1 steps GPU i holds
// the full sum of chunk i. All-gather step t: GPU i forwards complete chunk (i - t + 1) mod N.
const ringAllReduce = (n: number): Step[] => {
  const idx = (gpu: number, chunk: number) => chunk * n + gpu
  let parts: Set<number>[] = Array.from({ length: n * n }, (_, k) => new Set([k % n]))
  const label = (s: Set<number>) =>
    s.size === n
      ? 'Σ'
      : [...s]
          .sort((a, b) => a - b)
          .map(g => LETTERS[g])
          .join('+')
  const snapshot = (
    phase: Step['phase'],
    recv: Set<number>,
    senders: number[],
    sends: string,
    stepsDone: number,
  ): Step => ({
    phase,
    labels: parts.map(label),
    states: parts.map((s, k) =>
      recv.has(k) ? 'recv' : s.size === n ? 'full' : s.size > 1 ? 'partial' : 'own',
    ),
    senders,
    sends,
    sent: `${frac(stepsDone, n)} S`,
  })
  const mod = (a: number) => ((a % n) + n) % n
  const steps: Step[] = [snapshot('start', new Set(), [], '–', 0)]
  for (let phase = 0; phase < 2; phase++) {
    for (let t = 1; t < n; t++) {
      const next = parts.map(s => new Set(s))
      const recv = new Set<number>()
      const senders: number[] = []
      const sends: string[] = []
      for (let i = 0; i < n; i++) {
        const chunk = phase === 0 ? mod(i - t) : mod(i - t + 1)
        const to = mod(i + 1)
        const from = parts[idx(i, chunk)]
        next[idx(to, chunk)] =
          phase === 0 ? new Set([...next[idx(to, chunk)], ...from]) : new Set(from)
        recv.add(idx(to, chunk))
        senders.push(idx(i, chunk))
        sends.push(`${LETTERS[i]}→${LETTERS[to]} c${chunk}`)
      }
      parts = next
      steps.push(
        snapshot(
          phase === 0 ? 'reduce-scatter' : 'all-gather',
          recv,
          senders,
          sends.join(', '),
          phase * (n - 1) + t,
        ),
      )
    }
  }
  return steps
}

const STATES: Record<number, Step[]> = Object.fromEntries(
  GPU_COUNTS.map(n => [n, ringAllReduce(n)]),
)

const totalSent = (n: number) => `2(${n}−1)/${n} S = ${frac(2 * (n - 1), n)} S`
const CLIENT = Object.fromEntries(
  GPU_COUNTS.map(n => [
    n,
    { steps: STATES[n], chunk: `S/${n}`, total: totalSent(n), count: 2 * (n - 1) },
  ]),
)

const CELL_W = 58
const CELL_H = 26
const GAP_X = 12
const GAP_Y = 4
const HEAD_H = 20
const LABEL_W = 30
const RING_Y = 8
const GRID_Y = RING_Y + 12 + HEAD_H + 6

const geometry = (n: number) => {
  const colX = (g: number) => LABEL_W + g * (CELL_W + GAP_X)
  const width = colX(n) - GAP_X + 2
  const height = GRID_Y + n * (CELL_H + GAP_Y)
  return { colX, width, height }
}

let instance = 0

const AllReduceStepsImpl: QuartzMdxComponent<Props> = ({ caption, gpus = 4 }) => {
  const id = `ars-${++instance}`
  const n0 = (GPU_COUNTS as readonly number[]).includes(gpus) ? gpus : 4
  const s0 = STATES[n0][0]

  return (
    <figure
      class="all-reduce-steps"
      data-all-reduce-steps
      data-ars-n={String(n0)}
      data-ars-step="0"
      data-ars-states={JSON.stringify(CLIENT)}
    >
      <div class="ars-controls">
        <fieldset class="ars-segments">
          <legend class="ars-legend">
            <MathText text="GPUs $N$" mathClass="ars-math" />
          </legend>
          {GPU_COUNTS.map(n => (
            <label class="ars-segment">
              <input
                type="radio"
                name={`${id}-n`}
                value={String(n)}
                checked={n === n0}
                data-ars-n-input
              />
              <span>{String(n)}</span>
            </label>
          ))}
        </fieldset>
        <div class="ars-stepper">
          <button type="button" class="ars-nav" data-ars-nav="reset" disabled>
            reset
          </button>
          <button type="button" class="ars-nav" data-ars-nav="-1" disabled>
            prev
          </button>
          <button type="button" class="ars-nav" data-ars-nav="1">
            next
          </button>
        </div>
      </div>

      <p class="ars-status" aria-live="polite">
        <span class="ars-status-step" data-ars-out="step">{`step 0 of ${2 * (n0 - 1)}`}</span>
        {' · '}
        <span data-ars-out="phase">{s0.phase}</span>
      </p>

      {GPU_COUNTS.map(n => {
        const { colX, width, height } = geometry(n)
        const step = STATES[n][0]
        const headY = RING_Y + 12
        const mid = (g: number) => colX(g) + CELL_W / 2
        return (
          <svg
            class="ars-svg"
            data-ars-grid={String(n)}
            viewBox={`0 0 ${width} ${height}`}
            style={{ maxWidth: `${(width * 1.25).toFixed(0)}px` }}
            hidden={n !== n0}
            role="img"
            aria-label={`${n} GPUs in a ring, each holding ${n} chunks; columns are GPUs, rows are chunks.`}
          >
            <path
              class="ars-ring"
              d={`M ${mid(n - 1)} ${headY} V ${RING_Y} H ${mid(0)} V ${headY - 4}`}
            />
            <path
              class="ars-ring-head"
              d={`M ${mid(0) - 3} ${headY - 5} L ${mid(0)} ${headY} L ${mid(0) + 3} ${headY - 5} Z`}
            />
            {Array.from({ length: n }, (_, g) => (
              <g>
                <rect class="ars-head" x={colX(g)} y={headY} width={CELL_W} height={HEAD_H} />
                <text class="ars-head-label" x={mid(g)} y={headY + 13.5} text-anchor="middle">
                  {`GPU ${LETTERS[g]}`}
                </text>
                {g < n - 1 ? (
                  <g>
                    <path
                      class="ars-ring"
                      d={`M ${colX(g) + CELL_W} ${headY + HEAD_H / 2} H ${colX(g + 1) - 4}`}
                    />
                    <path
                      class="ars-ring-head"
                      d={`M ${colX(g + 1) - 5} ${headY + HEAD_H / 2 - 3} L ${colX(g + 1)} ${headY + HEAD_H / 2} L ${colX(g + 1) - 5} ${headY + HEAD_H / 2 + 3} Z`}
                    />
                  </g>
                ) : null}
              </g>
            ))}
            {Array.from({ length: n }, (_, c) => (
              <text
                class="ars-row-label"
                x={LABEL_W - 5}
                y={GRID_Y + c * (CELL_H + GAP_Y) + CELL_H / 2 + 3.5}
                text-anchor="end"
              >
                {`c${c}`}
              </text>
            ))}
            {Array.from({ length: n * n }, (_, k) => {
              const g = k % n
              const c = Math.floor(k / n)
              const x = colX(g)
              const y = GRID_Y + c * (CELL_H + GAP_Y)
              return (
                <g
                  class="ars-cell"
                  data-ars-cell={String(k)}
                  data-ars-state={step.states[k]}
                  data-ars-sender="false"
                >
                  <rect x={x} y={y} width={CELL_W} height={CELL_H} />
                  <text x={x + CELL_W / 2} y={y + CELL_H / 2 + 3.5} text-anchor="middle">
                    {step.labels[k]}
                  </text>
                </g>
              )
            })}
          </svg>
        )
      })}

      <ul class="ars-legend-list">
        <li>
          <span class="ars-swatch" data-ars-state="own" aria-hidden="true" />
          own chunk
        </li>
        <li>
          <span class="ars-swatch" data-ars-state="partial" aria-hidden="true" />
          partial sum
        </li>
        <li>
          <span class="ars-swatch" data-ars-state="recv" aria-hidden="true" />
          received this step
        </li>
        <li>
          <span class="ars-swatch ars-swatch--sender" aria-hidden="true" />
          sent this step
        </li>
        <li>
          <span class="ars-swatch" data-ars-state="full" aria-hidden="true" />
          Σ, sum over all GPUs
        </li>
      </ul>

      <dl class="ars-readout">
        <div>
          <dt>Sends this step</dt>
          <dd data-ars-out="sends">{s0.sends}</dd>
        </div>
        <div>
          <dt>Chunk per send</dt>
          <dd data-ars-out="chunk">{CLIENT[n0].chunk}</dd>
        </div>
        <div>
          <dt>Sent per GPU so far</dt>
          <dd data-ars-out="sent">{s0.sent}</dd>
        </div>
        <div>
          <dt>Sent per GPU, whole all-reduce</dt>
          <dd data-ars-out="total">{CLIENT[n0].total}</dd>
        </div>
      </dl>

      {caption ? (
        <figcaption class="ars-caption">
          <MathText text={caption} mathClass="ars-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const AllReduceStepsComponent = AllReduceStepsImpl as QuartzMdxComponent<Props>
AllReduceStepsComponent.css = style
AllReduceStepsComponent.afterDOMLoaded = script

export const AllReduceSteps = registerMdxComponent('AllReduceSteps', AllReduceStepsComponent)

export default (() => AllReduceSteps) satisfies (opts: undefined) => QuartzMdxComponent<Props>
