import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/grammar-bitmask-batch.inline'
import style from '../styles/grammarBitmaskBatch.scss'
import data from './jump-forward-boundaries.json'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; request?: string; workers?: number }

type Token = { id: number; text: string; p: number }
type Request = {
  id: string
  label: string
  state: number | null
  status: 'running' | 'waiting'
  allowed: Token[]
  raw: Token
  pick: Token
  pickShare: number
  words: number[]
  bins: Set<number>
}

const VOCAB = data.vocab
const WORDS = data.bitmaskWords
const WORDS_PER_BIN = 40
const BINS = Math.ceil(WORDS / WORDS_PER_BIN)
const WORKER_COUNTS = [1, 2, 4] as const

const stateData = (state: number) => {
  const s = data.states.find(x => x.state === state)
  if (!s) throw new Error(`state ${state} missing from data`)
  return s
}

// One engine step as vLLM v0.8.0 schedules it: rows follow the running list, a request whose
// grammar is still compiling waits in WAITING_FOR_FSM, and sampling here is greedy.
const buildRequests = (): Request[] => {
  const free = stateData(0).top[0]
  const make = (id: string, label: string, state: number | null): Request => {
    if (state === null) {
      return {
        id,
        label,
        state,
        status: 'running',
        allowed: [],
        raw: free,
        pick: free,
        pickShare: free.p,
        words: [],
        bins: new Set(),
      }
    }
    const s = stateData(state)
    const allowed = s.allowed
    const kept = allowed.reduce((a, t) => a + t.p, 0)
    const pick = allowed.reduce((a, t) => (t.p > a.p ? t : a))
    const words = [...new Set(allowed.map(t => t.id >> 5))].sort((a, b) => a - b)
    return {
      id,
      label,
      state,
      status: 'running',
      allowed,
      raw: s.top[0],
      pick,
      pickShare: pick.p / kept,
      words,
      bins: new Set(words.map(w => Math.floor(w / WORDS_PER_BIN))),
    }
  }
  return [
    make('r0', 'no grammar', null),
    make('r1', 'state 0', 0),
    make('r2', 'state 9', 9),
    make('r3', 'state 22', 22),
    { ...make('r4', 'compiling', null), status: 'waiting', raw: free, pick: free },
  ]
}

const REQUESTS = buildRequests()
const BATCH = REQUESTS.filter(r => r.status === 'running')
const STRUCTURED = BATCH.filter(r => r.state !== null)
const consumed = (r: Request) => (r.state === null ? '' : stateData(r.state).consumed)

const fmt = (n: number) => n.toLocaleString('en-US')
const pct = (p: number) => (p >= 0.01 ? `${(p * 100).toFixed(1)}%` : `${(p * 100).toPrecision(2)}%`)

const W = 560
const LABEL_X = 8
const STATE_X = 40
const STRIP_X = 148
const BIN_W = 2.8
const STRIP_W = BINS * BIN_W
const COUNT_X = W - 8
const TITLE_Y = 16
const HEAD_Y = 42
const ROW_Y0 = 54
const ROW_PITCH = 24
const ROW_H = 14
const SCHED = { x: 0, y: 0, w: W, h: ROW_Y0 + REQUESTS.length * ROW_PITCH + 4 }
const BUS_Y = SCHED.h + 40
const WORKER_Y = BUS_Y + 26
const WORKER_H = 104
const H = WORKER_Y + WORKER_H + 2
const RETURN_X = 18

const workerBoxes = (n: number) => {
  const gap = 10
  const left = RETURN_X + 14
  const w = (W - left - gap * (n - 1)) / n
  return Array.from({ length: n }, (_, k) => ({ k, x: left + k * (w + gap), w }))
}

let instance = 0

const GrammarBitmaskBatchImpl: QuartzMdxComponent<Props> = ({
  caption,
  request = 'r2',
  workers = 2,
}) => {
  const id = `gbb-${++instance}`
  const req0 = BATCH.find(r => r.id === request)?.id ?? 'r2'
  const n0 = (WORKER_COUNTS as readonly number[]).includes(Number(workers)) ? Number(workers) : 2
  const structuredRows = STRUCTURED.map(r => BATCH.indexOf(r)).join(', ')

  return (
    <figure
      class="grammar-bitmask-batch"
      data-grammar-bitmask-batch
      data-gbb-request={req0}
      data-gbb-workers={String(n0)}
    >
      <div class="gbb-controls">
        <fieldset class="gbb-segments">
          <legend class="gbb-legend">Request</legend>
          {BATCH.map(r => (
            <label class="gbb-segment">
              <input
                type="radio"
                name={`${id}-request`}
                value={r.id}
                checked={r.id === req0}
                data-gbb-input="request"
              />
              <span>{r.id}</span>
            </label>
          ))}
        </fieldset>
        <fieldset class="gbb-segments">
          <legend class="gbb-legend">GPU workers</legend>
          {WORKER_COUNTS.map(n => (
            <label class="gbb-segment">
              <input
                type="radio"
                name={`${id}-workers`}
                value={String(n)}
                checked={n === n0}
                data-gbb-input="workers"
              />
              <span>{`N = ${n}`}</span>
            </label>
          ))}
        </fieldset>
      </div>

      <svg
        class="gbb-svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`The scheduler fills one grammar bitmask row per structured request in a batch of ${BATCH.length}, broadcasts it with the scheduler output, and every GPU worker applies it to its logits before sampling.`}
      >
        <rect
          class="gbb-box"
          x={SCHED.x + 0.5}
          y={SCHED.y + 0.5}
          width={SCHED.w - 1}
          height={SCHED.h - 1}
        />
        <text class="gbb-title" x={LABEL_X} y={TITLE_Y}>
          Scheduler, N = 1
        </text>
        <text class="gbb-head" x={STATE_X} y={HEAD_Y}>
          grammar
        </text>
        <text class="gbb-head" x={STRIP_X} y={HEAD_Y}>
          {`grammar_bitmask row: ${fmt(WORDS)} int32 words, ${WORDS_PER_BIN} per cell`}
        </text>
        <text class="gbb-head" x={COUNT_X} y={HEAD_Y} text-anchor="end">
          allowed
        </text>
        {REQUESTS.map((r, i) => {
          const y = ROW_Y0 + i * ROW_PITCH
          const row = BATCH.indexOf(r)
          return (
            <g class="gbb-req" data-gbb-req={r.id} data-on={r.id === req0 ? 'true' : 'false'}>
              <rect class="gbb-req-band" x={4} y={y - 4} width={W - 8} height={ROW_H + 8} />
              <text class="gbb-label" x={LABEL_X} y={y + 11}>
                {r.id}
              </text>
              <text class="gbb-state" x={STATE_X} y={y + 11}>
                {r.label}
              </text>
              {r.status === 'waiting' ? (
                <text class="gbb-note" x={STRIP_X} y={y + 11}>
                  WAITING_FOR_FSM: stays in the waiting queue, not in this batch
                </text>
              ) : (
                <g>
                  <rect class="gbb-strip" x={STRIP_X} y={y} width={STRIP_W} height={ROW_H} />
                  {[...r.bins].map(b => (
                    <rect
                      class="gbb-bit"
                      x={STRIP_X + b * BIN_W}
                      y={y}
                      width={BIN_W}
                      height={ROW_H}
                      data-gbb-bin={String(b)}
                    />
                  ))}
                  {r.state === null ? (
                    <text class="gbb-note" x={STRIP_X + 6} y={y + 10.5}>
                      {`row ${row} not filled or applied`}
                    </text>
                  ) : null}
                </g>
              )}
              <text class="gbb-value" x={COUNT_X} y={y + 11} text-anchor="end">
                {r.status === 'waiting'
                  ? '—'
                  : r.state === null
                    ? fmt(VOCAB)
                    : `${r.allowed.length} / ${fmt(VOCAB)}`}
              </text>
            </g>
          )
        })}

        <line class="gbb-wire" x1={W / 2} y1={SCHED.h} x2={W / 2} y2={BUS_Y} />
        <text class="gbb-note" x={W / 2 + 6} y={SCHED.h + 14}>
          {`SchedulerOutput with grammar_bitmask int32[${BATCH.length} × ${fmt(WORDS)}]`}
        </text>
        <text class="gbb-note" x={W / 2 + 6} y={SCHED.h + 26}>
          one shared-memory broadcast to every worker
        </text>
        <line
          class="gbb-wire gbb-wire--return"
          x1={RETURN_X}
          y1={WORKER_Y + 14}
          x2={RETURN_X}
          y2={SCHED.h}
        />
        <text class="gbb-note" x={RETURN_X + 6} y={SCHED.h + 14}>
          rank 0 returns tokens;
        </text>
        <text class="gbb-note" x={RETURN_X + 6} y={SCHED.h + 26}>
          accept_tokens advances
        </text>

        {WORKER_COUNTS.map(n => {
          const boxes = workerBoxes(n)
          const first = boxes[0].x + boxes[0].w / 2
          const last = boxes[n - 1].x + boxes[n - 1].w / 2
          return (
            <g data-gbb-n={String(n)} data-on={n === n0 ? 'true' : 'false'}>
              <line
                class="gbb-wire"
                x1={Math.min(first, W / 2)}
                x2={Math.max(last, W / 2)}
                y1={BUS_Y}
                y2={BUS_Y}
              />
              <line
                class="gbb-wire gbb-wire--return"
                x1={RETURN_X}
                x2={boxes[0].x}
                y1={WORKER_Y + 14}
                y2={WORKER_Y + 14}
              />
              {boxes.map(b => (
                <g>
                  <line
                    class="gbb-wire"
                    x1={b.x + b.w / 2}
                    x2={b.x + b.w / 2}
                    y1={BUS_Y}
                    y2={WORKER_Y}
                  />
                  <rect class="gbb-box" x={b.x} y={WORKER_Y} width={b.w} height={WORKER_H} />
                  <text class="gbb-title" x={b.x + 6} y={WORKER_Y + 16}>
                    {n === 1 ? 'GPU worker' : `rank ${b.k}`}
                  </text>
                  {BATCH.map((r, i) => (
                    <rect
                      class="gbb-logit"
                      data-gbb-req={r.id}
                      data-on={r.id === req0 ? 'true' : 'false'}
                      data-masked={r.state === null ? 'false' : 'true'}
                      x={b.x + 6}
                      y={WORKER_Y + 34 + i * 9}
                      width={b.w - 12}
                      height={7}
                    />
                  ))}
                  <text class="gbb-step" x={b.x + 6} y={WORKER_Y + 82}>
                    {n === 4
                      ? `mask ${structuredRows}`
                      : `logits ${BATCH.length} × ${fmt(VOCAB)}, mask rows ${structuredRows}`}
                  </text>
                  <text class="gbb-step" x={b.x + 6} y={WORKER_Y + 94}>
                    {b.k === 0 ? 'sample, return' : 'sample, return None'}
                  </text>
                </g>
              ))}
            </g>
          )
        })}
      </svg>

      {BATCH.map(r => (
        <div class="gbb-detail" data-gbb-req={r.id} data-on={r.id === req0 ? 'true' : 'false'}>
          <dl class="gbb-readout">
            <div>
              <dt>Text after the prompt</dt>
              <dd>{r.state === null ? 'no grammar' : consumed(r) || 'nothing yet'}</dd>
            </div>
            <div>
              <dt>Bitmask words with a set bit</dt>
              <dd>{r.state === null ? 'row not applied' : `${r.words.length} of ${fmt(WORDS)}`}</dd>
            </div>
            <div>
              <dt>Unmasked argmax</dt>
              <dd>{`${r.raw.text}  ${pct(r.raw.p)}`}</dd>
            </div>
            <div>
              <dt>Sampled (greedy, after mask)</dt>
              <dd>{`${r.pick.text}  ${pct(r.pickShare)}`}</dd>
            </div>
          </dl>
          {r.allowed.length ? (
            <table class="gbb-table">
              <thead>
                <tr>
                  <th scope="col">allowed token</th>
                  <th scope="col">id</th>
                  <th scope="col">word, bit</th>
                  <th scope="col">model p</th>
                </tr>
              </thead>
              <tbody>
                {r.allowed.map(t => (
                  <tr>
                    <th scope="row">{t.text}</th>
                    <td>{t.id}</td>
                    <td>{`${t.id >> 5}, ${t.id & 31}`}</td>
                    <td>{pct(t.p)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
        </div>
      ))}

      {caption ? (
        <figcaption class="gbb-caption">
          <MathText text={caption} mathClass="gbb-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const GrammarBitmaskBatchComponent = GrammarBitmaskBatchImpl as QuartzMdxComponent<Props>
GrammarBitmaskBatchComponent.css = style
GrammarBitmaskBatchComponent.afterDOMLoaded = script

export const GrammarBitmaskBatch = registerMdxComponent(
  'GrammarBitmaskBatch',
  GrammarBitmaskBatchComponent,
)

export default (() => GrammarBitmaskBatch) satisfies (opts: undefined) => QuartzMdxComponent<Props>
