import { MathText } from '../../util/math-text'
import style from '../styles/vllmSamplingSequence.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; id?: string }

type Lane =
  | 'serving'
  | 'request'
  | 'async'
  | 'engine'
  | 'scheduler'
  | 'executor'
  | 'worker'
  | 'runner'
  | 'model'
  | 'logits'

type Message =
  | { kind: 'found'; to: Lane; label: string[] }
  | { kind: 'call' | 'return'; from: Lane; to: Lane; label: string[] }
  | { kind: 'self'; lane: Lane; label: string[]; back?: string[]; critical?: boolean }

type Activation = { lane: Lane; from: number; to: number; pad?: number }

// vLLM v0 sampling path, transcribed from the draw.io diagram in vllm-project/vllm#5329.
const LANES: { id: Lane; name: string[] }[] = [
  { id: 'serving', name: ['OpenAIServingCompletion'] },
  { id: 'request', name: ['CompletionRequest'] },
  { id: 'async', name: ['AsyncLLMEngine'] },
  { id: 'engine', name: ['LLMEngine'] },
  { id: 'scheduler', name: ['Scheduler'] },
  { id: 'executor', name: ['ModelExecutor'] },
  { id: 'worker', name: ['Worker'] },
  { id: 'runner', name: ['ModelRunner'] },
  { id: 'model', name: ['Model'] },
  { id: 'logits', name: ['LogitsProcessor', '(nn.Module)'] },
]

const MESSAGES: Message[] = [
  { kind: 'found', to: 'serving', label: ['create_completion(request)'] },
  { kind: 'call', from: 'serving', to: 'request', label: ['request.to_sampling_params()'] },
  { kind: 'return', from: 'request', to: 'serving', label: ['SamplingParams'] },
  { kind: 'self', lane: 'serving', label: ['append logits', 'processors'] },
  { kind: 'call', from: 'serving', to: 'async', label: ['generate(prompt, sampling_params)'] },
  { kind: 'call', from: 'async', to: 'engine', label: ['add_request(prompt, sp)'] },
  { kind: 'self', lane: 'engine', label: ['create seq'] },
  { kind: 'self', lane: 'engine', label: ['create seq_group'] },
  { kind: 'self', lane: 'async', label: ['event loop'] },
  { kind: 'call', from: 'async', to: 'engine', label: ['step()'] },
  { kind: 'call', from: 'engine', to: 'scheduler', label: ['schedule()'] },
  {
    kind: 'return',
    from: 'scheduler',
    to: 'engine',
    label: ['List(SeqGroupMetadata),', 'SchedulerOutputs'],
  },
  { kind: 'call', from: 'engine', to: 'executor', label: ['execute_model(ExecuteModelRequest)'] },
  {
    kind: 'call',
    from: 'executor',
    to: 'worker',
    label: ['execute_model(', 'ExecuteModelRequest)'],
  },
  {
    kind: 'call',
    from: 'worker',
    to: 'runner',
    label: ['execute_model(', 'List(SeqGroupMetadata))'],
  },
  {
    kind: 'self',
    lane: 'runner',
    label: ['prepare_input_tensors(', 'List(SeqGroupMetadata))'],
    back: ['SamplingMetadata'],
  },
  { kind: 'call', from: 'runner', to: 'model', label: ['compute_logits(', 'SamplingMetadata)'] },
  { kind: 'call', from: 'model', to: 'logits', label: ['forward(', 'SamplingMetadata)'] },
  {
    kind: 'self',
    lane: 'logits',
    label: [
      'call sampling_params',
      '.logits_processors',
      'in sequence passing',
      'token_ids and logits',
    ],
    critical: true,
  },
]

// Message indices from the source diagram; pad extends a caller past the callee it waits on.
const ACTIVATIONS: Activation[] = [
  { lane: 'serving', from: 0, to: 4 },
  { lane: 'request', from: 1, to: 2 },
  { lane: 'async', from: 4, to: 5 },
  { lane: 'async', from: 8, to: 11 },
  { lane: 'engine', from: 5, to: 7 },
  { lane: 'engine', from: 9, to: 14, pad: 30 },
  { lane: 'scheduler', from: 10, to: 11 },
  { lane: 'executor', from: 12, to: 14, pad: 22 },
  { lane: 'worker', from: 13, to: 14, pad: 14 },
  { lane: 'runner', from: 14, to: 18, pad: 26 },
  { lane: 'model', from: 16, to: 18, pad: 18 },
  { lane: 'logits', from: 17, to: 18, pad: 10 },
]

// Gaps between neighbouring lifelines, sized to the widest label each gap carries.
const GAPS = [184, 150, 164, 156, 140, 140, 156, 184, 150]
const CHAR_W = 6
const HEAD_CHAR_W = 6.4
const LINE_H = 12
const BAR_W = 10
const HALF = BAR_W / 2
const LOOP_W = 24
const X0 = 182
const HEAD_Y = 12
const HEAD_H = 36
const TOP = HEAD_Y + HEAD_H + 26
const FOUND_X = 10

const laneX = (() => {
  const xs = new Map<Lane, number>()
  let x = X0
  LANES.forEach((lane, i) => {
    xs.set(lane.id, x)
    x += GAPS[i] ?? 0
  })
  return xs
})()

const xOf = (lane: Lane) => laneX.get(lane) ?? 0

type Placed = { msg: Message; y0: number; y1: number }

function place(): { rows: Placed[]; height: number } {
  const rows: Placed[] = []
  let y = TOP
  for (const msg of MESSAGES) {
    if (msg.kind === 'self') {
      const top = y + 4
      const loopH = Math.max(18, msg.label.length * LINE_H + (msg.critical ? LINE_H + 8 : 4))
      rows.push({ msg, y0: top, y1: top + loopH })
      y = top + loopH + (msg.back ? LINE_H + 4 : 0) + 14
    } else {
      const arrow = y + msg.label.length * LINE_H + 4
      rows.push({ msg, y0: arrow, y1: arrow })
      y = arrow + 14
    }
  }
  return { rows, height: y }
}

const { rows: ROWS, height: BODY_H } = place()
const LAST_X = xOf('logits')
const CRIT_W = 21 * CHAR_W + 16
const VIEW_W = LAST_X + HALF + LOOP_W + 6 + CRIT_W + 12
const BAR_END = Math.max(...ACTIVATIONS.map(a => (ROWS[a.to]?.y1 ?? 0) + 8 + (a.pad ?? 0)))
const VIEW_H = Math.max(BODY_H, BAR_END) + 24

const Label = ({
  x,
  y,
  lines,
  anchor,
}: {
  x: number
  y: number
  lines: string[]
  anchor: 'start' | 'middle'
}) => (
  <text class="vsq-label" x={x} y={y} text-anchor={anchor}>
    {lines.map((line, i) => (
      <tspan x={x} dy={i === 0 ? 0 : LINE_H}>
        {line}
      </tspan>
    ))}
  </text>
)

const VllmSamplingSequenceImpl: QuartzMdxComponent<Props> = ({
  caption,
  id = 'vllm-sampling-seq',
}) => {
  const callHead = `${id}-call`
  const returnHead = `${id}-return`
  const lifelineEnd = VIEW_H - 10

  const renderMessage = ({ msg, y0, y1 }: Placed, i: number) => {
    if (msg.kind === 'self') {
      const x = xOf(msg.lane) + HALF
      const xr = x + LOOP_W
      const labelX = xr + 6
      const cls = msg.critical ? 'vsq-msg vsq-msg--critical' : 'vsq-msg'
      return (
        <g key={i} class={cls} data-step={i + 1}>
          {msg.critical ? (
            <>
              <rect
                class="vsq-critical"
                x={labelX}
                y={y0 - 2}
                width={CRIT_W}
                height={(msg.label.length + 1) * LINE_H + 14}
              />
              <text class="vsq-label vsq-label--tag" x={labelX + 8} y={y0 + 10}>
                critical path
              </text>
            </>
          ) : null}
          {msg.back ? (
            <>
              <path class="vsq-line" d={`M${x},${y0} H${xr} V${y1}`} />
              <path
                class="vsq-line vsq-line--return"
                d={`M${xr},${y1} H${x + 1}`}
                marker-end={`url(#${returnHead})`}
              />
              <Label x={xr - 14} y={y1 + LINE_H + 2} lines={msg.back} anchor="start" />
            </>
          ) : (
            <path
              class="vsq-line"
              d={`M${x},${y0} H${xr} V${y1} H${x + 1}`}
              marker-end={`url(#${callHead})`}
            />
          )}
          <Label
            x={msg.critical ? labelX + 8 : labelX}
            y={msg.critical ? y0 + 10 + LINE_H + 2 : y0 + 9}
            lines={msg.label}
            anchor="start"
          />
        </g>
      )
    }

    const toX = xOf(msg.to)
    const fromX = msg.kind === 'found' ? FOUND_X : xOf(msg.from)
    const dir = toX > fromX ? 1 : -1
    const x1 = msg.kind === 'found' ? FOUND_X : fromX + dir * HALF
    const x2 = toX - dir * (HALF + 1)
    const isReturn = msg.kind === 'return'
    return (
      <g key={i} class="vsq-msg" data-step={i + 1}>
        {msg.kind === 'found' ? <circle class="vsq-found" cx={FOUND_X} cy={y0} r={3.5} /> : null}
        <line
          class={isReturn ? 'vsq-line vsq-line--return' : 'vsq-line'}
          x1={x1}
          y1={y0}
          x2={x2}
          y2={y0}
          marker-end={`url(#${isReturn ? returnHead : callHead})`}
        />
        <Label
          x={(x1 + x2) / 2}
          y={y0 - 5 - (msg.label.length - 1) * LINE_H}
          lines={msg.label}
          anchor="middle"
        />
      </g>
    )
  }

  return (
    <figure class="vllm-sampling-sequence" data-vllm-sampling-sequence>
      <div class="vsq-scroll" tabIndex={0} role="group" aria-label="vLLM sampling sequence diagram">
        <svg
          class="vsq-graph"
          width={VIEW_W}
          height={VIEW_H}
          viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
          role="img"
          aria-labelledby={`${id}-desc`}
        >
          <desc id={`${id}-desc`}>
            {`UML sequence diagram of the vLLM v0 sampling path across ${LANES.length} lifelines. ` +
              MESSAGES.map((m, i) => `${i + 1}. ${m.label.join(' ')}`).join('; ') +
              '. The last step is the critical path.'}
          </desc>
          <defs>
            <marker
              id={callHead}
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="7"
              markerHeight="7"
              orient="auto-start-reverse"
            >
              <path class="vsq-arrowhead" d="M0,1 L10,5 L0,9 z" />
            </marker>
            <marker
              id={returnHead}
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="7"
              markerHeight="7"
              orient="auto-start-reverse"
            >
              <path class="vsq-arrowhead--open" d="M1,1 L9,5 L1,9" />
            </marker>
          </defs>

          {LANES.map(lane => {
            const x = xOf(lane.id)
            const w = Math.max(...lane.name.map(n => n.length)) * HEAD_CHAR_W + 18
            const lines = lane.name.length
            return (
              <g key={lane.id} class="vsq-lane">
                <line class="vsq-lifeline" x1={x} y1={HEAD_Y + HEAD_H} x2={x} y2={lifelineEnd} />
                <rect class="vsq-head" x={x - w / 2} y={HEAD_Y} width={w} height={HEAD_H} />
                <text
                  class="vsq-head-label"
                  x={x}
                  y={HEAD_Y + HEAD_H / 2 + 3.5 - ((lines - 1) * LINE_H) / 2}
                  text-anchor="middle"
                >
                  {lane.name.map((n, i) => (
                    <tspan x={x} dy={i === 0 ? 0 : LINE_H}>
                      {n}
                    </tspan>
                  ))}
                </text>
              </g>
            )
          })}

          {ACTIVATIONS.map((a, i) => {
            const y0 = (ROWS[a.from]?.y0 ?? 0) - 6
            const y1 = (ROWS[a.to]?.y1 ?? 0) + 8 + (a.pad ?? 0)
            const critical = a.lane === 'logits'
            return (
              <rect
                key={i}
                class={critical ? 'vsq-bar vsq-bar--critical' : 'vsq-bar'}
                x={xOf(a.lane) - HALF}
                y={y0}
                width={BAR_W}
                height={y1 - y0}
              />
            )
          })}

          {ROWS.map(renderMessage)}
        </svg>
      </div>
      {caption ? (
        <figcaption class="vsq-caption">
          <MathText text={caption} mathClass="vsq-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const VllmSamplingSequenceComponent = VllmSamplingSequenceImpl as QuartzMdxComponent<Props>
VllmSamplingSequenceComponent.css = style

export const VllmSamplingSequence = registerMdxComponent(
  'VllmSamplingSequence',
  VllmSamplingSequenceComponent,
)

export default (() => VllmSamplingSequence) satisfies (opts: undefined) => QuartzMdxComponent<Props>
