import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/logits-processor-trace.inline'
import style from '../styles/logitsProcessorTrace.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; guided?: number; step?: number }

type Kind = 'call' | 'self' | 'attach' | 'row'
type Call = { from: number; to: number; code: string; src: string; does: string; kind: Kind }

const LANES = [
  'OpenAIServingCompletion',
  'CompletionRequest',
  'AsyncLLMEngine',
  'LLMEngine',
  'Scheduler',
  'GPUExecutorAsync',
  'Worker',
  'ModelRunner',
  'LlamaForCausalLM',
  'LogitsProcessor',
  'JSONLogitsProcessor',
  'Sampler',
] as const
const lane = (name: (typeof LANES)[number]) => LANES.indexOf(name)

const BATCH = 4
const GUIDED_COUNTS = [0, 1, 2, 4] as const
// Which rows of the batch carry a guided processor for each count.
const GUIDED_ROWS: Record<number, number[]> = { 0: [], 1: [0], 2: [0, 2], 4: [0, 1, 2, 3] }

// Calls in vLLM v0.5.0 source order for one request, then one engine step.
const BEFORE: Call[] = [
  {
    from: lane('OpenAIServingCompletion'),
    to: lane('CompletionRequest'),
    code: 'request.to_sampling_params()',
    src: 'entrypoints/openai/serving_completion.py:97',
    does: 'Builds the SamplingParams of the request.',
    kind: 'call',
  },
  {
    from: lane('OpenAIServingCompletion'),
    to: lane('OpenAIServingCompletion'),
    code: 'sampling_params.logits_processors.append(guided_decode_logit_processor)',
    src: 'entrypoints/openai/serving_completion.py:102-110',
    does: 'get_guided_decoding_logits_processor builds a JSONLogitsProcessor in a thread pool; its RegexGuide is the compiled FSM. The processor object rides in this request’s SamplingParams.',
    kind: 'attach',
  },
  {
    from: lane('OpenAIServingCompletion'),
    to: lane('AsyncLLMEngine'),
    code: 'engine.generate(inputs, sampling_params, request_id)',
    src: 'entrypoints/openai/serving_completion.py:128',
    does: 'Queues the request on the request tracker and streams its outputs.',
    kind: 'call',
  },
  {
    from: lane('AsyncLLMEngine'),
    to: lane('LLMEngine'),
    code: 'engine.add_request_async(**new_request)',
    src: 'engine/async_llm_engine.py:488',
    does: 'engine_step moves new requests into the engine, which creates the Sequence and SequenceGroup.',
    kind: 'call',
  },
  {
    from: lane('LLMEngine'),
    to: lane('Scheduler'),
    code: 'scheduler.add_seq_group(seq_group)',
    src: 'engine/llm_engine.py:467',
    does: 'Adds the sequence group to the waiting queue.',
    kind: 'call',
  },
  {
    from: lane('AsyncLLMEngine'),
    to: lane('LLMEngine'),
    code: 'engine.step_async()',
    src: 'engine/async_llm_engine.py:503',
    does: 'One decoding iteration for the whole batch.',
    kind: 'call',
  },
  {
    from: lane('LLMEngine'),
    to: lane('Scheduler'),
    code: 'scheduler.schedule()',
    src: 'engine/async_llm_engine.py:223',
    does: 'Returns the SequenceGroupMetadata list and SchedulerOutputs for this step.',
    kind: 'call',
  },
  {
    from: lane('LLMEngine'),
    to: lane('GPUExecutorAsync'),
    code: 'model_executor.execute_model_async(execute_model_req)',
    src: 'engine/async_llm_engine.py:235',
    does: 'Hands the ExecuteModelRequest to the executor.',
    kind: 'call',
  },
  {
    from: lane('GPUExecutorAsync'),
    to: lane('Worker'),
    code: 'driver_worker.execute_model(execute_model_req=execute_model_req)',
    src: 'executor/gpu_executor.py:117',
    does: 'Runs on the driver worker.',
    kind: 'call',
  },
  {
    from: lane('Worker'),
    to: lane('ModelRunner'),
    code: 'model_runner.execute_model(seq_group_metadata_list, self.gpu_cache)',
    src: 'worker/worker.py:272',
    does: 'After cache swaps and copies, the model runner executes the batch.',
    kind: 'call',
  },
  {
    from: lane('ModelRunner'),
    to: lane('ModelRunner'),
    code: 'prepare_input_tensors(seq_group_metadata_list)',
    src: 'worker/model_runner.py:724',
    does: 'Builds input tokens, positions, attention metadata and SamplingMetadata.',
    kind: 'self',
  },
  {
    from: lane('ModelRunner'),
    to: lane('LlamaForCausalLM'),
    code: 'model_executable(input_ids, positions, kv_caches, attn_metadata)',
    src: 'worker/model_runner.py:738',
    does: 'The forward pass on the GPU; returns hidden states.',
    kind: 'call',
  },
  {
    from: lane('ModelRunner'),
    to: lane('LlamaForCausalLM'),
    code: 'model.compute_logits(hidden_states, sampling_metadata)',
    src: 'worker/model_runner.py:747',
    does: 'Starts the logits computation.',
    kind: 'call',
  },
  {
    from: lane('LlamaForCausalLM'),
    to: lane('LogitsProcessor'),
    code: 'logits_processor(lm_head.weight, hidden_states, sampling_metadata)',
    src: 'model_executor/models/llama.py:377',
    does: 'LogitsProcessor.forward, an nn.Module on the model.',
    kind: 'call',
  },
  {
    from: lane('LogitsProcessor'),
    to: lane('LogitsProcessor'),
    code: '_get_logits(hidden_states, embedding, embedding_bias)',
    src: 'model_executor/layers/logits_processor.py:52',
    does: 'Matmul with the LM head, then tensor_model_parallel_gather; logits for the batch.',
    kind: 'self',
  },
]

const rowCall = (row: number): Call => ({
  from: lane('LogitsProcessor'),
  to: lane('JSONLogitsProcessor'),
  code: `logits_row = logits_processor(past_tokens_ids, logits[${row}])`,
  src: 'model_executor/layers/logits_processor.py:116',
  does: `_apply_logits_processors loops over seq groups and their rows in Python. For row ${row} the outlines processor advances its FSM state, then builds torch.full((vocab,), -inf), zeroes the allowed IDs and adds the mask (outlines_logits_processors.py:44-63).`,
  kind: 'row',
})

const AFTER: Call[] = [
  {
    from: lane('ModelRunner'),
    to: lane('LlamaForCausalLM'),
    code: 'model.sample(logits=logits, sampling_metadata=sampling_metadata)',
    src: 'worker/model_runner.py:754',
    does: 'Sampling starts only after every processor row has returned.',
    kind: 'call',
  },
  {
    from: lane('LlamaForCausalLM'),
    to: lane('Sampler'),
    code: 'self.sampler(logits, sampling_metadata)',
    src: 'model_executor/models/llama.py:386',
    does: 'Samples the next token of every sequence in the batch; returns SamplerOutput.',
    kind: 'call',
  },
  {
    from: lane('LLMEngine'),
    to: lane('LLMEngine'),
    code: '_process_model_outputs(output, scheduled_seq_groups, ...)',
    src: 'engine/async_llm_engine.py:240',
    does: 'Appends the sampled tokens and checks stop conditions; the next step starts from schedule().',
    kind: 'self',
  },
]

const callsFor = (guided: number): Call[] => [
  ...BEFORE,
  ...GUIDED_ROWS[guided].map(rowCall),
  ...AFTER,
]
const MAX_CALLS = callsFor(BATCH).length
const FORWARD = BEFORE.findIndex(c => c.code.startsWith('model_executable'))

const W = 640
const LABEL_W = 132
const X0 = LABEL_W + 14
const PITCH = (W - X0 - 10) / (MAX_CALLS - 1)
const LANE_Y0 = 26
const LANE_PITCH = 19
const laneY = (l: number) => LANE_Y0 + l * LANE_PITCH
const H = laneY(LANES.length - 1) + 26
const NODE = 6
const xOf = (i: number) => X0 + i * PITCH

const stateFor = (guided: number) => {
  const calls = callsFor(guided)
  const rows = calls.filter(c => c.kind === 'row').length
  const sample = calls.findIndex(c => c.code.startsWith('model.sample'))
  return calls.map((c, i) => ({
    step: `${i + 1} / ${calls.length}`,
    who: c.from === c.to ? LANES[c.from] : `${LANES[c.from]} → ${LANES[c.to]}`,
    code: c.code,
    src: `vllm/${c.src}`,
    does: c.does,
    between: `${sample - FORWARD - 1} calls, ${rows} per-row`,
  }))
}
const CLIENT_STATES = Object.fromEntries(GUIDED_COUNTS.map(g => [String(g), stateFor(g)]))

const CallMark = ({ c, i, on }: { c: Call; i: number; on: boolean }) => {
  const x = xOf(i)
  const y1 = laneY(c.from)
  const y2 = laneY(c.to)
  const attrs = {
    'data-lpt-call': String(i),
    'data-kind': c.kind,
    'data-on': on ? 'true' : 'false',
  }
  if (c.from === c.to)
    return (
      <g class="lpt-call" {...attrs}>
        <rect class="lpt-self" x={x - 5} y={y1 - 5} width={10} height={10} />
      </g>
    )
  return (
    <g class="lpt-call" {...attrs}>
      <line class="lpt-wire" x1={x} x2={x} y1={y1} y2={y2} />
      <rect class="lpt-from" x={x - NODE / 2} y={y1 - NODE / 2} width={NODE} height={NODE} />
      <rect
        class="lpt-to"
        x={x - NODE / 2 - 1}
        y={y2 - NODE / 2 - 1}
        width={NODE + 2}
        height={NODE + 2}
      />
    </g>
  )
}

let instance = 0

const LogitsProcessorTraceImpl: QuartzMdxComponent<Props> = ({ caption, guided = 2, step }) => {
  const id = `lpt-${++instance}`
  const g0 = (GUIDED_COUNTS as readonly number[]).includes(Number(guided)) ? Number(guided) : 2
  const calls0 = callsFor(g0)
  const firstRow = calls0.findIndex(c => c.kind === 'row')
  const s0 = Math.min(
    Math.max(1, Number(step ?? (firstRow >= 0 ? firstRow + 1 : FORWARD + 1))),
    calls0.length,
  )
  const st = CLIENT_STATES[String(g0)][s0 - 1]

  return (
    <figure
      class="logits-processor-trace"
      data-logits-processor-trace
      data-lpt-states={JSON.stringify(CLIENT_STATES)}
    >
      <div class="lpt-controls">
        <fieldset class="lpt-segments">
          <legend class="lpt-legend">{`Guided rows in a batch of ${BATCH}`}</legend>
          {GUIDED_COUNTS.map(g => (
            <label class="lpt-segment">
              <input
                type="radio"
                name={`${id}-guided`}
                value={String(g)}
                checked={g === g0}
                data-lpt-guided
              />
              <span>{String(g)}</span>
            </label>
          ))}
        </fieldset>
        <label class="lpt-range">
          <span class="lpt-legend-inline">Call</span>
          <input
            type="range"
            min={1}
            max={calls0.length}
            step={1}
            value={s0}
            aria-valuetext={`call ${st.step}: ${st.code}`}
            data-lpt-step
          />
          <output data-lpt-out="step">{st.step}</output>
        </label>
      </div>

      <svg
        class="lpt-svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Sequence lanes for a completion request in vLLM v0.5.0. Per-row logits processor calls run between the forward pass and sampling."
      >
        {LANES.map((name, l) => (
          <g>
            <text class="lpt-lane" x={LABEL_W} y={laneY(l) + 3} text-anchor="end">
              {name}
            </text>
            <line class="lpt-track" x1={X0 - 8} x2={W - 2} y1={laneY(l)} y2={laneY(l)} />
          </g>
        ))}
        {GUIDED_COUNTS.map(g => {
          const calls = callsFor(g)
          const sample = calls.findIndex(c => c.code.startsWith('model.sample'))
          return (
            <g data-lpt-set={String(g)} data-on={g === g0 ? 'true' : 'false'}>
              {calls.map((_, i) => (
                <text class="lpt-num" x={xOf(i)} y={12} text-anchor="middle">
                  {String(i + 1)}
                </text>
              ))}
              {calls.map((c, i) => (
                <CallMark c={c} i={i} on={g === g0 && i === s0 - 1} />
              ))}
              <text class="lpt-tag" x={xOf(FORWARD)} y={H - 4} text-anchor="middle">
                forward
              </text>
              <text class="lpt-tag" x={xOf(sample)} y={H - 4} text-anchor="middle">
                sample
              </text>
            </g>
          )
        })}
      </svg>

      <dl class="lpt-readout">
        <div>
          <dt>Caller and callee</dt>
          <dd data-lpt-out="who">{st.who}</dd>
        </div>
        <div>
          <dt>Call</dt>
          <dd class="lpt-code" data-lpt-out="code">
            {st.code}
          </dd>
        </div>
        <div>
          <dt>Source, vLLM v0.5.0</dt>
          <dd data-lpt-out="src">{st.src}</dd>
        </div>
        <div>
          <dt>What it does</dt>
          <dd class="lpt-prose" data-lpt-out="does">
            {st.does}
          </dd>
        </div>
        <div>
          <dt>Between forward and sample</dt>
          <dd data-lpt-out="between">{st.between}</dd>
        </div>
      </dl>

      {caption ? (
        <figcaption class="lpt-caption">
          <MathText text={caption} mathClass="lpt-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const LogitsProcessorTraceComponent = LogitsProcessorTraceImpl as QuartzMdxComponent<Props>
LogitsProcessorTraceComponent.css = style
LogitsProcessorTraceComponent.afterDOMLoaded = script

export const LogitsProcessorTrace = registerMdxComponent(
  'LogitsProcessorTrace',
  LogitsProcessorTraceComponent,
)

export default (() => LogitsProcessorTrace) satisfies (opts: undefined) => QuartzMdxComponent<Props>
