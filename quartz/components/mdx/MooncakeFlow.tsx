import type { FunctionalComponent } from 'preact'
import { MathText } from '../../util/math-text'
import {
  DECODE_INSTANCES,
  DECODE_VRAM_SLOTS,
  MOONCAKE_REQUESTS,
  MOONCAKE_STAGES,
  POOL_SLOTS,
  PREFILL_INSTANCES,
  PROMPT_BLOCKS,
  QUEUE_SLOTS,
  mooncakePlan,
  mooncakeView,
  type MooncakeView,
} from '../../util/mooncake-flow'
//@ts-ignore
import script from '../scripts/mooncake-flow.inline'
import style from '../styles/mooncakeFlow.scss'
import {
  DECODE_MSG_Y,
  MOONCAKE_LAYOUTS,
  MSG_H,
  PREFILL_MSG_Y,
  type MooncakeLayout,
} from './MooncakeFlow/layout'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

const STAGE_NOTES: Record<string, string> = {
  route:
    'The conductor matches the prompt against each prefill pool and picks the lowest estimated TTFT: queued work plus the uncached suffix. It pairs a decode instance whose VRAM can hold the request.',
  prefill:
    "The chosen instance loads the cached prefix from its pool (sage) and computes only the uncached blocks (salmon). Its messenger streams each layer's KV over RDMA as soon as that layer finishes, so the transfer overlaps prefill.",
  transfer:
    "Prefill also stores the new KV in its own pool for later reuse (sage). The last layer's KV lands in the decode node's CPU DRAM (salmon).",
  decode:
    'Decode loads the transferred KV into VRAM, joins the continuous batch, and appends new KV blocks as it generates (salmon).',
}

const SCHEDULERS = [
  { id: 'sched-prefill', lines: ['cache-aware', 'prefill scheduler'] },
  { id: 'sched-balance', lines: ['KVCache balance', 'scheduler'] },
  { id: 'sched-decode', lines: ['load-balance', 'decode scheduler'] },
]

type CellRowProps = {
  view: MooncakeView
  keyPrefix: string
  count: number
  x: number
  y: number
  w: number
  h: number
  perRow?: number
  gap?: number
}

const CellRow: FunctionalComponent<CellRowProps> = ({
  view,
  keyPrefix,
  count,
  x,
  y,
  w,
  h,
  perRow = count,
  gap = 2,
}) => (
  <g>
    {Array.from({ length: count }, (_, i) => {
      const key = `${keyPrefix}-${i}`
      return (
        <rect
          class="mck-cell"
          data-mck-cell={key}
          data-state={view.cells[key] ?? 'empty'}
          x={x + (i % perRow) * (w + gap)}
          y={y + Math.floor(i / perRow) * (h + gap)}
          width={w}
          height={h}
        />
      )
    })}
  </g>
)

const Label: FunctionalComponent<{
  x: number
  y: number
  text: string
  cls?: string
  anchor?: 'start' | 'end'
}> = ({ x, y, text, cls, anchor }) => (
  <text class={`mck-label ${cls ?? ''}`.trim()} x={x} y={y} text-anchor={anchor}>
    {text}
  </text>
)

const fmtMs = (ms: number) => `${ms} ms`

const Messenger: FunctionalComponent<{
  x: number
  y: number
  w: number
  id: string
  on: string
}> = ({ x, y, w, id, on }) => (
  <g>
    <rect
      class="mck-node mck-msg"
      data-mck-active-key={`link-${id}`}
      data-active={on}
      x={x + 8}
      y={y}
      width={w - 16}
      height={MSG_H}
    />
    <Label x={x + 14} y={y + 13} text="messenger · RDMA" />
  </g>
)

const Graph: FunctionalComponent<{ layout: MooncakeLayout; view: MooncakeView }> = ({
  layout: L,
  view,
}) => {
  const on = (key: string) => (view.active[key] ? 'true' : 'false')
  return (
    <svg
      class={`mck-graph mck-graph--${L.id}`}
      viewBox={`0 0 ${L.viewW} ${L.viewH}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label="Conductor schedulers; two prefill instances with GPU paged KV, a CPU DRAM pool and a messenger; an RDMA junction; two decode instances that receive KV into CPU DRAM before GPU VRAM."
    >
      <Label x={L.prompt.x} y={L.prompt.y - 8} text={`prompt · ${PROMPT_BLOCKS} × 512 tok`} />
      <CellRow
        view={view}
        keyPrefix="prompt"
        count={PROMPT_BLOCKS}
        x={L.prompt.x}
        y={L.prompt.y}
        w={L.prompt.w}
        h={16}
        gap={L.prompt.gap}
      />
      <Label x={L.prompt.x} y={L.conductorY} text="conductor" cls="mck-label--head" />

      {L.links.map(link => (
        <polyline
          class="mck-link"
          data-mck-active-key={link.key}
          data-active={link.key ? on(link.key) : undefined}
          points={link.points.map(p => p.join(',')).join(' ')}
        />
      ))}

      {SCHEDULERS.map((sched, k) => {
        const box = L.scheds[k]
        return (
          <g>
            <rect
              class="mck-node"
              data-mck-active-key={sched.id}
              data-active={on(sched.id)}
              x={box.x}
              y={box.y}
              width={box.w}
              height={box.h}
            />
            {sched.lines.map((line, i) => (
              <Label x={box.x + 8} y={box.y + 18 + i * 15} text={line} />
            ))}
          </g>
        )
      })}

      <circle
        class="mck-rdma"
        data-mck-active-key="rdma"
        data-active={on('rdma')}
        cx={L.rdma.x}
        cy={L.rdma.y}
        r={L.rdma.r}
      />
      <Label x={L.rdmaLabel.x} y={L.rdmaLabel.y} text="RDMA" anchor={L.rdmaLabel.anchor} />

      {PREFILL_INSTANCES.map((instance, k) => {
        const { x, y, w, h } = L.prefill[k]
        return (
          <g>
            <rect
              class="mck-node"
              data-mck-active-key={instance.id}
              data-active={on(instance.id)}
              x={x}
              y={y}
              width={w}
              height={h}
            />
            <Label x={x + 8} y={y + 17} text={instance.label} cls="mck-label--head" />
            <Label x={x + 8} y={y + 37} text="queue" />
            <CellRow
              view={view}
              keyPrefix={`${instance.id}-queue`}
              count={QUEUE_SLOTS}
              x={x + 64}
              y={y + 27}
              w={16}
              h={14}
            />
            <Label x={x + 8} y={y + 60} text="GPU VRAM · paged KV" />
            <CellRow
              view={view}
              keyPrefix={`${instance.id}-vram`}
              count={PROMPT_BLOCKS}
              x={x + 8}
              y={y + 66}
              w={24}
              h={20}
            />
            <Label x={x + 8} y={y + 106} text="CPU DRAM / SSD pool" />
            <CellRow
              view={view}
              keyPrefix={`${instance.id}-pool`}
              count={POOL_SLOTS}
              perRow={8}
              x={x + 8}
              y={y + 112}
              w={24}
              h={16}
            />
            <Messenger
              x={x}
              y={y + PREFILL_MSG_Y}
              w={w}
              id={instance.id}
              on={on(`link-${instance.id}`)}
            />
          </g>
        )
      })}

      {DECODE_INSTANCES.map((instance, k) => {
        const { x, y, w, h } = L.decode[k]
        return (
          <g>
            <rect
              class="mck-node"
              data-mck-active-key={instance.id}
              data-active={on(instance.id)}
              x={x}
              y={y}
              width={w}
              height={h}
            />
            <Label x={x + 8} y={y + 17} text={instance.label} cls="mck-label--head" />
            <Messenger
              x={x}
              y={y + DECODE_MSG_Y}
              w={w}
              id={instance.id}
              on={on(`link-${instance.id}`)}
            />
            <Label x={x + 8} y={y + 62} text="CPU DRAM pool" />
            <CellRow
              view={view}
              keyPrefix={`${instance.id}-pool`}
              count={POOL_SLOTS}
              perRow={8}
              x={x + 8}
              y={y + 68}
              w={24}
              h={14}
            />
            <Label x={x + 8} y={y + 116} text="GPU VRAM · paged KV" />
            <CellRow
              view={view}
              keyPrefix={`${instance.id}-vram`}
              count={DECODE_VRAM_SLOTS}
              perRow={12}
              x={x + 8}
              y={y + 122}
              w={15.5}
              h={16}
              gap={1.7}
            />
          </g>
        )
      })}
    </svg>
  )
}

let figureCount = 0

const MooncakeFlowImpl: QuartzMdxComponent<Props> = ({ caption }) => {
  const plan = mooncakePlan(undefined)
  const view = mooncakeView(plan, 'route')
  const id = `mck-${++figureCount}`

  return (
    <figure
      class="mooncake-flow"
      data-mooncake-flow
      data-mck-request={plan.request.id}
      data-mck-stage="route"
    >
      <div class="mck-controls">
        <fieldset class="mck-seg-group">
          <legend>request</legend>
          {MOONCAKE_REQUESTS.map((request, i) => (
            <label class="mck-seg">
              <input
                type="radio"
                name={`${id}-request`}
                value={request.id}
                checked={i === 0}
                data-mck-request-input
              />
              <span>{request.label}</span>
            </label>
          ))}
        </fieldset>
        <fieldset class="mck-seg-group">
          <legend>stage</legend>
          {MOONCAKE_STAGES.map((stage, i) => (
            <label class="mck-seg">
              <input
                type="radio"
                name={`${id}-stage`}
                value={stage}
                checked={i === 0}
                data-mck-stage-input
              />
              <span>{stage}</span>
            </label>
          ))}
        </fieldset>
      </div>

      <div class="mck-drawings">
        {MOONCAKE_LAYOUTS.map(layout => (
          <Graph layout={layout} view={view} />
        ))}
      </div>

      <div class="mck-readouts">
        <table class="mck-table" aria-label="Prefill routing">
          <thead>
            <tr>
              <th scope="col">prefill</th>
              <th scope="col">cached prefix</th>
              <th scope="col">queued</th>
              <th scope="col">est. TTFT</th>
            </tr>
          </thead>
          <tbody>
            {plan.prefill.map(route => (
              <tr
                data-mck-row={route.id}
                data-selected={route.id === plan.prefillId ? 'true' : 'false'}
              >
                <th scope="row">{route.label}</th>
                <td data-mck-val={`${route.id}-hit`}>{`${route.hit}/${PROMPT_BLOCKS}`}</td>
                <td>{route.queued}</td>
                <td data-mck-val={`${route.id}-ttft`}>{fmtMs(route.ttftMs)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <table class="mck-table" aria-label="Decode admission">
          <thead>
            <tr>
              <th scope="col">decode</th>
              <th scope="col">resident</th>
              <th scope="col">after admit</th>
              <th scope="col">VRAM</th>
            </tr>
          </thead>
          <tbody>
            {plan.decode.map(route => (
              <tr
                data-mck-row={route.id}
                data-selected={route.id === plan.decodeId ? 'true' : 'false'}
              >
                <th scope="row">{route.label}</th>
                <td>{route.resident}</td>
                <td data-mck-val={`${route.id}-after`}>{route.after}</td>
                <td>{DECODE_VRAM_SLOTS}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div class="mck-notes" aria-live="polite">
        {MOONCAKE_STAGES.map(stage => (
          <p class="mck-note" data-mck-note={stage}>
            {STAGE_NOTES[stage]}
          </p>
        ))}
      </div>

      {caption ? (
        <figcaption class="mck-caption">
          <MathText text={caption} mathClass="mck-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const MooncakeFlowComponent = MooncakeFlowImpl as QuartzMdxComponent<Props>
MooncakeFlowComponent.css = style
MooncakeFlowComponent.afterDOMLoaded = script

export const MooncakeFlow = registerMdxComponent('MooncakeFlow', MooncakeFlowComponent)

export default (() => MooncakeFlow) satisfies (opts: undefined) => QuartzMdxComponent<Props>
