import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/flash-data-flow.inline'
import style from '../styles/flashDataFlow.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

const MODES = [
  {
    id: 'fa1',
    label: 'FA-1',
    sub: 'Serial: load a $K, V$ tile into SRAM, compute, then store $O$ — one tile at a time, transfer and compute never overlap.',
  },
  {
    id: 'fa2',
    label: 'FA-2',
    sub: 'Parallel: query blocks are split across SMs (split-Q), so several $K, V$ tiles stream from HBM at once.',
  },
  {
    id: 'fa3',
    label: 'FA-3',
    sub: 'Async: producer warps (TMA) load tile $t{+}1$ while consumer warps (WGMMA) compute tile $t$ — transfer hides under compute.',
  },
] as const

const CELLS = ['Q', 'K', 'V', 'O'] as const
const INITIAL = 'fa1'

const FlashDataFlowImpl: QuartzMdxComponent<Props> = ({ caption }) => {
  return (
    <figure class="flash-data-flow" data-flash-data-flow data-fdf-mode={INITIAL}>
      <header class="fdf-head">
        <div class="fdf-tablist" role="tablist" aria-label="FlashAttention generation">
          {MODES.map(m => (
            <button
              type="button"
              class="fdf-tab"
              data-fdf-tab={m.id}
              role="tab"
              aria-selected={m.id === INITIAL ? 'true' : 'false'}
              tabindex={m.id === INITIAL ? 0 : -1}
            >
              {m.label}
            </button>
          ))}
        </div>
      </header>

      {MODES.map(m => (
        <div class={`fdf-sub fdf-sub--${m.id}`}>
          <MathText text={m.sub} mathClass="fdf-math" />
        </div>
      ))}

      <svg
        class="fdf-svg"
        viewBox="0 0 660 300"
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="HBM to SRAM data transfer across FlashAttention generations"
      >
        <rect class="fdf-tier fdf-tier--hbm" x={24} y={60} width={160} height={180} rx={6} />
        <text class="fdf-label--big" x={104} y={50}>
          HBM
        </text>
        <text class="fdf-bw" x={104} y={256}>
          1.5 TB/s, 40 GB
        </text>
        {CELLS.map((_, i) => (
          <rect class="fdf-cell" x={44} y={83 + i * 36} width={120} height={26} rx={3} />
        ))}
        {CELLS.map((c, i) => (
          <text class="fdf-label" x={104} y={101 + i * 36}>
            {c}
          </text>
        ))}

        <rect class="fdf-tier fdf-tier--sram" x={476} y={80} width={160} height={140} rx={6} />
        <text class="fdf-label--big" x={556} y={70}>
          SRAM
        </text>
        <text class="fdf-bw" x={556} y={236}>
          19 TB/s, 20 MB
        </text>
        <rect class="fdf-core" x={506} y={120} width={100} height={60} rx={4} />
        <text class="fdf-label" x={556} y={155}>
          compute
        </text>

        <line class="fdf-guide" x1={184} y1={96} x2={476} y2={96} />
        <line class="fdf-guide" x1={184} y1={132} x2={476} y2={132} />
        <line class="fdf-guide" x1={184} y1={168} x2={476} y2={168} />
        <line class="fdf-guide" x1={476} y1={204} x2={184} y2={204} />

        <g class="fdf-lane--1">
          <rect class="fdf-pkt fdf-pkt--load" x={184} y={89} width={18} height={14} />
        </g>
        <g class="fdf-lane--2">
          <rect class="fdf-pkt fdf-pkt--load" x={184} y={125} width={18} height={14} />
        </g>
        <g class="fdf-lane--3">
          <rect class="fdf-pkt fdf-pkt--load" x={184} y={161} width={18} height={14} />
        </g>
        <g class="fdf-lane--store">
          <rect class="fdf-pkt fdf-pkt--store" x={458} y={197} width={18} height={14} />
        </g>
      </svg>

      {caption ? (
        <figcaption class="fdf-caption">
          <MathText text={caption} mathClass="fdf-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const FlashDataFlowComponent = FlashDataFlowImpl as QuartzMdxComponent<Props>
FlashDataFlowComponent.css = style
FlashDataFlowComponent.afterDOMLoaded = script

export const FlashDataFlow = registerMdxComponent('FlashDataFlow', FlashDataFlowComponent)

export default (() => FlashDataFlow) satisfies (opts: undefined) => QuartzMdxComponent<Props>
