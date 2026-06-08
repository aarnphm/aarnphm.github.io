import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/flash-attention-tiles.inline'
import style from '../styles/flashAttentionTiles.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

const SEQ_LEN = 4
const HBM_X = 30
const HBM_LABEL_W = 28
const HBM_CELL = 22
const BLOCK_GAP = 26
const HBM_TOP = 40
const BLOCK_H = SEQ_LEN * HBM_CELL
const SRAM_X = 410
const SRAM_W = 240
const SRAM_PAD = 16
const SRAM_TILE_H = 36
const SRAM_TILE_GAP = 12

const BLOCKS = [
  { id: 'q', label: 'Q', attr: 'data-fat-q-row' },
  { id: 'k', label: 'K', attr: 'data-fat-k-row' },
  { id: 'v', label: 'V', attr: 'data-fat-v-row' },
  { id: 'o', label: 'O', attr: 'data-fat-o-row' },
] as const

const SRAM_TILES = [
  { label: 'Q_i', attr: 'data-fat-sram-q', aria: 'Q tile in SRAM' },
  { label: 'K_j', attr: 'data-fat-sram-k', aria: 'K tile in SRAM' },
  { label: 'V_j', attr: 'data-fat-sram-v', aria: 'V tile in SRAM' },
] as const

const STATS: { tex: string; key: string }[] = [
  { tex: 'm_1', key: 'm-0' },
  { tex: 'm_2', key: 'm-1' },
  { tex: 'l_1', key: 'l-0' },
  { tex: 'l_2', key: 'l-1' },
  { tex: 'O_1', key: 'o-0' },
  { tex: 'O_2', key: 'o-1' },
]

const RECURRENCE = [
  'm_i^{\\text{new}} = \\max\\!\\big(m_i^{\\text{old}},\\, \\max_j S_{ij}^{(t)}\\big)',
  'l_i^{\\text{new}} = e^{m_i^{\\text{old}} - m_i^{\\text{new}}} l_i^{\\text{old}} + \\sum_j e^{S_{ij}^{(t)} - m_i^{\\text{new}}}',
  'O_i^{\\text{new}} = e^{m_i^{\\text{old}} - m_i^{\\text{new}}} O_i^{\\text{old}} + \\sum_j e^{S_{ij}^{(t)} - m_i^{\\text{new}}} V_j^{(t)}',
]

function tex(t: string, display = false): string {
  return katex.renderToString(t, {
    ...katexOptions,
    displayMode: display,
    output: 'html',
    macros: customMacros,
    strict: false,
    throwOnError: false,
  })
}

const Math: FunctionalComponent<{ t: string; d?: boolean; cls?: string }> = ({ t, d, cls }) => (
  <span class={cls} dangerouslySetInnerHTML={{ __html: tex(t, d) }} />
)

const HbmCol = HBM_X + HBM_LABEL_W
const blockTop = (i: number) => HBM_TOP + i * (BLOCK_H + BLOCK_GAP)
const sramTop = (i: number) => HBM_TOP + i * (SRAM_TILE_H + SRAM_TILE_GAP)
const SRAM_LEFT = SRAM_X + SRAM_PAD
const SRAM_INNER = SRAM_W - SRAM_PAD * 2
const PANEL_TOP = HBM_TOP - 8
const PANEL_H = sramTop(BLOCKS.length - 1) + SRAM_TILE_H + SRAM_PAD - PANEL_TOP

const FlashAttentionTilesImpl: QuartzMdxComponent<Props> = ({ caption }) => {
  const arrowId = 'fat-arrow-head'
  return (
    <figure class="flash-attention-tiles" data-flash-attention-tiles>
      <div class="fat-stage">
        <div>
          <svg
            class="fat-graph"
            viewBox="0 0 720 492"
            preserveAspectRatio="xMidYMid meet"
            role="img"
            aria-label="FlashAttention tiled streaming: HBM holds full Q, K, V, O matrices while SRAM holds the active Q_i, K_j, V_j tiles plus running stats m, l, O_i that the online softmax recurrence updates each step."
          >
            <defs>
              <marker
                id={arrowId}
                viewBox="0 0 10 10"
                refX="9"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path class="fat-arrowhead is-active" d="M0,0 L10,5 L0,10 z" />
              </marker>
            </defs>

            <foreignObject x={HbmCol - 35} y={4} width={92} height={14}>
              <div
                class="fat-fo fat-fo--axis"
                dangerouslySetInnerHTML={{ __html: tex('\\mathrm{HBM}') }}
              />
            </foreignObject>
            <foreignObject x={HbmCol - 50} y={18} width={120} height={12}>
              <div
                class="fat-fo fat-fo--sublabel"
                dangerouslySetInnerHTML={{ __html: tex('\\text{slow, full }L{\\times}d') }}
              />
            </foreignObject>
            <foreignObject x={SRAM_X + SRAM_W / 2 - 46} y={4} width={92} height={14}>
              <div
                class="fat-fo fat-fo--axis"
                dangerouslySetInnerHTML={{ __html: tex('\\mathrm{SRAM}') }}
              />
            </foreignObject>
            <foreignObject x={SRAM_X + SRAM_W / 2 - 70} y={18} width={140} height={12}>
              <div
                class="fat-fo fat-fo--sublabel"
                dangerouslySetInnerHTML={{
                  __html: tex('\\text{fast, }B_m{\\cdot}B_n\\text{ on-chip}'),
                }}
              />
            </foreignObject>

            <rect
              class="fat-sram-panel"
              x={SRAM_X}
              y={PANEL_TOP}
              width={SRAM_W}
              height={PANEL_H}
              rx={6}
            />

            {BLOCKS.map((b, idx) => {
              const top = blockTop(idx)
              return (
                <g class="fat-hbm-block" data-fat-block={b.id}>
                  <foreignObject
                    x={HBM_X - 6}
                    y={top + BLOCK_H / 2 - 9}
                    width={HBM_LABEL_W}
                    height={18}
                  >
                    <div
                      class="fat-fo fat-fo--block-label"
                      dangerouslySetInnerHTML={{ __html: tex(b.label) }}
                    />
                  </foreignObject>
                  {Array.from({ length: SEQ_LEN }, (_, i) => (
                    <rect
                      class="fat-cell"
                      x={HbmCol}
                      y={top + i * HBM_CELL}
                      width={HBM_CELL}
                      height={HBM_CELL}
                      rx={2}
                      {...{ [b.attr]: i }}
                    />
                  ))}
                  <foreignObject x={HbmCol - 20} y={top + BLOCK_H + 4} width={60} height={12}>
                    <div
                      class="fat-fo fat-fo--sublabel"
                      dangerouslySetInnerHTML={{ __html: tex(`${SEQ_LEN}{\\times}d`) }}
                    />
                  </foreignObject>
                </g>
              )
            })}

            {SRAM_TILES.map((t, idx) => {
              const y = sramTop(idx)
              return (
                <g>
                  <rect
                    class="fat-cell is-active"
                    x={SRAM_LEFT}
                    y={y}
                    width={SRAM_INNER}
                    height={SRAM_TILE_H}
                    rx={4}
                    aria-label={t.aria}
                  />
                  <foreignObject x={SRAM_LEFT} y={y} width={SRAM_INNER} height={SRAM_TILE_H}>
                    <div
                      class="fat-fo fat-fo--tile"
                      {...{ [t.attr]: '' }}
                      dangerouslySetInnerHTML={{ __html: tex(t.label) }}
                    />
                  </foreignObject>
                </g>
              )
            })}

            <g>
              <rect
                class="fat-cell is-active"
                x={SRAM_LEFT}
                y={sramTop(3)}
                width={SRAM_INNER}
                height={SRAM_TILE_H}
                rx={4}
                aria-label="online-softmax stats in SRAM"
              />
              <foreignObject x={SRAM_LEFT} y={sramTop(3)} width={SRAM_INNER} height={SRAM_TILE_H}>
                <div
                  class="fat-fo"
                  dangerouslySetInnerHTML={{ __html: tex('m_i,\\, l_i,\\, O_i') }}
                />
              </foreignObject>
            </g>

            {BLOCKS.map((_, idx) => {
              const fy = blockTop(idx) + BLOCK_H / 2
              const ty = sramTop(idx) + SRAM_TILE_H / 2
              const mx = (HbmCol + HBM_CELL + SRAM_LEFT) / 2
              return (
                <path
                  class="fat-arrow"
                  data-fat-arrow
                  d={`M ${HbmCol + HBM_CELL} ${fy} C ${mx} ${fy}, ${mx} ${ty}, ${SRAM_LEFT} ${ty}`}
                  marker-end={`url(#${arrowId})`}
                />
              )
            })}

            <foreignObject x={190} y={428} width={350} height={22}>
              <div
                class="fat-fo fat-fo--caption-line"
                dangerouslySetInnerHTML={{
                  __html: tex(
                    '\\text{outer: }K,V\\text{ tiles; inner: }Q\\text{ tiles; online softmax accumulates}',
                  ),
                }}
              />
            </foreignObject>
          </svg>

          <div class="fat-controls">
            <button type="button" class="fat-button" data-fat-prev aria-label="Previous tile step">
              prev
            </button>
            <button type="button" class="fat-button" data-fat-next aria-label="Next tile step">
              next
            </button>
            <button
              type="button"
              class="fat-button"
              data-fat-reset
              aria-label="Reset to first step"
            >
              reset
            </button>
            <span
              class="fat-step-readout"
              data-fat-step-readout
              dangerouslySetInnerHTML={{
                __html: tex('\\text{step } 1/4,\\ \\text{outer } j{=}1,\\ \\text{inner } i{=}1'),
              }}
            />
          </div>
        </div>

        <aside class="fat-sidebar">
          <div class="fat-card">
            <dl class="fat-stats">
              {STATS.map(s => [
                <dt>
                  <Math t={s.tex} />
                </dt>,
                <dd data-fat-stat={s.key} dangerouslySetInnerHTML={{ __html: tex('\\text{-}') }} />,
              ])}
            </dl>
          </div>

          <div class="fat-card">
            <p
              class="fat-ratio"
              data-fat-ratio
              dangerouslySetInnerHTML={{ __html: tex('\\text{HBM: streamed in tiles}') }}
            />
          </div>
        </aside>
      </div>

      <section class="fat-recurrence" aria-label="online softmax recurrence">
        {RECURRENCE.map(t => (
          <Math t={t} d cls="fat-math" />
        ))}
      </section>

      {caption ? (
        <figcaption class="fat-caption">
          <MathText text={caption} mathClass="fat-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const FlashAttentionTilesComponent = FlashAttentionTilesImpl as QuartzMdxComponent<Props>
FlashAttentionTilesComponent.css = style
FlashAttentionTilesComponent.afterDOMLoaded = script

export const FlashAttentionTiles = registerMdxComponent(
  'FlashAttentionTiles',
  FlashAttentionTilesComponent,
)

export default (() => FlashAttentionTiles) satisfies (opts: undefined) => QuartzMdxComponent<Props>
