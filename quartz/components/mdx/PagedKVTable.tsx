import katex from 'katex'
import { type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../../cfg'
import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/paged-kv-table.inline'
import style from '../styles/pagedKVTable.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }
type SeqId = 'a' | 'b'
type Tone = 'salmon' | 'sage'
type Seq = { id: SeqId; label: string; tokens: number; blocks: number; tone: Tone }
type Occupant = { seq: SeqId; blk: number }

const BLOCK_SIZE = 4
const PHYSICAL_SLOTS = 8

const SEQS: Seq[] = [
  { id: 'a', label: 'seq A', tokens: 9, blocks: 3, tone: 'salmon' },
  { id: 'b', label: 'seq B', tokens: 5, blocks: 2, tone: 'sage' },
]

const INITIAL_MAP: Array<{ seq: SeqId; blk: number; phys: number }> = [
  { seq: 'a', blk: 0, phys: 5 },
  { seq: 'a', blk: 1, phys: 2 },
  { seq: 'a', blk: 2, phys: 7 },
  { seq: 'b', blk: 0, phys: 1 },
  { seq: 'b', blk: 1, phys: 4 },
]

const renderMath = (tex: string, display = false): string =>
  katex.renderToString(tex, {
    ...katexOptions,
    displayMode: display,
    output: 'html',
    macros: customMacros,
    strict: false,
    throwOnError: false,
  })

const MathLabel: FunctionalComponent<{ tex: string; display?: boolean }> = ({
  tex,
  display = false,
}) => {
  const Tag = display ? 'div' : 'span'
  return (
    <Tag
      class={`pkt-math${display ? ' pkt-math--display' : ''}`}
      dangerouslySetInnerHTML={{ __html: renderMath(tex, display) }}
    />
  )
}

const mappingFor = (seq: SeqId, blk: number): number | undefined =>
  INITIAL_MAP.find(m => m.seq === seq && m.blk === blk)?.phys

const occupantOf = (idx: number): Occupant | undefined => {
  const m = INITIAL_MAP.find(x => x.phys === idx)
  return m ? { seq: m.seq, blk: m.blk } : undefined
}

const tokensInBlock = (s: Seq, blk: number): number =>
  Math.max(0, Math.min(BLOCK_SIZE, s.tokens - blk * BLOCK_SIZE))

const initialState = JSON.stringify({
  seqs: SEQS.map(s => ({
    id: s.id,
    tokens: s.tokens,
    blocks: Array.from({ length: s.blocks }, (_, blk) => ({
      blk,
      filled: tokensInBlock(s, blk),
      phys: mappingFor(s.id, blk) ?? null,
    })),
  })),
  blockSize: BLOCK_SIZE,
  physicalSlots: PHYSICAL_SLOTS,
})

const LogicalBlock: FunctionalComponent<{ seq: Seq; blk: number }> = ({ seq, blk }) => {
  const filled = tokensInBlock(seq, blk)
  const phys = mappingFor(seq.id, blk)
  const mapped = phys !== undefined
  return (
    <button
      type="button"
      class={`pkt-lblock pkt-lblock--${seq.tone}`}
      data-pkt-lblock
      data-pkt-seq={seq.id}
      data-pkt-blk={String(blk)}
      data-pkt-mapped={mapped ? 'true' : 'false'}
      aria-label={`${seq.label} logical block ${blk}, ${filled} of ${BLOCK_SIZE} tokens, ${
        mapped ? `mapped to physical slot ${phys}` : 'paged out'
      }`}
    >
      <span class="pkt-lblock-tag">
        {seq.id}
        <sub>{blk}</sub>
      </span>
      <span class="pkt-lblock-cells" aria-hidden="true">
        {Array.from({ length: BLOCK_SIZE }, (_, i) => (
          <span class={`pkt-tok${i < filled ? ' is-filled' : ''}`} />
        ))}
      </span>
      <span class="pkt-lblock-meta" data-pkt-lblock-meta>
        {mapped ? `to p${phys}` : 'paged out'}
      </span>
    </button>
  )
}

const PhysicalSlot: FunctionalComponent<{ idx: number }> = ({ idx }) => {
  const occ = occupantOf(idx)
  const tone: Tone | 'free' = occ ? (occ.seq === 'a' ? 'salmon' : 'sage') : 'free'
  return (
    <li
      class={`pkt-pblock pkt-pblock--${tone}`}
      data-pkt-pblock
      data-pkt-phys={String(idx)}
      data-pkt-occupied={occ ? 'true' : 'false'}
    >
      <span class="pkt-pblock-idx">{idx}</span>
      <span class="pkt-pblock-label" data-pkt-pblock-label>
        {occ ? `${occ.seq}${occ.blk}` : 'free'}
      </span>
      <button
        type="button"
        class="pkt-evict"
        data-pkt-evict
        aria-label={`Evict physical slot ${idx}`}
        disabled={!occ}
      >
        evict
      </button>
    </li>
  )
}

const PagedKVTableImpl: QuartzMdxComponent<Props> = ({ caption }) => (
  <figure class="paged-kv-table" data-paged-kv-table data-pkt-initial={initialState}>
    <div class="pkt-stage">
      <section class="pkt-col" aria-label="Logical KV blocks per sequence">
        <header class="pkt-col-head">logical KV blocks</header>
        <div class="pkt-seqs">
          {SEQS.map(seq => (
            <div class={`pkt-seq pkt-seq--${seq.tone}`} aria-label={seq.label}>
              <span class="pkt-seq-title">{seq.label}</span>
              <div class="pkt-seq-blocks" data-pkt-seq-blocks={seq.id}>
                {Array.from({ length: seq.blocks }, (_, blk) => (
                  <LogicalBlock seq={seq} blk={blk} />
                ))}
              </div>
              <button
                type="button"
                class="pkt-append"
                data-pkt-append
                data-pkt-seq={seq.id}
                aria-label={`Append a token to ${seq.label}`}
              >
                + token
              </button>
            </div>
          ))}
        </div>
      </section>

      <section class="pkt-col" aria-label="Page table">
        <header class="pkt-col-head">page table</header>
        <table class="pkt-pt">
          <thead>
            <tr>
              <th scope="col">
                <MathLabel tex="(\text{seq},\, b)" />
              </th>
              <th scope="col" aria-hidden="true" />
              <th scope="col">phys</th>
            </tr>
          </thead>
          <tbody data-pkt-pt-body>
            {INITIAL_MAP.map(m => (
              <tr
                class="pkt-pt-row"
                data-pkt-pt-row
                data-pkt-seq={m.seq}
                data-pkt-blk={String(m.blk)}
                data-pkt-phys={String(m.phys)}
              >
                <td class="pkt-pt-key">
                  <span class="pkt-pt-id">
                    {m.seq}
                    <sub>{m.blk}</sub>
                  </span>
                </td>
                <td class="pkt-pt-arrow" aria-hidden="true">
                  to
                </td>
                <td class="pkt-pt-val">p{m.phys}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section class="pkt-col" aria-label="Physical GPU memory">
        <header class="pkt-col-head">physical memory</header>
        <ol class="pkt-pblocks">
          {Array.from({ length: PHYSICAL_SLOTS }, (_, idx) => (
            <PhysicalSlot idx={idx} />
          ))}
        </ol>
      </section>
    </div>

    <aside class="pkt-side" aria-label="Per-block attention and fragmentation readout">
      <div class="pkt-card">
        <h4>per-block attention</h4>
        <MathLabel
          display
          tex="A_{ij} = \dfrac{\exp\!\left(q_i^{\top} K_j / \sqrt{d}\right)}{\sum_{t}\exp\!\left(q_i^{\top} K_t / \sqrt{d}\right)}"
        />
        <p class="pkt-card-note">
          <MathLabel tex="K_j" /> is the j-th physical block, fetched through the page table on
          demand.
        </p>
      </div>
      <div class="pkt-card">
        <h4>live readout</h4>
        <dl class="pkt-readout">
          <div class="pkt-readout-row">
            <dt>tokens (A, B)</dt>
            <dd>
              <span data-pkt-tokens-a class="pkt-val">
                9
              </span>
              <span class="pkt-sep">/</span>
              <span data-pkt-tokens-b class="pkt-val">
                5
              </span>
            </dd>
          </div>
          <div class="pkt-readout-row">
            <dt>allocated blocks</dt>
            <dd>
              <span data-pkt-alloc class="pkt-val">
                5
              </span>
              <span class="pkt-sep">/</span>
              <span class="pkt-muted">{PHYSICAL_SLOTS}</span>
            </dd>
          </div>
          <div class="pkt-readout-row">
            <dt>internal fragmentation</dt>
            <dd>
              <MathLabel tex="1 - \dfrac{\text{tokens}}{B\cdot\#\text{blocks}}" />
              <span class="pkt-sep">=</span>
              <span data-pkt-frag class="pkt-val pkt-val--big">
                0%
              </span>
            </dd>
          </div>
          <div class="pkt-readout-row">
            <dt>contiguous baseline</dt>
            <dd>
              up to <MathLabel tex="(B-1)/B" />
              <span class="pkt-sep">=</span>
              <span class="pkt-val pkt-val--muted">75%</span>
            </dd>
          </div>
        </dl>
      </div>
      <div class="pkt-card pkt-card--reset">
        <button type="button" class="pkt-reset" data-pkt-reset aria-label="Reset to initial state">
          reset
        </button>
      </div>
    </aside>

    <p class="pkt-intuition">
      Virtual memory for the KV cache. Two sequences of different lengths share one pool of physical
      blocks through the page table; growth needs no contiguous region, eviction frees a slot
      anywhere, and cold blocks can spill to host. Click a logical block to trace its mapping, press{' '}
      <strong>+ token</strong> to grow, or <strong>evict</strong> to free a slot.
    </p>

    {caption ? (
      <figcaption class="pkt-caption">
        <MathText text={caption} mathClass="pkt-math" />
      </figcaption>
    ) : null}
  </figure>
)

const PagedKVTableComponent = PagedKVTableImpl as QuartzMdxComponent<Props>
PagedKVTableComponent.css = style
PagedKVTableComponent.afterDOMLoaded = script

export const PagedKVTable = registerMdxComponent('PagedKVTable', PagedKVTableComponent)

export default (() => PagedKVTable) satisfies (opts: undefined) => QuartzMdxComponent<Props>
