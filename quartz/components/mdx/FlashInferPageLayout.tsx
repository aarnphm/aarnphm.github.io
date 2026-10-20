import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/flashinfer-page-layout.inline'
import style from '../styles/flashInferPageLayout.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string }

const PAGE_SIZE = 8
const MAX_PAGES = 9
const KV_INDPTR = [0, 3, 6, 8] as const
const KV_PAGE_INDICES = [0, 5, 8, 1, 6, 7, 3, 4] as const
const KV_LAST_PAGE_LENS = [6, 4, 7] as const
const QO_INDPTR = [0, 4, 6, 9] as const
const REQ_TONES = ['salmon', 'sage', 'ink'] as const

type Tone = (typeof REQ_TONES)[number]

type Request = {
  id: number
  tone: Tone
  pages: number[]
  lastLen: number
  qoStart: number
  qoEnd: number
}

const REQUESTS: Request[] = KV_INDPTR.slice(0, -1).map((start, id) => ({
  id,
  tone: REQ_TONES[id],
  pages: KV_PAGE_INDICES.slice(start, KV_INDPTR[id + 1]),
  lastLen: KV_LAST_PAGE_LENS[id],
  qoStart: QO_INDPTR[id],
  qoEnd: QO_INDPTR[id + 1],
}))

const pageOwner = (page: number): Request | undefined => REQUESTS.find(r => r.pages.includes(page))

const filledSlots = (page: number): number => {
  const owner = pageOwner(page)
  if (!owner) return 0
  const isLast = owner.pages[owner.pages.length - 1] === page
  return isLast ? owner.lastLen : PAGE_SIZE
}

const CELL = 18
const GAP = 3
const PAGE_W = PAGE_SIZE * CELL + (PAGE_SIZE - 1) * GAP
const PAGE_H = CELL
const PAGE_GAP_Y = 10
const LABEL_W = 54
const GRID_X = LABEL_W + 8
const GRID_Y = 28
const VIEW_W = GRID_X + PAGE_W + 16
const VIEW_H = GRID_Y + MAX_PAGES * (PAGE_H + PAGE_GAP_Y)

let instance = 0

const FlashInferPageLayoutImpl: QuartzMdxComponent<Props> = ({ caption }) => {
  const id = `fpl-${++instance}`
  const state = {
    pageSize: PAGE_SIZE,
    maxPages: MAX_PAGES,
    kvIndptr: [...KV_INDPTR],
    kvPageIndices: [...KV_PAGE_INDICES],
    kvLastPageLens: [...KV_LAST_PAGE_LENS],
    qoIndptr: [...QO_INDPTR],
    requests: REQUESTS.map(r => ({
      id: r.id,
      tone: r.tone,
      pages: r.pages,
      lastLen: r.lastLen,
      qoStart: r.qoStart,
      qoEnd: r.qoEnd,
    })),
  }

  return (
    <figure
      class="flashinfer-page-layout"
      data-flashinfer-page-layout
      data-fpl-state={JSON.stringify(state)}
      data-fpl-req=""
    >
      <div class="fpl-controls">
        <fieldset class="fpl-segments">
          <legend class="fpl-legend">request</legend>
          <label class="fpl-segment">
            <input type="radio" name={`${id}-req`} value="" checked data-fpl-req-input />
            <span>all</span>
          </label>
          {REQUESTS.map(r => (
            <label class="fpl-segment">
              <input type="radio" name={`${id}-req`} value={String(r.id)} data-fpl-req-input />
              <span>r{r.id}</span>
            </label>
          ))}
        </fieldset>
      </div>

      <div class="fpl-body">
        <svg
          class="fpl-svg"
          viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
          role="img"
          aria-label="FlashInfer paged KV layout: nine physical pages of size eight, three requests mapped through kv_indptr and kv_page_indices."
        >
          <text class="fpl-title" x={GRID_X} y={16}>
            kv_data layout
          </text>
          {Array.from({ length: MAX_PAGES }, (_, page) => {
            const owner = pageOwner(page)
            const filled = filledSlots(page)
            const y = GRID_Y + page * (PAGE_H + PAGE_GAP_Y)
            return (
              <g
                key={page}
                data-fpl-page={String(page)}
                data-fpl-owner={owner ? String(owner.id) : ''}
                data-fpl-tone={owner?.tone ?? 'empty'}
              >
                <text class="fpl-page-label" x={LABEL_W} y={y + PAGE_H / 2 + 4} text-anchor="end">
                  page {page}
                </text>
                {Array.from({ length: PAGE_SIZE }, (_, slot) => {
                  const x = GRID_X + slot * (CELL + GAP)
                  const live = slot < filled
                  return (
                    <rect
                      key={slot}
                      class={`fpl-cell${live ? ' fpl-cell--live' : ''}`}
                      data-fpl-slot={String(slot)}
                      data-fpl-live={live ? 'true' : 'false'}
                      x={x}
                      y={y}
                      width={CELL}
                      height={PAGE_H}
                    />
                  )
                })}
              </g>
            )
          })}
        </svg>

        <div class="fpl-side">
          <table class="fpl-table">
            <thead>
              <tr>
                <th>req</th>
                <th>pages</th>
                <th>last len</th>
              </tr>
            </thead>
            <tbody>
              {REQUESTS.map(r => (
                <tr data-fpl-row={String(r.id)} data-fpl-tone={r.tone}>
                  <td>{r.id}</td>
                  <td>{r.pages.join(', ')}</td>
                  <td>{r.lastLen}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <dl class="fpl-readout">
            <div>
              <dt>kv_indptr</dt>
              <dd data-fpl-out="indptr">[{KV_INDPTR.join(', ')}]</dd>
            </div>
            <div>
              <dt>kv_page_indices</dt>
              <dd data-fpl-out="pages">[{KV_PAGE_INDICES.join(', ')}]</dd>
            </div>
            <div>
              <dt>kv_last_page_lens</dt>
              <dd data-fpl-out="lens">[{KV_LAST_PAGE_LENS.join(', ')}]</dd>
            </div>
            <div>
              <dt>qo_indptr</dt>
              <dd data-fpl-out="qo">[{QO_INDPTR.join(', ')}]</dd>
            </div>
            <div>
              <dt>shape</dt>
              <dd>
                ({MAX_PAGES}, 2, {PAGE_SIZE}, H, D)
              </dd>
            </div>
          </dl>

          <div class="fpl-qo" aria-label="query output layout">
            <span class="fpl-qo-label">qo</span>
            {Array.from({ length: QO_INDPTR[QO_INDPTR.length - 1] }, (_, i) => {
              const req = REQUESTS.find(r => i >= r.qoStart && i < r.qoEnd)!
              return (
                <span
                  class="fpl-qo-cell"
                  data-fpl-qo={String(i)}
                  data-fpl-owner={String(req.id)}
                  data-fpl-tone={req.tone}
                />
              )
            })}
          </div>
        </div>
      </div>

      {caption ? (
        <figcaption class="fpl-caption">
          <MathText text={caption} mathClass="fpl-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const FlashInferPageLayoutComponent = FlashInferPageLayoutImpl as QuartzMdxComponent<Props>
FlashInferPageLayoutComponent.css = style
FlashInferPageLayoutComponent.afterDOMLoaded = script

export const FlashInferPageLayout = registerMdxComponent(
  'FlashInferPageLayout',
  FlashInferPageLayoutComponent,
)

export default (() => FlashInferPageLayout) satisfies (opts: undefined) => QuartzMdxComponent<Props>
