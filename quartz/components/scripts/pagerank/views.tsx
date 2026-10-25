// Markup shared by the emitter, which renders it into the static page, and the page script, which
// re-renders it with preact as filters and the selection change.
import {
  logTicks,
  type PageRankLink,
  type PageRankModel,
  type PageRankNode,
} from '../../../util/pagerank-view'
import { resolveRelative, type FullSlug } from '../../../util/path'

export const pageSize = 100
const oneSource = 0.6

export const formatTimes = (times: number) => times.toFixed(2)
export const formatShare = (share: number) =>
  share >= 0.1 || share === 0 ? `${Math.round(share * 100)}%` : `${(share * 100).toFixed(1)}%`
export const formatDelta = (delta: number) =>
  delta > 0 ? `+${delta}` : delta < 0 ? `−${-delta}` : '0'

export const noteHref = (slug: string) => resolveRelative('pagerank' as FullSlug, slug as FullSlug)

export const NoteLink = ({ node }: { node: PageRankNode }) => (
  <a class="internal" href={noteHref(node.slug)} data-slug={node.slug}>
    {node.title}
  </a>
)

const Row = ({ node, selected }: { node: PageRankNode; selected: boolean }) => (
  <tr data-id={node.id} aria-current={selected ? 'true' : undefined}>
    <td class="pagerank-col-position">
      <button
        type="button"
        class="pagerank-select"
        data-pr-select={node.id}
        aria-pressed={selected}
        aria-label={`inspect ${node.title}, rank ${node.position}`}
      >
        {node.position}
      </button>
    </td>
    <td class="pagerank-col-note">
      <span class="pagerank-note-line">
        <NoteLink node={node} />
        {node.courseIndex && (
          <span class="pagerank-chip" title="course navigation index">
            course index
          </span>
        )}
        {node.inbound > 0 && node.topShare > oneSource && (
          <span
            class="pagerank-chip"
            title={`one note gives ${formatShare(node.topShare)} of this score`}
          >
            one source
          </span>
        )}
      </span>
      <span class="pagerank-slug">{node.slug}</span>
    </td>
    <td class="pagerank-col-times">{formatTimes(node.times)}</td>
    <td class="pagerank-col-axis">
      <span class="pagerank-axis-mark" style={{ '--px': node.axis.toFixed(4) }} />
    </td>
    <td class="pagerank-col-inbound">{node.inbound}</td>
    <td class="pagerank-col-outbound">{node.outbound}</td>
    <td class={node.delta > 0 ? 'pagerank-col-delta pagerank-rise' : 'pagerank-col-delta'}>
      {formatDelta(node.delta)}
    </td>
  </tr>
)

const Status = ({ span, text }: { span: number; text: string }) => (
  <tr class="pagerank-status">
    <td colSpan={span}>{text}</td>
  </tr>
)

export const RankRows = ({ rows, selected }: { rows: readonly PageRankNode[]; selected: number }) =>
  rows.length === 0 ? (
    <Status
      span={7}
      text="no notes match. global positions are unchanged; clear a filter to see them."
    />
  ) : (
    <>
      {rows.map(node => (
        <Row key={node.id} node={node} selected={node.id === selected} />
      ))}
    </>
  )

export const LinkRows = ({ links, empty }: { links: readonly PageRankLink[]; empty: string }) =>
  links.length === 0 ? (
    <Status span={3} text={empty} />
  ) : (
    <>
      {links.map(link => (
        <tr key={link.node.id}>
          <td>
            {/* The rank selects the note here; the title opens it. */}
            <button
              type="button"
              class="pagerank-link-rank"
              data-pr-select={link.node.id}
              aria-label={`inspect ${link.node.title}, rank ${link.node.position}`}
            >
              {link.node.position}
            </button>
            <NoteLink node={link.node} />
          </td>
          <td>{formatTimes(link.gives)}</td>
          <td class="pagerank-share">
            <span class="pagerank-bar" style={{ '--share': Math.min(1, link.share).toFixed(4) }} />
            <span>{formatShare(link.share)}</span>
          </td>
        </tr>
      ))}
    </>
  )

const height = 240
const margin = { top: 12, right: 14, bottom: 34, left: 46 }
const tickLabel = (value: number, unit = '') =>
  (value >= 1000 ? `${value / 1000}k` : value >= 1 ? String(value) : value.toPrecision(1)) + unit

/**
 * Rank–size plot: log position against log ×mean, one polyline for the whole corpus. Without a
 * measured width (the static page) it draws at `fallbackWidth` and scales to its box.
 */
export function RankSizePlot({
  model,
  width,
  selected,
  pointer,
}: {
  model: PageRankModel
  width?: number
  selected: PageRankNode | null
  pointer: PageRankNode | null
}) {
  const { byPosition, summary } = model
  const n = byPosition.length
  if (n === 0) return null
  const w = width ?? 960
  const yMin = byPosition[n - 1].times
  const yMax = summary.max || 1
  const plotW = w - margin.left - margin.right
  const plotH = height - margin.top - margin.bottom
  const lx = Math.log10(Math.max(n, 10))
  const ly0 = Math.log10(yMin)
  const ly1 = Math.log10(yMax)
  const sx = (position: number) => margin.left + (Math.log10(position) / lx) * plotW
  const sy = (times: number) =>
    margin.top + (1 - (Math.log10(times) - ly0) / (ly1 - ly0 || 1)) * plotH

  let d = ''
  let lastX = -1
  for (const [index, node] of byPosition.entries()) {
    const x = sx(node.position)
    // Sub-pixel steps in the long tail draw nothing visible.
    if (x - lastX < 0.5 && index !== n - 1) continue
    d += `${d ? 'L' : 'M'}${x.toFixed(1)} ${sy(node.times).toFixed(1)}`
    lastX = x
  }
  const fitAt = (position: number) =>
    sy(10 ** (summary.intercept - summary.alpha * Math.log10(position)))
  const mark = (node: PageRankNode | null, r: number, className: string) => (
    <circle
      r={r}
      class={className}
      hidden={!node}
      style={node ? { transform: `translate(${sx(node.position)}px, ${sy(node.times)}px)` } : {}}
    />
  )
  const xTicks = logTicks(1, n)
  const yTicks = logTicks(yMin, yMax)
  return (
    <svg
      viewBox={`0 0 ${w} ${height}`}
      width={width ?? '100%'}
      height={width ? height : undefined}
      role="img"
      aria-label={`rank–size plot of ${n} notes: score from ${yMax.toFixed(2)}× down to ${yMin.toFixed(2)}× the mean, both axes logarithmic`}
    >
      <clipPath id="pagerank-plot-clip">
        <rect x={margin.left} y={margin.top} width={plotW} height={plotH} />
      </clipPath>
      <g class="pagerank-grid">
        {xTicks.map(value => (
          <line
            key={`x${value}`}
            x1={sx(value)}
            x2={sx(value)}
            y1={margin.top}
            y2={margin.top + plotH}
          />
        ))}
        {yTicks.map(value => (
          <line
            key={`y${value}`}
            x1={margin.left}
            x2={margin.left + plotW}
            y1={sy(value)}
            y2={sy(value)}
          />
        ))}
      </g>
      <rect
        x={margin.left}
        y={margin.top}
        width={plotW}
        height={plotH}
        class="pagerank-plot-frame"
      />
      <path
        d={`M${sx(1)} ${fitAt(1)}L${sx(n)} ${fitAt(n)}`}
        class="pagerank-fit"
        clip-path="url(#pagerank-plot-clip)"
      />
      <path d={d} class="pagerank-curve" />
      <g class="pagerank-ticks">
        {xTicks.map(value => (
          <text
            key={`x${value}`}
            x={sx(value)}
            y={height - margin.bottom + 15}
            text-anchor="middle"
          >
            {tickLabel(value)}
          </text>
        ))}
        {yTicks.map(value => (
          <text key={`y${value}`} x={margin.left - 6} y={sy(value) + 4} text-anchor="end">
            {tickLabel(value, '×')}
          </text>
        ))}
      </g>
      <text x={margin.left + plotW} y={height - 4} text-anchor="end" class="pagerank-axis-title">
        rank (log)
      </text>
      {/* The curve falls from the top left, so the top right corner stays clear for the label. */}
      <text
        x={margin.left + plotW - 4}
        y={margin.top + 14}
        text-anchor="end"
        class="pagerank-axis-title"
      >
        ×mean (log)
      </text>
      {mark(pointer && pointer !== selected ? pointer : null, 5.5, 'pagerank-mark-pointer')}
      {mark(selected, 4.5, 'pagerank-mark-selected')}
    </svg>
  )
}
