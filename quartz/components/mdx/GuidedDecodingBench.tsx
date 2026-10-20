import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/guided-decoding-bench.inline'
import style from '../styles/guidedDecodingBench.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; id?: string }

// Michael Goin's vLLM runs: Llama-3.1-8B, one H100, NousResearch/json-mode-eval, 50% guided.
const QPS = [1, 5, 10, 15, 20, 25] as const
type Metric = {
  key: 'ttft' | 'tpot'
  tab: string
  name: string
  axis: string
  log: boolean
  xgrammar: readonly number[]
  outlines: readonly number[]
}
const METRICS: readonly Metric[] = [
  {
    key: 'ttft',
    tab: 'TTFT',
    name: 'time to first token',
    axis: 'mean TTFT (ms, log)',
    log: true,
    xgrammar: [75, 68, 80, 118, 220, 446],
    outlines: [10507, 139455, 144131, 204710, 192325, 244884],
  },
  {
    key: 'tpot',
    tab: 'TPOT',
    name: 'time per output token',
    axis: 'mean TPOT (ms)',
    log: false,
    xgrammar: [8.4, 9.6, 12.1, 16.1, 26.5, 32.0],
    outlines: [7.9, 13.4, 19.6, 73.8, 68.6, 152.3],
  },
]

const VIEW_W = 560
const VIEW_H = 250
const X0 = 60
const X1 = 548
const Y0 = 206
const Y1 = 22
const BAR_W = 22
const BAR_GAP = 4
// The log axis spans 10 ms to 10⁶ ms; a log bar has no zero, so bars rise from the 10 ms floor.
const LOG_DEC = [1, 6] as const
const LIN_MAX = 200

const r1 = (n: number) => Math.round(n * 10) / 10
const groupW = (X1 - X0) / QPS.length
const groupCx = (i: number) => X0 + groupW * (i + 0.5)
const yOf = (m: Metric, v: number) =>
  m.log
    ? Y0 - ((Math.log10(v) - LOG_DEC[0]) / (LOG_DEC[1] - LOG_DEC[0])) * (Y0 - Y1)
    : Y0 - (v / LIN_MAX) * (Y0 - Y1)
const ticksOf = (m: Metric) => (m.log ? [10, 100, 1e3, 1e4, 1e5, 1e6] : [0, 50, 100, 150, 200])
const tickLabel = (v: number) => (v >= 1e6 ? `${v / 1e6}M` : v >= 1e3 ? `${v / 1e3}k` : String(v))
const ms = (m: Metric, v: number) => (m.log ? v.toLocaleString('en-US') : v.toFixed(1))
const ratioLabel = (r: number) => (r >= 100 ? `${Math.round(r)}×` : `${r.toFixed(2)}×`)

const rowsOf = (m: Metric) =>
  QPS.map((q, i) => {
    const xg = m.xgrammar[i]
    const ol = m.outlines[i]
    return { q, i, xg, ol, ratio: ratioLabel(ol / xg) }
  })

const GuidedDecodingBenchImpl: QuartzMdxComponent<Props> = ({ caption, id = 'gdb' }) => (
  <figure class="guided-decoding-bench" data-guided-decoding-bench>
    <div class="gdb-head">
      <div class="gdb-tablist" role="tablist" aria-label="Latency metric">
        {METRICS.map((m, k) => (
          <button
            type="button"
            class="gdb-tab"
            role="tab"
            id={`${id}-tab-${m.key}`}
            aria-controls={`${id}-panel-${m.key}`}
            aria-selected={k === 0 ? 'true' : 'false'}
            tabindex={k === 0 ? 0 : -1}
            title={m.name}
            data-gdb-tab
          >
            {m.tab}
          </button>
        ))}
      </div>
      <ul class="gdb-legend" aria-label="Series">
        <li>
          <span class="gdb-swatch gdb-swatch--xgrammar" aria-hidden="true" />
          vLLM 0.6.5, XGrammar
        </li>
        <li>
          <span class="gdb-swatch gdb-swatch--outlines" aria-hidden="true" />
          vLLM 0.6.4, Outlines
        </li>
      </ul>
    </div>

    {METRICS.map((m, k) => {
      const rows = rowsOf(m)
      const aria = `Grouped bar chart of mean ${m.name} in milliseconds${m.log ? ' on a log axis' : ''} at QPS ${QPS.join(', ')}. Outlines over XGrammar: ${rows.map(r => `${r.ratio} at QPS ${r.q}`).join(', ')}.`
      return (
        <div
          class="gdb-panel"
          role="tabpanel"
          id={`${id}-panel-${m.key}`}
          aria-labelledby={`${id}-tab-${m.key}`}
          hidden={k !== 0}
          data-gdb-panel
        >
          <svg
            class="gdb-graph"
            viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
            preserveAspectRatio="xMidYMid meet"
            role="img"
            aria-label={aria}
          >
            <g class="gdb-grid">
              {ticksOf(m).map(t => (
                <line x1={X0} x2={X1} y1={r1(yOf(m, t))} y2={r1(yOf(m, t))} />
              ))}
            </g>
            <g class="gdb-tick" text-anchor="end">
              {ticksOf(m).map(t => (
                <text x={X0 - 8} y={r1(yOf(m, t)) + 3.5}>
                  {tickLabel(t)}
                </text>
              ))}
            </g>
            <g class="gdb-tick" text-anchor="middle">
              {QPS.map((q, i) => (
                <text x={r1(groupCx(i))} y={Y0 + 16}>
                  {q}
                </text>
              ))}
            </g>
            <text class="gdb-axis" x={(X0 + X1) / 2} y={VIEW_H - 4} text-anchor="middle">
              requests per second (QPS)
            </text>
            <text class="gdb-axis" x={4} y={10}>
              {m.axis}
            </text>

            {rows.map(r => {
              const cx = groupCx(r.i)
              const yx = r1(yOf(m, r.xg))
              const yo = r1(yOf(m, r.ol))
              return (
                <g class="gdb-group" data-gdb-q={String(r.i)}>
                  <rect
                    class="gdb-bar gdb-bar--xgrammar"
                    x={r1(cx - BAR_GAP / 2 - BAR_W)}
                    y={yx}
                    width={BAR_W}
                    height={r1(Y0 - yx)}
                  />
                  <rect
                    class="gdb-bar gdb-bar--outlines"
                    x={r1(cx + BAR_GAP / 2)}
                    y={yo}
                    width={BAR_W}
                    height={r1(Y0 - yo)}
                  />
                  <text class="gdb-ratio" x={r1(cx)} y={Math.min(yx, yo) - 7} text-anchor="middle">
                    {r.ratio}
                  </text>
                  <rect
                    class="gdb-hit"
                    x={r1(cx - groupW / 2)}
                    y={Y1 - 12}
                    width={r1(groupW)}
                    height={Y0 - Y1 + 12}
                  />
                </g>
              )
            })}
            <line class="gdb-baseline" x1={X0} x2={X1} y1={Y0} y2={Y0} />
          </svg>

          <table class="gdb-table">
            <caption class="gdb-sr">Mean {m.name}, milliseconds</caption>
            <thead>
              <tr>
                <th scope="col">QPS</th>
                <th scope="col">
                  <span class="gdb-swatch gdb-swatch--xgrammar" aria-hidden="true" />
                  XGrammar ms
                </th>
                <th scope="col">
                  <span class="gdb-swatch gdb-swatch--outlines" aria-hidden="true" />
                  Outlines ms
                </th>
                <th scope="col">Outlines ÷ XGrammar</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(r => (
                <tr data-gdb-q={String(r.i)}>
                  <th scope="row">{r.q}</th>
                  <td>{ms(m, r.xg)}</td>
                  <td>{ms(m, r.ol)}</td>
                  <td class="gdb-ratio-cell">{r.ratio}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )
    })}

    {caption ? (
      <figcaption class="gdb-caption">
        <MathText text={caption} mathClass="gdb-math" />
      </figcaption>
    ) : null}
  </figure>
)

const GuidedDecodingBenchComponent = GuidedDecodingBenchImpl as QuartzMdxComponent<Props>
GuidedDecodingBenchComponent.css = style
GuidedDecodingBenchComponent.afterDOMLoaded = script

export const GuidedDecodingBench = registerMdxComponent(
  'GuidedDecodingBench',
  GuidedDecodingBenchComponent,
)

export default (() => GuidedDecodingBench) satisfies (opts: undefined) => QuartzMdxComponent<Props>
