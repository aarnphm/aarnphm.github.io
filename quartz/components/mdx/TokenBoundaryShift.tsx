import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/token-boundary-shift.inline'
import style from '../styles/tokenBoundaryShift.scss'
import data from './jump-forward-boundaries.json'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; boundary?: string }

type Boundary = {
  id: string
  label: string
  pieces: string[]
  natural: boolean
  probs: number[]
  allowedMass: number
  paul: number
  kl: number
  tv: number
}

// Bars run Paul's branch, then John's, longest token first.
const ORDER = ['Paul', 'Pa', 'P', 'John', 'Jo', 'J']
const TOKENS = ORDER.map(text => {
  const index = data.allowed.findIndex(a => a.text === text)
  if (index < 0) throw new Error(`allowed token ${text} missing from data`)
  return { text, index }
})

const buildBoundaries = (): Boundary[] => {
  const rows = data.boundaries.map(b => {
    const sum = b.probs.reduce((a, p) => a + p, 0)
    const probs = b.probs.map(p => p / sum)
    return {
      id: b.id,
      label: b.id === 'chars' ? 'chars' : b.id.split('|').join('·'),
      pieces: b.pieces,
      natural: b.natural,
      probs,
      allowedMass: b.allowedMass,
      paul: TOKENS.slice(0, 3).reduce((a, t) => a + probs[t.index], 0),
    }
  })
  const base = rows.find(r => r.natural)
  if (!base) throw new Error('no natural tokenization in data')
  return rows.map(r => ({
    ...r,
    kl: base.probs.reduce((a, p, i) => a + (p > 0 ? p * Math.log(p / r.probs[i]) : 0), 0),
    tv: 0.5 * base.probs.reduce((a, p, i) => a + Math.abs(p - r.probs[i]), 0),
  }))
}

const BOUNDARIES = buildBoundaries()
const NATURAL = BOUNDARIES.find(b => b.natural)!

const pct = (p: number) => (p >= 0.01 ? `${(p * 100).toFixed(1)}%` : `${(p * 100).toPrecision(2)}%`)
const readout = (b: Boundary) => ({
  forced: `${NATURAL.pieces.length} → ${b.pieces.length}`,
  mass: `${pct(NATURAL.allowedMass)} → ${pct(b.allowedMass)}`,
  paul: `${pct(NATURAL.paul)} → ${pct(b.paul)}`,
  kl: `${b.kl.toFixed(4)} nats`,
  tv: b.tv.toFixed(3),
  tokens: TOKENS.map(t => pct(b.probs[t.index])),
})
// The inline script only applies these; the model above is the single source.
const CLIENT_STATES = Object.fromEntries(BOUNDARIES.map(b => [b.id, readout(b)]))

const W = 520
const TAPE_X = 76
const TAPE_Y = [12, 46]
const CELL_H = 22
const CHAR_W = 7
const cellW = (text: string) => 10 + CHAR_W * text.length
const NODE = { x: 318, y: 29, s: 22 }
const PLOT = { x: 44, y: 104, w: W - 52, h: 150 }
const Y_MAX = 0.8
const GROUP_W = PLOT.w / 7
const BAR_W = 29
const groupX = (i: number) => PLOT.x + GROUP_W * (i + (i >= 3 ? 1 : 0.5))
const yOf = (p: number) => PLOT.y + PLOT.h * (1 - Math.min(p, Y_MAX) / Y_MAX)
const H = PLOT.y + PLOT.h + 40

const Tape = ({ pieces, y, kind }: { pieces: string[]; y: number; kind: string }) => {
  let x = TAPE_X
  const cells = pieces.map(p => {
    const cell = { p, x, w: cellW(p) }
    x += cell.w + 3
    return cell
  })
  const end = x - 3
  return (
    <g>
      {cells.map(c => (
        <g>
          <rect class={`tbs-cell tbs-cell--${kind}`} x={c.x} y={y} width={c.w} height={CELL_H} />
          <text class="tbs-piece" x={c.x + c.w / 2} y={y + 15} text-anchor="middle">
            {c.p}
          </text>
        </g>
      ))}
      <line class="tbs-link" x1={end} y1={y + CELL_H / 2} x2={NODE.x} y2={NODE.y + NODE.s / 2} />
      <text class="tbs-count" x={end + 6} y={y + (y < NODE.y ? -3 : CELL_H + 10)}>
        {`${pieces.length} tokens`}
      </text>
    </g>
  )
}

let instance = 0

const TokenBoundaryShiftImpl: QuartzMdxComponent<Props> = ({ caption, boundary = 'n|a|m|e' }) => {
  const id = `tbs-${++instance}`
  const b0 = BOUNDARIES.find(b => b.id === boundary) ?? BOUNDARIES[1]
  const r0 = readout(b0)

  return (
    <figure
      class="token-boundary-shift"
      data-token-boundary-shift
      data-tbs-states={JSON.stringify(CLIENT_STATES)}
    >
      <fieldset class="tbs-segments">
        <legend class="tbs-legend">Segmentation of name</legend>
        {BOUNDARIES.map(b => (
          <label class="tbs-segment">
            <input
              type="radio"
              name={`${id}-boundary`}
              value={b.id}
              checked={b.id === b0.id}
              data-tbs-input
            />
            <span>{b.label}</span>
          </label>
        ))}
      </fieldset>

      <ul class="tbs-key">
        <li>
          <span class="tbs-swatch tbs-swatch--natural" />
          natural tokenization
        </li>
        <li>
          <span class="tbs-swatch tbs-swatch--split" />
          this split
        </li>
      </ul>

      <svg
        class="tbs-svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Two tokenizations reach grammar state 9; the chart shows probabilities for its six allowed next tokens."
      >
        <text class="tbs-row" x={0} y={TAPE_Y[0] + 15}>
          natural
        </text>
        <text class="tbs-row" x={0} y={TAPE_Y[1] + 15}>
          this split
        </text>
        <Tape pieces={NATURAL.pieces} y={TAPE_Y[0]} kind="natural" />
        {BOUNDARIES.map(b => (
          <g data-tbs-boundary={b.id} data-on={b.id === b0.id ? 'true' : 'false'}>
            <Tape pieces={b.pieces} y={TAPE_Y[1]} kind="split" />
          </g>
        ))}
        <rect class="tbs-node" x={NODE.x} y={NODE.y} width={NODE.s} height={NODE.s} />
        <text class="tbs-node-label" x={NODE.x + NODE.s / 2} y={NODE.y + 15} text-anchor="middle">
          9
        </text>
        <text class="tbs-note" x={NODE.x + NODE.s + 8} y={NODE.y + 9}>
          same grammar state:
        </text>
        <text class="tbs-note" x={NODE.x + NODE.s + 8} y={NODE.y + 21}>
          Paul or John next
        </text>

        {[0, 0.2, 0.4, 0.6, 0.8].map(v => (
          <g>
            <line class="tbs-grid" x1={PLOT.x} x2={PLOT.x + PLOT.w} y1={yOf(v)} y2={yOf(v)} />
            <text class="tbs-axis" x={PLOT.x - 6} y={yOf(v) + 3} text-anchor="end">
              {v.toFixed(1)}
            </text>
          </g>
        ))}
        <text class="tbs-axis" x={PLOT.x} y={PLOT.y - 10}>
          probability after the mask
        </text>
        {TOKENS.map((t, i) => {
          const cx = groupX(i)
          const pn = NATURAL.probs[t.index]
          return (
            <g>
              <rect
                class="tbs-bar tbs-bar--natural"
                x={cx - BAR_W - 2}
                y={yOf(pn)}
                width={BAR_W}
                height={PLOT.y + PLOT.h - yOf(pn)}
              />
              {pn >= 0.05 ? (
                <text class="tbs-value" x={cx - BAR_W / 2 - 2} y={yOf(pn) - 4} text-anchor="middle">
                  {pct(pn)}
                </text>
              ) : null}
              {BOUNDARIES.map(b => {
                const p = b.probs[t.index]
                return (
                  <g data-tbs-boundary={b.id} data-on={b.id === b0.id ? 'true' : 'false'}>
                    <rect
                      class="tbs-bar tbs-bar--split"
                      x={cx + 2}
                      y={yOf(p)}
                      width={BAR_W}
                      height={PLOT.y + PLOT.h - yOf(p)}
                      data-tbs-bar={t.text}
                      data-p={p.toFixed(6)}
                    />
                    {p >= 0.05 ? (
                      <text
                        class="tbs-value"
                        x={cx + BAR_W / 2 + 2}
                        y={yOf(p) - 4}
                        text-anchor="middle"
                      >
                        {pct(p)}
                      </text>
                    ) : null}
                  </g>
                )
              })}
              <text class="tbs-tick" x={cx} y={PLOT.y + PLOT.h + 14} text-anchor="middle">
                {t.text}
              </text>
            </g>
          )
        })}
        <line
          class="tbs-axis-line"
          x1={PLOT.x}
          x2={PLOT.x + PLOT.w}
          y1={PLOT.y + PLOT.h}
          y2={PLOT.y + PLOT.h}
        />
        <text class="tbs-branch" x={groupX(1)} y={PLOT.y + PLOT.h + 32} text-anchor="middle">
          Paul branch
        </text>
        <text class="tbs-branch" x={groupX(4)} y={PLOT.y + PLOT.h + 32} text-anchor="middle">
          John branch
        </text>
      </svg>

      <div class="tbs-tables">
        <table class="tbs-table">
          <thead>
            <tr>
              <th scope="col">token</th>
              <th scope="col">natural</th>
              <th scope="col">this split</th>
            </tr>
          </thead>
          <tbody>
            {TOKENS.map((t, i) => (
              <tr>
                <th scope="row">{t.text}</th>
                <td>{pct(NATURAL.probs[t.index])}</td>
                <td data-tbs-token={String(i)}>{r0.tokens[i]}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <dl class="tbs-readout">
          <div>
            <dt>Forced tokens</dt>
            <dd data-tbs-out="forced">{r0.forced}</dd>
          </div>
          <div>
            <dt>Mass the mask keeps</dt>
            <dd data-tbs-out="mass">{r0.mass}</dd>
          </div>
          <div>
            <dt>Paul branch after mask</dt>
            <dd data-tbs-out="paul">{r0.paul}</dd>
          </div>
          <div>
            <dt>
              <MathText
                text="$D_{\mathrm{KL}}(p_{\text{nat}}\,\|\,p_{\text{split}})$"
                mathClass="tbs-math"
              />
            </dt>
            <dd data-tbs-out="kl">{r0.kl}</dd>
          </div>
          <div>
            <dt>
              <MathText
                text="$\mathrm{TV}=\tfrac12\sum|p_{\text{nat}}-p_{\text{split}}|$"
                mathClass="tbs-math"
              />
            </dt>
            <dd data-tbs-out="tv">{r0.tv}</dd>
          </div>
        </dl>
      </div>

      {caption ? (
        <figcaption class="tbs-caption">
          <MathText text={caption} mathClass="tbs-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const TokenBoundaryShiftComponent = TokenBoundaryShiftImpl as QuartzMdxComponent<Props>
TokenBoundaryShiftComponent.css = style
TokenBoundaryShiftComponent.afterDOMLoaded = script

export const TokenBoundaryShift = registerMdxComponent(
  'TokenBoundaryShift',
  TokenBoundaryShiftComponent,
)

export default (() => TokenBoundaryShift) satisfies (opts: undefined) => QuartzMdxComponent<Props>
