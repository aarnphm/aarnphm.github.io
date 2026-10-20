import { renderInlineMath, MathText } from '../../util/math-text'
import {
  E2M1,
  E2M3,
  E3M2,
  E4M3,
  emax,
  floorLog2,
  levels,
  maxFinite,
  quantizeMXBlock,
  totalBits,
  type FloatFormat,
} from '../../util/minifloat'
//@ts-ignore
import script from '../scripts/mx-block-scale.inline'
import style from '../styles/mxBlockScale.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; format?: string; block?: string }

const BLOCK = 32
const SCALE_BITS = 8

const FORMATS: readonly { id: string; label: string; fmt: FloatFormat }[] = [
  { id: 'e2m1', label: 'MXFP4 E2M1', fmt: E2M1 },
  { id: 'e2m3', label: 'MXFP6 E2M3', fmt: E2M3 },
  { id: 'e3m2', label: 'MXFP6 E3M2', fmt: E3M2 },
  { id: 'e4m3', label: 'MXFP8 E4M3', fmt: E4M3 },
]

// Synthetic block: 32 standard normal draws from a fixed seed. Two variants overwrite one element,
// once with the note's example maximum of 7 and once with an outlier of 24.
const SEED = 11
const SPIKE_AT = 11
const mulberry32 = (seed: number) => () => {
  seed = (seed + 0x6d2b79f5) | 0
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}
const gaussian = (n: number, seed: number) => {
  const rand = mulberry32(seed)
  return Array.from({ length: n }, () => {
    const u = 1 - rand()
    const v = rand()
    // Two decimals keep the inputs readable; the format is applied to these values exactly.
    return Math.round(Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * 100) / 100
  })
}
const BASE = gaussian(BLOCK, SEED)
const withSpike = (v: number) => BASE.map((x, i) => (i === SPIKE_AT ? v : x))

const BLOCKS: readonly { id: string; label: string; values: number[] }[] = [
  { id: 'normal', label: 'Normal draws', values: BASE },
  { id: 'max7', label: 'One value 7', values: withSpike(7) },
  { id: 'spike', label: 'One value 24', values: withSpike(24) },
]

const VIEW_W = 560
const X0 = 58
const PITCH = 15
const BAR_W = 11
const FILL_W = 5
const TOP = 22
const VAL_H = 188
const ZERO_Y = TOP + VAL_H / 2
const ERR_TOP = TOP + VAL_H + 30
const ERR_H = 64
const ERR_ZERO = ERR_TOP + ERR_H / 2
const AXIS_X = X0 - 6
const VIEW_H = ERR_TOP + ERR_H + 22
const colX = (i: number) => X0 + i * PITCH + (PITCH - BAR_W) / 2

const bar = (v: number, zero: number, unit: number) => {
  const h = Math.abs(v) * unit
  return { y: v >= 0 ? zero - h : zero, h }
}
const r2 = (n: number) => Math.round(n * 100) / 100

const fmtNum = (x: number) => {
  if (x === 0) return '0'
  const a = Math.abs(x)
  return a >= 0.01 && a < 1000 ? String(Number(x.toPrecision(4))) : x.toExponential(2)
}

type Geometry = { orig: number[][]; deq: number[][]; err: number[][]; ticks: string; max: number }
type Readout = Record<string, string>
type State = Geometry & { errLabel: string; valLabel: string; out: Readout }

const STATES: Record<string, State> = {}
for (const b of BLOCKS) {
  const results = FORMATS.map(f => quantizeMXBlock(b.values, f.fmt))
  // One value axis and one error axis per block, so switching formats compares honestly.
  const vmax = Math.max(
    ...b.values.map(Math.abs),
    ...results.flatMap(r => r.dequantized.map(Math.abs)),
  )
  const emaxErr = Math.max(
    ...results.flatMap(r => r.dequantized.map((d, i) => Math.abs(d - b.values[i]))),
  )
  const unit = VAL_H / 2 / vmax
  const errUnit = ERR_H / 2 / emaxErr
  const amax = Math.max(...b.values.map(Math.abs))
  const maxIndex = b.values.findIndex(v => Math.abs(v) === amax)
  FORMATS.forEach((f, k) => {
    const r = results[k]
    const ticks = levels(f.fmt)
      .map(l => l * r.scale)
      .filter(l => l <= vmax)
      .flatMap(l => (l === 0 ? [0] : [l, -l]))
      .map(l => `M${AXIS_X - 4} ${r2(ZERO_Y - l * unit)}H${AXIS_X}`)
      .join('')
    const flushed = b.values.filter((v, i) => v !== 0 && r.dequantized[i] === 0).length
    const sq = (a: number[]) => a.reduce((s, x) => s + x * x, 0)
    const relRms = Math.sqrt(sq(r.dequantized.map((d, i) => d - b.values[i])) / sq(b.values))
    const e = emax(f.fmt)
    const fl = floorLog2(amax)
    STATES[`${b.id}:${f.id}`] = {
      orig: b.values.map(v => bar(v, ZERO_Y, unit)).map(g => [r2(g.y), r2(g.h)]),
      deq: r.dequantized.map(v => bar(v, ZERO_Y, unit)).map(g => [r2(g.y), r2(g.h)]),
      err: r.dequantized
        .map((d, i) => bar(d - b.values[i], ERR_ZERO, errUnit))
        .map(g => [r2(g.y), r2(g.h)]),
      ticks,
      max: maxIndex,
      errLabel: `±${fmtNum(emaxErr)}`,
      valLabel: `±${fmtNum(vmax)}`,
      out: {
        amax: renderInlineMath(`\\max_i|V_i| = ${fmtNum(amax)}`),
        shared: renderInlineMath(
          `\\lfloor\\log_2 ${fmtNum(amax)}\\rfloor - e_{\\max} = ${fl} - ${e} = ${r.shared}`,
        ),
        scale: renderInlineMath(`2^{${r.shared}} = ${fmtNum(r.scale)}`),
        code: renderInlineMath(`${r.shared} + 127 = ${r.shared + 127}`),
        top: renderInlineMath(
          `${fmtNum(maxFinite(f.fmt))}\\,S = ${fmtNum(maxFinite(f.fmt) * r.scale)}`,
        ),
        flushed: renderInlineMath(`${flushed}/${BLOCK}`),
        clipped: renderInlineMath(`${r.clipped}/${BLOCK}`),
        rms: renderInlineMath(
          `\\lVert\\hat V - V\\rVert/\\lVert V\\rVert = ${(relRms * 100).toFixed(1)}\\%`,
        ),
        bits: renderInlineMath(
          `(${BLOCK}\\cdot${totalBits(f.fmt)}+${SCALE_BITS})/${BLOCK} = ${(BLOCK * totalBits(f.fmt) + SCALE_BITS) / BLOCK}`,
        ),
      },
    }
  })
}

const READOUT: readonly [string, string][] = [
  ['amax', 'Block maximum'],
  ['shared', 'Shared exponent'],
  ['scale', 'Scale $S$'],
  ['code', 'E8M0 code'],
  ['top', 'Largest level'],
  ['bits', 'Bits per element'],
  ['flushed', 'Nonzero values flushed to zero'],
  ['clipped', 'Values clamped to the largest level'],
  ['rms', 'Relative RMS error'],
]

let instance = 0

const MXBlockScaleImpl: QuartzMdxComponent<Props> = ({
  caption,
  format = 'e2m1',
  block = 'max7',
}) => {
  const id = `mxb-${++instance}`
  const f0 = FORMATS.some(f => f.id === format) ? format : 'e2m1'
  const b0 = BLOCKS.some(b => b.id === block) ? block : 'max7'
  const s0 = STATES[`${b0}:${f0}`]

  return (
    <figure class="mx-block-scale" data-mx-block-scale data-mxb-states={JSON.stringify(STATES)}>
      <div class="mxb-controls">
        <fieldset class="mxb-segments">
          <legend class="mxb-legend">Element format</legend>
          {FORMATS.map(f => (
            <label class="mxb-segment">
              <input
                type="radio"
                name={`${id}-format`}
                value={f.id}
                checked={f.id === f0}
                data-mxb-format-input
              />
              <span>{f.label}</span>
            </label>
          ))}
        </fieldset>
        <fieldset class="mxb-segments">
          <legend class="mxb-legend">Block</legend>
          {BLOCKS.map(b => (
            <label class="mxb-segment">
              <input
                type="radio"
                name={`${id}-block`}
                value={b.id}
                checked={b.id === b0}
                data-mxb-block-input
              />
              <span>{b.label}</span>
            </label>
          ))}
        </fieldset>
      </div>

      <svg
        class="mxb-svg"
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        role="img"
        aria-label={`One MX block of ${BLOCK} values sharing a power-of-two scale: inputs, dequantized values and per-element error.`}
      >
        <text class="mxb-axis" x={AXIS_X - 8} y={ZERO_Y + 3} text-anchor="end">
          0
        </text>
        <text class="mxb-axis" x={AXIS_X - 8} y={ERR_ZERO + 3} text-anchor="end">
          0
        </text>
        <text class="mxb-panel" x={AXIS_X} y={TOP - 9}>
          value
        </text>
        <text class="mxb-panel" x={AXIS_X} y={ERR_TOP - 9}>
          error
        </text>
        <text class="mxb-axis" x={AXIS_X - 8} y={TOP + 7} text-anchor="end" data-mxb-valmax>
          {s0.valLabel}
        </text>
        <text class="mxb-axis" x={AXIS_X - 8} y={ERR_TOP + 7} text-anchor="end" data-mxb-errmax>
          {s0.errLabel}
        </text>
        <path class="mxb-ticks" d={s0.ticks} data-mxb-ticks />
        <line class="mxb-rule" x1={AXIS_X} y1={TOP} x2={AXIS_X} y2={TOP + VAL_H} />
        <line class="mxb-zero" x1={AXIS_X} y1={ZERO_Y} x2={X0 + BLOCK * PITCH} y2={ZERO_Y} />
        <line class="mxb-rule" x1={AXIS_X} y1={ERR_TOP} x2={AXIS_X} y2={ERR_TOP + ERR_H} />
        <line class="mxb-zero" x1={AXIS_X} y1={ERR_ZERO} x2={X0 + BLOCK * PITCH} y2={ERR_ZERO} />
        {Array.from({ length: BLOCK }, (_, i) => {
          const x = colX(i)
          return (
            <g data-mxb-col={String(i)} data-mxb-max={i === s0.max ? 'true' : 'false'}>
              <rect class="mxb-orig" x={x} y={s0.orig[i][0]} width={BAR_W} height={s0.orig[i][1]} />
              <rect
                class="mxb-deq"
                x={x + (BAR_W - FILL_W) / 2}
                y={s0.deq[i][0]}
                width={FILL_W}
                height={s0.deq[i][1]}
              />
              <rect
                class="mxb-err"
                x={x + 2}
                y={s0.err[i][0]}
                width={BAR_W - 4}
                height={s0.err[i][1]}
              />
              {i % 8 === 0 || i === BLOCK - 1 ? (
                <text class="mxb-axis" x={x + BAR_W / 2} y={VIEW_H - 6} text-anchor="middle">
                  {String(i)}
                </text>
              ) : null}
            </g>
          )
        })}
      </svg>

      <ul class="mxb-key">
        <li>
          <span class="mxb-swatch mxb-swatch--orig" />
          input <MathText text="$V_i$" mathClass="mxb-math" />
        </li>
        <li>
          <span class="mxb-swatch mxb-swatch--deq" />
          <MathText text="$\hat V_i = S P_i$" mathClass="mxb-math" />
        </li>
        <li>
          <span class="mxb-swatch mxb-swatch--max" />
          block maximum, which sets <MathText text="$S$" mathClass="mxb-math" />
        </li>
        <li>
          <span class="mxb-swatch mxb-swatch--tick" />
          levels of <MathText text="$S P$" mathClass="mxb-math" />
        </li>
      </ul>

      <dl class="mxb-readout">
        {READOUT.map(([key, label]) => (
          <div>
            <dt>
              <MathText text={label} mathClass="mxb-math" />
            </dt>
            <dd data-mxb-out={key} dangerouslySetInnerHTML={{ __html: s0.out[key] }} />
          </div>
        ))}
      </dl>

      {caption ? (
        <figcaption class="mxb-caption">
          <MathText text={caption} mathClass="mxb-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const MXBlockScaleComponent = MXBlockScaleImpl as QuartzMdxComponent<Props>
MXBlockScaleComponent.css = style
MXBlockScaleComponent.afterDOMLoaded = script

export const MXBlockScale = registerMdxComponent('MXBlockScale', MXBlockScaleComponent)

export default (() => MXBlockScale) satisfies (opts: undefined) => QuartzMdxComponent<Props>
