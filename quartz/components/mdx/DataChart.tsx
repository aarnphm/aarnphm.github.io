import { MathText } from '../../util/math-text'
//@ts-ignore
import script from '../scripts/data-chart.inline'
import style from '../styles/dataChart.scss'
import {
  BAR_GAP,
  BAR_MAX,
  MAX_SLOTS,
  SAMPLES,
  VIEW_H,
  VIEW_W,
  X0,
  X1,
  Y0,
  Y_TOP,
  evalTerms,
  expTex,
  fwd,
  json,
  niceTicks,
  num,
  r1,
  siTick,
  trim,
  type Json,
  type MarkSpec,
  type Position,
  type ProbeSpec,
  type Props,
  type Pt,
  type Scale,
  type Series,
  type SeriesSpec,
  type XScale,
} from './data-chart/model'
import { Marker, Readouts, Swatch } from './data-chart/parts'
import { probeStatesFor } from './data-chart/probe'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'
let instance = 0

const DataChartImpl: QuartzMdxComponent<Props> = props => {
  const id = `dch-${++instance}`
  const mode = props.mode === 'line' ? 'line' : 'bar'
  const yscale: Scale = props.yscale === 'log' ? 'log' : 'linear'
  const xscale: XScale = mode === 'bar' ? 'linear' : (props.xscale ?? 'linear')
  const decimals = num(props.decimals) ?? 0
  const unit = props.unit ? ` ${props.unit}` : ''
  const fmt = (v: number) => v.toFixed(decimals)
  const xs = json('x', props.x, [] as (string | number)[])
  const specs = json('series', props.series, [] as SeriesSpec[])
  if (specs.length > MAX_SLOTS)
    throw new Error(`DataChart: at most ${MAX_SLOTS} series; split the rest into another chart`)
  const marks = json('marks', props.marks, [] as MarkSpec[])
  const probe = props.probe ? json<ProbeSpec | null>('probe', props.probe, null) : null

  // Domains: explicit props win; otherwise the data extent, padded in scale space.
  const dataX = specs.flatMap(s =>
    s.points ? s.points.map(p => p[0]) : s.terms ? [s.from ?? 0, s.to ?? 1] : xs.map(Number),
  )
  const xminP = num(props.xmin)
  const xmaxP = num(props.xmax)
  const tx0 = xminP !== undefined ? fwd(xscale, xminP) : Math.min(...dataX.map(v => fwd(xscale, v)))
  const tx1 = xmaxP !== undefined ? fwd(xscale, xmaxP) : Math.max(...dataX.map(v => fwd(xscale, v)))
  const xpad = xminP !== undefined && xmaxP !== undefined ? 0 : (tx1 - tx0) * 0.05
  const [dx0, dx1] = [
    tx0 - (xminP !== undefined ? 0 : xpad),
    tx1 + (xmaxP !== undefined ? 0 : xpad),
  ]

  const sampleTerms = (s: SeriesSpec): Pt[] => {
    const from = s.from ?? 0
    const to = s.to ?? 1
    return Array.from({ length: SAMPLES + 1 }, (_, i) => {
      const x = from + ((to - from) * i) / SAMPLES
      return { x, y: evalTerms(s.terms!, x) }
    })
  }
  const series: Series[] = specs.map((s, slot) => ({
    name: s.name,
    slot,
    spec: s,
    analytic: Boolean(s.terms),
    pts: s.terms
      ? sampleTerms(s)
      : s.points
        ? s.points.map(([x, y]) => ({ x, y }))
        : (s.values ?? []).flatMap((y, i) =>
            y === null || y === undefined
              ? []
              : [{ x: mode === 'bar' ? i : Number(xs[i]), y: Number(y) }],
          ),
  }))

  const ys = series.flatMap(s => s.pts.map(p => p.y))
  const yminP = num(props.ymin)
  const ymaxP = num(props.ymax)
  const ymin =
    yminP ?? (yscale === 'log' ? 10 ** Math.floor(Math.log10(Math.min(...ys))) : Math.min(0, ...ys))
  const ymax =
    ymaxP ??
    (yscale === 'log'
      ? 10 ** Math.ceil(Math.log10(Math.max(...ys)))
      : (niceTicks(ymin, Math.max(...ys) * 1.08).at(-1) ?? 1))
  const ty0 = fwd(yscale, ymin)
  const ty1 = fwd(yscale, ymax)
  const yPx = (v: number) => r1(Y0 - ((fwd(yscale, v) - ty0) / (ty1 - ty0)) * (Y0 - Y_TOP))

  const n = xs.length
  const band = mode === 'bar' ? (X1 - X0) / Math.max(1, n) : 0
  const xPx = (v: number) =>
    mode === 'bar'
      ? r1(X0 + band * (v + 0.5))
      : r1(X0 + ((fwd(xscale, v) - dx0) / (dx1 - dx0)) * (X1 - X0))

  const yTicks = json(
    'yticks',
    props.yticks,
    yscale === 'log'
      ? Array.from(
          { length: Math.floor(ty1) - Math.ceil(ty0) + 1 },
          (_, i) => 10 ** (Math.ceil(ty0) + i),
        )
      : niceTicks(ymin, ymax),
  )
  const xTicks: number[] =
    mode === 'bar'
      ? []
      : json(
          'xticks',
          props.xticks,
          xscale === 'linear'
            ? niceTicks(dx0, dx1)
            : (() => {
                const k0 = Math.ceil(dx0)
                const k1 = Math.floor(dx1)
                const step = Math.max(1, Math.ceil((k1 - k0 + 1) / 7))
                const base = xscale === 'log2' ? 2 : 10
                const out: number[] = []
                for (let k = k0; k <= k1; k += step) out.push(base ** k)
                return out
              })(),
        )

  const xText = (v: number) => (xscale === 'log2' ? expTex(v).replace(/[{}]/g, '') : trim(v))
  const xTex = (v: number) => (xscale === 'log2' ? expTex(v) : trim(v))

  // One readout state per x position: a bar group, or every distinct x a data series visits.
  const measured = series.filter(s => !s.analytic)
  const positions: Position[] = (() => {
    if (probe) return []
    if (mode === 'bar')
      return xs.map((x, i) => ({
        x: i,
        cx: xPx(i),
        lo: r1(X0 + band * i),
        hi: r1(X0 + band * (i + 1)),
        tex: String(x),
        text: String(x),
        values: measured.map(s => s.pts.find(p => p.x === i)?.y ?? null),
      }))
    const uniq = [...new Set(measured.flatMap(s => s.pts.map(p => p.x)))].sort((a, b) => a - b)
    const cxs = uniq.map(xPx)
    return uniq.map((x, i) => ({
      x,
      cx: cxs[i],
      lo: i === 0 ? X0 : r1((cxs[i - 1] + cxs[i]) / 2),
      hi: i === uniq.length - 1 ? X1 : r1((cxs[i] + cxs[i + 1]) / 2),
      tex: `$${xTex(x)}$`,
      text: xText(x),
      values: measured.map(s => s.pts.find(p => p.x === x)?.y ?? null),
    }))
  })()
  const xname = props.xname ?? props.xlabel ?? 'x'
  const ariaFor = (p: Position) =>
    `${xname} ${p.text}: ${measured
      .map((s, j) => `${s.name} ${p.values[j] === null ? 'no data' : fmt(p.values[j]!) + unit}`)
      .join('; ')}`
  const activeP = num(props.active)
  const active = Math.min(
    Math.max(0, activeP ?? positions.length - 1),
    Math.max(0, positions.length - 1),
  )

  const labelSpec = props.labels ?? 'none'
  const labelled = new Set<number>(
    labelSpec === 'all'
      ? positions.map((_, i) => i)
      : labelSpec === 'last'
        ? [positions.length - 1]
        : labelSpec === 'none'
          ? []
          : json('labels', labelSpec as Json<number[]>, []),
  )

  // Labels in one group that would overlap keep only the larger value; the readout has the rest.
  const LABEL_CLEAR = 16
  const shownLabel = (p: Position, j: number) =>
    p.values[j] !== null &&
    p.values.every(
      (v, o) =>
        o === j ||
        v === null ||
        v < p.values[j]! ||
        (v === p.values[j] && o > j) ||
        Math.abs(yPx(v) - yPx(p.values[j]!)) >= LABEL_CLEAR,
    )

  // A label wider than its bar would cross a taller neighbour, so it hangs off the outer edge.
  const labelAt = (p: Position, i: number, j: number) => {
    if (mode === 'line') return { x: r1(p.cx + 7), 'text-anchor': 'start' }
    const taller = p.values.some((v, o) => o !== j && v !== null && v > p.values[j]!)
    if (!taller || k === 1) return { x: r1(barX(i, j) + bw / 2), 'text-anchor': 'middle' }
    return j === 0
      ? { x: r1(barX(i, j) + bw), 'text-anchor': 'end' }
      : { x: barX(i, j), 'text-anchor': 'start' }
  }

  // Bars: square, at most 24 units thick, 2 units of surface between neighbours in a group.
  const k = measured.length
  const bw = Math.min(BAR_MAX, (band * 0.72 - BAR_GAP * (k - 1)) / Math.max(1, k))
  const gw = k * bw + (k - 1) * BAR_GAP
  const barX = (i: number, j: number) => r1(xPx(i) - gw / 2 + j * (bw + BAR_GAP))

  // Probe: a point that walks the analytic series in fixed steps; every step is precomputed.
  const curve = series.find(s => s.analytic)
  const probeStates = probeStatesFor(probe, curve, xPx, yPx, fmt)
  const probeIndex = probe
    ? probeStates.reduce(
        (best, s, i) =>
          Math.abs(s.x - probe.x) < Math.abs(probeStates[best].x - probe.x) ? i : best,
        0,
      )
    : 0
  const p0 = probeStates[probeIndex]

  const pathFor = (pts: Pt[]) =>
    pts
      .filter(p => p.y >= ymin && p.y <= ymax)
      .map((p, i) => `${i === 0 ? 'M' : 'L'}${xPx(p.x)} ${yPx(p.y)}`)
      .join('')

  const showLegend = measured.length >= 2

  return (
    <figure
      class="data-chart"
      data-data-chart
      data-dch-mode={mode}
      data-dch-probe={probe ? JSON.stringify(probeStates.map(s => [s.cx, s.cy])) : undefined}
    >
      {props.label ? <p class="dch-title">{props.label}</p> : null}
      {showLegend ? (
        <ul class="dch-legend" aria-label="series">
          {measured.map(s => (
            <li>
              <Swatch slot={s.slot} mode={mode} />
              <span>{s.name}</span>
            </li>
          ))}
        </ul>
      ) : null}

      <svg
        class="dch-svg"
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        preserveAspectRatio="xMidYMid meet"
        role="group"
        aria-label={props.label ?? `${props.ylabel ?? 'value'} by ${xname}`}
        data-dch-svg
      >
        <g class="dch-grid" aria-hidden="true">
          {yTicks.map(t => (
            <line x1={X0} x2={X1} y1={yPx(t)} y2={yPx(t)} />
          ))}
          {xTicks.map(t => (
            <line x1={xPx(t)} x2={xPx(t)} y1={Y_TOP} y2={Y0} />
          ))}
        </g>
        <g class="dch-tick" text-anchor="end" aria-hidden="true">
          {yTicks.map(t => (
            <text x={X0 - 7} y={yPx(t) + 3.5}>
              {siTick(t)}
            </text>
          ))}
        </g>
        <g class="dch-tick" text-anchor="middle" aria-hidden="true">
          {mode === 'bar'
            ? xs.map((x, i) => (
                <text x={xPx(i)} y={Y0 + 15}>
                  {String(x)}
                </text>
              ))
            : xTicks.map(t =>
                xscale === 'log2' ? (
                  <text x={xPx(t)} y={Y0 + 16}>
                    2
                    <tspan class="dch-sup" dy="-0.5em">
                      {expTex(t).slice(3, -1)}
                    </tspan>
                  </text>
                ) : (
                  <text x={xPx(t)} y={Y0 + 15}>
                    {siTick(t)}
                  </text>
                ),
              )}
        </g>
        {props.ylabel ? (
          <text class="dch-axis" x={4} y={12} aria-hidden="true">
            {props.ylabel}
          </text>
        ) : null}
        {props.xlabel ? (
          <text
            class="dch-axis"
            x={(X0 + X1) / 2}
            y={VIEW_H - 6}
            text-anchor="middle"
            aria-hidden="true"
          >
            {props.xlabel}
          </text>
        ) : null}
        <path class="dch-rule" d={`M${X0} ${Y_TOP}V${Y0}H${X1}`} aria-hidden="true" />

        {positions.length ? (
          <g class="dch-bands" aria-hidden="true">
            {positions.map((p, i) => (
              <g
                class="dch-band"
                data-dch-band={String(i)}
                data-dch-on={i === active ? '' : undefined}
              >
                <rect x={p.lo} y={Y_TOP} width={r1(p.hi - p.lo)} height={Y0 - Y_TOP} />
                {mode === 'line' ? <line x1={p.cx} x2={p.cx} y1={Y_TOP} y2={Y0} /> : null}
              </g>
            ))}
          </g>
        ) : null}

        <g class="dch-marks" aria-hidden="true">
          {mode === 'bar'
            ? measured.map((s, j) =>
                s.pts.map(p => (
                  <rect
                    class={`dch-bar dch-s${s.slot}`}
                    x={barX(p.x, j)}
                    y={yPx(p.y)}
                    width={r1(bw)}
                    height={r1(Y0 - yPx(p.y))}
                  />
                )),
              )
            : series.map(s => (
                <path class={`dch-line dch-l${s.analytic ? 'c' : s.slot}`} d={pathFor(s.pts)} />
              ))}
          {curve ? (
            <text
              class="dch-curve-label"
              x={xPx(curve.pts.at(-1)!.x)}
              y={yPx(curve.pts.at(-1)!.y) - 8}
              text-anchor="end"
            >
              {curve.name}
            </text>
          ) : null}
          {mode === 'line'
            ? measured.map(s => s.pts.map(p => <Marker slot={s.slot} x={xPx(p.x)} y={yPx(p.y)} />))
            : null}
        </g>

        <g class="dch-value" aria-hidden="true">
          {positions.flatMap((p, i) =>
            labelled.has(i)
              ? measured.map((_, j) =>
                  !shownLabel(p, j) ? null : (
                    <text {...labelAt(p, i, j)} y={yPx(p.values[j]!) - 5}>
                      {fmt(p.values[j]!)}
                    </text>
                  ),
                )
              : [],
          )}
        </g>

        {curve
          ? marks.map(m => {
              const y = evalTerms(curve.spec.terms!, m.x)
              const cx = xPx(m.x)
              const cy = yPx(y)
              return (
                <g class="dch-ref" aria-hidden="true">
                  <path d={`M${X0} ${cy}H${cx}V${Y0}`} />
                  <g transform={`translate(${cx} ${cy})`}>
                    <rect class="dch-ref-mark" x={-2.5} y={-2.5} width={5} height={5} />
                  </g>
                  {m.xlabel ? (
                    <foreignObject x={cx - 30} y={Y0 + 2} width={60} height={24}>
                      <div class="dch-fo dch-fo--center">
                        <MathText text={`$${m.xlabel}$`} mathClass="dch-math" />
                      </div>
                    </foreignObject>
                  ) : null}
                  {m.ylabel ? (
                    <foreignObject x={X0 - 44} y={cy - 12} width={38} height={24}>
                      <div class="dch-fo dch-fo--end">
                        <MathText text={`$${m.ylabel}$`} mathClass="dch-math" />
                      </div>
                    </foreignObject>
                  ) : null}
                </g>
              )
            })
          : null}

        {p0 ? (
          <g class="dch-probe" aria-hidden="true">
            <rect
              class="dch-probe-hit"
              x={X0}
              y={Y_TOP}
              width={X1 - X0}
              height={Y0 - Y_TOP}
              data-dch-probe-hit
            />
            <path class="dch-probe-drop" d={`M${X0} ${p0.cy}H${p0.cx}V${Y0}`} data-dch-drop />
            <circle class="dch-probe-dot" cx={p0.cx} cy={p0.cy} r={5} data-dch-dot />
          </g>
        ) : null}

        {positions.length ? (
          <g class="dch-hits" role="listbox" aria-label={xname} data-dch-listbox>
            {positions.map((p, i) => (
              <rect
                class="dch-hit"
                role="option"
                aria-selected={i === active ? 'true' : 'false'}
                aria-label={ariaFor(p)}
                tabindex={i === active ? 0 : -1}
                x={p.lo}
                y={Y_TOP}
                width={r1(p.hi - p.lo)}
                height={Y0 - Y_TOP}
                data-dch-pos={String(i)}
                data-dch-cx={String(p.cx)}
              />
            ))}
          </g>
        ) : null}
      </svg>

      {p0 && probe ? (
        <div class="dch-probe-control">
          <label class="dch-probe-label" for={`${id}-probe`}>
            <MathText text={probe.xname ?? 'x'} mathClass="dch-math" />
          </label>
          <input
            id={`${id}-probe`}
            class="dch-slider"
            type="range"
            min="0"
            max={String(probeStates.length - 1)}
            step="1"
            value={String(probeIndex)}
            aria-valuetext={`${p0.xs}, ${p0.ys}, ${p0.regime}`}
            data-dch-probe-input
            data-dch-valuetext={JSON.stringify(
              probeStates.map(s => `${s.xs}, ${s.ys}, ${s.regime}`),
            )}
          />
        </div>
      ) : null}

      <Readouts
        mode={mode}
        positions={positions}
        active={active}
        xname={xname}
        measured={measured}
        curve={curve}
        marks={marks}
        probe={probe}
        probeStates={probeStates}
        probeIndex={probeIndex}
        fmt={fmt}
        unit={unit}
        unitName={props.unit}
        ratio={props.ratio}
      />

      {props.caption ? (
        <figcaption class="dch-caption">
          <MathText text={props.caption} mathClass="dch-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const DataChartComponent = DataChartImpl as QuartzMdxComponent<Props>
DataChartComponent.css = style
DataChartComponent.afterDOMLoaded = script

export const DataChart = registerMdxComponent('DataChart', DataChartComponent)

export default (() => DataChart) satisfies (opts: undefined) => QuartzMdxComponent<Props>
