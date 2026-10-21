import type { ActivityAnalysisRange, StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriathlonPresentation } from '../triathlon-presentation'
import type { TriNodeFactory } from './construction-contracts'
import { RUN_STRIDE_BIN_M, RUN_STRIDE_CADENCE_BIN_SPM, type RunStrideCell } from '../run-mechanics'
import { triText } from '../triathlon-i18n'
import { activityTraceUsesElapsedAxis, runStrideLengthValue } from './activity-metric-data'
import { axisFrame, type ActivityGraphDomain, type AxisYTick } from './chart-geometry'
import { clock, formatStrideLength, isImperial, zoneClock } from './metric-text'
import { buildRunStrideTrace } from './run-mechanics-traces'

// Plot range keeps this share of moving time, so a stray GPS sample cannot stretch the axes.
const RUN_STRIDE_TIME_SHARE = 0.95

const runPaceText = (presentation: TriathlonPresentation, speedMps: number): string => {
  const imperial = isImperial(presentation)
  return `${clock((imperial ? 1_609.344 : 1_000) / speedMps)} ${imperial ? '/mi' : '/km'}`
}

const signedPercent = (fraction: number): string =>
  `${fraction < 0 ? '−' : '+'}${Math.abs(fraction * 100).toFixed(1)}%`

const runStrideAxisRange = (
  cells: readonly RunStrideCell[],
  pick: (cell: RunStrideCell) => number,
  bin: number,
): [number, number] => {
  const sorted = [...cells].sort((a, b) => pick(a) - pick(b))
  const tail =
    (sorted.reduce((total, cell) => total + cell.seconds, 0) * (1 - RUN_STRIDE_TIME_SHARE)) / 2
  let seconds = 0
  let low = pick(sorted[0])
  for (const cell of sorted) {
    seconds += cell.seconds
    if (seconds > tail) {
      low = pick(cell)
      break
    }
  }
  seconds = 0
  let high = pick(sorted[sorted.length - 1]) + bin
  for (let index = sorted.length - 1; index >= 0; index--) {
    seconds += sorted[index].seconds
    if (seconds > tail) {
      high = pick(sorted[index]) + bin
      break
    }
  }
  return [low, high]
}

/** Late-third change against the first third, split exactly into stride and cadence (log terms). */
export const runStrideShift = (
  trace: Pick<NonNullable<StravaActivityDetail['runStrideCadence']>, 'early' | 'late'>,
): { stride: number; cadence: number; speed: number } | null => {
  const { early, late } = trace
  if (!early || !late) return null
  const stride = Math.log(late.strideM / early.strideM)
  const cadence = Math.log(late.cadenceSpm / early.cadenceSpm)
  return { stride, cadence, speed: stride + cadence }
}

export const buildRunStrideCadenceChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  if (d.sport !== 'run') return null
  const text = (key: string): string => triText(f.presentation.locale, key)
  const trace = d.runStrideCadence
  const hasDensity = trace != null && trace.cells.length > 0
  const hasDistance = d.route.filter(point => runStrideLengthValue(d, point) != null).length >= 2
  if (!hasDensity) return hasDistance ? buildRunStrideTrace(f, d, selection, graphDomain) : null
  const mode = 'cadence'
  const controls = (selected: 'distance' | 'cadence'): N[] => {
    if (!hasDistance) return []
    const group = f.el('div', 'tri-chart-controls tri-torque-controls', undefined, {
      role: 'group',
      'aria-label': text('stride view'),
    })
    for (const option of ['distance', 'cadence'])
      f.add(
        group,
        f.el(
          'button',
          'tri-torque-mode tri-curve-range',
          text(option === 'distance' && activityTraceUsesElapsedAxis(d) ? 'time' : option),
          {
            type: 'button',
            'data-torque-mode': option,
            'aria-pressed': String(option === selected),
          },
        ),
      )
    return [group]
  }
  const panel = f.el('section', 'tri-zone tri-torque-panel tri-run-stride-panel', undefined, {
    'data-torque-mode': mode,
  })
  const distance = hasDistance
    ? buildRunStrideTrace(f, d, selection, graphDomain, {
        capExtra: controls('distance'),
        wrapAttrs: { 'data-torque-pane': 'distance', hidden: '' },
      })
    : null
  const cells = trace.cells
  const [cadenceLow, cadenceHigh] = runStrideAxisRange(
    cells,
    cell => cell.cadenceSpm,
    RUN_STRIDE_CADENCE_BIN_SPM,
  )
  const [strideLow, strideHigh] = runStrideAxisRange(cells, cell => cell.strideM, RUN_STRIDE_BIN_M)
  const xMin = Math.floor(cadenceLow / 10) * 10
  const xMax = Math.max(xMin + 10, Math.ceil(cadenceHigh / 10) * 10)
  const yMin = Math.floor(strideLow * 10) / 10
  const yMax = Math.max(yMin + 0.1, Math.round(Math.ceil(strideHigh * 10)) / 10)
  // Log axes turn each iso-pace curve (stride = 60 v / cadence) into a straight line.
  const lx = (spm: number): number => (Math.log(spm / xMin) / Math.log(xMax / xMin)) * 100
  const ly = (meters: number): number => 30 - (Math.log(meters / yMin) / Math.log(yMax / yMin)) * 30
  const visible = cells.filter(
    cell =>
      cell.cadenceSpm >= xMin &&
      cell.cadenceSpm + RUN_STRIDE_CADENCE_BIN_SPM <= xMax &&
      cell.strideM >= yMin - 1e-9 &&
      cell.strideM + RUN_STRIDE_BIN_M <= yMax + 1e-9,
  )
  const maxSeconds = Math.max(...visible.map(cell => cell.seconds), 1)
  const graph = f.svg('svg', {
    class: 'tri-torque-density tri-run-stride-density',
    viewBox: '0 0 100 30',
    preserveAspectRatio: 'none',
    tabindex: '0',
    role: 'slider',
    'aria-label': text('stride by cadence'),
    'aria-valuemin': 0,
    'aria-valuemax': Math.max(0, visible.length - 1),
    'aria-valuenow': 0,
    'aria-valuetext': text('time in cadence and stride bins'),
  })
  const unitM = isImperial(f.presentation) ? 1_609.344 : 1_000
  // Only lines that reach the right axis carry a label, so the pace range comes from that edge.
  const slowestS = unitM / ((xMax * yMin) / 60)
  const fastestS = unitM / ((xMax * yMax) / 60)
  const paceStepS = [15, 30, 60, 120, 300].find(step => (slowestS - fastestS) / step <= 5) ?? 600
  const paceTicks: AxisYTick[] = []
  for (
    let paceS = Math.ceil(fastestS / paceStepS) * paceStepS;
    paceS <= slowestS;
    paceS += paceStepS
  ) {
    const speedMps = unitM / paceS
    // The line enters where its stride falls to yMax (or at xMin) and ends on the right axis.
    const fromSpm = Math.max(xMin, (60 * speedMps) / yMax)
    const rightStride = (60 * speedMps) / xMax
    f.add(
      graph,
      f.svg('line', {
        class: 'tri-run-stride-pace',
        x1: lx(fromSpm).toFixed(3),
        y1: ly((60 * speedMps) / fromSpm).toFixed(3),
        x2: 100,
        y2: ly(rightStride).toFixed(3),
      }),
    )
    paceTicks.push({ label: runPaceText(f.presentation, speedMps), vbY: ly(rightStride) })
  }
  for (const cell of visible) {
    const centerSpeed =
      ((cell.cadenceSpm + RUN_STRIDE_CADENCE_BIN_SPM / 2) * (cell.strideM + RUN_STRIDE_BIN_M / 2)) /
      60
    const label = `${cell.cadenceSpm}–${cell.cadenceSpm + RUN_STRIDE_CADENCE_BIN_SPM} spm · ${formatStrideLength(f.presentation, cell.strideM)}–${formatStrideLength(f.presentation, cell.strideM + RUN_STRIDE_BIN_M)} · ${runPaceText(f.presentation, centerSpeed)} · ${zoneClock(cell.seconds)}`
    f.add(
      graph,
      f.svg('rect', {
        class: 'tri-torque-cell',
        x: lx(cell.cadenceSpm).toFixed(3),
        y: ly(cell.strideM + RUN_STRIDE_BIN_M).toFixed(3),
        width: (lx(cell.cadenceSpm + RUN_STRIDE_CADENCE_BIN_SPM) - lx(cell.cadenceSpm)).toFixed(3),
        height: (ly(cell.strideM) - ly(cell.strideM + RUN_STRIDE_BIN_M)).toFixed(3),
        opacity: (0.15 + 0.85 * Math.sqrt(cell.seconds / maxSeconds)).toFixed(3),
        'data-torque-cell': label,
      }),
    )
  }
  const shift = runStrideShift(trace)
  const legend = f.el('div', 'tri-run-stride-legend')
  if (shift && trace.early && trace.late) {
    const point = (centroid: { cadenceSpm: number; strideM: number }): string =>
      `${lx(centroid.cadenceSpm).toFixed(3)} ${ly(centroid.strideM).toFixed(3)}`
    f.add(
      graph,
      f.svg('path', {
        class: 'tri-run-stride-shift',
        d: `M ${point(trace.early)} L ${point(trace.late)}`,
      }),
      f.svg('path', {
        class: 'tri-run-stride-centroid tri-run-stride-centroid--early',
        d: `M ${point(trace.early)} L ${point(trace.early)}`,
      }),
      f.svg('path', {
        class: 'tri-run-stride-centroid tri-run-stride-centroid--late',
        d: `M ${point(trace.late)} L ${point(trace.late)}`,
      }),
    )
    f.add(
      legend,
      f.el('span', 'tri-run-stride-key tri-run-stride-key--early', text('first third')),
      f.el('span', 'tri-run-stride-key tri-run-stride-key--late', text('last third')),
      f.el(
        'span',
        'tri-run-stride-delta',
        `${text('stride')} ${signedPercent(shift.stride)} · ${text('cadence')} ${signedPercent(shift.cadence)} · ${text('speed')} ${signedPercent(shift.speed)}`,
        {
          'data-stride-shift': shift.stride.toFixed(4),
          'data-cadence-shift': shift.cadence.toFixed(4),
        },
      ),
    )
  }
  f.add(legend, f.el('span', 'tri-run-stride-key tri-run-stride-key--pace', text('iso-pace')))
  const cadence = f.el(
    'div',
    'tri-elev-wrap tri-torque-cadence tri-run-stride-cadence',
    undefined,
    { 'data-torque-pane': 'cadence' },
  )
  const caption = f.el('div', 'tri-elev-cap')
  f.add(
    caption,
    f.el('span', 'tri-elev-d', text('stride by cadence'), {
      'data-gloss': '',
      'data-gloss-def': text('stride by cadence definition'),
      tabindex: '0',
    }),
    f.el('span', 'tri-elev-range', `${text('darker means more time')} · ${text('whole run')}`),
    ...controls('cadence'),
  )
  const foot = f.el('div', 'tri-run-stride-foot')
  f.add(foot, legend, f.el('div', 'tri-torque-readout', text('time in cadence and stride bins')))
  const xTicks = [xMin, Math.round(Math.sqrt(xMin * xMax) / 5) * 5, xMax]
  f.add(
    cadence,
    caption,
    axisFrame(
      f,
      graph,
      [yMin, Math.sqrt(yMin * yMax), yMax].map(value => ({
        label: formatStrideLength(f.presentation, value),
        vbY: ly(value),
      })),
      30,
      xTicks.map(value => ({ label: `${value} spm`, pct: lx(value) })),
      true,
      undefined,
      [],
      paceTicks,
    ),
    foot,
  )
  if (distance) f.add(panel, distance)
  f.add(panel, cadence)
  return panel
}
