import type { ActivityAnalysisRange, StravaActivityDetail } from '../../plugins/stores/strava'
import type { TriNodeFactory } from './construction-contracts'
import { triText } from '../triathlon-i18n'
import { triathlonTraceName } from '../triathlon-trace-settings'
import {
  axisFrame,
  distanceXTicks,
  graphViewForDistance,
  type ActivityGraphDomain,
} from './chart-geometry'
import { buildAnalysisSelection } from './chart-selection'
import { zoneClock } from './metric-text'
import {
  dominantGearPairing,
  gearStepPath,
  gearTeethDurations,
  gearY,
  sampledGearTicks,
  shiftingWattsData,
  shiftingWattsTicks,
} from './shifting-data'
import { buildWattsHeatmap } from './watts-heatmap'

// Every label and value variant is in the markup; the chart's data-shift-* attributes pick one.
const buildShiftingHistogramRow = <N>(
  f: TriNodeFactory<N>,
  labels: { teeth: string; number: string },
  durationS: number,
  peakS: number,
  totalS: number,
  ratio?: number,
): N => {
  const share = totalS > 0 ? (durationS / totalS) * 100 : 0
  const row = f.el('div', 'tri-shift-hist-row', undefined, {
    role: 'listitem',
    title: `${labels.teeth} · ${zoneClock(durationS)} · ${Math.round(share)}%`,
  })
  const label = f.el('span', 'tri-shift-hist-teeth')
  f.add(
    label,
    f.el('span', undefined, labels.teeth, { 'data-shift-label': 'teeth' }),
    f.el('span', undefined, labels.number, { 'data-shift-label': 'number' }),
  )
  f.add(row, label)
  if (ratio != null) f.add(row, f.el('span', 'tri-shift-hist-ratio', ratio.toFixed(2)))
  const track = f.el('span', 'tri-shift-hist-track', undefined, { 'aria-hidden': 'true' })
  f.add(
    track,
    f.el('span', 'tri-shift-hist-bar', undefined, {
      style: `width: ${peakS > 0 ? ((durationS / peakS) * 100).toFixed(2) : '0'}%`,
    }),
  )
  const value = f.el('span', 'tri-shift-hist-time')
  f.add(
    value,
    f.el('span', undefined, zoneClock(durationS), { 'data-shift-value': 'duration' }),
    f.el('span', undefined, `${Math.round(share)}%`, { 'data-shift-value': 'percent' }),
  )
  f.add(row, track, value)
  return row
}

const buildShiftingHistogram = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  graphDomain?: ActivityGraphDomain | null,
): N => {
  const { front, rear, combos, totalS } = gearTeethDurations(d, graphDomain)
  const hist = f.el('div', 'tri-shift-hist', undefined, {
    'data-shift-pane': 'hist',
    style: `--tri-shift-hist-rows: ${Math.max(front.length, rear.length, 1)}; --tri-shift-hist-combo-rows: ${Math.max(combos.length, 1)}`,
  })
  for (const [kind, gears] of [
    ['front', front],
    ['rear', rear],
  ] as const) {
    const group = f.el('div', `tri-shift-hist-group tri-shift-hist-group--${kind}`, undefined, {
      role: 'list',
      'aria-label': triText(f.presentation.locale, kind),
      'data-i18n-aria-label': kind,
    })
    // Bars scale to the busiest gear in the group; the percentage carries the share of time.
    const peakS = Math.max(...gears.map(gear => gear.durationS), 0)
    for (const gear of gears)
      f.add(
        group,
        buildShiftingHistogramRow(
          f,
          { teeth: `${gear.teeth}T`, number: `${gear.gearNum}` },
          gear.durationS,
          peakS,
          totalS,
        ),
      )
    f.add(hist, group)
  }
  const comboGroup = f.el('div', 'tri-shift-hist-group tri-shift-hist-group--combos', undefined, {
    role: 'list',
    'aria-label': triText(f.presentation.locale, 'combinations'),
    'data-i18n-aria-label': 'combinations',
  })
  const comboPeakS = Math.max(...combos.map(combo => combo.durationS), 0)
  for (const combo of combos)
    f.add(
      comboGroup,
      buildShiftingHistogramRow(
        f,
        {
          teeth: `${combo.front.teeth}×${combo.rear.teeth}`,
          number: `${combo.front.gearNum}×${combo.rear.gearNum}`,
        },
        combo.durationS,
        comboPeakS,
        totalS,
        combo.front.teeth / combo.rear.teeth,
      ),
    )
  f.add(hist, comboGroup)
  return hist
}

export const SHIFTING_SETTINGS = [
  { setting: 'view', label: 'view', options: ['graph', 'hist'] },
  { setting: 'group', label: 'group by', options: ['sides', 'combos'] },
  { setting: 'label', label: 'show gears as', options: ['teeth', 'number'] },
  { setting: 'value', label: 'show data as', options: ['duration', 'percent'] },
] as const

export type ShiftingSetting = (typeof SHIFTING_SETTINGS)[number]['setting']

const SHIFTING_OPTION_LABELS: Record<string, string> = {
  graph: 'graph',
  hist: 'hist',
  sides: 'front / rear',
  combos: 'combinations',
  teeth: 'teeth',
  number: 'gear number',
  percent: 'percentage',
  duration: 'duration',
}

const buildShiftingSettings = <N>(f: TriNodeFactory<N>, id: string): N => {
  const text = (key: string) => triText(f.presentation.locale, key)
  const picker = f.el('div', 'tri-chart-controls g-select tri-shift-settings')
  const trigger = f.el('button', 'g-select-trigger tri-shift-settings-trigger', undefined, {
    type: 'button',
    'aria-haspopup': 'menu',
    'aria-expanded': 'false',
    'aria-controls': `${id}-shift-settings`,
    'aria-label': text('electronic shifting view'),
    'data-i18n-aria-label': 'electronic shifting view',
  })
  const value = f.el('span', 'g-select-value')
  f.add(
    value,
    f.el('span', undefined, text('graph'), { 'data-shift-trigger': 'graph', 'data-i18n': 'graph' }),
    f.el('span', undefined, text('hist'), { 'data-shift-trigger': 'hist', 'data-i18n': 'hist' }),
  )
  const chevron = f.svg('svg', {
    class: 'g-select-chevron',
    viewBox: '0 0 16 16',
    fill: 'none',
    'aria-hidden': 'true',
    focusable: 'false',
  })
  f.add(
    chevron,
    f.svg('path', {
      d: 'm4 6 4 4 4-4',
      stroke: 'currentColor',
      'stroke-width': 1.4,
      'stroke-linecap': 'round',
      'stroke-linejoin': 'round',
    }),
  )
  f.add(trigger, value, chevron)
  const menu = f.el('div', 'g-select-menu tri-shift-settings-menu', undefined, {
    id: `${id}-shift-settings`,
    role: 'menu',
    hidden: '',
    'aria-label': text('electronic shifting view'),
    'data-i18n-aria-label': 'electronic shifting view',
  })
  for (const { setting, label, options } of SHIFTING_SETTINGS) {
    const group = f.el('div', 'g-select-group', undefined, {
      role: 'group',
      'aria-label': text(label),
      'data-i18n-aria-label': label,
    })
    f.add(
      group,
      f.el('span', 'g-select-heading', text(label), { 'aria-hidden': 'true', 'data-i18n': label }),
    )
    for (const [index, option] of options.entries()) {
      const item = f.el('button', 'g-select-option tri-shift-settings-option', undefined, {
        type: 'button',
        role: 'menuitemradio',
        tabindex: '-1',
        'aria-checked': String(index === 0),
        'data-shift-setting': setting,
        'data-shift-option': option,
      })
      f.add(
        item,
        f.el('span', 'g-select-check', '✓', { 'aria-hidden': 'true' }),
        f.el('span', 'g-select-label', text(SHIFTING_OPTION_LABELS[option]), {
          'data-i18n': SHIFTING_OPTION_LABELS[option],
        }),
      )
      f.add(group, item)
    }
    f.add(menu, group)
  }
  f.add(picker, trigger, menu)
  return picker
}

export const buildShiftingChart = <N>(
  f: TriNodeFactory<N>,
  d: StravaActivityDetail,
  selection?: ActivityAnalysisRange | null,
  graphDomain?: ActivityGraphDomain | null,
): N | null => {
  const shifts = d.gearShifts
  if (d.sport !== 'bike' || shifts.length === 0) return null
  const width = 100
  const height = 30
  const maxDistanceKm = Math.max(d.route.at(-1)?.d ?? d.distanceKm, 0.001)
  const frontValues = shifts.map(shift => shift.frontTeeth)
  const rearValues = shifts.map(shift => shift.rearTeeth)
  const frontMin = Math.min(...frontValues)
  const frontMax = Math.max(...frontValues)
  const rearMin = Math.min(...rearValues)
  const rearMax = Math.max(...rearValues)
  const dominant = dominantGearPairing(
    shifts,
    Math.max(d.movingTimeS, d.route.at(-1)?.elapsedS ?? 0),
  )
  const view = graphViewForDistance(maxDistanceKm, graphDomain)
  const frontY = (value: number): number => gearY(value, frontMin, frontMax, height)
  const rearY = (value: number): number => gearY(value, rearMin, rearMax, height)
  const wattsData = shiftingWattsData(d, shifts, graphDomain)
  const heatAvailable = wattsData.samples.length > 0 && wattsData.rows.length > 0
  const heatmap = heatAvailable
    ? buildWattsHeatmap(
        f,
        d,
        'electronic shifting',
        'tri-shift-heatmap',
        'electronic-shifting',
        [{ side: 'single', samples: wattsData.samples }],
        0,
        wattsData.rows.length === 1 ? 1 : wattsData.rows.length - 1,
        height,
      )
    : null
  const svgEl = f.svg('svg', {
    class: 'tri-elev tri-shift-svg',
    viewBox: `${view.start.toFixed(4)} 0 ${view.width.toFixed(4)} ${height}`,
    preserveAspectRatio: 'none',
    'data-domain-start-distance-km': view.startDistanceKm,
    'data-domain-end-distance-km': view.endDistanceKm,
  })
  if (selection !== undefined) f.add(svgEl, buildAnalysisSelection(f, d, height, selection))
  f.add(
    svgEl,
    f.svg('path', {
      d: gearStepPath(shifts, shift => shift.frontTeeth, frontY, maxDistanceKm, width),
      class: 'tri-shift-line tri-shift-line--front',
    }),
    f.svg('path', {
      d: gearStepPath(shifts, shift => shift.rearTeeth, rearY, maxDistanceKm, width),
      class: 'tri-shift-line tri-shift-line--rear',
    }),
    f.svg('line', { class: 'tri-elev-cursor', x1: 0, y1: 0, x2: 0, y2: height }),
  )
  const wrap = f.el(
    'div',
    `tri-zone tri-elev-wrap tri-shift-chart${heatmap ? ' tri-cycling-mode-chart' : ''}`,
    undefined,
    {
      'data-tri-trace': triathlonTraceName('electronic shifting'),
      'data-shift-view': 'graph',
      'data-shift-group': 'sides',
      'data-shift-label': 'teeth',
      'data-shift-value': 'duration',
      ...(heatmap ? { 'data-cycling-chart-mode': 'distance' } : {}),
    },
  )
  const cap = f.el('div', 'tri-elev-cap tri-elev-cap--summary')
  const summary = f.el('span', 'tri-shift-summary')
  if (dominant) {
    const range = f.el('span', 'tri-elev-range')
    f.add(
      range,
      f.el('span', undefined, `${dominant.frontTeeth}×${dominant.rearTeeth}`, {
        'data-shift-label': 'teeth',
      }),
      f.el('span', undefined, `${dominant.frontGearNum}×${dominant.rearGearNum}`, {
        'data-shift-label': 'number',
      }),
      f.el('span', undefined, ` · ${zoneClock(dominant.durationS)}`),
    )
    f.add(summary, range)
  }
  for (const kind of ['front', 'rear'] as const) {
    const item = f.el('span', `tri-shift-legend-item tri-shift-legend-item--${kind}`)
    f.add(
      item,
      f.el('span', 'tri-shift-legend-line', undefined, { 'aria-hidden': 'true' }),
      f.el('span', 'tri-shift-legend-label', kind, { 'data-i18n': kind }),
    )
    f.add(summary, item)
  }
  f.add(
    cap,
    f.el('span', 'tri-elev-d', 'electronic shifting', { 'data-i18n': 'electronic shifting' }),
    summary,
    buildShiftingSettings(f, `tri-shift-${d.id}`),
  )
  const histogram = buildShiftingHistogram(f, d, graphDomain)
  const distanceFrame = axisFrame(
    f,
    svgEl,
    sampledGearTicks(frontValues).flatMap(value => [
      { label: `${value}T`, vbY: frontY(value), attrs: { 'data-shift-label': 'teeth' } },
      {
        label: `${shifts.find(shift => shift.frontTeeth === value)?.frontGearNum ?? '—'}`,
        vbY: frontY(value),
        attrs: { 'data-shift-label': 'number' },
      },
    ]),
    height,
    distanceXTicks(f.presentation, view.startDistanceKm, view.endDistanceKm),
    true,
    { top: 0, bottom: height },
  )
  if (!heatmap) {
    f.add(wrap, cap, distanceFrame, histogram)
    return wrap
  }
  f.add(cap, heatmap.readout)
  const distancePane = f.el(
    'div',
    'tri-cycling-chart-pane tri-cycling-chart-pane--distance tri-shift-pane--distance',
    undefined,
    { 'data-cycling-chart-mode': 'distance', 'aria-hidden': 'false' },
  )
  f.add(distancePane, distanceFrame)
  const powerPane = f.el(
    'div',
    'tri-cycling-chart-pane tri-cycling-chart-pane--power tri-shift-pane--power',
    undefined,
    { 'data-cycling-chart-mode': 'power', 'aria-hidden': 'true', hidden: '' },
  )
  f.add(
    powerPane,
    axisFrame(
      f,
      heatmap.svg,
      shiftingWattsTicks(wattsData.rows, height),
      height,
      heatmap.wattsTicks,
      true,
      { top: 0, bottom: height },
    ),
  )
  f.add(wrap, cap, distancePane, powerPane, histogram)
  return wrap
}
