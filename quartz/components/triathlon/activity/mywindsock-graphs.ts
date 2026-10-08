import type {
  MyWindsockArchiveReference,
  MyWindsockGraphs,
  MyWindsockGraphPlot,
  MyWindsockGraphSeries,
} from '../../../util/mywindsock-graphs'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import {
  isMyWindsockArchiveReference,
  MYWINDSOCK_GRAPH_CATEGORIES,
  parseMyWindsockGraphs,
} from '../../../util/mywindsock-graphs'
import { triText } from '../../../util/triathlon-i18n'
import { buildLabDateChevron } from '../analytics/panels/body-composition'
import { applyI18n, el, svg } from '../runtime/dom'
import { WIND_TRACE_COLORS } from './trace-colors'

export async function loadMyWindsockGraphs(
  reference: MyWindsockArchiveReference,
  signal: AbortSignal,
): Promise<MyWindsockGraphs> {
  if (!isMyWindsockArchiveReference(reference, reference.activityId))
    throw new Error('Invalid archive reference.')
  const response = await fetch(reference.path, { signal })
  if (!response.ok) throw new Error(`Archive unavailable (${response.status}).`)
  const value: unknown = await response.json()
  const archive = parseMyWindsockGraphs(value, reference.activityId)
  if (!archive) throw new Error('Archive identity or graph data is invalid.')
  return archive
}

// Reader help per native graph key. Sign and angle conventions were checked against the archived
// series: Head/Tail Wind reads 360° where air speed exceeds ground speed, and wImpact% is positive there.
const GRAPH_DESCRIPTIONS: Record<string, string> = {
  interval_designer:
    'Power through the ride with the CdA and total weight that the model uses. myWindsock uses this view to plan target power for course sections.',
  delta_compare:
    'Time difference in seconds against your rides before the last recorded performance change, such as a new position or new equipment.',
  delta_compare_avg: 'Spread of the time difference through the ride.',
  weather:
    'Temperature, precipitation and air density at your position and time. Denser air increases drag at the same speed.',
  windspeed:
    'Wind speed and gusts along the route at the standard 10 m measurement height. Wind at rider height is lower.',
  virt_elev:
    'Your elevation profile with the wind changed into climbing. A headwind adds height and a tailwind removes it. The gap to the real profile is the cost of the wind as climbing.',
  virt_grade:
    'Road gradient adjusted for the wind, next to the actual gradient. The gap shows how much steeper or flatter the wind made the road feel.',
  '3dcourse_virt': 'Elevation profile that the 3D route view uses.',
  airdist_acc:
    'Extra distance of air that you rode through, added up over the ride. The line rises in headwinds and falls in tailwinds. The last value is how much longer the ride was in air than on the road.',
  effective:
    'Speed of the air against you: ground speed plus the headwind part of the wind at rider height. Aerodynamic drag follows this speed.',
  sidewind:
    'Part of the wind that crosses your direction of travel. A strong crosswind pushes the bike sideways and increases yaw.',
  groundspd: 'Speed over the road from the recorded activity.',
  ground_dist: 'Minutes spent in each ground speed band.',
  diff: 'Air speed minus ground speed. Above zero is a net headwind. Below zero is a net tailwind.',
  direction:
    'Wind angle relative to your direction of travel. 360° is a direct headwind, 270° is a direct crosswind and 180° is a direct tailwind.',
  rollingavg:
    'Moving average of ground speed. It removes short changes so that the trend is clear.',
  wwatts:
    'Percent of your power that the wind added or removed at each moment. Above zero means that the wind made you work harder.',
  yaw: 'Angle between your direction of travel and the air that hits you. 0° is air from straight ahead. Crosswind makes the angle larger.',
  yawdist:
    'Time spent at each yaw angle. Compare it with the yaw range in wind tunnel data for wheels and frames.',
  cda: 'Drag area (CdA, m²) that the model calculates from power, speed, wind and gradient. Lower is more aerodynamic. CdA is the ride average and Live CdA is the estimate at each moment. Where available, Test Average and the shaded Test Range show the provider’s aero-test results.',
  cda_dist: 'Time spent at each CdA value.',
  brake:
    'Places where the model detects braking: the bike slowed more than power, gradient and drag can explain.',
  kj_acc:
    'Mechanical work that you did, added up over the ride in kilojoules. On a bike, 1 kJ of work is approximately 1 kcal of food energy.',
  power: 'Power output through the ride.',
  rollingavg_power: 'Moving average of power. It removes short surges so that the trend is clear.',
  pdc: 'Best average power that you held for each duration in this ride, from short sprints on the left to the full ride on the right.',
  wprime:
    'W′ balance: the energy above Critical Power that you have left, in joules. It falls when you ride above Critical Power and refills below it. Near zero means that you are close to exhaustion.',
  pcp: 'Moving average of power that gives more weight to hard efforts, as Normalized Power does. It shows the physiological cost of a variable effort.',
  grade: 'Road gradient through the ride, in percent.',
  gradient_dist: 'Minutes spent in each gradient band.',
  watts:
    'Share of the resistance from rolling, gravity, acceleration and air at each moment. When air resistance is the largest part, aerodynamics matter more than power to weight.',
}

const number = (value: number, maximumFractionDigits = 3): string =>
  value.toLocaleString('en-US', { maximumFractionDigits }).replace('-', '\u2212')
const clock = (milliseconds: number): string => {
  const seconds = Math.max(0, Math.round(milliseconds / 1000))
  return `${Math.floor(seconds / 3600)}:${String(Math.floor(seconds / 60) % 60).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`
}
const extent = (values: number[]): [number, number] => {
  let lo = Infinity
  let hi = -Infinity
  for (const value of values) {
    lo = Math.min(lo, value)
    hi = Math.max(hi, value)
  }
  return Number.isFinite(lo) ? [lo, hi === lo ? lo + 1 : hi] : [0, 1]
}
const yDomain = (plot: MyWindsockGraphPlot, series: MyWindsockGraphSeries[]): [number, number] => {
  const axis = plot.axes[series[0]?.axis ?? 'scale-y']
  const values = series.flatMap(series =>
    series.points.flatMap(point =>
      [point.y, point.upper].flatMap(value => (value == null ? [] : [value])),
    ),
  )
  const [low, high] = extent(values)
  // Native axis limits only widen the domain: Feels Like elevation drops below the provider floor.
  const minimum = Math.min(
    low,
    axis?.minimum ?? (series.some(series => series.kind === 'bar') ? Math.min(0, low) : low),
  )
  const maximum = Math.max(high, axis?.maximum ?? high)
  return maximum > minimum ? [minimum, maximum] : [minimum, minimum + 1]
}
const hasSeriesData = (series: MyWindsockGraphSeries): boolean =>
  series.points.some(point => point.y != null && (series.kind !== 'range' || point.upper != null))

const renderCartesian = (
  host: HTMLElement,
  plot: MyWindsockGraphPlot,
  active: Set<string>,
  color: (id: string) => string,
  text: (key: string) => string,
  elapsedTimeS: number,
): void => {
  const series = plot.series.filter(series => active.has(series.id))
  const categorical = plot.xKind === 'category'
  const categories = [
    ...new Set(plot.series.flatMap(series => series.points.map(point => String(point.x)))),
  ]
  const xs = plot.series.flatMap(series =>
    series.points.flatMap(point => (typeof point.x === 'number' ? [point.x] : [])),
  )
  const [nativeLow, nativeHigh] = categorical
    ? [-0.5, Math.max(1, categories.length) - 0.5]
    : extent(xs)
  const elapsed = plot.xKind === 'elapsed'
  const xLow = elapsed ? Math.min(0, nativeLow) : nativeLow
  const xHigh = elapsed ? Math.max(nativeHigh, elapsedTimeS * 1000) : nativeHigh
  const position = (x: number | string): number =>
    categorical ? categories.indexOf(String(x)) : typeof x === 'number' ? x : 0
  const x = (value: number | string): number => ((position(value) - xLow) / (xHigh - xLow)) * 100
  const panel = el('div', 'tri-workspace-plot-panel')
  const chart = el('div', 'tri-workspace-chart')
  const plotHead = el('div', 'tri-workspace-plot-head')
  const readout = el('div', 'tri-workspace-position')
  const location = el('span', 'tri-workspace-location')
  const values = el('div', 'tri-workspace-values')
  readout.append(location, values)
  plotHead.append(readout)
  const startAxes = el('div', 'tri-workspace-y-axes tri-workspace-y-axes--start')
  const endAxes = el('div', 'tri-workspace-y-axes tri-workspace-y-axes--end')
  const graph = svg('svg', {
    viewBox: '0 0 100 100',
    preserveAspectRatio: 'none',
    class: 'tri-workspace-graph',
    role: 'slider',
    tabindex: 0,
    'aria-label': text(plot.xLabel),
    'aria-valuemin': 0,
    'aria-valuemax': 100,
    'aria-valuenow': 0,
    ...(elapsed
      ? {
          'data-domain-start-elapsed-s': String(xLow / 1000),
          'data-domain-end-elapsed-s': String(xHigh / 1000),
        }
      : {}),
  })
  const grid = svg('g', { class: 'tri-workspace-grid' })
  for (const y of [0, 25, 50, 75, 100]) grid.append(svg('line', { x1: 0, x2: 100, y1: y, y2: y }))
  graph.append(grid)
  // Series sharing a native axis share one scale; axes alternate sides like the workspace graph.
  const domains = new Map<string, [number, number]>()
  for (const axis of new Set(series.map(series => series.axis))) {
    const group = series.filter(series => series.axis === axis)
    const domain = yDomain(plot, group)
    domains.set(axis, domain)
    const label =
      text(plot.axes[axis]?.label ?? '') ||
      (group.some(series => series.label.endsWith('Resistance'))
        ? '%'
        : group.map(series => text(series.label)).join(', '))
    const column = el('div', 'tri-workspace-y-axis', undefined, {
      role: 'group',
      'aria-label': label,
      title: label,
    })
    const fractions = [0, 0.25, 0.5, 0.75, 1]
    const reversed = plot.axes[axis]?.reversed ?? false
    const labels = fractions.map(fraction =>
      number(
        reversed
          ? domain[0] + fraction * (domain[1] - domain[0])
          : domain[1] - fraction * (domain[1] - domain[0]),
      ),
    )
    column.style.setProperty(
      '--tri-workspace-axis-chars',
      String(Math.max(...labels.map(value => value.length))),
    )
    const scale = el('div', 'tri-workspace-y-ticks')
    scale.append(
      ...fractions.map((fraction, index) =>
        el('span', undefined, labels[index], { style: `inset-block-start: ${fraction * 100}%` }),
      ),
    )
    column.append(scale)
    ;(startAxes.childElementCount <= endAxes.childElementCount ? startAxes : endAxes).append(column)
  }
  const y = (value: number, domain: [number, number], reversed = false): number =>
    Math.min(
      100,
      Math.max(
        0,
        ((reversed ? value - domain[0] : domain[1] - value) / (domain[1] - domain[0])) * 100,
      ),
    )
  // Terrain and range fills stay behind the native data lines.
  const layer = (line: MyWindsockGraphSeries): number =>
    line.label === 'Elevation' ? 2 : line.kind === 'range' ? 1 : 0
  for (const line of [...series].sort((a, b) => layer(b) - layer(a))) {
    const domain = domains.get(line.axis) ?? [0, 1]
    const reversed = plot.axes[line.axis]?.reversed ?? false
    const elevation = line.label === 'Elevation'
    const filled = line.kind === 'area' || elevation
    const group = svg('g', {
      'data-native-series': line.id,
      style: `--trace-color: ${color(line.id)}`,
    })
    let path = ''
    let area = ''
    let areaOpen = false
    let areaEnd = ''
    const closeArea = (): void => {
      if (areaOpen) area += ` L ${areaEnd} 100 Z`
      areaOpen = false
    }
    let lower = ''
    let upper: string[] = []
    const closeBand = (): void => {
      if (lower && upper.length)
        group.append(
          svg('path', {
            d: `${lower} ${upper.reverse().join(' ')} Z`,
            class: 'tri-mywindsock-band',
          }),
        )
      lower = ''
      upper = []
    }
    let continuous = false
    let previousX = 0
    let previousY = ''
    let previousUpper = ''
    for (const point of line.points) {
      if (point.y == null || (line.kind === 'range' && point.upper == null)) {
        continuous = false
        closeArea()
        closeBand()
        continue
      }
      const currentX = x(point.x)
      const px = currentX.toFixed(3)
      const stepX = (
        line.step === 'before'
          ? previousX
          : line.step === 'after'
            ? currentX
            : (previousX + currentX) / 2
      ).toFixed(3)
      const py = y(point.y, domain, reversed).toFixed(3)
      if (filled) {
        if (!areaOpen) {
          areaOpen = true
          area += ` M ${px} 100`
        }
        area += ` L ${px} ${py}`
        areaEnd = px
      }
      if (line.kind === 'bar') {
        const base = y(0, domain, reversed)
        const width = categorical ? 50 / Math.max(1, categories.length) : 0.5
        group.append(
          svg('rect', {
            x: Number(px) - width / 2,
            y: Math.min(Number(py), base),
            width,
            height: Math.abs(base - Number(py)),
            class: 'tri-mywindsock-bar',
          }),
        )
      } else if (line.kind === 'range' && point.upper != null) {
        const currentUpper = y(point.upper, domain, reversed).toFixed(3)
        if (continuous && line.step) {
          lower += ` L ${stepX} ${previousY} L ${stepX} ${py}`
          upper.push(`L ${stepX} ${previousUpper}`, `L ${stepX} ${currentUpper}`)
        }
        lower += `${lower ? ' L' : 'M'} ${px} ${py}`
        previousUpper = currentUpper
        upper.push(`L ${px} ${previousUpper}`)
        if (elevation) path += `${continuous ? ' L' : ' M'} ${px} ${py}`
      } else
        path +=
          continuous && line.step
            ? ` H ${stepX} V ${py} H ${px}`
            : `${continuous ? ' L' : ' M'} ${px} ${py}`
      previousY = py
      previousX = currentX
      continuous = true
    }
    closeBand()
    closeArea()
    if (area)
      group.prepend(
        svg('path', {
          d: area,
          class: elevation ? 'tri-cycling-power-elevation' : 'tri-mywindsock-band',
        }),
      )
    if (path)
      group.append(
        svg('path', {
          d: path,
          class: elevation ? 'tri-cycling-power-elevation-line' : 'tri-workspace-line',
        }),
      )
    graph.append(group)
  }
  const formatX = (value: number): string =>
    categorical
      ? (categories[Math.max(0, Math.min(categories.length - 1, Math.round(value)))] ?? '')
      : plot.xKind === 'elapsed' || plot.xKind === 'duration'
        ? clock(value)
        : number(value)
  const ticks = el(
    'div',
    `tri-workspace-ticks${elapsed ? ' tri-mywindsock-ticks--elapsed' : ''}${categorical ? ' tri-mywindsock-ticks--category' : ''}`,
  )
  const tickValues = categorical
    ? [
        ...new Set(
          [0, 0.25, 0.5, 0.75, 1].map(fraction => Math.round(fraction * (categories.length - 1))),
        ),
      ]
    : (elapsed ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1]).map(
        fraction => xLow + fraction * (xHigh - xLow),
      )
  for (const value of tickValues)
    ticks.append(
      el('span', undefined, formatX(value), {
        style: `inset-inline-start: ${((value - xLow) / (xHigh - xLow)) * 100}%`,
      }),
    )
  const cursor = svg('line', { x1: 0, x2: 0, y1: 0, y2: 100, class: 'tri-workspace-cursor' })
  graph.append(cursor)
  let fraction = 0
  const show = (next: number): void => {
    fraction = Math.max(0, Math.min(1, next))
    const at = xLow + fraction * (xHigh - xLow)
    const descriptions = [`${text(plot.xLabel)} ${formatX(at)}`]
    location.textContent = `${text(plot.xLabel).toLocaleLowerCase()} ${formatX(at)}`
    values.replaceChildren()
    for (const line of series) {
      let nearest: MyWindsockGraphSeries['points'][number] | undefined =
        line.step === 'after' || line.step === 'before' ? undefined : line.points[0]
      for (const point of line.points)
        if (
          line.step === 'after'
            ? position(point.x) <= at && (!nearest || position(point.x) > position(nearest.x))
            : line.step === 'before'
              ? position(point.x) >= at && (!nearest || position(point.x) < position(nearest.x))
              : !nearest || Math.abs(position(point.x) - at) < Math.abs(position(nearest.x) - at)
        )
          nearest = point
      const first = line.points[0]
      const last = line.points.at(-1)
      if (elapsed && first && last && (at < position(first.x) || at > position(last.x)))
        nearest = undefined
      const precision = plot.axes[line.axis]?.label === 'CdA' ? 4 : 3
      // Feels Like's range pairs actual elevation with the separately named Feels Like trace.
      const showRange = line.kind === 'range' && line.label !== 'Elevation'
      const formatted =
        nearest?.y == null || (showRange && nearest.upper == null)
          ? '—'
          : showRange && nearest.upper != null
            ? `${number(nearest.y, precision)}–${number(nearest.upper, precision)}`
            : number(nearest.y, precision)
      const description = `${text(line.label)} ${formatted}`
      // The readout doubles as the legend: colour key, series name, then the value under the cursor.
      const item = el('span', 'tri-workspace-current-value', undefined, {
        title: description,
        'aria-label': description,
      })
      item.style.setProperty('--trace-color', color(line.id))
      item.append(el('span', 'tri-mywindsock-legend-label', text(line.label)), formatted)
      values.append(item)
      descriptions.push(description)
    }
    graph.setAttribute('aria-valuenow', String(Math.round(fraction * 100)))
    graph.setAttribute('aria-valuetext', descriptions.join(' · '))
    cursor.setAttribute('x1', String(fraction * 100))
    cursor.setAttribute('x2', String(fraction * 100))
  }
  graph.addEventListener('pointermove', event => {
    const bounds = graph.getBoundingClientRect()
    show((event.clientX - bounds.left) / bounds.width)
  })
  graph.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
    event.preventDefault()
    show(
      event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? 1
          : fraction + (event.key === 'ArrowLeft' ? -0.01 : 0.01),
    )
  })
  chart.append(plotHead, startAxes, graph, endAxes, ticks)
  panel.append(chart)
  host.append(panel)
  show(0)
}

const renderPolar = (
  host: HTMLElement,
  plot: MyWindsockGraphPlot,
  active: Set<string>,
  color: (id: string) => string,
  text: (key: string) => string,
): void => {
  const lines = plot.series.filter(series => active.has(series.id))
  const graph = svg('svg', {
    viewBox: '0 0 120 120',
    class: 'tri-mywindsock-polar',
    role: 'img',
    'aria-label': `${text(plot.kind)} · ${text('provider values')}`,
  })
  const list = el('dl', 'tri-mywindsock-polar-values')
  const domain = extent(
    lines.flatMap(series => series.points.flatMap(point => (point.y == null ? [] : [point.y]))),
  )[1]
  if (plot.kind === 'radar') {
    for (const radius of [10, 20, 30, 40])
      graph.append(svg('circle', { cx: 60, cy: 60, r: radius, class: 'tri-mywindsock-grid' }))
    for (const line of lines) {
      const count = line.points.length
      let path = ''
      let continuous = false
      line.points.forEach((point, index) => {
        if (point.y == null) {
          continuous = false
          return
        }
        const angle = (index / count) * Math.PI * 2 - Math.PI / 2
        const r = (point.y / Math.max(domain, 1)) * 40
        path += `${continuous ? ' L' : ' M'} ${(60 + Math.cos(angle) * r).toFixed(2)} ${(60 + Math.sin(angle) * r).toFixed(2)}`
        continuous = true
      })
      if (path)
        graph.append(
          svg('path', {
            d: `${path}${line.points.every(point => point.y != null) ? ' Z' : ''}`,
            class: 'tri-workspace-line',
            style: `--trace-color: ${color(line.id)}`,
          }),
        )
    }
  } else if (plot.kind === 'pie' && lines.every(line => line.points[0]?.y != null)) {
    const total = lines.reduce((sum, line) => sum + Math.max(0, line.points[0]?.y ?? 0), 0)
    let angle = -Math.PI / 2
    for (const line of lines) {
      const share = total > 0 ? Math.max(0, line.points[0]?.y ?? 0) / total : 0
      const end = angle + share * Math.PI * 2
      const startX = 60 + 40 * Math.cos(angle),
        startY = 60 + 40 * Math.sin(angle)
      const endX = 60 + 40 * Math.cos(end),
        endY = 60 + 40 * Math.sin(end)
      if (share >= 1) graph.append(svg('circle', { cx: 60, cy: 60, r: 40, fill: color(line.id) }))
      else if (share > 0)
        graph.append(
          svg('path', {
            d: `M 60 60 L ${startX} ${startY} A 40 40 0 ${share > 0.5 ? 1 : 0} 1 ${endX} ${endY} Z`,
            fill: color(line.id),
          }),
        )
      angle = end
    }
  } else if (plot.kind === 'gauge') {
    const axis = plot.axes['scale-r']
    for (const line of lines) {
      const value = line.points[0]?.y
      if (value != null)
        host.append(
          el('meter', 'tri-mywindsock-gauge', undefined, {
            min: String(axis?.minimum ?? 0),
            max: String(axis?.maximum ?? Math.max(1, value)),
            value: String(value),
            'aria-label': text('Activity weather ranking'),
          }),
        )
    }
  }
  for (const line of lines)
    for (const point of line.points) {
      list.append(
        el('dt', 'tri-mywindsock-legend-key', `${text(line.label)} · ${text(String(point.x))}`, {
          style: `--trace-color: ${color(line.id)}`,
        }),
        el('dd', undefined, point.y == null ? '—' : number(point.y)),
      )
    }
  if (plot.kind !== 'gauge') host.append(graph)
  host.append(list)
}

function mountMyWindsockGraphs(
  host: HTMLElement,
  archive: MyWindsockGraphs,
  pickerHost: HTMLElement,
  presentation: () => TriathlonPresentation,
  elapsedTimeS: number,
): (() => void) | null {
  const text = (key: string): string => triText(presentation().locale, key)
  const visibleGraphs = archive.graphs.filter(
    graph =>
      !['ai_power', 'bearing', 'inline:pointsgraph', 'inline:summary_windrose_chart'].includes(
        graph.key,
      ) &&
      graph.state === 'captured' &&
      graph.plots.some(plot => plot.series.some(hasSeriesData)),
  )
  if (!visibleGraphs.length) {
    pickerHost.replaceChildren()
    host.replaceChildren()
    return null
  }
  const pickerId = `tri-mywindsock-${archive.activityId}`
  const trigger = el('button', 'tri-lab-date-trigger', undefined, {
    type: 'button',
    id: `${pickerId}-trigger`,
    'aria-haspopup': 'listbox',
    'aria-expanded': 'false',
    'aria-controls': `${pickerId}-menu`,
  })
  const value = el('span', 'tri-lab-date-value')
  trigger.append(value, buildLabDateChevron())
  const menu = el('div', 'tri-lab-date-menu', undefined, {
    id: `${pickerId}-menu`,
    role: 'listbox',
    'aria-label': text('wind graph'),
    'data-i18n-aria-label': 'wind graph',
  })
  menu.hidden = true
  const graphLabel = (graph: MyWindsockGraphs['graphs'][number]): string => text(graph.label)
  const options: HTMLElement[] = []
  for (const category of [...Object.keys(MYWINDSOCK_GRAPH_CATEGORIES), 'Other']) {
    const graphs = visibleGraphs.filter(graph => graph.category === category)
    if (!graphs.length) continue
    const group = el('div', 'tri-mywindsock-picker-group', undefined, {
      role: 'group',
      'aria-label': text(category),
      'data-i18n-aria-label': category,
    })
    group.append(
      el('span', 'tri-mywindsock-picker-heading', text(category), {
        'aria-hidden': 'true',
        'data-i18n': category,
      }),
    )
    for (const graph of graphs) {
      const option = el('button', 'tri-lab-date-option', undefined, {
        type: 'button',
        role: 'option',
        'aria-selected': 'false',
        'data-mywindsock-graph': graph.key,
        tabindex: '-1',
      })
      option.append(
        el('span', 'tri-lab-date-check', '✓', { 'aria-hidden': 'true' }),
        el('span', 'tri-lab-date-option-value', graphLabel(graph)),
      )
      options.push(option)
      group.append(option)
    }
    menu.append(group)
  }
  const picker = el('div', 'tri-lab-date-picker')
  picker.append(trigger, menu)
  pickerHost.replaceChildren(picker)
  // Every description shares one grid cell, so the block keeps the tallest one's height at the
  // current width and switching graphs never moves the chart.
  const descriptions = el('div', 'tri-mywindsock-descriptions')
  const descriptionItems = new Map<string, HTMLElement>()
  for (const graph of visibleGraphs) {
    if (!GRAPH_DESCRIPTIONS[graph.key]) continue
    const item = el('p', 'tri-mywindsock-description', undefined, {
      id: `${pickerId}-description-${graph.key.replace(/[^\w-]/g, '-')}`,
    })
    descriptionItems.set(graph.key, item)
    descriptions.append(item)
  }
  const status = el('p', 'tri-mywindsock-note', undefined, { role: 'status' })
  const plots = el('div', 'tri-mywindsock-plots')
  host.replaceChildren(descriptions, plots, status)
  let selected =
    visibleGraphs.find(graph => graph.key === (archive.cyclingCda ? 'cda' : 'virt_elev')) ??
    visibleGraphs[0]
  const active = new Set<string>()
  const colors = new Map<string, string>()
  const color = (id: string): string => colors.get(id) ?? WIND_TRACE_COLORS[0]
  const updateStatus = (): void => {
    for (const [key, item] of descriptionItems) {
      item.textContent = text(GRAPH_DESCRIPTIONS[key])
      item.toggleAttribute('data-active', key === selected.key)
    }
    const description = descriptionItems.get(selected.key)
    if (description) trigger.setAttribute('aria-describedby', description.id)
    else trigger.removeAttribute('aria-describedby')
    status.textContent = [
      selected.note ? text(selected.note) : null,
      archive.sport === 'run' && (selected.key === 'cda' || selected.key === 'interval_designer')
        ? text('Run model output · aerodynamic meaning unverified.')
        : null,
    ]
      .filter(Boolean)
      .join(' · ')
    status.hidden = !status.textContent
  }
  const draw = (): void => {
    plots.replaceChildren()
    for (const plot of selected.plots) {
      if (!plot.series.some(hasSeriesData)) continue
      const panel = el('div', 'tri-mywindsock-native-graph', undefined, {
        'data-native-kind': plot.kind,
        'data-native-axis': plot.xKind,
      })
      if (plot.kind === 'cartesian') renderCartesian(panel, plot, active, color, text, elapsedTimeS)
      else if (plot.kind === 'unsupported')
        panel.append(
          el(
            'p',
            'tri-mywindsock-note',
            text('This chart type is only kept as native configuration.'),
          ),
        )
      else renderPolar(panel, plot, active, color, text)
      plots.append(panel)
    }
  }
  const update = (): void => {
    value.textContent = graphLabel(selected)
    trigger.setAttribute('aria-label', `${text('wind graph')}: ${graphLabel(selected)}`)
    for (const option of options)
      option.setAttribute('aria-selected', String(option.dataset.mywindsockGraph === selected.key))
    active.clear()
    colors.clear()
    const series = selected.plots.flatMap(plot => plot.series)
    series.forEach((line, index) => {
      if (hasSeriesData(line)) active.add(line.id)
      colors.set(
        line.id,
        line.label === 'Elevation'
          ? 'var(--gray)'
          : selected.key === 'cda' && (line.label === 'Test Average' || line.label === 'Test Range')
            ? WIND_TRACE_COLORS[1]
            : selected.key === 'cda' && line.label === 'CdA'
              ? WIND_TRACE_COLORS[0]
              : WIND_TRACE_COLORS[index % WIND_TRACE_COLORS.length],
      )
    })
    updateStatus()
    draw()
  }
  const close = (restoreFocus = false): void => {
    menu.hidden = true
    trigger.setAttribute('aria-expanded', 'false')
    if (restoreFocus) trigger.focus({ preventScroll: true })
  }
  const focusOption = (index: number): void => {
    const option = options[index]
    if (!option) return
    for (const candidate of options) candidate.tabIndex = candidate === option ? 0 : -1
    option.focus({ preventScroll: true })
    option.scrollIntoView({ block: 'nearest' })
  }
  const open = (): void => {
    if (!menu.hidden) return
    menu.hidden = false
    trigger.setAttribute('aria-expanded', 'true')
    focusOption(options.findIndex(option => option.getAttribute('aria-selected') === 'true'))
  }
  const onTriggerClick = (): void => (menu.hidden ? open() : close())
  const onTriggerKeydown = (event: KeyboardEvent): void => {
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return
    event.preventDefault()
    open()
  }
  const onMenuClick = (event: MouseEvent): void => {
    const option =
      event.target instanceof Element
        ? event.target.closest<HTMLElement>('[data-mywindsock-graph]')
        : null
    const next = visibleGraphs.find(graph => graph.key === option?.dataset.mywindsockGraph)
    if (!next) return
    selected = next
    update()
    close(true)
  }
  const onMenuKeydown = (event: KeyboardEvent): void => {
    if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      close(true)
      return
    }
    const current = options.findIndex(option => option === document.activeElement)
    const target =
      event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? options.length - 1
          : event.key === 'ArrowDown'
            ? Math.min(options.length - 1, current + 1)
            : event.key === 'ArrowUp'
              ? Math.max(0, current - 1)
              : -1
    if (target < 0) return
    event.preventDefault()
    focusOption(target)
  }
  const onFocusout = (event: FocusEvent): void => {
    if (event.relatedTarget instanceof Node && picker.contains(event.relatedTarget)) return
    close()
  }
  const onPointerdown = (event: PointerEvent): void => {
    if (menu.hidden || event.composedPath().includes(picker)) return
    close()
  }
  const onLocale = (): void => {
    if (!host.isConnected) return
    applyI18n(pickerHost, presentation())
    applyI18n(host, presentation())
    value.textContent = graphLabel(selected)
    trigger.setAttribute('aria-label', `${text('wind graph')}: ${graphLabel(selected)}`)
    for (const option of options) {
      const graph = visibleGraphs.find(graph => graph.key === option.dataset.mywindsockGraph)
      const label = option.querySelector('.tri-lab-date-option-value')
      if (graph && label) label.textContent = graphLabel(graph)
    }
    updateStatus()
    draw()
  }
  trigger.addEventListener('click', onTriggerClick)
  trigger.addEventListener('keydown', onTriggerKeydown)
  menu.addEventListener('click', onMenuClick)
  menu.addEventListener('keydown', onMenuKeydown)
  picker.addEventListener('focusout', onFocusout)
  document.addEventListener('pointerdown', onPointerdown)
  window.addEventListener('tri:locale', onLocale)
  update()
  return () => {
    trigger.removeEventListener('click', onTriggerClick)
    trigger.removeEventListener('keydown', onTriggerKeydown)
    menu.removeEventListener('click', onMenuClick)
    menu.removeEventListener('keydown', onMenuKeydown)
    picker.removeEventListener('focusout', onFocusout)
    document.removeEventListener('pointerdown', onPointerdown)
    window.removeEventListener('tri:locale', onLocale)
  }
}

export function setupMyWindsockGraphs(
  root: HTMLElement,
  signal: AbortSignal,
  presentation: () => TriathlonPresentation,
): () => void {
  const requests = new Map<HTMLElement, AbortController>()
  const mounted = new Map<HTMLElement, () => void>()
  const localizeSection = (section: Element): void => {
    section.setAttribute('aria-label', triText(presentation().locale, 'wind graphs'))
    applyI18n(section, presentation())
  }
  const load = async (section: HTMLElement): Promise<void> => {
    const content = section.querySelector<HTMLElement>('[data-mywindsock-content]')
    const pickerHost = section.querySelector<HTMLElement>('[data-mywindsock-picker]')
    if (!content || !pickerHost || !root.contains(section) || signal.aborted) return
    if (requests.has(section) || mounted.has(section)) return
    visibility.unobserve(section)
    const request = new AbortController()
    requests.set(section, request)
    section.dataset.mywindsockState = 'loading'
    section.setAttribute('aria-busy', 'true')
    content.replaceChildren(
      el('p', 'tri-mywindsock-note', triText(presentation().locale, 'Loading graphs…'), {
        role: 'status',
        'data-i18n': 'Loading graphs…',
      }),
    )
    try {
      const archive = await loadMyWindsockGraphs(
        {
          activityId: Number(section.dataset.mywindsockId),
          path: section.dataset.mywindsockPath ?? '',
          capturedAt: section.dataset.mywindsockCapturedAt ?? '',
        },
        request.signal,
      )
      if (signal.aborted || !section.isConnected) return
      const elapsedTimeS = Number(section.dataset.mywindsockElapsedS)
      const cleanup = mountMyWindsockGraphs(
        content,
        archive,
        pickerHost,
        presentation,
        Number.isFinite(elapsedTimeS) && elapsedTimeS > 0 ? elapsedTimeS : 0,
      )
      section.hidden = cleanup === null
      if (cleanup) mounted.set(section, cleanup)
      section.dataset.mywindsockState = cleanup ? 'ready' : 'empty'
    } catch (error) {
      if (signal.aborted || request.signal.aborted || !section.isConnected) return
      section.dataset.mywindsockState = 'error'
      content.replaceChildren(
        el(
          'p',
          'tri-mywindsock-note',
          triText(
            presentation().locale,
            error instanceof Error ? error.message : 'Archive unavailable.',
          ),
          { role: 'alert' },
        ),
        el('button', undefined, triText(presentation().locale, 'Retry graphs'), {
          type: 'button',
          'data-mywindsock-retry': '',
          'data-i18n': 'Retry graphs',
        }),
      )
    } finally {
      requests.delete(section)
      section.removeAttribute('aria-busy')
    }
  }
  const visibility = new IntersectionObserver(
    entries => {
      for (const entry of entries)
        if (entry.isIntersecting && entry.target instanceof HTMLElement) void load(entry.target)
    },
    { rootMargin: '200px 0px' },
  )
  const observe = (element: Element): void => {
    const sections = element.matches('[data-mywindsock-id]')
      ? [element]
      : element.querySelectorAll('[data-mywindsock-id]')
    for (const section of sections)
      if (section instanceof HTMLElement && section.dataset.mywindsockState === 'pending') {
        localizeSection(section)
        visibility.observe(section)
      }
  }
  const additions = new MutationObserver(records => {
    for (const record of records)
      for (const node of record.addedNodes) if (node instanceof Element) observe(node)
  })
  const onClick = (event: MouseEvent): void => {
    const button =
      event.target instanceof Element
        ? event.target.closest<HTMLButtonElement>('[data-mywindsock-retry]')
        : null
    const section = button?.closest<HTMLElement>('[data-mywindsock-id]')
    if (section) void load(section)
  }
  root.addEventListener('click', onClick)
  const onLocale = (): void => {
    for (const section of root.querySelectorAll('[data-mywindsock-id]')) localizeSection(section)
  }
  window.addEventListener('tri:locale', onLocale)
  additions.observe(root, { childList: true, subtree: true })
  observe(root)
  return () => {
    root.removeEventListener('click', onClick)
    window.removeEventListener('tri:locale', onLocale)
    visibility.disconnect()
    additions.disconnect()
    for (const request of requests.values()) request.abort()
    for (const cleanup of mounted.values()) cleanup()
    requests.clear()
    mounted.clear()
  }
}
