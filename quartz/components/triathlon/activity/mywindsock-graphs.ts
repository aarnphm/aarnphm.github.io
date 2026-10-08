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

const number = (value: number): string =>
  value.toLocaleString('en-US', { maximumFractionDigits: 3 }).replace('-', '\u2212')
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

const renderCartesian = (
  host: HTMLElement,
  plot: MyWindsockGraphPlot,
  active: Set<string>,
  color: (id: string) => string,
  text: (key: string) => string,
): void => {
  const series = plot.series.filter(series => active.has(series.id))
  const categories = [
    ...new Set(plot.series.flatMap(series => series.points.map(point => String(point.x)))),
  ]
  const xs = plot.series.flatMap(series =>
    series.points.flatMap(point => (typeof point.x === 'number' ? [point.x] : [])),
  )
  const [xLow, xHigh] =
    plot.xKind === 'category' ? [0, Math.max(1, categories.length - 1)] : extent(xs)
  const position = (x: number | string): number =>
    plot.xKind === 'category' ? categories.indexOf(String(x)) : typeof x === 'number' ? x : 0
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
  for (const line of series) {
    const domain = domains.get(line.axis) ?? [0, 1]
    const reversed = plot.axes[line.axis]?.reversed ?? false
    const group = svg('g', {
      'data-native-series': line.id,
      style: `--trace-color: ${color(line.id)}`,
    })
    let path = ''
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
    for (const point of line.points) {
      if (point.y == null || (line.kind === 'range' && point.upper == null)) {
        continuous = false
        closeBand()
        continue
      }
      const px = x(point.x).toFixed(3)
      const py = y(point.y, domain, reversed).toFixed(3)
      if (line.kind === 'bar') {
        const base = y(0, domain, reversed)
        const width = plot.xKind === 'category' ? 85 / Math.max(1, categories.length) : 0.5
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
        lower += `${lower ? ' L' : 'M'} ${px} ${py}`
        upper.push(`L ${px} ${y(point.upper, domain, reversed).toFixed(3)}`)
      } else path += `${continuous ? ' L' : ' M'} ${px} ${py}`
      continuous = true
    }
    closeBand()
    if (path) group.append(svg('path', { d: path, class: 'tri-workspace-line' }))
    graph.append(group)
  }
  const formatX = (value: number): string =>
    plot.xKind === 'category'
      ? (categories[Math.round(value)] ?? '')
      : plot.xKind === 'elapsed' || plot.xKind === 'duration'
        ? clock(value)
        : number(value)
  const ticks = el('div', 'tri-workspace-ticks')
  for (const fraction of [0, 0.25, 0.5, 0.75, 1])
    ticks.append(el('span', undefined, formatX(xLow + fraction * (xHigh - xLow))))
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
      let nearest = line.points[0]
      for (const point of line.points)
        if (!nearest || Math.abs(position(point.x) - at) < Math.abs(position(nearest.x) - at))
          nearest = point
      // A range band's upper end is plotted as its own series, so the readout names the lower end.
      const formatted = nearest?.y == null ? '—' : number(nearest.y)
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
): () => void {
  const text = (key: string): string => triText(presentation().locale, key)
  const visibleGraphs = archive.graphs.filter(
    graph =>
      !['ai_power', 'bearing', 'inline:pointsgraph', 'inline:summary_windrose_chart'].includes(
        graph.key,
      ) && graph.state !== 'unavailable',
  )
  if (!visibleGraphs.length) {
    pickerHost.replaceChildren()
    host.replaceChildren(
      el('p', 'tri-mywindsock-note', text('No graphs available.'), {
        'data-i18n': 'No graphs available.',
      }),
    )
    return () => {}
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
  const graphLabel = (graph: MyWindsockGraphs['graphs'][number]): string =>
    graph.state === 'captured' ? text(graph.label) : `${text(graph.label)} · ${text(graph.state)}`
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
  const status = el('p', 'tri-mywindsock-note', undefined, { role: 'status' })
  const plots = el('div', 'tri-mywindsock-plots')
  host.replaceChildren(plots, status)
  let selected =
    visibleGraphs.find(graph => graph.key === (archive.cyclingCda ? 'cda' : 'virt_elev')) ??
    visibleGraphs[0]
  const active = new Set<string>()
  const colors = new Map<string, string>()
  const color = (id: string): string => colors.get(id) ?? WIND_TRACE_COLORS[0]
  const updateStatus = (): void => {
    status.textContent = [
      selected.state === 'captured' ? null : text(selected.state),
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
    if (selected.state !== 'captured') return
    for (const plot of selected.plots) {
      const panel = el('div', 'tri-mywindsock-native-graph', undefined, {
        'data-native-kind': plot.kind,
        'data-native-axis': plot.xKind,
      })
      if (plot.kind === 'cartesian') renderCartesian(panel, plot, active, color, text)
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
      if (selected.key === 'cda' && (line.label === 'Test Average' || line.label === 'Test Range'))
        return
      if (line.points.some(point => point.y != null)) active.add(line.id)
      colors.set(line.id, WIND_TRACE_COLORS[index % WIND_TRACE_COLORS.length])
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
      mounted.set(section, mountMyWindsockGraphs(content, archive, pickerHost, presentation))
      section.dataset.mywindsockState = 'ready'
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
