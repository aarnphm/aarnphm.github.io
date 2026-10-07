import type {
  MyWindsockArchiveReference,
  MyWindsockGraphs,
  MyWindsockGraphPlot,
  MyWindsockGraphSeries,
} from '../../../util/mywindsock-graphs'
import {
  isMyWindsockArchiveReference,
  parseMyWindsockGraphs,
} from '../../../util/mywindsock-graphs'
import { el, svg } from '../runtime/dom'

export const MYWINDSOCK_COLORS = ['#205ea6', '#da702c', '#3aa99f', '#8b6fd6', '#d14d41', '#a47c1b']

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
  value.toLocaleString('en-US', { maximumFractionDigits: 3 })
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
  const minimum =
    axis?.minimum ?? (series.some(series => series.kind === 'bar') ? Math.min(0, low) : low)
  const maximum = axis?.maximum ?? high
  return maximum > minimum ? [minimum, maximum] : [minimum, minimum + 1]
}

const renderCartesian = (
  host: HTMLElement,
  plot: MyWindsockGraphPlot,
  active: Set<string>,
  color: (id: string) => string,
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
  const frame = el('div', 'tri-mywindsock-frame')
  const axes = el('div', 'tri-mywindsock-y-axes')
  const graph = svg('svg', {
    viewBox: '0 0 100 60',
    preserveAspectRatio: 'none',
    class: 'tri-mywindsock-plot',
    role: 'slider',
    tabindex: 0,
    'aria-label': plot.xLabel,
    'aria-valuemin': 0,
    'aria-valuemax': 100,
    'aria-valuenow': 0,
  })
  for (const y of [0, 15, 30, 45, 60])
    graph.append(svg('line', { x1: 0, x2: 100, y1: y, y2: y, class: 'tri-mywindsock-grid' }))
  const domains = new Map<string, [number, number]>()
  for (const axis of new Set(series.map(series => series.axis))) {
    const group = series.filter(series => series.axis === axis)
    const domain = yDomain(plot, group)
    domains.set(axis, domain)
    const column = el('div', 'tri-mywindsock-y-axis', undefined, {
      'aria-label': plot.axes[axis]?.label || group.map(series => series.label).join(', '),
    })
    column.append(
      el(
        'span',
        'tri-mywindsock-axis-title',
        plot.axes[axis]?.label ||
          (group.some(series => series.label.endsWith('Resistance')) ? '%' : group[0].label),
      ),
    )
    for (const fraction of [0, 0.25, 0.5, 0.75, 1])
      column.append(el('span', undefined, number(domain[1] - fraction * (domain[1] - domain[0]))))
    axes.append(column)
  }
  const y = (value: number, domain: [number, number]): number =>
    Math.min(60, Math.max(0, ((domain[1] - value) / (domain[1] - domain[0])) * 60))
  for (const line of series) {
    const domain = domains.get(line.axis) ?? [0, 1]
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
      const py = y(point.y, domain).toFixed(3)
      if (line.kind === 'bar') {
        const base = y(0, domain)
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
        upper.push(`L ${px} ${y(point.upper, domain).toFixed(3)}`)
      } else path += `${continuous ? ' L' : ' M'} ${px} ${py}`
      continuous = true
    }
    closeBand()
    if (path) group.append(svg('path', { d: path, class: 'tri-mywindsock-line' }))
    graph.append(group)
  }
  const formatX = (value: number): string =>
    plot.xKind === 'category'
      ? (categories[Math.round(value)] ?? '')
      : plot.xKind === 'elapsed' || plot.xKind === 'duration'
        ? clock(value)
        : number(value)
  const ticks = el('div', 'tri-mywindsock-ticks')
  for (const fraction of [0, 0.25, 0.5, 0.75, 1])
    ticks.append(el('span', undefined, formatX(xLow + fraction * (xHigh - xLow))))
  const readout = el('div', 'tri-mywindsock-readout', undefined, { 'aria-live': 'off' })
  const cursor = svg('line', { x1: 0, x2: 0, y1: 0, y2: 60, class: 'tri-mywindsock-cursor' })
  graph.append(cursor)
  let fraction = 0
  const show = (next: number): void => {
    fraction = Math.max(0, Math.min(1, next))
    const at = xLow + fraction * (xHigh - xLow)
    const values = [formatX(at)]
    for (const line of series) {
      let nearest = line.points[0]
      for (const point of line.points)
        if (!nearest || Math.abs(position(point.x) - at) < Math.abs(position(nearest.x) - at))
          nearest = point
      values.push(
        `${line.label}: ${nearest?.y == null ? '—' : number(nearest.y)}${nearest?.upper == null ? '' : ` to ${number(nearest.upper)}`}`,
      )
    }
    readout.textContent = values.join(' · ')
    graph.setAttribute('aria-valuenow', String(Math.round(fraction * 100)))
    graph.setAttribute('aria-valuetext', values.join(' · '))
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
  frame.append(axes, graph)
  host.append(frame, ticks, el('div', 'tri-mywindsock-x-label', plot.xLabel), readout)
  if (!series.length) host.append(el('p', 'tri-mywindsock-note', 'All series are hidden.'))
  show(0)
}

const renderPolar = (
  host: HTMLElement,
  plot: MyWindsockGraphPlot,
  active: Set<string>,
  color: (id: string) => string,
): void => {
  const lines = plot.series.filter(series => active.has(series.id))
  const graph = svg('svg', {
    viewBox: '0 0 120 120',
    class: 'tri-mywindsock-polar',
    role: 'img',
    'aria-label': `${plot.kind} · provider values`,
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
            class: 'tri-mywindsock-line',
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
            'aria-label': 'Activity weather ranking',
          }),
        )
    }
  }
  for (const line of lines)
    for (const point of line.points) {
      list.append(
        el('dt', undefined, `${line.label} · ${String(point.x)}`),
        el('dd', undefined, point.y == null ? '—' : number(point.y)),
      )
    }
  if (plot.kind !== 'gauge') host.append(graph)
  host.append(list)
}

export function mountMyWindsockRawGraphs(
  host: HTMLElement,
  archive: MyWindsockGraphs,
  source: HTMLDetailsElement,
): () => void {
  const selectId = `mywindsock-graph-${archive.activityId}-${Math.random().toString(36).slice(2, 7)}`
  const picker = document.createElement('select')
  picker.className = 'tri-mywindsock-picker'
  picker.id = selectId
  picker.setAttribute('aria-label', 'myWindsock graph')
  for (const graph of archive.graphs)
    picker.append(
      el(
        'option',
        undefined,
        `${graph.label} · ${graph.key}${graph.state === 'captured' ? '' : ` (${graph.state})`}`,
        { value: graph.key },
      ),
    )
  picker.value = archive.cyclingCda ? 'cda' : 'virt_elev'
  const controls = el('div', 'tri-mywindsock-toggles', undefined, {
    role: 'group',
    'aria-label': 'myWindsock graph series',
  })
  const status = el('p', 'tri-mywindsock-note', undefined, { role: 'status' })
  const plots = el('div', 'tri-mywindsock-plots')
  const json = el('pre')
  source.append(el('p', 'tri-mywindsock-note', 'Native graph configuration'), json)
  host.replaceChildren(
    el('label', undefined, 'Captured graph', { for: selectId }),
    picker,
    status,
    controls,
    plots,
  )
  let selected = archive.graphs.find(graph => graph.key === picker.value) ?? archive.graphs[0]
  const active = new Set<string>()
  const colors = new Map<string, string>()
  const color = (id: string): string => colors.get(id) ?? MYWINDSOCK_COLORS[0]
  const draw = (): void => {
    plots.replaceChildren()
    if (selected.state !== 'captured') return
    for (const plot of selected.plots) {
      const panel = el('div', 'tri-mywindsock-native-graph', undefined, {
        'data-native-kind': plot.kind,
        'data-native-axis': plot.xKind,
      })
      if (plot.kind === 'cartesian') renderCartesian(panel, plot, active, color)
      else if (plot.kind === 'unsupported')
        panel.append(
          el('p', undefined, 'This native chart type is available in Provider analysis.'),
        )
      else renderPolar(panel, plot, active, color)
      plots.append(panel)
    }
  }
  const update = (): void => {
    selected = archive.graphs.find(graph => graph.key === picker.value) ?? archive.graphs[0]
    active.clear()
    colors.clear()
    controls.replaceChildren()
    const series = selected.plots.flatMap(plot => plot.series)
    series.forEach((line, index) => {
      if (line.points.some(point => point.y != null)) active.add(line.id)
      colors.set(line.id, MYWINDSOCK_COLORS[index % MYWINDSOCK_COLORS.length])
      const button = el('button', 'tri-mywindsock-toggle', line.label, {
        type: 'button',
        'data-native-toggle': line.id,
        'aria-pressed': String(active.has(line.id)),
        ...(line.points.length ? {} : { disabled: '' }),
        title: `${line.points.length.toLocaleString()} native points`,
      })
      button.style.setProperty('--trace-color', color(line.id))
      controls.append(button)
    })
    status.textContent = [
      selected.state === 'captured'
        ? `${series.length} series · native units and coordinates`
        : selected.state,
      selected.note,
      archive.sport === 'run' && (selected.key === 'cda' || selected.key === 'interval_designer')
        ? 'Run model output · aerodynamic meaning unverified.'
        : null,
    ]
      .filter(Boolean)
      .join(' · ')
    json.textContent = source.open ? JSON.stringify(selected.configuration, null, 2) : ''
    draw()
  }
  const toggle = (event: MouseEvent): void => {
    const button =
      event.target instanceof Element
        ? event.target.closest<HTMLButtonElement>('[data-native-toggle]')
        : null
    const id = button?.dataset.nativeToggle
    if (!id || !button) return
    if (active.has(id)) active.delete(id)
    else active.add(id)
    button.setAttribute('aria-pressed', String(active.has(id)))
    draw()
  }
  const showJson = (): void => {
    if (source.open) json.textContent = JSON.stringify(selected.configuration, null, 2)
  }
  picker.addEventListener('change', update)
  controls.addEventListener('click', toggle)
  source.addEventListener('toggle', showJson)
  update()
  return () => {
    picker.removeEventListener('change', update)
    controls.removeEventListener('click', toggle)
    source.removeEventListener('toggle', showJson)
  }
}

export function setupMyWindsockGraphs(root: HTMLElement, signal: AbortSignal): () => void {
  const requests = new Map<HTMLElement, AbortController>()
  const mounted = new Map<HTMLElement, () => void>()
  const load = async (section: HTMLElement): Promise<void> => {
    const content = section.querySelector<HTMLElement>('[data-mywindsock-content]')
    const provenance = section.querySelector<HTMLDetailsElement>('[data-mywindsock-provenance]')
    if (!content || !provenance || !root.contains(section) || signal.aborted) return
    if (requests.has(section) || mounted.has(section)) return
    visibility.unobserve(section)
    const request = new AbortController()
    requests.set(section, request)
    section.dataset.mywindsockState = 'loading'
    section.setAttribute('aria-busy', 'true')
    content.replaceChildren(el('p', 'tri-mywindsock-note', 'Loading graphs…', { role: 'status' }))
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
      mounted.set(section, mountMyWindsockRawGraphs(content, archive, provenance))
      section.dataset.mywindsockState = 'ready'
    } catch (error) {
      if (signal.aborted || request.signal.aborted || !section.isConnected) return
      section.dataset.mywindsockState = 'error'
      content.replaceChildren(
        el(
          'p',
          'tri-mywindsock-note',
          error instanceof Error ? error.message : 'Archive unavailable.',
          { role: 'alert' },
        ),
        el('button', undefined, 'Retry graphs', { type: 'button', 'data-mywindsock-retry': '' }),
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
      if (section instanceof HTMLElement && section.dataset.mywindsockState === 'pending')
        visibility.observe(section)
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
  additions.observe(root, { childList: true, subtree: true })
  observe(root)
  return () => {
    root.removeEventListener('click', onClick)
    visibility.disconnect()
    additions.disconnect()
    for (const request of requests.values()) request.abort()
    for (const cleanup of mounted.values()) cleanup()
    requests.clear()
    mounted.clear()
  }
}
