import type {
  MyWindsockArchiveReference,
  MyWindsockGraphPlot,
  MyWindsockGraphs,
  MyWindsockGraphSeries,
} from '../../../util/mywindsock-graphs'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import { parseElapsedPauses, type ElapsedPause } from '../../../util/elapsed-axis'
import {
  isMyWindsockArchiveReference,
  MYWINDSOCK_GRAPH_CATEGORIES,
  parseMyWindsockGraphs,
} from '../../../util/mywindsock-graphs'
import { triText } from '../../../util/triathlon-i18n'
import { buildSelect, type Select } from '../../controls/select'
import { applyI18n, el, svg } from '../runtime/dom'
import { renderCartesian } from './mywindsock-graphs/cartesian'
import { GRAPH_DESCRIPTIONS } from './mywindsock-graphs/descriptions'
import { extent, number } from './mywindsock-graphs/plot-values'
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

const hasSeriesData = (series: MyWindsockGraphSeries): boolean =>
  series.points.some(point => point.y != null && (series.kind !== 'range' || point.upper != null))

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
  pauses: readonly ElapsedPause[],
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
  const graphLabel = (graph: MyWindsockGraphs['graphs'][number]): string => text(graph.label)
  let selected =
    visibleGraphs.find(graph => graph.key === (archive.cyclingCda ? 'cda' : 'virt_elev')) ??
    visibleGraphs[0]
  // Labels follow the locale, so a locale change rebuilds the picker.
  const buildPicker = (): Select<string> =>
    buildSelect({
      id: `${pickerId}-menu`,
      label: text('wind graph'),
      groups: [...Object.keys(MYWINDSOCK_GRAPH_CATEGORIES), 'Other'].flatMap(category => {
        const graphs = visibleGraphs.filter(graph => graph.category === category)
        return graphs.length
          ? [
              {
                label: text(category),
                options: graphs.map(graph => ({ value: graph.key, label: graphLabel(graph) })),
              },
            ]
          : []
      }),
      selected: selected.key,
      onSelect: key => {
        selected = visibleGraphs.find(graph => graph.key === key) ?? selected
        update()
      },
    })
  let picker = buildPicker()
  let disposePicker = picker.mount()
  pickerHost.replaceChildren(picker.element)
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
  const active = new Set<string>()
  const colors = new Map<string, string>()
  const color = (id: string): string => colors.get(id) ?? WIND_TRACE_COLORS[0]
  const updateStatus = (): void => {
    for (const [key, item] of descriptionItems) {
      item.textContent = text(GRAPH_DESCRIPTIONS[key])
      item.toggleAttribute('data-active', key === selected.key)
    }
    const description = descriptionItems.get(selected.key)
    if (description) picker.trigger.setAttribute('aria-describedby', description.id)
    else picker.trigger.removeAttribute('aria-describedby')
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
      if (plot.kind === 'cartesian')
        renderCartesian(panel, plot, active, color, text, elapsedTimeS, pauses, presentation())
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
    picker.setSelected(selected.key)
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
  const onLocale = (): void => {
    if (!host.isConnected) return
    disposePicker()
    picker = buildPicker()
    disposePicker = picker.mount()
    pickerHost.replaceChildren(picker.element)
    applyI18n(host, presentation())
    updateStatus()
    draw()
  }
  window.addEventListener('tri:locale', onLocale)
  update()
  return () => {
    disposePicker()
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
          activityId: section.dataset.mywindsockId!,
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
        parseElapsedPauses(section.dataset.mywindsockPauses),
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
