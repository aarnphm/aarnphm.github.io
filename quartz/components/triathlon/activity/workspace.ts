import type { StravaActivityDetail } from '../../../plugins/stores/strava'
import type { TriathlonContext } from '../runtime/context'
import { zoneClock, scrubDist } from '../../../util/triathlon-card'
import { triText } from '../../../util/triathlon-i18n'
import { gpsSegments } from '../maps/model'
import { el, svg } from '../runtime/dom'
import { activityScrubElapsedIndexAt, activityScrubIndexAt } from './analysis'
import { bindActivityComparisonGraph, type ActivityComparisonScrubState } from './comparison-graph'
import { mountActivityComparisonMap, type ActivityComparisonMapController } from './comparison-map'
import {
  workspaceTraceDomain,
  workspaceTracePaths,
  workspaceTraces,
  workspaceValueAt,
  type WorkspaceAxis,
} from './workspace-data'

const buildWorkspace = (
  host: HTMLElement,
  activity: StravaActivityDetail,
  context: TriathlonContext,
): (() => void) => {
  const presentation = context.presentation
  const text = (key: string): string => triText(presentation.locale, key)
  const d = activity
  const traces = workspaceTraces(d, presentation)
  const selected = new Set<string>(
    traces.some(trace => trace.id === 'elevation') ? ['elevation'] : [],
  )
  let axis: WorkspaceAxis = 'time'
  let range: StravaActivityDetail['analysisRanges'][number] | null = null
  let map: ActivityComparisonMapController | null = null
  const state: ActivityComparisonScrubState = { fraction: 0, selectedFraction: 0 }
  const maxTime = Math.max(
    d.elapsedTimeS,
    ...traces.map(trace => trace.samples.at(-1)?.elapsedS ?? 0),
    1,
  )
  const maxDistance = Math.max(
    d.distanceKm,
    ...traces.map(trace => trace.samples.at(-1)?.distanceKm ?? 0),
  )
  const hasDistance = maxDistance > 0
  const header = el('div', 'tri-workspace-summary')
  header.append(
    el(
      'span',
      undefined,
      `${scrubDist(presentation, d.distanceKm, d.sport)} · ${zoneClock(d.elapsedTimeS)} ${text('elapsed time')}`,
    ),
  )
  const toolbar = el('div', 'tri-workspace-toolbar')
  const layouts = el('div', 'tri-workspace-switches', undefined, {
    role: 'group',
    'aria-label': text('analysis layout'),
  })
  const axes = el('div', 'tri-workspace-switches', undefined, {
    role: 'group',
    'aria-label': text('graph axis'),
  })
  const stage = el('div', 'tri-workspace-stage', undefined, { 'data-layout': 'both' })
  const mapPanel = el('div', 'tri-workspace-map-panel')
  const mapHost = el('div', 'tri-workspace-map', undefined, {
    'aria-label': text('activity route'),
  })
  const attribution = el('div', 'tri-workspace-attribution')
  attribution.append(
    el('a', undefined, '© Mapbox', {
      href: 'https://www.mapbox.com/about/maps/',
      target: '_blank',
      rel: 'noopener',
    }),
    el('a', undefined, '© OpenStreetMap', {
      href: 'https://www.openstreetmap.org/copyright',
      target: '_blank',
      rel: 'noopener',
    }),
  )
  mapPanel.append(mapHost, attribution)
  const plotPanel = el('div', 'tri-workspace-plot-panel')
  const readout = el('div', 'tri-workspace-position')
  const graph = svg('svg', {
    class: 'tri-workspace-graph',
    viewBox: '0 0 100 100',
    preserveAspectRatio: 'none',
    role: 'slider',
    tabindex: 0,
    'aria-label': text('activity graph'),
    'aria-valuemin': 0,
    'aria-valuemax': 100,
    'aria-valuenow': 0,
  })
  const grid = svg('g', { class: 'tri-workspace-grid' })
  for (const y of [0, 25, 50, 75, 100]) grid.append(svg('line', { x1: 0, x2: 100, y1: y, y2: y }))
  const lines = svg('g', { class: 'tri-workspace-traces' })
  const cursor = svg('line', { class: 'tri-workspace-cursor', x1: 0, x2: 0, y1: 0, y2: 100 })
  graph.append(grid, lines, cursor)
  const ticks = el('div', 'tri-workspace-ticks')
  const scaleNote = el(
    'p',
    'tri-workspace-note',
    text('Each trace uses its own scale. Ranges and cursor values keep their original units.'),
  )
  plotPanel.append(readout, graph, ticks, scaleNote)
  const controls = el('div', 'tri-workspace-trace-controls', undefined, {
    role: 'group',
    'aria-label': text('graph overlays'),
  })
  const values = new Map<string, HTMLElement>()
  for (const trace of traces) {
    const isBase = trace.id === 'elevation'
    const button = el('button', 'tri-workspace-trace-toggle', undefined, {
      type: 'button',
      'data-workspace-trace': trace.id,
      'aria-pressed': String(isBase),
      'aria-label': text(trace.label),
      ...(isBase ? { disabled: '', title: text('elevation base') } : {}),
    })
    button.style.setProperty('--trace-color', trace.color)
    const label = el('span', 'tri-workspace-trace-label', text(trace.label))
    const value = el('span', 'tri-workspace-trace-value', '—')
    const [low, high] = workspaceTraceDomain(trace)
    const scale = el(
      'span',
      'tri-workspace-trace-scale',
      `${trace.format(low)}–${trace.format(high)}`,
    )
    button.append(label, value, scale)
    if (isBase || trace.estimated)
      button.append(
        el('span', 'tri-workspace-trace-source', text(isBase ? 'base' : 'Garden estimate')),
      )
    values.set(trace.id, value)
    controls.append(button)
  }
  const bounds = (): [number, number] =>
    axis === 'time'
      ? [range?.startElapsedS ?? 0, range?.endElapsedS ?? maxTime]
      : [range?.startDistanceKm ?? 0, range?.endDistanceKm ?? maxDistance]
  const formatAxis = (value: number): string =>
    axis === 'time' ? zoneClock(value) : scrubDist(presentation, value, d.sport)
  const sampleAt = (fraction: number): { elapsedS: number; distanceKm: number } => {
    const [start, end] = bounds()
    const position = start + fraction * (end - start)
    const samples = d.route.length
      ? d.route.map(p => ({ elapsedS: p.elapsedS, d: p.d }))
      : (traces[0]?.samples.map(p => ({ elapsedS: p.elapsedS, d: p.distanceKm })) ?? [])
    const index =
      axis === 'time'
        ? activityScrubElapsedIndexAt(samples, position)
        : activityScrubIndexAt(samples, position)
    const point = samples[index]
    return {
      elapsedS: axis === 'time' ? position : (point?.elapsedS ?? 0),
      distanceKm: axis === 'distance' ? position : (point?.d ?? 0),
    }
  }
  const show = (fraction: number): void => {
    state.fraction = fraction
    const point = sampleAt(fraction)
    const location = `${zoneClock(point.elapsedS)}${hasDistance ? ` · ${scrubDist(presentation, point.distanceKm, d.sport)}` : ''}`
    readout.textContent = location
    const descriptions = [location]
    for (const trace of traces) {
      const value = workspaceValueAt(trace, point.elapsedS)
      const formatted = value == null ? '—' : trace.format(value)
      const target = values.get(trace.id)
      if (target) target.textContent = formatted
      if (selected.has(trace.id)) descriptions.push(`${text(trace.label)} ${formatted}`)
    }
    const x = (fraction * 100).toFixed(3)
    cursor.setAttribute('x1', x)
    cursor.setAttribute('x2', x)
    graph.setAttribute('aria-valuenow', `${Math.round(fraction * 100)}`)
    graph.setAttribute('aria-valuetext', descriptions.join(' · '))
    map?.showCursors(point.distanceKm)
  }
  const draw = (): void => {
    const [start, end] = bounds()
    lines.replaceChildren()
    for (const trace of traces) {
      if (!selected.has(trace.id)) continue
      const paths = workspaceTracePaths(trace, axis, start, end)
      const group = svg('g', { 'data-overlay': trace.id, style: `--trace-color: ${trace.color}` })
      if (trace.id === 'elevation')
        group.append(svg('path', { class: 'tri-workspace-elevation', d: paths.area }))
      group.append(svg('path', { class: 'tri-workspace-line', d: paths.line }))
      lines.append(group)
    }
    ticks.replaceChildren(
      ...[0, 0.25, 0.5, 0.75, 1].map(fraction =>
        el('span', undefined, formatAxis(start + fraction * (end - start))),
      ),
    )
    show(state.selectedFraction ?? state.fraction)
  }
  const layoutButtons = new Map<string, HTMLElement>()
  for (const [mode, label] of [
    ['both', 'map + graphs'],
    ['graphs', 'graphs'],
    ['map', 'map'],
  ]) {
    const button = el('button', undefined, text(label), {
      type: 'button',
      'data-workspace-layout': mode,
      'aria-pressed': String(mode === 'both'),
    })
    layoutButtons.set(mode, button)
    layouts.append(button)
  }
  const axisButtons = new Map<WorkspaceAxis, HTMLElement>()
  for (const mode of ['time', 'distance'] satisfies WorkspaceAxis[]) {
    const button = el('button', undefined, text(mode), {
      type: 'button',
      'data-workspace-axis': mode,
      'aria-pressed': String(mode === axis),
      ...(mode === 'distance' && !hasDistance ? { disabled: '' } : {}),
    })
    axisButtons.set(mode, button)
    axes.append(button)
  }
  const rangeSelect = document.createElement('select')
  rangeSelect.className = 'tri-workspace-range'
  rangeSelect.setAttribute('aria-label', text('activity range'))
  rangeSelect.add(new Option(text('entire activity'), ''))
  for (const [kind, label] of [
    ['lap', 'Laps'],
    ['climb', 'Summit Freeride'],
    ['segment', 'Segments'],
  ]) {
    const group = document.createElement('optgroup')
    group.label = text(label)
    for (const [index, candidate] of d.analysisRanges.entries()) {
      if (candidate.kind !== kind || candidate.endElapsedS <= candidate.startElapsedS) continue
      group.append(
        new Option(
          `${text(candidate.kind)} · ${candidate.label} · ${zoneClock(candidate.durationS)}`,
          `${index}`,
        ),
      )
    }
    if (group.children.length) rangeSelect.append(group)
  }
  toolbar.append(layouts, axes, rangeSelect)
  stage.append(mapPanel, plotPanel)
  host.append(header, toolbar, stage, controls)
  if (!traces.some(trace => trace.id === 'elevation'))
    header.append(
      el(
        'span',
        'tri-workspace-note',
        text('No recorded elevation. Enable an available trace below.'),
      ),
    )
  if (!traces.length)
    host.append(el('p', 'tri-workspace-note', text('No recorded telemetry for this activity.')))
  const onClick = (event: MouseEvent): void => {
    if (!(event.target instanceof Element)) return
    const button = event.target.closest<HTMLButtonElement>('button')
    if (!button || !host.contains(button)) return
    const traceId = button.dataset.workspaceTrace
    if (traceId && traceId !== 'elevation') {
      if (selected.has(traceId)) selected.delete(traceId)
      else selected.add(traceId)
      button.setAttribute('aria-pressed', String(selected.has(traceId)))
      draw()
    }
    const layout = button.dataset.workspaceLayout
    if (layout) {
      stage.dataset.layout = layout
      for (const [key, item] of layoutButtons)
        item.setAttribute('aria-pressed', String(key === layout))
    }
    const nextAxis = button.dataset.workspaceAxis
    if (nextAxis === 'time' || nextAxis === 'distance') {
      axis = nextAxis
      for (const [key, item] of axisButtons) item.setAttribute('aria-pressed', String(key === axis))
      draw()
    }
  }
  const onRange = (): void => {
    range = rangeSelect.value ? d.analysisRanges[Number(rangeSelect.value)] : null
    axisButtons
      .get('distance')
      ?.toggleAttribute(
        'disabled',
        !hasDistance || (range != null && range.endDistanceKm <= range.startDistanceKm),
      )
    if (axis === 'distance' && range && range.endDistanceKm <= range.startDistanceKm) {
      axis = 'time'
      for (const [key, item] of axisButtons) item.setAttribute('aria-pressed', String(key === axis))
    }
    state.selectedFraction = 0
    draw()
  }
  host.addEventListener('click', onClick)
  rangeSelect.addEventListener('change', onRange)
  const cleanupGraph = bindActivityComparisonGraph(
    graph,
    state,
    show,
    (_source, restore) => restore(),
    (_source, restore) => restore(),
    () => {},
    0.005,
  )
  draw()
  if (gpsSegments(d).length)
    map = mountActivityComparisonMap(mapHost, [d], {
      unavailableText: text('map unavailable'),
      onScrub: distanceKm => {
        const [start, end] = bounds()
        const index = activityScrubIndexAt(d.route, distanceKm)
        const point = d.route[index]
        const position = axis === 'time' ? point?.elapsedS : distanceKm
        if (position == null) return
        show(Math.min(1, Math.max(0, (position - start) / Math.max(end - start, 0.001))))
      },
      onLeave: () => show(state.selectedFraction ?? state.fraction),
    })
  else {
    mapHost.classList.add('tri-workspace-map--empty')
    mapHost.textContent = text('No recorded GPS route for this activity.')
    attribution.hidden = true
  }
  return () => {
    cleanupGraph()
    map?.destroy()
    host.removeEventListener('click', onClick)
    rangeSelect.removeEventListener('change', onRange)
  }
}

export const setupActivityWorkspace = (context: TriathlonContext): (() => void) => {
  let dialog: HTMLDialogElement | null = null
  let disposeView: (() => void) | null = null
  let generation = 0
  const close = (): void => {
    generation += 1
    disposeView?.()
    disposeView = null
    dialog?.close()
    dialog?.remove()
    dialog = null
  }
  const open = async (button: HTMLButtonElement): Promise<void> => {
    const id = button.dataset.activityAnalyze
    const path =
      button.closest<HTMLElement>('[data-detail-path]')?.dataset.detailPath ??
      context.root?.dataset.detailPath ??
      '/static/strava-detail.json'
    if (!id || !path) return
    close()
    const current = document.createElement('dialog')
    current.className = 'tri-workspace'
    current.setAttribute('aria-label', context.formatter.text('activity analysis'))
    const header = el('div', 'tri-workspace-head')
    const title = el('div', 'tri-workspace-title', context.formatter.text('loading activity'))
    const closeButton = el('button', 'tri-workspace-close', '×', {
      type: 'button',
      'aria-label': context.formatter.text('Close'),
    })
    header.append(title, closeButton)
    const content = el('div', 'tri-workspace-content', context.formatter.text('loading activity'), {
      'aria-busy': 'true',
    })
    current.append(header, content)
    document.body.append(current)
    dialog = current
    current.showModal()
    closeButton.addEventListener('click', close)
    current.addEventListener('cancel', event => {
      event.preventDefault()
      close()
    })
    // The dialog owns keyboard input while open, including Escape before the underlying panels.
    current.addEventListener('keydown', event => event.stopPropagation())
    const request = generation
    const result = await context.resources.detail.load(path)
    if (context.signal.aborted || dialog !== current || request !== generation) return
    content.setAttribute('aria-busy', 'false')
    const activity = result.status === 'ready' ? result.value.details[id] : null
    if (!activity) {
      content.textContent = context.formatter.text('activity data unavailable')
      return
    }
    title.replaceChildren(
      el('span', 'tri-workspace-date', context.formatter.shortDate(activity.date)),
      el('h2', undefined, activity.name || activity.sport),
    )
    content.replaceChildren()
    disposeView = buildWorkspace(content, activity, context)
  }
  const onClick = (event: MouseEvent): void => {
    if (!(event.target instanceof Element)) return
    const button = event.target.closest<HTMLButtonElement>('[data-activity-analyze]')
    if (!button) return
    event.stopPropagation()
    void open(button)
  }
  context.scope.addEventListener('click', onClick)
  return () => {
    close()
    context.scope.removeEventListener('click', onClick)
  }
}
