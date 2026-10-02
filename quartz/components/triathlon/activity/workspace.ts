import type { StravaActivityDetail } from '../../../plugins/stores/strava'
import type { TriathlonContext } from '../runtime/context'
import {
  zoneClock,
  scrubDist,
  cyclingWorkoutLaps,
  cyclingWorkoutPowerSummary,
} from '../../../util/triathlon-card'
import { triText } from '../../../util/triathlon-i18n'
import { gpsSegments } from '../maps/model'
import { el, svg } from '../runtime/dom'
import { activityScrubElapsedIndexAt, activityScrubIndexAt } from './analysis'
import { bindActivityComparisonGraph, type ActivityComparisonScrubState } from './comparison-graph'
import { mountActivityComparisonMap, type ActivityComparisonMapController } from './comparison-map'
import { buildActivityMapControls } from './map-controls'
import { buildActivityRangePicker } from './range-picker'
import {
  workspaceTraceDomain,
  workspaceTracePaths,
  workspaceTraces,
  workspaceValueAt,
  type WorkspaceAxis,
  type WorkspaceTrace,
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
  const laps = cyclingWorkoutLaps(d)
  const lapColor = 'color-mix(in srgb, var(--tri-bike) 55%, var(--dark))'
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
  const axes = el('div', 'tri-workspace-switches', undefined, {
    role: 'group',
    'aria-label': text('graph axis'),
  })
  const stage = el('div', 'tri-workspace-stage')
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
  const mapControls = buildActivityMapControls(text)
  mapPanel.append(mapHost, mapControls.element, attribution)
  const plotPanel = el('div', 'tri-workspace-plot-panel', undefined, {
    role: 'region',
    'aria-label': text('activity graph axes'),
    tabindex: '0',
  })
  const scrollHint = el('span', 'tri-workspace-scroll-hint', text('scroll to see all axes'))
  scrollHint.hidden = true
  const readout = el('div', 'tri-workspace-position')
  const position = el('span', 'tri-workspace-location')
  const currentValues = el('div', 'tri-workspace-values')
  readout.append(position, currentValues)
  const chart = el('div', 'tri-workspace-chart')
  const startAxes = el('div', 'tri-workspace-y-axes tri-workspace-y-axes--start')
  const endAxes = el('div', 'tri-workspace-y-axes tri-workspace-y-axes--end')
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
  const lapBands = svg('g', { class: 'tri-workspace-laps' })
  const lapAverage = svg('line', {
    class: 'tri-workspace-lap-average',
    x1: 0,
    x2: 100,
    y1: 100,
    y2: 100,
  })
  const lapHighlight = svg('rect', { class: 'tri-workspace-lap-highlight' })
  const cursor = svg('line', { class: 'tri-workspace-cursor', x1: 0, x2: 0, y1: 0, y2: 100 })
  graph.append(lapBands, grid, lines, lapAverage, lapHighlight, cursor)
  const lapLabels = el('div', 'tri-workspace-lap-labels', undefined, { 'aria-hidden': 'true' })
  const ticks = el('div', 'tri-workspace-ticks')
  const plotHead = el('div', 'tri-workspace-plot-head')
  const lapStats = el('div', 'tri-workspace-lap-stats')
  plotHead.append(readout, axes, lapStats)
  chart.append(plotHead, startAxes, graph, endAxes, lapLabels, ticks)
  plotPanel.append(chart)
  const controls = el('div', 'tri-workspace-trace-controls', undefined, {
    role: 'group',
    'aria-label': text('graph overlays'),
  })
  for (const trace of traces) {
    if (trace.id === 'elevation') continue
    const button = el('button', 'tri-workspace-trace-toggle', undefined, {
      type: 'button',
      'data-workspace-trace': trace.id,
      'aria-pressed': 'false',
      'aria-label': text(trace.label).toLocaleLowerCase(),
      ...(trace.estimated ? { title: text('Garden estimate') } : {}),
    })
    button.style.setProperty('--trace-color', trace.color)
    const label = el('span', 'tri-workspace-trace-label', text(trace.label).toLocaleLowerCase())
    button.append(label)
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
    const lap = laps.find(
      lap =>
        point.elapsedS >= lap.range.startElapsedS &&
        (point.elapsedS < lap.range.endElapsedS ||
          (lap === laps.at(-1) && point.elapsedS === lap.range.endElapsedS)),
    )
    const lapLabel = lap ? `${text('lap')} ${lap.index}` : ''
    position.textContent = `${location}${lap ? ` · ${lapLabel}` : ''}`
    currentValues.replaceChildren()
    const descriptions = [location]
    if (lap) descriptions.push(lapLabel)
    const addValue = (id: string, color: string, label: string, formatted: string): void => {
      const description = `${label} ${formatted}`
      const item = el('span', 'tri-workspace-current-value', formatted, {
        'data-cursor-trace': id,
        title: description,
        'aria-label': description,
      })
      item.style.setProperty('--trace-color', color)
      currentValues.append(item)
      descriptions.push(description)
    }
    if (lap?.powerWatts != null)
      addValue(
        'laps',
        lapColor,
        `${lapLabel} ${text('average power')}`,
        `${Math.round(lap.powerWatts)} W`,
      )
    for (const trace of traces) {
      if (!selected.has(trace.id)) continue
      const value = workspaceValueAt(trace, point.elapsedS)
      const formatted = value == null ? '—' : trace.format(value)
      const label = text(trace.label)
      const source = trace.estimated ? ` · ${text('Garden estimate')}` : ''
      addValue(trace.id, trace.color, `${label}${source}`, formatted)
    }
    let activeBand: SVGRectElement | null = null
    for (const element of lapBands.querySelectorAll<SVGRectElement>('rect[data-lap]')) {
      const active = element.getAttribute('data-lap') === lap?.range.id
      element.toggleAttribute('data-active', active)
      if (active) activeBand = element
    }
    lapHighlight.style.display = activeBand && activeBand.height.baseVal.value > 0 ? '' : 'none'
    for (const attribute of ['x', 'y', 'width', 'height']) {
      const value = activeBand?.getAttribute(attribute)
      if (value != null) lapHighlight.setAttribute(attribute, value)
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
    const visibleLaps = laps.filter(lap => {
      const left = axis === 'time' ? lap.range.startElapsedS : lap.range.startDistanceKm
      const right = axis === 'time' ? lap.range.endElapsedS : lap.range.endDistanceKm
      return right > start && left < end
    })
    const lapSummary = cyclingWorkoutPowerSummary(visibleLaps)
    lapBands.replaceChildren()
    lapLabels.replaceChildren()
    lapLabels.hidden = !visibleLaps.length
    lapAverage.style.display = lapSummary ? '' : 'none'
    lapStats.replaceChildren()
    if (lapSummary) {
      const y = 100 - (lapSummary.average / lapSummary.maximum) * 100
      lapAverage.setAttribute('y1', String(y))
      lapAverage.setAttribute('y2', String(y))
      lapStats.append(
        ...[
          `${text('highest')} ${Math.round(lapSummary.highest)} W`,
          `${text('avg')} ${Math.round(lapSummary.average)} W`,
          `${text('lowest')} ${Math.round(lapSummary.lowest)} W`,
        ].map(value => el('span', undefined, value)),
      )
      lapStats.title = `${text('Laps').toLocaleLowerCase()} · ${text('average power')}`
    }
    for (const lap of visibleLaps) {
      const lapStart = axis === 'time' ? lap.range.startElapsedS : lap.range.startDistanceKm
      const lapEnd = axis === 'time' ? lap.range.endElapsedS : lap.range.endDistanceKm
      const left = Math.max(start, lapStart)
      const right = Math.min(end, lapEnd)
      if (right <= left) continue
      const x = ((left - start) / (end - start)) * 100
      const width = ((right - left) / (end - start)) * 100
      const height =
        lapSummary && lap.powerWatts != null ? (lap.powerWatts / lapSummary.maximum) * 100 : 0
      const band = svg('rect', {
        x,
        y: 100 - height,
        width,
        height,
        'data-lap': lap.range.id,
        'data-lap-watts': lap.powerWatts ?? '',
        class: 'tri-workspace-lap',
      })
      const title = svg('title', {})
      title.textContent = `${text('lap')} ${lap.index} · ${zoneClock(lap.range.durationS)}${lap.powerWatts == null ? '' : ` · ${Math.round(lap.powerWatts)} W`}`
      band.append(title)
      lapBands.append(band, svg('line', { x1: x, x2: x, y1: 0, y2: 100 }))
      lapLabels.append(
        el('span', undefined, String(lap.index), {
          style: `inset-inline-start:${x + width / 2}%;inline-size:${width}%`,
        }),
      )
    }
    startAxes.replaceChildren()
    endAxes.replaceChildren()
    const addAxis = (
      parent: HTMLElement,
      trace: Pick<WorkspaceTrace, 'id' | 'label' | 'format'>,
      low: number,
      high: number,
    ): void => {
      const label = `${text(trace.label)} ${text('graph axis')}`
      const verticalAxis = el('div', 'tri-workspace-y-axis', undefined, {
        role: 'group',
        'data-axis-trace': trace.id,
        'aria-label': label,
        title: label,
      })
      const scale = el('div', 'tri-workspace-y-ticks')
      const fractions = low === high ? [0.5] : [0, 0.25, 0.5, 0.75, 1]
      const labels = fractions.map(fraction => {
        const formatted = trace.format(high - fraction * (high - low))
        return formatted.match(/^[+-]?\d[\d,]*(?:\.\d+)?/)?.[0] ?? formatted
      })
      verticalAxis.style.setProperty(
        '--tri-workspace-axis-chars',
        String(Math.max(...labels.map(value => value.length))),
      )
      scale.append(
        ...fractions.map((fraction, index) =>
          el('span', undefined, labels[index], { style: `inset-block-start: ${fraction * 100}%` }),
        ),
      )
      verticalAxis.append(scale)
      parent.append(verticalAxis)
    }
    const activeTraces = [...selected].flatMap(id => traces.filter(trace => trace.id === id))
    const firstTrace = activeTraces[0]
    if (firstTrace) addAxis(startAxes, firstTrace, ...workspaceTraceDomain(firstTrace))
    if (lapSummary)
      addAxis(
        endAxes,
        { id: 'laps', label: 'lap average power', format: value => `${Math.round(value)} W` },
        0,
        lapSummary.maximum,
      )
    for (const trace of activeTraces.slice(1)) {
      const side = startAxes.childElementCount <= endAxes.childElementCount ? startAxes : endAxes
      addAxis(side, trace, ...workspaceTraceDomain(trace))
    }
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
  const rangePicker = buildActivityRangePicker(d, text, nextRange => {
    range = nextRange
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
  })
  toolbar.append(header, rangePicker.element)
  stage.append(plotPanel, scrollHint, controls)
  host.append(toolbar, mapPanel, stage)
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
    const nextAxis = button.dataset.workspaceAxis
    if (nextAxis === 'time' || nextAxis === 'distance') {
      axis = nextAxis
      for (const [key, item] of axisButtons) item.setAttribute('aria-pressed', String(key === axis))
      draw()
    }
  }
  host.addEventListener('click', onClick)
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
  const resize = new ResizeObserver(() => {
    scrollHint.hidden = chart.scrollWidth <= plotPanel.clientWidth
  })
  resize.observe(plotPanel)
  resize.observe(chart)
  if (gpsSegments(d).length)
    map = mountActivityComparisonMap(mapHost, [d], {
      unavailableText: text('map unavailable'),
      distance: presentation.distance,
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
    resize.disconnect()
    cleanupGraph()
    map?.destroy()
    host.removeEventListener('click', onClick)
    rangePicker.dispose()
    mapControls.dispose()
  }
}

export const setupActivityWorkspace = (context: TriathlonContext): (() => void) => {
  type WorkspaceView = {
    button: HTMLButtonElement
    owner: HTMLElement
    region: HTMLElement
    siblings: HTMLElement[]
    scrollHost: HTMLElement | null
    scrollTop: number
    dispose: (() => void) | null
    observer: MutationObserver
  }
  let active: WorkspaceView | null = null
  let loadingButton: HTMLButtonElement | null = null
  let generation = 0
  let pendingUpdate = Promise.resolve()
  let moving: {
    transition: ViewTransition
    surface: HTMLElement
    head: HTMLElement | null
  } | null = null
  const clearTransition = (): void => {
    moving?.surface.removeAttribute('data-activity-transition')
    moving?.head?.removeAttribute('data-activity-transition-head')
    document.documentElement.classList.remove('tri-activity-transition')
    moving = null
  }
  const morph = async (owner: HTMLElement, update: () => void, animate: boolean): Promise<void> => {
    moving?.transition.skipTransition()
    await pendingUpdate
    clearTransition()
    if (
      !animate ||
      !owner.isConnected ||
      window.matchMedia('(prefers-reduced-motion: reduce)').matches ||
      typeof document.startViewTransition !== 'function'
    ) {
      update()
      return
    }
    const surface = owner.closest<HTMLElement>('.tri-analytics') ?? owner
    const head = owner.querySelector<HTMLElement>(':scope > .tri-pop-head, :scope > .tri-act-head')
    surface.setAttribute('data-activity-transition', '')
    head?.setAttribute('data-activity-transition-head', '')
    document.documentElement.classList.add('tri-activity-transition')
    const transition = document.startViewTransition(update)
    moving = { transition, surface, head }
    pendingUpdate = transition.updateCallbackDone.catch(() => undefined)
    void transition.ready.catch(() => undefined)
    void transition.finished
      .catch(() => undefined)
      .then(() => {
        if (moving?.transition === transition) clearTransition()
      })
    await pendingUpdate
  }
  const close = async (animate = false, restoreFocus = false): Promise<void> => {
    generation += 1
    loadingButton?.removeAttribute('aria-busy')
    loadingButton = null
    const view = active
    active = null
    if (!view) return
    view.observer.disconnect()
    await morph(
      view.owner,
      () => {
        view.dispose?.()
        view.region.remove()
        for (const sibling of view.siblings) sibling.removeAttribute('data-workspace-hidden')
        view.owner.classList.remove('tri-activity-analysis-open')
        view.button.setAttribute('aria-expanded', 'false')
        view.button.removeAttribute('aria-controls')
        if (view.scrollHost) view.scrollHost.scrollTop = view.scrollTop
        if (restoreFocus && view.button.isConnected) view.button.focus({ preventScroll: true })
      },
      animate,
    )
  }
  const open = async (button: HTMLButtonElement, animate: boolean): Promise<void> => {
    if (active?.button === button || loadingButton === button) {
      await close(animate, true)
      return
    }
    await close()
    const id = button.dataset.activityAnalyze
    const owner =
      button.closest<HTMLElement>('.tri-act') ?? button.closest<HTMLElement>('.tri-pop-card')
    const path =
      button.closest<HTMLElement>('[data-detail-path]')?.dataset.detailPath ??
      context.root?.dataset.detailPath ??
      '/static/strava-detail.json'
    if (!id || !path || !owner) return
    loadingButton = button
    button.setAttribute('aria-busy', 'true')
    const request = ++generation
    const result = await context.resources.detail.load(path)
    if (context.signal.aborted || !owner.isConnected || request !== generation) return
    button.removeAttribute('aria-busy')
    loadingButton = null
    const activity = result.status === 'ready' ? result.value.details[id] : null
    if (activity && !gpsSegments(activity).length) {
      button.disabled = true
      button.title = context.formatter.text('No recorded GPS route for this activity.')
      return
    }
    const region = el('section', 'tri-workspace', undefined, {
      id: `tri-workspace-${id}`,
      'aria-label': context.formatter.text('activity analysis'),
      tabindex: '-1',
    })
    const content = el('div', 'tri-workspace-content')
    region.append(content)
    const head = owner.querySelector(':scope > .tri-pop-head, :scope > .tri-act-head')
    const siblings = Array.from(owner.children).filter(
      (child): child is HTMLElement => child instanceof HTMLElement && child !== head,
    )
    const scrollHost = owner.closest<HTMLElement>('.tri-ana-body')
    const observer = new MutationObserver(() => {
      if (active?.owner === owner && !owner.isConnected) void close()
    })
    if (owner.parentElement) observer.observe(owner.parentElement, { childList: true })
    const view: WorkspaceView = {
      button,
      owner,
      region,
      siblings,
      scrollHost,
      scrollTop: scrollHost?.scrollTop ?? 0,
      dispose: null,
      observer,
    }
    active = view
    await morph(
      owner,
      () => {
        if (active !== view || context.signal.aborted) return
        for (const sibling of siblings) sibling.setAttribute('data-workspace-hidden', '')
        owner.classList.add('tri-activity-analysis-open')
        owner.append(region)
        button.setAttribute('aria-expanded', 'true')
        button.setAttribute('aria-controls', region.id)
        if (scrollHost) scrollHost.scrollTop = 0
        if (activity) view.dispose = buildWorkspace(content, activity, context)
        else content.textContent = context.formatter.text('activity data unavailable')
        region.focus({ preventScroll: true })
      },
      animate,
    )
  }
  const onClick = (event: MouseEvent): void => {
    if (!(event.target instanceof Element)) return
    const button = event.target.closest<HTMLButtonElement>('[data-activity-analyze]')
    if (!button) {
      if (event.target.closest('.tri-ana-close, .tri-ana-back')) void close()
      return
    }
    if (button.disabled) return
    event.stopPropagation()
    void open(button, event.detail > 0)
  }
  const onKey = (event: KeyboardEvent): void => {
    if (
      event.key !== 'Escape' ||
      !active ||
      !(event.target instanceof Node) ||
      !active.owner.contains(event.target)
    )
      return
    if (active.region.querySelector('.tri-workspace-range-trigger[aria-expanded="true"]')) return
    event.preventDefault()
    event.stopImmediatePropagation()
    void close(false, true)
  }
  const onPresentationChange = (): void => {
    void close()
  }
  context.scope.addEventListener('click', onClick)
  context.scope.addEventListener('keydown', onKey, true)
  window.addEventListener('tri:unit', onPresentationChange)
  window.addEventListener('tri:locale', onPresentationChange)
  return () => {
    void close()
    moving?.transition.skipTransition()
    context.scope.removeEventListener('click', onClick)
    context.scope.removeEventListener('keydown', onKey, true)
    window.removeEventListener('tri:unit', onPresentationChange)
    window.removeEventListener('tri:locale', onPresentationChange)
  }
}
