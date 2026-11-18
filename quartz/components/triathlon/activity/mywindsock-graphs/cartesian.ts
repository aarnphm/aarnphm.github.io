import type { MyWindsockGraphPlot, MyWindsockGraphSeries } from '../../../../util/mywindsock-graphs'
import type { TriathlonPresentation } from '../../../../util/triathlon-presentation'
import { elapsedAxis, formatElapsedPauses, type ElapsedPause } from '../../../../util/elapsed-axis'
import { addElapsedBreaks } from '../../../../util/triathlon-card/elapsed-chart'
import { createDomFactory, el, setText, svg } from '../../runtime/dom'
import { clock, extent, number, yDomain } from './plot-values'

export const renderCartesian = (
  host: HTMLElement,
  plot: MyWindsockGraphPlot,
  active: Set<string>,
  color: (id: string) => string,
  text: (key: string) => string,
  elapsedTimeS: number,
  pauses: readonly ElapsedPause[],
  presentation: TriathlonPresentation,
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
  // Elapsed x is in milliseconds. A recording pause takes no width; without pauses the map is
  // the linear one, so a ride with no pause renders the same paths.
  const shownPauses = elapsed
    ? elapsedAxis(xLow / 1000, xHigh / 1000, pauses).pauses
    : ([] as ElapsedPause[])
  const axis = elapsedAxis(
    xLow,
    xHigh,
    shownPauses.map(pause => ({ startS: pause.startS * 1000, endS: pause.endS * 1000 })),
  )
  const x = (value: number | string): number => axis.fraction(position(value)) * 100
  const insidePause = (at: number): boolean =>
    axis.pauses.some(pause => at > pause.startS && at < pause.endS)
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
          ...(shownPauses.length ? { 'data-domain-pauses': formatElapsedPauses(shownPauses) } : {}),
        }
      : {}),
  })
  const grid = svg('g', { class: 'tri-workspace-grid' })
  for (const y of [0, 25, 50, 75, 100]) grid.append(svg('line', { x1: 0, x2: 100, y1: y, y2: y }))
  graph.append(grid)
  if (shownPauses.length)
    addElapsedBreaks(
      createDomFactory(presentation),
      graph,
      elapsedAxis(xLow / 1000, xHigh / 1000, shownPauses),
      100,
    )
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
    let previousPosition = 0
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
      // Samples inside a removed pause have no place on the axis; the pen lifts at the join.
      const at = position(point.x)
      if (insidePause(at) || (continuous && axis.crossesPause(previousPosition, at))) {
        continuous = false
        closeArea()
        closeBand()
        if (insidePause(at)) continue
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
      previousPosition = at
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
    : (elapsed ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1]).map(axis.elapsedAt)
  // Labels keep elapsed clock time, so a tick past a pause includes the stopped time.
  for (const value of tickValues)
    ticks.append(
      el('span', undefined, formatX(value), {
        style: `inset-inline-start: ${categorical ? ((value - xLow) / (xHigh - xLow)) * 100 : x(value)}%`,
      }),
    )
  const cursor = svg('line', { x1: 0, x2: 0, y1: 0, y2: 100, class: 'tri-workspace-cursor' })
  graph.append(cursor)
  let fraction = 0
  // The readout doubles as the legend: colour key, series name, then the value under the cursor.
  // Its nodes are built once and rewritten in place, so a pointer move adds and removes no nodes.
  const items = series.map(line => {
    const item = el('span', 'tri-workspace-current-value')
    item.style.setProperty('--trace-color', color(line.id))
    const label = el('span', 'tri-mywindsock-legend-label')
    const value = document.createTextNode('')
    item.append(label, value)
    return { item, label, value }
  })
  values.replaceChildren(...items.map(entry => entry.item))
  const show = (next: number): void => {
    fraction = Math.max(0, Math.min(1, next))
    // The readout jumps by the stopped time across a join.
    const at = categorical ? xLow + fraction * (xHigh - xLow) : axis.elapsedAt(fraction)
    const descriptions = [`${text(plot.xLabel)} ${formatX(at)}`]
    setText(location, `${text(plot.xLabel).toLocaleLowerCase()} ${formatX(at)}`)
    for (const [index, line] of series.entries()) {
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
      const { item, label, value } = items[index]
      item.title = description
      item.setAttribute('aria-label', description)
      setText(label, text(line.label))
      if (value.data !== formatted) value.data = formatted
      descriptions.push(description)
    }
    graph.setAttribute('aria-valuenow', String(Math.round(fraction * 100)))
    graph.setAttribute('aria-valuetext', descriptions.join(' · '))
    cursor.setAttribute('x1', String(fraction * 100))
    cursor.setAttribute('x2', String(fraction * 100))
  }
  let pendingX: number | null = null
  graph.addEventListener('pointermove', event => {
    if (pendingX == null)
      requestAnimationFrame(() => {
        const clientX = pendingX
        pendingX = null
        if (clientX == null || !graph.isConnected) return
        const bounds = graph.getBoundingClientRect()
        if (bounds.width > 0) show((clientX - bounds.left) / bounds.width)
      })
    pendingX = event.clientX
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
