import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import { cyclingWattsCellText } from '../../../util/triathlon-card'

export function setupCyclingWattsCharts(
  scope: HTMLElement,
  presentation: () => TriathlonPresentation,
): () => void {
  let activeCell: SVGRectElement | null = null
  let activeChart: HTMLElement | null = null
  let focusedGraph: SVGSVGElement | null = null
  const cellCache = new WeakMap<SVGSVGElement, SVGRectElement[]>()
  const cellsFor = (graph: SVGSVGElement): SVGRectElement[] => {
    const cached = cellCache.get(graph)
    if (cached) return cached
    const cells = Array.from(graph.querySelectorAll<SVGRectElement>('.tri-cycling-watts-heat-cell'))
    cellCache.set(graph, cells)
    return cells
  }
  const clear = (): void => {
    activeCell?.classList.remove('tri-cycling-watts-heat-cell--active')
    activeChart?.classList.remove('tri-chart--hover')
    const readout = activeChart?.querySelector<HTMLElement>('.tri-cycling-watts-readout')
    if (readout) readout.hidden = true
    activeCell = null
    activeChart = null
  }
  const show = (graph: SVGSVGElement, index: number): void => {
    const chart = graph.closest<HTMLElement>('.tri-cycling-mode-chart')
    if (!chart || chart.dataset.cyclingChartMode !== 'power') {
      clear()
      return
    }
    const readout = chart.querySelector<HTMLElement>('.tri-cycling-watts-readout')
    const cells = cellsFor(graph)
    const selected = Math.max(0, Math.min(cells.length - 1, index))
    const cell = cells[selected]
    if (!cell || !readout) return
    const side = cell.dataset.side
    if (side !== 'left' && side !== 'right' && side !== 'single') return
    const value = cyclingWattsCellText(
      {
        count: Number(cell.dataset.samples),
        minWatts: Number(cell.dataset.wattsMin),
        maxWatts: Number(cell.dataset.wattsMax),
        value: Number(cell.dataset.value),
        label: cell.dataset.valueLabel ?? '',
        side,
      },
      chart.dataset.triTrace === 'power-balance',
      presentation().locale,
    )
    if (activeCell !== cell) clear()
    activeCell = cell
    activeChart = chart
    cell.classList.add('tri-cycling-watts-heat-cell--active')
    chart.classList.add('tri-chart--hover')
    readout.textContent = value
    readout.hidden = false
    graph.setAttribute('aria-valuenow', String(selected))
    graph.setAttribute('aria-valuetext', value)
  }
  const restoreFocus = (): void => {
    if (
      focusedGraph?.isConnected &&
      focusedGraph.closest<HTMLElement>('.tri-cycling-mode-chart')?.dataset.cyclingChartMode ===
        'power'
    )
      show(focusedGraph, Number(focusedGraph.getAttribute('aria-valuenow')))
    else clear()
  }
  const pointer = (event: PointerEvent): void => {
    const cell =
      event.target instanceof Element
        ? event.target.closest<SVGRectElement>('.tri-cycling-watts-heat-cell')
        : null
    const graph = cell?.closest<SVGSVGElement>('.tri-cycling-watts-heatmap')
    if (!cell || !graph) {
      if (activeCell) restoreFocus()
      return
    }
    show(graph, cellsFor(graph).indexOf(cell))
  }
  const focus = (event: FocusEvent): void => {
    if (
      !(event.target instanceof SVGSVGElement) ||
      !event.target.matches('.tri-cycling-watts-heatmap')
    )
      return
    focusedGraph = event.target
    show(focusedGraph, Number(focusedGraph.getAttribute('aria-valuenow')))
  }
  const blur = (event: FocusEvent): void => {
    if (event.target !== focusedGraph) return
    focusedGraph = null
    clear()
  }
  const key = (event: KeyboardEvent): void => {
    if (
      !(event.target instanceof SVGSVGElement) ||
      !event.target.matches('.tri-cycling-watts-heatmap') ||
      event.altKey ||
      event.ctrlKey ||
      event.metaKey
    )
      return
    const graph = event.target
    const index = Number(graph.getAttribute('aria-valuenow'))
    let next: number
    if (event.key === 'ArrowRight' || event.key === 'ArrowUp') next = index + 1
    else if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') next = index - 1
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = cellsFor(graph).length - 1
    else if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      graph.blur()
      clear()
      return
    } else return
    event.preventDefault()
    event.stopPropagation()
    show(graph, next)
  }
  const mode = (event: MouseEvent): void => {
    if (!(event.target instanceof Element) || !event.target.closest('.tri-cycling-chart-mode'))
      return
    focusedGraph = null
    clear()
  }
  scope.addEventListener('pointermove', pointer)
  scope.addEventListener('pointerdown', pointer)
  scope.addEventListener('pointerleave', restoreFocus)
  scope.addEventListener('pointercancel', restoreFocus)
  scope.addEventListener('focusin', focus)
  scope.addEventListener('focusout', blur)
  scope.addEventListener('keydown', key)
  scope.addEventListener('click', mode)
  window.addEventListener('tri:locale', restoreFocus)
  return () => {
    clear()
    scope.removeEventListener('pointermove', pointer)
    scope.removeEventListener('pointerdown', pointer)
    scope.removeEventListener('pointerleave', restoreFocus)
    scope.removeEventListener('pointercancel', restoreFocus)
    scope.removeEventListener('focusin', focus)
    scope.removeEventListener('focusout', blur)
    scope.removeEventListener('keydown', key)
    scope.removeEventListener('click', mode)
    window.removeEventListener('tri:locale', restoreFocus)
  }
}
