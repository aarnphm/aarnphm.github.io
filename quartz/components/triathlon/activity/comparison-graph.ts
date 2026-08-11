import type { ActivityComparisonMetric } from '../../../util/triathlon-card'
import { activityComparisonFractionForKey } from '../../../util/triathlon-card'

export const activityComparisonMetric = (
  value: string | undefined,
): ActivityComparisonMetric | null => {
  if (value === 'elevation') return value
  if (value === 'speed') return value
  if (value === 'hr') return value
  if (value === 'power') return value
  if (value === 'cadence') return value
  if (value === 'respiration') return value
  if (value === 'temperature') return value
  if (value === 'skin-temperature') return value
  if (value === 'stride-length') return value
  if (value === 'ground-contact-time') return value
  if (value === 'vertical-oscillation') return value
  if (value === 'swim-pace') return value
  if (value === 'stroke-rate') return value
  return null
}

export const activityComparisonMetricLabel = (metric: ActivityComparisonMetric): string =>
  metric === 'hr'
    ? 'heart rate'
    : metric === 'swim-pace'
      ? 'pace /100m'
      : metric.replaceAll('-', ' ')

export type ActivityComparisonScrubState = { fraction: number; selectedFraction?: number }

export type ActivityComparisonSelectionRange = { startFraction: number; endFraction: number }

export const positionActivityComparisonCursor = (graph: SVGElement, fraction: number): void => {
  const normalized = Math.min(1, Math.max(0, fraction))
  const x = normalized * 100
  const cursor = graph.querySelector<SVGElement>('.tri-compare-cursor')
  cursor?.setAttribute('x1', x.toFixed(2))
  cursor?.setAttribute('x2', x.toFixed(2))
}

export type ActivityComparisonDragSelection = {
  preview: (anchorFraction: number, focusFraction: number) => void
  commit: () => void
  clear: () => void
  restore: () => void
}

export const bindActivityComparisonGraph = (
  graph: SVGElement,
  state: ActivityComparisonScrubState,
  show: (fraction: number) => void,
  activate: (source: SVGElement, restore: () => void) => void,
  render: (source: SVGElement, restore: () => void) => void,
  deactivate: (source: SVGElement) => void,
  keyboardStep = 0.01,
  selection?: ActivityComparisonDragSelection,
): (() => void) => {
  let pendingFraction: number | null = null
  let pendingClientX: number | null = null
  let frame = 0
  let pointerActive = false
  let focused = false
  let drag: {
    pointerId: number
    startClientX: number
    anchorFraction: number
    selected: boolean
  } | null = null
  const selectedFraction = (): number => state.selectedFraction ?? state.fraction
  const restore = () => show(selectedFraction())
  const flush = () => {
    frame = 0
    if (pendingFraction == null) return
    state.fraction = pendingFraction
    if (drag && pendingClientX != null) {
      drag.selected = Math.abs(pendingClientX - drag.startClientX) >= 3
      if (drag.selected) selection?.preview(drag.anchorFraction, state.fraction)
      else selection?.restore()
    }
    render(graph, restore)
    pendingFraction = null
    pendingClientX = null
  }
  const queue = (fraction: number, clientX?: number) => {
    pendingFraction = Math.min(1, Math.max(0, fraction))
    pendingClientX = clientX ?? null
    if (!frame) frame = window.requestAnimationFrame(flush)
  }
  const fractionAt = (clientX: number): number | null => {
    const bounds = graph.getBoundingClientRect()
    if (bounds.width <= 0) return null
    return Math.min(1, Math.max(0, (clientX - bounds.left) / bounds.width))
  }
  const onPointerMove = (event: PointerEvent) => {
    if (drag && event.pointerId !== drag.pointerId) return
    const fraction = fractionAt(event.clientX)
    if (fraction == null) return
    if (!pointerActive) {
      pointerActive = true
      activate(graph, restore)
    }
    queue(fraction, event.clientX)
  }
  const release = () => {
    if (pointerActive || focused) return
    pendingFraction = null
    pendingClientX = null
    if (frame) {
      window.cancelAnimationFrame(frame)
      frame = 0
    }
    deactivate(graph)
  }
  const onKeyDown = (event: KeyboardEvent) => {
    const next = activityComparisonFractionForKey(event.key, selectedFraction(), keyboardStep)
    if (next == null) return
    event.preventDefault()
    if (state.selectedFraction != null) state.selectedFraction = next
    queue(next)
  }
  const onPointerLeave = () => {
    if (drag) return
    if (state.selectedFraction != null) {
      state.fraction = state.selectedFraction
      show(state.selectedFraction)
    }
    pointerActive = false
    release()
  }
  const onFocus = () => {
    focused = true
    state.fraction = selectedFraction()
    activate(graph, restore)
    render(graph, restore)
  }
  const onBlur = () => {
    focused = false
    release()
  }
  const onPointerDown = (event: PointerEvent) => {
    if (!event.isPrimary || event.button !== 0 || drag) return
    const fraction = fractionAt(event.clientX)
    if (fraction == null) return
    if (!selection) {
      if (state.selectedFraction == null) return
      state.fraction = fraction
      state.selectedFraction = fraction
      pointerActive = true
      activate(graph, restore)
      render(graph, restore)
      graph.focus({ preventScroll: true })
      return
    }
    drag = {
      pointerId: event.pointerId,
      startClientX: event.clientX,
      anchorFraction: fraction,
      selected: false,
    }
    pointerActive = true
    activate(graph, restore)
    graph.setPointerCapture(event.pointerId)
    queue(fraction, event.clientX)
  }
  const onPointerUp = (event: PointerEvent) => {
    if (!drag || event.pointerId !== drag.pointerId) return
    const fraction = fractionAt(event.clientX)
    if (frame) window.cancelAnimationFrame(frame)
    frame = 0
    if (fraction != null) {
      pendingFraction = fraction
      pendingClientX = event.clientX
      flush()
    }
    const selected = drag.selected
    const pointerId = drag.pointerId
    drag = null
    if (selected) selection?.commit()
    else selection?.clear()
    if (graph.hasPointerCapture(pointerId)) graph.releasePointerCapture(pointerId)
    if (!graph.matches(':hover')) {
      pointerActive = false
      release()
    }
  }
  const onPointerCancel = (event: PointerEvent) => {
    if (!drag || event.pointerId !== drag.pointerId) return
    const pointerId = drag.pointerId
    drag = null
    selection?.restore()
    if (graph.hasPointerCapture(pointerId)) graph.releasePointerCapture(pointerId)
    pointerActive = false
    release()
  }
  graph.addEventListener('pointerdown', onPointerDown)
  graph.addEventListener('pointermove', onPointerMove)
  graph.addEventListener('pointerup', onPointerUp)
  graph.addEventListener('pointerleave', onPointerLeave)
  graph.addEventListener('pointercancel', onPointerCancel)
  graph.addEventListener('focus', onFocus)
  graph.addEventListener('keydown', onKeyDown)
  graph.addEventListener('blur', onBlur)
  return () => {
    graph.removeEventListener('pointerdown', onPointerDown)
    graph.removeEventListener('pointermove', onPointerMove)
    graph.removeEventListener('pointerup', onPointerUp)
    graph.removeEventListener('pointerleave', onPointerLeave)
    graph.removeEventListener('pointercancel', onPointerCancel)
    graph.removeEventListener('focus', onFocus)
    graph.removeEventListener('keydown', onKeyDown)
    graph.removeEventListener('blur', onBlur)
    if (frame) window.cancelAnimationFrame(frame)
    if (drag && graph.hasPointerCapture(drag.pointerId)) graph.releasePointerCapture(drag.pointerId)
    pendingFraction = null
    pendingClientX = null
    drag = null
    deactivate(graph)
  }
}
