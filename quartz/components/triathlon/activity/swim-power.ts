import type { SwimPowerCurvePoint } from '../../../util/swim-power'
import { dlabel, powerCurveFraction } from '../../../util/triathlon-card'
import { isRecord } from '../../../util/type-guards'

export function setupSwimPowerCharts(scope: HTMLElement): () => void {
  const curves = new WeakMap<SVGSVGElement, SwimPowerCurvePoint[]>()
  const data = (graph: SVGSVGElement): SwimPowerCurvePoint[] => {
    const cached = curves.get(graph)
    if (cached) return cached
    const parsed: unknown = JSON.parse(graph.dataset.swimDragCurve ?? '[]')
    const curve: SwimPowerCurvePoint[] = []
    if (Array.isArray(parsed))
      for (const p of parsed)
        if (
          isRecord(p) &&
          typeof p.durationS === 'number' &&
          Number.isFinite(p.durationS) &&
          p.durationS > 0 &&
          typeof p.index === 'number' &&
          Number.isFinite(p.index) &&
          p.index > 0 &&
          typeof p.startElapsedS === 'number' &&
          typeof p.endElapsedS === 'number'
        )
          curve.push({
            durationS: p.durationS,
            index: p.index,
            startElapsedS: p.startElapsedS,
            endElapsedS: p.endElapsedS,
          })
    curves.set(graph, curve)
    return curve
  }
  const select = (graph: SVGSVGElement, requested: number): void => {
    const curve = data(graph)
    const wrap = graph.closest<HTMLElement>('.tri-swim-drag-chart')
    if (!curve.length || !wrap) return
    wrap.classList.add('tri-chart--hover')
    const point = curve.reduce(
      (best, p) =>
        Math.abs(Math.log(p.durationS / requested)) < Math.abs(Math.log(best.durationS / requested))
          ? p
          : best,
      curve[0],
    )
    const x =
      100 *
      powerCurveFraction(point.durationS, curve[0].durationS, curve[curve.length - 1].durationS)
    const max = Number(graph.dataset.swimDragDomainMax)
    const height = graph.viewBox.baseVal.height
    const cursor = graph.querySelector('.tri-chart-cursor')
    cursor?.setAttribute('x1', String(x))
    cursor?.setAttribute('x2', String(x))
    const marker = wrap.querySelector<HTMLElement>('.tri-curve-point')
    if (marker) {
      marker.style.left = `${x}%`
      marker.style.top = `${((height - (point.index / max) * (height - 1)) / height) * 100}%`
    }
    const duration = wrap.querySelector('.tri-curve-readout-duration')
    const value = wrap.querySelector('.tri-swim-drag-value')
    if (duration) duration.textContent = dlabel(point.durationS)
    if (value) value.textContent = `${point.index.toFixed(1)} idx`
    graph.setAttribute('aria-valuenow', String(point.durationS))
    graph.setAttribute(
      'aria-valuetext',
      `${dlabel(point.durationS)} · ${point.index.toFixed(1)} idx · ${dlabel(point.startElapsedS)} to ${dlabel(point.endElapsedS)}`,
    )
    for (const tick of wrap.querySelectorAll<HTMLButtonElement>('[data-swim-drag-seconds]'))
      tick.setAttribute(
        'aria-pressed',
        String(Number(tick.dataset.swimDragSeconds) === point.durationS),
      )
  }
  const graphAt = (target: EventTarget | null): SVGSVGElement | null =>
    target instanceof Element
      ? (target
          .closest('.tri-swim-drag-chart')
          ?.querySelector<SVGSVGElement>('.tri-swim-drag-svg') ?? null)
      : null
  const move = (event: PointerEvent): void => {
    if (!(event.target instanceof Element) || !event.target.closest('.tri-swim-drag-svg')) return
    const graph = graphAt(event.target)
    if (!graph) return
    const curve = data(graph),
      rect = graph.getBoundingClientRect()
    if (!curve.length || rect.width <= 0) return
    const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width))
    select(
      graph,
      curve[0].durationS * (curve[curve.length - 1].durationS / curve[0].durationS) ** fraction,
    )
  }
  const click = (event: MouseEvent): void => {
    if (!(event.target instanceof Element)) return
    const tick = event.target.closest<HTMLButtonElement>('[data-swim-drag-seconds]')
    const graph = graphAt(tick)
    if (graph && tick) select(graph, Number(tick.dataset.swimDragSeconds))
  }
  const key = (event: KeyboardEvent): void => {
    if (
      !(event.target instanceof SVGSVGElement) ||
      !event.target.matches('.tri-swim-drag-svg') ||
      !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)
    )
      return
    const graph = event.target,
      curve = data(graph)
    if (!curve.length) return
    event.preventDefault()
    event.stopPropagation()
    const current = curve.findIndex(
      p => p.durationS === Number(graph.getAttribute('aria-valuenow')),
    )
    const index =
      event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? curve.length - 1
          : Math.max(0, Math.min(curve.length - 1, current + (event.key === 'ArrowLeft' ? -1 : 1)))
    select(graph, curve[index].durationS)
  }
  scope.addEventListener('pointermove', move)
  scope.addEventListener('click', click)
  scope.addEventListener('keydown', key)
  return () => {
    scope.removeEventListener('pointermove', move)
    scope.removeEventListener('click', click)
    scope.removeEventListener('keydown', key)
  }
}
