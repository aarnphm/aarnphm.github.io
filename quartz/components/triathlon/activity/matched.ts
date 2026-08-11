import { marqueeCtl } from '../analytics/search'

export const setupMatchedActivities = (scope: HTMLElement): (() => void) => {
  const marquee = marqueeCtl()
  const indexOf = (element: HTMLElement): number | null => {
    const index = Number(element.dataset.matchedIndex)
    return Number.isInteger(index) && index >= 0 ? index : null
  }
  const show = (section: HTMLElement, index: number): void => {
    const points = section.querySelectorAll<HTMLElement>('.tri-matched-point')
    const rows = section.querySelectorAll<HTMLElement>('.tri-matched-table [data-matched-index]')
    let selected: HTMLElement | null = null
    for (const point of points) {
      const active = indexOf(point) === index
      point.dataset.selected = String(active)
      point.setAttribute('aria-pressed', String(active))
      if (active) selected = point
    }
    for (const row of rows) row.dataset.selected = String(indexOf(row) === index)
    if (!selected) return
    const cursor = section.querySelector<SVGLineElement>('.tri-matched-cursor')
    const x = selected.dataset.matchedX
    if (cursor && x) {
      cursor.setAttribute('x1', x)
      cursor.setAttribute('x2', x)
    }
    const readout = section.querySelector<HTMLElement>('.tri-matched-readout')
    if (readout && x) {
      readout.style.left = `${x}%`
      readout.dataset.direction = selected.dataset.matchedDirection ?? 'equal'
    }
    const title = section.querySelector<HTMLElement>('.tri-matched-readout-title')
    const value = section.querySelector<HTMLElement>('.tri-matched-readout-value')
    const delta = section.querySelector<HTMLElement>('.tri-matched-readout-delta')
    if (title) title.textContent = selected.dataset.matchedTitle ?? ''
    if (value) value.textContent = selected.dataset.matchedValue ?? ''
    if (delta) delta.textContent = selected.dataset.matchedDelta ?? ''
  }
  const restore = (section: HTMLElement): void => {
    const index = Number(section.dataset.matchedCurrentIndex)
    if (Number.isInteger(index)) show(section, index)
  }
  const sectionFor = (target: EventTarget | null): HTMLElement | null =>
    target instanceof Element ? target.closest<HTMLElement>('.tri-matched') : null
  const selectableFor = (target: EventTarget | null): HTMLElement | null =>
    target instanceof Element
      ? target.closest<HTMLElement>('.tri-matched-point, .tri-matched-table [data-matched-index]')
      : null
  const activityFor = (target: EventTarget | null): HTMLElement | null =>
    target instanceof Element ? target.closest<HTMLElement>('.tri-matched-activity') : null
  const runActivityMarquee = (target: EventTarget | null): void => {
    const activity = activityFor(target)
    if (activity) marquee.run(activity)
  }
  const stopActivityMarquee = (event: PointerEvent | FocusEvent): void => {
    const activity = activityFor(event.target)
    if (!activity) return
    if (event.relatedTarget instanceof Node && activity.contains(event.relatedTarget)) return
    marquee.stop()
  }
  const selectTarget = (target: EventTarget | null): boolean => {
    const selectable = selectableFor(target)
    const section = sectionFor(selectable)
    if (!selectable || !section) return false
    const index = indexOf(selectable)
    if (index == null) return false
    show(section, index)
    return true
  }
  const onPointerMove = (event: PointerEvent): void => {
    if (selectTarget(event.target)) return
    if (!(event.target instanceof Element)) return
    const graph = event.target.closest<SVGSVGElement>('.tri-matched-svg')
    const section = sectionFor(graph)
    if (!graph || !section) return
    const bounds = graph.getBoundingClientRect()
    if (bounds.width <= 0) return
    const x = Math.min(100, Math.max(0, ((event.clientX - bounds.left) / bounds.width) * 100))
    let nearest: HTMLElement | null = null
    let nearestDistance = Infinity
    for (const point of section.querySelectorAll<HTMLElement>('.tri-matched-point')) {
      const pointX = Number(point.dataset.matchedX)
      if (!Number.isFinite(pointX)) continue
      const distance = Math.abs(pointX - x)
      if (distance < nearestDistance) {
        nearest = point
        nearestDistance = distance
      }
    }
    if (nearest) selectTarget(nearest)
  }
  const restoreAfterExit = (event: PointerEvent | FocusEvent): void => {
    const section = sectionFor(event.target)
    if (!section) return
    if (event.relatedTarget instanceof Node && section.contains(event.relatedTarget)) return
    restore(section)
  }
  const onPointerOver = (event: PointerEvent): void => {
    runActivityMarquee(event.target)
  }
  const onPointerOut = (event: PointerEvent): void => {
    stopActivityMarquee(event)
    restoreAfterExit(event)
  }
  const onFocusIn = (event: FocusEvent): void => {
    selectTarget(event.target)
    runActivityMarquee(event.target)
  }
  const onFocusOut = (event: FocusEvent): void => {
    stopActivityMarquee(event)
    restoreAfterExit(event)
  }
  const onClick = (event: MouseEvent): void => {
    selectTarget(event.target)
  }
  const onKeyDown = (event: KeyboardEvent): void => {
    if (!(event.target instanceof Element)) return
    const point = event.target.closest<HTMLButtonElement>('.tri-matched-point')
    const section = sectionFor(point)
    if (!point || !section) return
    const points = [...section.querySelectorAll<HTMLButtonElement>('.tri-matched-point')]
    const index = points.indexOf(point)
    if (index < 0) return
    let nextIndex: number | null = null
    if (event.key === 'ArrowLeft' || event.key === 'ArrowDown') nextIndex = Math.max(0, index - 1)
    if (event.key === 'ArrowRight' || event.key === 'ArrowUp')
      nextIndex = Math.min(points.length - 1, index + 1)
    if (event.key === 'Home') nextIndex = 0
    if (event.key === 'End') nextIndex = points.length - 1
    if (event.key === 'Escape') {
      event.preventDefault()
      restore(section)
      const currentIndex = Number(section.dataset.matchedCurrentIndex)
      if (Number.isInteger(currentIndex)) points[currentIndex]?.focus()
      return
    }
    if (nextIndex == null) return
    event.preventDefault()
    points[nextIndex].focus()
    show(section, nextIndex)
  }

  scope.addEventListener('pointerover', onPointerOver)
  scope.addEventListener('pointermove', onPointerMove)
  scope.addEventListener('pointerout', onPointerOut)
  scope.addEventListener('focusin', onFocusIn)
  scope.addEventListener('focusout', onFocusOut)
  scope.addEventListener('click', onClick)
  scope.addEventListener('keydown', onKeyDown)
  return () => {
    marquee.stop()
    scope.removeEventListener('pointerover', onPointerOver)
    scope.removeEventListener('pointermove', onPointerMove)
    scope.removeEventListener('pointerout', onPointerOut)
    scope.removeEventListener('focusin', onFocusIn)
    scope.removeEventListener('focusout', onFocusOut)
    scope.removeEventListener('click', onClick)
    scope.removeEventListener('keydown', onKeyDown)
  }
}
