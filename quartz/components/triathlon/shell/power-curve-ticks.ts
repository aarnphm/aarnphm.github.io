import { POWER_CURVE_EXTRA_AXIS_MARKERS } from '../../../util/triathlon-card'

interface PowerCurveTickBounds {
  left: number
  right: number
  optional: boolean
}

export const powerCurveTickVisibility = (ticks: readonly PowerCurveTickBounds[]): boolean[] => {
  const occupied = ticks.filter(tick => !tick.optional && tick.right > tick.left)
  return ticks.map(tick => {
    if (!tick.optional) return true
    if (
      tick.right <= tick.left ||
      occupied.some(other => tick.left < other.right + 1 && tick.right > other.left - 1)
    )
      return false
    occupied.push(tick)
    return true
  })
}

export const setupPowerCurveTicks = (root: ParentNode): (() => void) => {
  const axes = Array.from(root.querySelectorAll<HTMLElement>('.tri-cax-xax')).flatMap(axis => {
    const ticks = Array.from(
      axis.querySelectorAll<HTMLButtonElement>('.tri-curve-tick, .tri-best-power-tick'),
    )
    const optional = ticks.filter(
      tick =>
        POWER_CURVE_EXTRA_AXIS_MARKERS.includes(
          Number(tick.dataset.curveSeconds ?? tick.dataset.powerSeconds),
        ) && !tick.matches('.tri-cax-xt--first, .tri-cax-xt--last'),
    )
    return optional.length > 0 ? [{ axis, ticks, optional }] : []
  })
  if (axes.length === 0) return () => {}
  let frame = 0
  const update = (): void => {
    frame = 0
    for (const { axis, ticks, optional } of axes) {
      if (axis.getBoundingClientRect().width === 0) continue
      for (const tick of optional) tick.hidden = false
      const visible = powerCurveTickVisibility(
        ticks.map(tick => {
          const { left, right } = tick.getBoundingClientRect()
          return { left, right, optional: optional.includes(tick) }
        }),
      )
      ticks.forEach((tick, index) => {
        if (optional.includes(tick)) tick.hidden = !visible[index]
      })
    }
  }
  const schedule = (): void => {
    if (frame === 0) frame = window.requestAnimationFrame(update)
  }
  const resize = new ResizeObserver(schedule)
  for (const { axis, ticks } of axes) {
    resize.observe(axis)
    for (const tick of ticks) resize.observe(tick)
  }
  window.addEventListener('resize', schedule, { passive: true })
  update()
  return () => {
    resize.disconnect()
    window.removeEventListener('resize', schedule)
    if (frame !== 0) window.cancelAnimationFrame(frame)
    for (const { optional } of axes) for (const tick of optional) tick.hidden = false
  }
}
