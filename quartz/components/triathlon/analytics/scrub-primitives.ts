import { setMath } from '../runtime/dom'
import { clampN } from './shared'

export const scrubBind = (
  hover: HTMLElement,
  svgEl: SVGElement,
  cursor: SVGElement,
  readout: HTMLElement,
  count: number,
  vbW: number,
  textOf: (i: number) => string,
): (() => void) => {
  if (count < 2) return () => {}
  const onMove = (event: MouseEvent) => {
    const r = svgEl.getBoundingClientRect()
    const frac = clampN((event.clientX - r.left) / r.width, 0, 1)
    const cx = (frac * vbW).toFixed(2)
    cursor.setAttribute('x1', cx)
    cursor.setAttribute('x2', cx)
    setMath(readout, textOf(Math.round(frac * (count - 1))))
    hover.classList.add('tri-chart--hover')
  }
  const onLeave = () => hover.classList.remove('tri-chart--hover')
  svgEl.addEventListener('mousemove', onMove)
  svgEl.addEventListener('mouseleave', onLeave)
  return () => {
    svgEl.removeEventListener('mousemove', onMove)
    svgEl.removeEventListener('mouseleave', onLeave)
  }
}

export type ScrubItem = {
  svgEl: SVGElement
  cursor: SVGElement
  readout: HTMLElement
  hover: HTMLElement
  textOf: (f: number) => string
}

export const scrubGroup = (
  items: ScrubItem[],
  cursorXOf: (f: number) => number,
  keyboardSteps?: number,
): (() => void) => {
  if (items.length === 0) return () => {}
  let fraction = 0
  const show = (f: number) => {
    fraction = f
    const cx = cursorXOf(f).toFixed(2)
    for (const it of items) {
      it.cursor.setAttribute('x1', cx)
      it.cursor.setAttribute('x2', cx)
      it.hover.classList.add('tri-chart--hover')
      const text = it.textOf(f)
      setMath(it.readout, text)
      if (keyboardSteps != null) {
        it.svgEl.setAttribute('aria-valuenow', String(Math.round(f * keyboardSteps)))
        it.svgEl.setAttribute('aria-valuetext', text)
      }
    }
  }
  const move = (event: PointerEvent, ref: SVGElement) => {
    const r = ref.getBoundingClientRect()
    show(clampN((event.clientX - r.left) / r.width, 0, 1))
  }
  const leave = () => {
    if (keyboardSteps != null && items.some(it => it.svgEl === document.activeElement)) return
    for (const it of items) it.hover.classList.remove('tri-chart--hover')
  }
  const focus = () => show(fraction)
  const key = (event: KeyboardEvent) => {
    if (keyboardSteps == null || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key))
      return
    event.preventDefault()
    event.stopPropagation()
    show(
      event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? 1
          : clampN(
              fraction + (event.key === 'ArrowLeft' ? -1 : 1) / Math.max(1, keyboardSteps),
              0,
              1,
            ),
    )
  }
  const offs: (() => void)[] = []
  for (const it of items) {
    const onMove = (e: PointerEvent) => move(e, it.svgEl)
    it.svgEl.addEventListener('pointermove', onMove)
    it.svgEl.addEventListener('pointerleave', leave)
    if (keyboardSteps != null) {
      it.svgEl.setAttribute('tabindex', '0')
      it.svgEl.setAttribute('role', 'slider')
      it.svgEl.setAttribute('aria-valuemin', '0')
      it.svgEl.setAttribute('aria-valuemax', String(keyboardSteps))
      it.svgEl.setAttribute('aria-valuenow', '0')
      it.svgEl.setAttribute('aria-valuetext', it.textOf(0))
      it.svgEl.addEventListener('keydown', key)
      it.svgEl.addEventListener('focus', focus)
      it.svgEl.addEventListener('blur', leave)
    }
    offs.push(() => {
      it.svgEl.removeEventListener('pointermove', onMove)
      it.svgEl.removeEventListener('pointerleave', leave)
      it.svgEl.removeEventListener('keydown', key)
      it.svgEl.removeEventListener('focus', focus)
      it.svgEl.removeEventListener('blur', leave)
    })
  }
  return () => offs.forEach(f => f())
}
