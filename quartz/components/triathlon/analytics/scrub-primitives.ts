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

export const scrubGroup = (items: ScrubItem[], cursorXOf: (f: number) => number): (() => void) => {
  if (items.length === 0) return () => {}
  const move = (event: MouseEvent, ref: SVGElement) => {
    const r = ref.getBoundingClientRect()
    const f = clampN((event.clientX - r.left) / r.width, 0, 1)
    const cx = cursorXOf(f).toFixed(2)
    for (const it of items) {
      it.cursor.setAttribute('x1', cx)
      it.cursor.setAttribute('x2', cx)
      it.hover.classList.add('tri-chart--hover')
      setMath(it.readout, it.textOf(f))
    }
  }
  const leave = () => {
    for (const it of items) it.hover.classList.remove('tri-chart--hover')
  }
  const offs: (() => void)[] = []
  for (const it of items) {
    const onMove = (e: MouseEvent) => move(e, it.svgEl)
    it.svgEl.addEventListener('mousemove', onMove)
    it.svgEl.addEventListener('mouseleave', leave)
    offs.push(() => {
      it.svgEl.removeEventListener('mousemove', onMove)
      it.svgEl.removeEventListener('mouseleave', leave)
    })
  }
  return () => offs.forEach(f => f())
}
