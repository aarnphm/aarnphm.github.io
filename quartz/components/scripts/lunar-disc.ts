import { lunarDiscPath } from '../../util/lunar-phase'

export function stopLunarTransition(path: SVGPathElement) {
  path.getAnimations().forEach(animation => animation.cancel())
  path.parentElement?.querySelectorAll('[data-moon-transition]').forEach(node => {
    node.getAnimations().forEach(animation => animation.cancel())
    node.remove()
  })
}

export function paintLunarDisc(
  path: SVGPathElement,
  phase: number,
  radius: number,
  pixel: number,
  animate: boolean,
) {
  const next = lunarDiscPath(phase, radius, pixel)
  const previous = path.getAttribute('d') ?? ''
  path.dataset.moonPhase = String(phase)
  if (next === previous) return
  const parent = path.parentElement
  stopLunarTransition(path)
  path.setAttribute('d', next)
  if (!animate || !parent) return
  const strength = Number(path.ownerDocument.defaultView?.getComputedStyle(path).opacity ?? 1)
  const outgoing = path.ownerDocument.createElementNS('http://www.w3.org/2000/svg', 'path')
  outgoing.setAttribute('d', previous)
  outgoing.setAttribute('class', path.getAttribute('class') ?? '')
  outgoing.setAttribute('data-moon-transition', '')
  outgoing.setAttribute('aria-hidden', 'true')
  parent.insertBefore(outgoing, path)
  const fade = outgoing.animate([{ opacity: strength }, { opacity: 0 }], {
    duration: 400,
    easing: 'ease-in-out',
    fill: 'forwards',
  })
  fade.onfinish = () => outgoing.remove()
  path.animate([{ opacity: 0 }, { opacity: strength }], { duration: 400, easing: 'ease-in-out' })
}
