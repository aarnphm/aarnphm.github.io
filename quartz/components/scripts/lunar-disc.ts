import { lunarDiscPath } from '../../util/lunar-phase'

const transitions = new WeakMap<
  SVGPathElement,
  { outgoing: SVGPathElement; fade: Animation; incoming: Animation }
>()

export function stopLunarTransition(path: SVGPathElement) {
  const transition = transitions.get(path)
  if (!transition) return
  transitions.delete(path)
  transition.fade.cancel()
  transition.incoming.cancel()
  transition.outgoing.remove()
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
  const incoming = path.animate([{ opacity: 0 }, { opacity: strength }], {
    duration: 400,
    easing: 'ease-in-out',
  })
  // Retain our handles: getAnimations() flushes the entire scene's pending style changes,
  // including when a newly created moon has no transition to cancel.
  transitions.set(path, { outgoing, fade, incoming })
  fade.onfinish = () => {
    if (transitions.get(path)?.fade === fade) stopLunarTransition(path)
  }
}
