import { el, mathFrag, setText } from '../runtime/dom'
import { clampN } from './shared'

let layoutEpoch = 1
let epochListenersBound = false
const rectCache = new WeakMap<Element, { epoch: number; rect: DOMRect }>()

const ensureEpochListeners = (): void => {
  if (epochListenersBound || typeof window === 'undefined') return
  epochListenersBound = true
  const bump = (): void => {
    layoutEpoch++
  }
  window.addEventListener('scroll', bump, { capture: true, passive: true })
  window.addEventListener('resize', bump, { passive: true })
}

export const invalidateRects = (): void => {
  layoutEpoch++
}

export const rectOf = (element: Element): DOMRect => {
  ensureEpochListeners()
  const cached = rectCache.get(element)
  if (cached && cached.epoch === layoutEpoch && cached.rect.width > 0) return cached.rect
  const rect = element.getBoundingClientRect()
  if (rect.width > 0) rectCache.set(element, { epoch: layoutEpoch, rect })
  return rect
}

export const rafEvent = <E extends Event>(
  handler: (event: E) => void,
): { onEvent: (event: E) => void; cancel: () => void } => {
  let frame = 0
  let latest: E | null = null
  const flush = (): void => {
    frame = 0
    const event = latest
    latest = null
    if (event) handler(event)
  }
  return {
    onEvent: (event: E): void => {
      latest = event
      if (frame === 0) frame = window.requestAnimationFrame(flush)
    },
    cancel: (): void => {
      if (frame !== 0) window.cancelAnimationFrame(frame)
      frame = 0
      latest = null
    },
  }
}

interface MathSlots {
  texts: [Text, Text, Text]
  maths: [HTMLElement, HTMLElement]
  exprs: [string, string]
}

const lastMathText = new WeakMap<HTMLElement, string>()
const mathSlotsMap = new WeakMap<HTMLElement, MathSlots>()
const katexCache = new Map<string, string>()

const renderMathOnce = (target: HTMLElement, expr: string): void => {
  let html = katexCache.get(expr)
  if (html === undefined) {
    const frag = mathFrag(`$${expr}$`)
    const first = frag[0]
    html = first instanceof HTMLElement ? first.innerHTML : expr
    katexCache.set(expr, html)
  }
  target.innerHTML = html
}

const normalizeMathTokens = (text: string): string =>
  text
    .replace(/\$([+-]?\d+(?:\.\d+)?)(\\[a-zA-Z]+)\$/g, '$1$$$2$$')
    .replace(/\$(\\[a-zA-Z]+)([+-]?\d+[^$]*)\$/g, '$$$1$$$2')

export const setFastMath = (host: HTMLElement, text: string): void => {
  if (lastMathText.get(host) === text) return
  lastMathText.set(host, text)
  let slots = mathSlotsMap.get(host)
  if (!text.includes('$') && !slots) {
    setText(host, text)
    return
  }
  const normalized = text.includes('$') ? normalizeMathTokens(text) : text
  const parts = normalized.split(/\$([^$]+)\$/)
  const mathCount = Math.floor(parts.length / 2)
  if (mathCount > 2) {
    mathSlotsMap.delete(host)
    host.replaceChildren(...mathFrag(text))
    return
  }
  if (!slots) {
    const t0 =
      host.firstChild && host.firstChild === host.lastChild && host.firstChild.nodeType === 3
        ? (host.firstChild as Text)
        : document.createTextNode('')
    if (t0.parentNode !== host) host.replaceChildren(t0)
    const m0 = el('span', 'tri-math')
    const t1 = document.createTextNode('')
    const m1 = el('span', 'tri-math')
    const t2 = document.createTextNode('')
    // Inline display, not `hidden`: `[hidden]` sits in site-wide `:has()` rules, so toggling it
    // costs ~20 ms of style recalc on triathlon pages.
    m0.style.display = 'none'
    m1.style.display = 'none'
    host.append(m0, t1, m1, t2)
    slots = { texts: [t0, t1, t2], maths: [m0, m1], exprs: ['', ''] }
    mathSlotsMap.set(host, slots)
  }
  const p0 = parts[0] ?? ''
  const e0 = parts[1] ?? ''
  const p1 = parts[2] ?? ''
  const e1 = parts[3] ?? ''
  const p2 = parts[4] ?? ''
  if (slots.texts[0].data !== p0) slots.texts[0].data = p0
  if (e0) {
    if (slots.exprs[0] !== e0) {
      slots.exprs[0] = e0
      renderMathOnce(slots.maths[0], e0)
    }
    slots.maths[0].style.display = ''
  } else {
    slots.maths[0].style.display = 'none'
  }
  if (slots.texts[1].data !== p1) slots.texts[1].data = p1
  if (e1) {
    if (slots.exprs[1] !== e1) {
      slots.exprs[1] = e1
      renderMathOnce(slots.maths[1], e1)
    }
    slots.maths[1].style.display = ''
  } else {
    slots.maths[1].style.display = 'none'
  }
  if (slots.texts[2].data !== p2) slots.texts[2].data = p2
}

export const scrubBind = (
  hover: HTMLElement,
  svgEl: SVGElement,
  cursor: SVGElement,
  readout: HTMLElement,
  count: number,
  vbW: number,
  textOf: (i: number) => string,
  indexAt: (fraction: number) => number = fraction => Math.round(fraction * (count - 1)),
): (() => void) => {
  if (count < 2) return () => {}
  let lastIndex = -1
  let isHovering = false
  const move = rafEvent((event: MouseEvent) => {
    const r = rectOf(svgEl)
    if (r.width <= 0) return
    const frac = clampN((event.clientX - r.left) / r.width, 0, 1)
    const cx = (frac * vbW).toFixed(2)
    cursor.setAttribute('x1', cx)
    cursor.setAttribute('x2', cx)
    const idx = indexAt(frac)
    if (idx !== lastIndex) {
      lastIndex = idx
      setFastMath(readout, textOf(idx))
    }
    if (!isHovering) {
      isHovering = true
      hover.classList.add('tri-chart--hover')
    }
  })
  const onLeave = () => {
    move.cancel()
    isHovering = false
    hover.classList.remove('tri-chart--hover')
  }
  svgEl.addEventListener('mousemove', move.onEvent)
  svgEl.addEventListener('mouseleave', onLeave)
  return () => {
    move.cancel()
    svgEl.removeEventListener('mousemove', move.onEvent)
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
  let lastStep = -1
  let isHovering = false
  const show = (f: number) => {
    fraction = f
    const cx = cursorXOf(f).toFixed(2)
    const step = keyboardSteps != null ? Math.round(f * keyboardSteps) : -1
    const stepChanged = keyboardSteps == null || step !== lastStep
    lastStep = step
    for (const it of items) {
      it.cursor.setAttribute('x1', cx)
      it.cursor.setAttribute('x2', cx)
      if (!isHovering) it.hover.classList.add('tri-chart--hover')
      if (stepChanged) {
        const text = it.textOf(f)
        setFastMath(it.readout, text)
        if (keyboardSteps != null) {
          it.svgEl.setAttribute('aria-valuenow', String(step))
          it.svgEl.setAttribute('aria-valuetext', text)
        }
      }
    }
    isHovering = true
  }
  let activeRef: SVGElement = items[0].svgEl
  const rafMove = rafEvent((event: PointerEvent) => {
    const r = rectOf(activeRef)
    if (r.width <= 0) return
    show(clampN((event.clientX - r.left) / r.width, 0, 1))
  })
  const leave = () => {
    rafMove.cancel()
    if (keyboardSteps != null && items.some(it => it.svgEl === document.activeElement)) return
    isHovering = false
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
    const onMove = (e: PointerEvent) => {
      activeRef = it.svgEl
      rafMove.onEvent(e)
    }
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
      rafMove.cancel()
      it.svgEl.removeEventListener('pointermove', onMove)
      it.svgEl.removeEventListener('pointerleave', leave)
      it.svgEl.removeEventListener('keydown', key)
      it.svgEl.removeEventListener('focus', focus)
      it.svgEl.removeEventListener('blur', leave)
    })
  }
  return () => offs.forEach(f => f())
}
