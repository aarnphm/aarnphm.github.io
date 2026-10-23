import { observeSvgScale } from './figure-scale'
import { registerMdxContent } from './mdx-content'
import {
  type Comp,
  type Lock,
  type Scale,
  type TimeState,
  CASES,
  LOCUS_SCALE,
  PLANE_X,
  PLANE_Y,
  TIME_ELEMENTS,
  TIME_SCALE,
  UNIT,
  designGain,
  dragTo,
  locusView,
  nearestGain,
  nudge,
  timeView,
} from './pole-response-model'

const setAttrs = (el: Element | undefined, attrs: Record<string, string | number>) => {
  if (!el) return
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v))
}

// Pointer position in s-plane units; the CTM keeps this right inside a zoomed figure.
const toS = (svg: SVGSVGElement, s: Scale, e: PointerEvent) => {
  const ctm = svg.getScreenCTM()
  if (!ctm) return null
  const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm.inverse())
  return { re: s.re[0] + (p.x - PLANE_X) / UNIT, im: s.im[1] - (p.y - PLANE_Y) / UNIT }
}

const setupTime = (root: HTMLElement): (() => void) | null => {
  const s = TIME_SCALE
  const plane = root.querySelector<SVGSVGElement>('[data-prs-plane]')
  const plot = root.querySelector<SVGSVGElement>('[data-prs-plot]')
  const raw = root.dataset.prsState
  if (!plane || !raw) return null
  const initial = JSON.parse(raw) as TimeState
  let st = initial
  let ref = initial
  let lock = (root.dataset.prsLock ?? 'none') as Lock
  const els = Object.fromEntries(
    TIME_ELEMENTS.map(k => [k, root.querySelector(`[data-prs-el="${k}"]`) ?? undefined]),
  )
  const handles = Array.from(root.querySelectorAll<SVGGElement>('[data-prs-handle]'))

  const apply = () => {
    const v = timeView(s, st, lock, ref)
    for (const k of TIME_ELEMENTS) setAttrs(els[k], v.attrs[k])
    for (const [k, text] of Object.entries(v.readout)) {
      const out = root.querySelector(`[data-prs-out="${k}"]`)
      if (out) out.textContent = text
    }
    root.dataset.prsCase = v.cls
    root.dataset.prsLock = lock
    for (const input of root.querySelectorAll<HTMLInputElement>('[data-prs-case-input]'))
      input.checked = input.value === v.cls
    for (const h of handles) h.setAttribute('aria-valuetext', v.aria)
    plot?.setAttribute('aria-label', `Unit-step response; ${v.aria}`)
  }

  let drag: { id: number; which: number } | null = null
  const nearest = (re: number, im: number) => {
    if (st.kind !== 'real') return 0
    return Math.abs(re - st.p1) + Math.abs(im) <= Math.abs(re - st.p2) + Math.abs(im) ? 0 : 1
  }
  const onDown = (e: PointerEvent) => {
    const p = toS(plane, s, e)
    if (!p) return
    const target = (e.target as Element).closest<SVGGElement>('[data-prs-handle]')
    const which = target ? Number(target.dataset.prsHandle) : nearest(p.re, p.im)
    drag = { id: e.pointerId, which }
    plane.setPointerCapture(e.pointerId)
    root.querySelector<SVGGElement>(`[data-prs-handle="${which}"]`)?.focus({ preventScroll: true })
    st = dragTo(s, st, lock, ref, which, p.re, p.im)
    apply()
    e.preventDefault()
  }
  const onMove = (e: PointerEvent) => {
    if (!drag || e.pointerId !== drag.id) return
    const p = toS(plane, s, e)
    if (!p) return
    st = dragTo(s, st, lock, ref, drag.which, p.re, p.im)
    apply()
  }
  const onUp = (e: PointerEvent) => {
    if (drag && e.pointerId === drag.id) drag = null
  }
  const onKey = (e: KeyboardEvent) => {
    const h = (e.target as Element).closest<SVGGElement>('[data-prs-handle]')
    if (!h) return
    const next = nudge(s, st, lock, ref, Number(h.dataset.prsHandle), e.key, e.shiftKey)
    if (!next) return
    e.preventDefault()
    st = next
    apply()
  }
  const onChange = (e: Event) => {
    const input = e.target as HTMLInputElement
    if (input.hasAttribute('data-prs-case-input')) {
      const c = CASES.find(x => x.id === input.value)
      if (c) st = ref = c.state
    } else if (input.hasAttribute('data-prs-lock-input')) {
      lock = input.value as Lock
      // A lock follows a conjugate pair; from real poles it restarts at the figure's first pair.
      if (st.kind !== 'pair') st = initial
      ref = st
    }
    apply()
  }

  plane.addEventListener('pointerdown', onDown)
  plane.addEventListener('pointermove', onMove)
  plane.addEventListener('pointerup', onUp)
  plane.addEventListener('pointercancel', onUp)
  root.addEventListener('keydown', onKey)
  root.addEventListener('change', onChange)
  apply()
  return () => {
    plane.removeEventListener('pointerdown', onDown)
    plane.removeEventListener('pointermove', onMove)
    plane.removeEventListener('pointerup', onUp)
    plane.removeEventListener('pointercancel', onUp)
    root.removeEventListener('keydown', onKey)
    root.removeEventListener('change', onChange)
  }
}

const setupLocus = (root: HTMLElement): (() => void) | null => {
  const s = LOCUS_SCALE
  const plane = root.querySelector<SVGSVGElement>('[data-prs-plane]')
  const plot = root.querySelector<SVGSVGElement>('[data-prs-plot]')
  const slider = root.querySelector<HTMLInputElement>('[data-prs-k-input]')
  const design = root.querySelector<HTMLButtonElement>('[data-prs-design]')
  if (!plane || !slider) return null
  let comp = (root.dataset.prsComp ?? 'none') as Comp
  const closed = Array.from(root.querySelectorAll<SVGCircleElement>('[data-prs-cl]'))
  const curve = root.querySelector('[data-prs-el="curve"]') ?? undefined

  const apply = () => {
    const k = Number(slider.value)
    const v = locusView(s, comp, k)
    v.poles.forEach((p, i) => setAttrs(closed[i], p))
    setAttrs(curve, { d: v.curve })
    for (const [key, text] of Object.entries(v.readout)) {
      const out = root.querySelector(`[data-prs-out="${key}"]`)
      if (out) out.textContent = text
    }
    root.dataset.prsComp = comp
    root.dataset.prsK = String(k)
    slider.setAttribute('aria-valuetext', v.aria)
    plot?.setAttribute('aria-label', `Closed-loop unit-step response; ${v.aria}`)
  }
  const setK = (k: number) => {
    slider.value = (Math.round(k * 10) / 10).toFixed(1)
    apply()
  }

  let dragId: number | null = null
  const fromPointer = (e: PointerEvent) => {
    const p = toS(plane, s, e)
    if (p) setK(nearestGain(comp, p.re, p.im))
  }
  const onDown = (e: PointerEvent) => {
    dragId = e.pointerId
    plane.setPointerCapture(e.pointerId)
    fromPointer(e)
    e.preventDefault()
  }
  const onMove = (e: PointerEvent) => {
    if (e.pointerId === dragId) fromPointer(e)
  }
  const onUp = (e: PointerEvent) => {
    if (e.pointerId === dragId) dragId = null
  }
  const onInput = () => apply()
  const onDesign = () => setK(designGain(comp))
  const onChange = (e: Event) => {
    const input = e.target as HTMLInputElement
    if (!input.hasAttribute('data-prs-comp-input')) return
    comp = input.value as Comp
    apply()
  }

  plane.addEventListener('pointerdown', onDown)
  plane.addEventListener('pointermove', onMove)
  plane.addEventListener('pointerup', onUp)
  plane.addEventListener('pointercancel', onUp)
  slider.addEventListener('input', onInput)
  design?.addEventListener('click', onDesign)
  root.addEventListener('change', onChange)
  apply()
  return () => {
    plane.removeEventListener('pointerdown', onDown)
    plane.removeEventListener('pointermove', onMove)
    plane.removeEventListener('pointerup', onUp)
    plane.removeEventListener('pointercancel', onUp)
    slider.removeEventListener('input', onInput)
    design?.removeEventListener('click', onDesign)
    root.removeEventListener('change', onChange)
  }
}

registerMdxContent('[data-pole-response]', root => {
  const cleanup = root.dataset.prsMode === 'locus' ? setupLocus(root) : setupTime(root)
  if (cleanup) window.addCleanup?.(cleanup)
  const svgs = root.querySelectorAll<SVGSVGElement>('.prs-plane, .prs-plot')
  window.addCleanup?.(observeSvgScale(svgs, '--prs-u'))
})
