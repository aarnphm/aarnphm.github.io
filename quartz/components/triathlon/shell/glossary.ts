import type { Locale } from '../../../util/triathlon-presentation'
import { glossFor } from '../../../util/triathlon-i18n'
import { rafEvent, rectOf, setFastMath } from '../analytics/scrub-primitives'
import { clampN } from '../analytics/shared'
import { el, mathFrag, setText } from '../runtime/dom'

export const renderGlossDef = (def: string): HTMLElement => {
  const span = el('span', 'tri-gloss-def')
  span.replaceChildren(...mathFrag(def))
  return span
}

export const setupGloss = (root: HTMLElement, locale: () => Locale): (() => void) => {
  const pop = el('div', 'tri-gloss')
  pop.setAttribute('role', 'tooltip')
  const headSpan = el('span', 'tri-gloss-h')
  const defSpan = el('span', 'tri-gloss-def')
  pop.append(headSpan, defSpan)
  root.appendChild(pop)
  let current: HTMLElement | null = null
  let currentFollowsPointer = false
  let popWidth = 0
  let popHeight = 0
  const sizeCache = new Map<string, { w: number; h: number }>()

  const place = (term: HTMLElement, pointer?: { x: number; y: number }) => {
    const followsPointer = pointer != null && currentFollowsPointer
    const r = followsPointer ? null : rectOf(term)
    let left = followsPointer ? pointer.x + 12 : (r?.left ?? 8)
    if (left + popWidth > window.innerWidth - 8)
      left = followsPointer ? pointer.x - 12 - popWidth : window.innerWidth - 8 - popWidth
    let top = followsPointer ? pointer.y + 12 : (r?.bottom ?? 8) + 6
    if (top + popHeight > window.innerHeight - 8)
      top = followsPointer ? pointer.y - 12 - popHeight : (r?.top ?? 8) - 6 - popHeight
    pop.style.left = `${clampN(left, 8, Math.max(8, window.innerWidth - 8 - popWidth))}px`
    pop.style.top = `${clampN(top, 8, Math.max(8, window.innerHeight - 8 - popHeight))}px`
  }
  const show = (term: HTMLElement) => {
    if (term === current) return
    const key = term.dataset.gloss ?? ''
    const definition = term.dataset.glossDef
    const g = definition
      ? { term: term.dataset.glossTerm ?? term.textContent?.trim() ?? '', def: definition }
      : glossFor(locale(), key)
    if (!g) return
    current = term
    currentFollowsPointer = term.closest('.tri-engine-cardio') != null
    const cacheKey = `${locale()}:${g.term}:${g.def}`
    const cached = sizeCache.get(cacheKey)
    if (!cached) rectOf(term)
    setText(headSpan, g.term)
    setFastMath(defSpan, g.def)
    pop.classList.add('tri-gloss--on')
    if (cached) {
      popWidth = cached.w
      popHeight = cached.h
    } else {
      const pr = pop.getBoundingClientRect()
      popWidth = pr.width
      popHeight = pr.height
      if (popWidth > 0) sizeCache.set(cacheKey, { w: popWidth, h: popHeight })
    }
    place(term)
  }
  const rafMove = rafEvent((event: MouseEvent) => {
    if (!current || !currentFollowsPointer) return
    place(current, { x: event.clientX, y: event.clientY })
  })
  const hide = (term?: HTMLElement) => {
    if (term && term !== current) return
    rafMove.cancel()
    current = null
    currentFollowsPointer = false
    pop.classList.remove('tri-gloss--on')
  }
  const onOver = (event: Event) => {
    const t = (event.target as HTMLElement | null)?.closest<HTMLElement>('[data-gloss]')
    if (t && t !== current) show(t)
  }
  const onOut = (event: Event) => {
    if (!current) return
    const t = (event.target as HTMLElement | null)?.closest<HTMLElement>('[data-gloss]')
    if (!t || t !== current) return
    const to = (event as MouseEvent).relatedTarget as Node | null
    if (to && t.contains(to)) return
    hide(t)
  }
  const onMove = (event: MouseEvent) => {
    if (!current || !currentFollowsPointer) return
    rafMove.onEvent(event)
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key === 'Escape') hide()
  }
  root.addEventListener('mouseover', onOver)
  root.addEventListener('mousemove', onMove)
  root.addEventListener('mouseout', onOut)
  root.addEventListener('focusin', onOver)
  root.addEventListener('focusout', onOut)
  document.addEventListener('keydown', onKey)
  return () => {
    rafMove.cancel()
    root.removeEventListener('mouseover', onOver)
    root.removeEventListener('mousemove', onMove)
    root.removeEventListener('mouseout', onOut)
    root.removeEventListener('focusin', onOver)
    root.removeEventListener('focusout', onOut)
    document.removeEventListener('keydown', onKey)
    pop.remove()
  }
}
