import type { Locale } from '../../../util/triathlon-presentation'
import { glossFor } from '../../../util/triathlon-i18n'
import { clampN } from '../analytics/shared'
import { el } from '../runtime/dom'
import { mathFrag } from '../runtime/dom'

export const renderGlossDef = (def: string): HTMLElement => {
  const span = el('span', 'tri-gloss-def')
  span.replaceChildren(...mathFrag(def))
  return span
}

export const setupGloss = (root: HTMLElement, locale: () => Locale): (() => void) => {
  const pop = el('div', 'tri-gloss')
  pop.setAttribute('role', 'tooltip')
  root.appendChild(pop)
  let current: HTMLElement | null = null
  const place = (term: HTMLElement, pointer?: { x: number; y: number }) => {
    const r = term.getBoundingClientRect()
    const pr = pop.getBoundingClientRect()
    const followsPointer = pointer != null && term.closest('.tri-engine-cardio') != null
    let left = followsPointer ? pointer.x + 12 : r.left
    if (left + pr.width > window.innerWidth - 8)
      left = followsPointer ? pointer.x - 12 - pr.width : window.innerWidth - 8 - pr.width
    let top = followsPointer ? pointer.y + 12 : r.bottom + 6
    if (top + pr.height > window.innerHeight - 8)
      top = followsPointer ? pointer.y - 12 - pr.height : r.top - 6 - pr.height
    pop.style.left = `${clampN(left, 8, Math.max(8, window.innerWidth - 8 - pr.width))}px`
    pop.style.top = `${clampN(top, 8, Math.max(8, window.innerHeight - 8 - pr.height))}px`
  }
  const show = (term: HTMLElement) => {
    const key = term.dataset.gloss ?? ''
    const definition = term.dataset.glossDef
    const g = definition
      ? { term: term.textContent?.trim() ?? '', def: definition }
      : glossFor(locale(), key)
    if (!g) return
    current = term
    pop.replaceChildren(el('span', 'tri-gloss-h', g.term), renderGlossDef(g.def))
    pop.classList.add('tri-gloss--on')
    place(term)
  }
  const hide = (term?: HTMLElement) => {
    if (term && term !== current) return
    current = null
    pop.classList.remove('tri-gloss--on')
  }
  const onOver = (event: Event) => {
    const t = (event.target as HTMLElement | null)?.closest<HTMLElement>('[data-gloss]')
    if (t) show(t)
  }
  const onOut = (event: Event) => {
    const t = (event.target as HTMLElement | null)?.closest<HTMLElement>('[data-gloss]')
    if (!t) return
    const to = (event as MouseEvent).relatedTarget as Node | null
    if (to && t.contains(to)) return
    hide(t)
  }
  const onMove = (event: MouseEvent) => {
    const t = (event.target as HTMLElement | null)?.closest<HTMLElement>('[data-gloss]')
    if (t && t === current) place(t, { x: event.clientX, y: event.clientY })
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
    root.removeEventListener('mouseover', onOver)
    root.removeEventListener('mousemove', onMove)
    root.removeEventListener('mouseout', onOut)
    root.removeEventListener('focusin', onOver)
    root.removeEventListener('focusout', onOut)
    document.removeEventListener('keydown', onKey)
    pop.remove()
  }
}
