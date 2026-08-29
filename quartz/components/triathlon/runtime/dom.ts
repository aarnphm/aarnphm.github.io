import katex from 'katex'
import type { TriNodeFactory } from '../../../util/triathlon-card'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import { triText } from '../../../util/triathlon-i18n'

export const applyI18n = (root: ParentNode, presentation: TriathlonPresentation): void => {
  for (const node of root.querySelectorAll<HTMLElement>('[data-i18n]')) {
    const key = node.dataset.i18n
    if (key) node.textContent = triText(presentation.locale, key)
  }
  for (const node of root.querySelectorAll<HTMLElement>('[data-i18n-aria-label]')) {
    const key = node.dataset.i18nAriaLabel
    if (key) node.setAttribute('aria-label', triText(presentation.locale, key))
  }
}

export const SVGNS = 'http://www.w3.org/2000/svg'

export const el = (
  tag: string,
  cls?: string,
  text?: string,
  attrs?: Record<string, string>,
): HTMLElement => {
  const e = document.createElement(tag)
  if (cls) e.className = cls
  if (text !== undefined) e.textContent = text
  if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k])
  return e
}

export const svg = (tag: string, attrs: Record<string, string | number>): SVGElement => {
  const e = document.createElementNS(SVGNS, tag)
  for (const k in attrs) e.setAttribute(k, String(attrs[k]))
  return e
}

export const mathFrag = (text: string): Node[] => {
  const out: Node[] = []
  text.split(/\$([^$]+)\$/).forEach((part, i) => {
    if (i % 2 === 1) {
      const m = el('span', 'tri-math')
      m.innerHTML = katex.renderToString(part, {
        displayMode: false,
        output: 'html',
        throwOnError: false,
        strict: false,
      })
      out.push(m)
    } else if (part) {
      out.push(document.createTextNode(part))
    }
  })
  return out
}

export const setMath = (host: HTMLElement, text: string): void => {
  host.replaceChildren(...mathFrag(text))
}

export const mathK = (cls: string, text: string): HTMLElement => {
  const span = el('span', cls)
  setMath(span, text)
  return span
}

export const createDomFactory = (
  presentation: TriathlonPresentation,
): TriNodeFactory<HTMLElement | SVGElement> => ({
  presentation,
  el,
  math: mathK,
  svg,
  add: (parent, ...children) => parent.append(...children),
})
