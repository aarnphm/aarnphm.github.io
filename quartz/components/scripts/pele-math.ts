import { type MountOptions } from 'pele/lazy'
import { loadPageStyles } from './page-styles'

type MathLabel = { element: HTMLElement; width: number; height: number }

const SVG_NS = 'http://www.w3.org/2000/svg'
const MEASURE_SIZE = 16

async function visible(figure: HTMLElement, signal: AbortSignal): Promise<void> {
  if (signal.aborted || figure.getBoundingClientRect().width > 0) return
  await new Promise<void>(resolve => {
    const finish = () => {
      observer.disconnect()
      signal.removeEventListener('abort', finish)
      resolve()
    }
    const observer = new ResizeObserver(entries => {
      if (entries.some(entry => entry.contentRect.width > 0)) finish()
    })
    observer.observe(figure)
    signal.addEventListener('abort', finish, { once: true })
  })
}

/** Measure pre-rendered KaTeX without putting a TeX parser in the browser bundle. */
export async function prepareDiagramMath(figure: HTMLElement, signal: AbortSignal) {
  const cache = figure.querySelector<HTMLElement>(':scope > .pele-math-cache')
  if (!cache?.dataset.source || !cache.dataset.prefix) return
  await loadPageStyles(figure)
  if (signal.aborted) return
  const probe = cache.cloneNode(true)
  if (!(probe instanceof HTMLElement)) return
  probe.hidden = false
  probe.className = 'pele-math-probe'
  const css = getComputedStyle(figure)
  probe.style.fontFamily = css.getPropertyValue('--pele-font').trim() || css.fontFamily
  // Measure outside collapsed sections and scaled canvas cards.
  document.body.append(probe)
  const removeProbe = () => probe.remove()
  signal.addEventListener('abort', removeProbe, { once: true })
  const labels = new Map<string, MathLabel>()
  try {
    // Resolve layout first so this probe starts loading every KaTeX face it uses.
    probe.getBoundingClientRect()
    await document.fonts.ready
    if (signal.aborted) return
    for (const label of probe.querySelectorAll<HTMLElement>('.pele-math-label[data-token]')) {
      const token = label.dataset.token
      if (!token) continue
      labels.set(token, { element: label, width: label.offsetWidth, height: label.offsetHeight })
    }
  } finally {
    signal.removeEventListener('abort', removeProbe)
    probe.remove()
  }
  const pattern = new RegExp(`${cache.dataset.prefix}\\d+\uE001`, 'g')
  const canvas = document.createElement('canvas').getContext('2d')
  if (!canvas) return
  // SVG character positions are unavailable under display:none.
  await visible(figure, signal)
  if (signal.aborted) return

  const dimensions = (label: MathLabel, size: number) => {
    // Pele fixes each label row at 1.5em. Keep tall fractions/matrices inside that row.
    const scale = Math.min(size / MEASURE_SIZE, (Math.round(size * 1.5) - 2) / label.height)
    return { width: label.width * scale + 2, fontSize: MEASURE_SIZE * scale }
  }
  const width = (text: string, size: number, style: number): number => {
    const css = getComputedStyle(figure)
    const family = css.getPropertyValue(style & 4 ? '--pele-font-mono' : '--pele-font').trim()
    canvas.font = `${style & 2 ? 'italic ' : ''}${style & 1 ? 'bold ' : ''}${size}px ${family || css.fontFamily}`
    let total = 0
    let start = 0
    for (const match of text.matchAll(pattern)) {
      total += canvas.measureText(text.slice(start, match.index)).width
      const label = labels.get(match[0])
      if (label) total += dimensions(label, size).width
      start = match.index + match[0].length
    }
    return total + canvas.measureText(text.slice(start)).width
  }
  const options = (container: HTMLElement): Pick<MountOptions, 'measurer' | 'onRender'> => ({
    measurer: { width },
    onRender: () => {
      const svg = container.querySelector('svg.pele')
      if (!svg) return
      const slots: { span: SVGTSpanElement; label: MathLabel; size: number }[] = []
      for (const text of svg.querySelectorAll('text')) {
        const walker = document.createTreeWalker(text, NodeFilter.SHOW_TEXT)
        const nodes: Text[] = []
        while (walker.nextNode())
          if (walker.currentNode instanceof Text) nodes.push(walker.currentNode)
        for (const node of nodes) {
          const value = node.data
          const matches = [...value.matchAll(pattern)]
          if (matches.length === 0) continue
          const fragment = document.createDocumentFragment()
          const size = Number.parseFloat(getComputedStyle(node.parentElement ?? text).fontSize)
          let start = 0
          for (const match of matches) {
            fragment.append(value.slice(start, match.index))
            const label = labels.get(match[0])
            if (label) {
              const span = document.createElementNS(SVG_NS, 'tspan')
              span.textContent = '\u2003'
              span.setAttribute('textLength', String(dimensions(label, size).width))
              span.setAttribute('lengthAdjust', 'spacingAndGlyphs')
              span.setAttribute('aria-hidden', 'true')
              slots.push({ span, label, size })
              fragment.append(span)
            }
            start = match.index + match[0].length
          }
          fragment.append(value.slice(start))
          node.replaceWith(fragment)
        }
      }
      for (const { span, label, size } of slots) {
        const text = span.closest('text')
        if (!text?.parentElement) continue
        const point = span.getStartPositionOfChar(0)
        const box = dimensions(label, size)
        const height = Math.round(size * 1.5)
        const fo = document.createElementNS(SVG_NS, 'foreignObject')
        fo.classList.add('pele-math')
        fo.setAttribute('x', String(point.x))
        fo.setAttribute('y', String(point.y - size * 0.35 - height / 2))
        fo.setAttribute('width', String(box.width))
        fo.setAttribute('height', String(height))
        const transform = text.getAttribute('transform')
        if (transform) fo.setAttribute('transform', transform)
        const content = label.element.cloneNode(true)
        if (!(content instanceof HTMLElement)) continue
        content.removeAttribute('data-token')
        content.style.fontSize = `${box.fontSize}px`
        fo.append(content)
        text.parentElement.append(fo)
      }
    },
  })
  return { source: cache.dataset.source, options }
}
