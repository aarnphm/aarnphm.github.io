import { observeSvgScale } from './figure-scale'
import { registerMdxContent } from './mdx-content'

const ROOT_SELECTOR = '[data-block-diagram]'

// Every step is rendered on the server; the script only shows the chosen variant and step.
const setupBlockDiagram = (root: HTMLElement): (() => void) => {
  const variants = Array.from(root.querySelectorAll<HTMLElement>('[data-bd-variant]'))

  const apply = () => {
    const v = root.querySelector<HTMLInputElement>('[data-bd-variant-input]:checked')?.value ?? '0'
    for (const panel of variants) {
      panel.hidden = panel.getAttribute('data-bd-variant') !== v
      const s = panel.querySelector<HTMLInputElement>('[data-bd-step-input]:checked')?.value ?? '0'
      for (const step of panel.querySelectorAll<HTMLElement>('[data-bd-step]'))
        step.hidden = step.getAttribute('data-bd-step') !== s
    }
  }

  root.addEventListener('change', apply)
  apply()
  return () => root.removeEventListener('change', apply)
}

// Labels hold their px size while blocks shrink with the svg, so below about its viewBox width the
// labels outgrow their blocks. The svg keeps a minimum width inside its own scroller, which leaves
// the step text at the frame width.
const MIN_PX_PER_UNIT = 1.1

const holdMinWidth = (svg: SVGSVGElement) => {
  svg.style.minWidth = `${Math.ceil(svg.viewBox.baseVal.width * MIN_PX_PER_UNIT)}px`
  if (svg.parentElement?.classList.contains('bd-scroll')) return
  const scroller = document.createElement('div')
  scroller.className = 'bd-scroll'
  svg.before(scroller)
  scroller.append(svg)
}

registerMdxContent(ROOT_SELECTOR, root => {
  window.addCleanup?.(setupBlockDiagram(root))
  const svgs = root.querySelectorAll<SVGSVGElement>('.bd-svg')
  for (const svg of svgs) holdMinWidth(svg)
  window.addCleanup?.(observeSvgScale(svgs, '--bd-u'))
})
