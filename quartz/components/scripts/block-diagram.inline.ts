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

registerMdxContent(ROOT_SELECTOR, root => {
  window.addCleanup?.(setupBlockDiagram(root))
  window.addCleanup?.(observeSvgScale(root.querySelectorAll<SVGSVGElement>('.bd-svg'), '--bd-u'))
})
