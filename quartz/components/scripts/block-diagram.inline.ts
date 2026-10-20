export {}

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

document.addEventListener('nav', () => {
  for (const root of document.querySelectorAll<HTMLElement>(ROOT_SELECTOR))
    window.addCleanup?.(setupBlockDiagram(root))
})
