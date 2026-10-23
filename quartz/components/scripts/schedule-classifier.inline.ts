import { registerMdxContent } from './mdx-content'

const SCL_ROOT_SELECTOR = '[data-schedule-classifier]'

const setupScheduleClassifier = (root: HTMLElement): (() => void) | null => {
  const raw = root.getAttribute('data-scl-regions')
  if (!raw || !root.querySelector('[data-scl-input]')) return null
  const regions = JSON.parse(raw) as Record<string, string>
  const panels = Array.from(root.querySelectorAll<HTMLElement>('[data-scl-panel]'))

  // Every panel is server-rendered from the classifier; this only picks which one shows.
  const apply = () => {
    const preset = root.querySelector<HTMLInputElement>('[data-scl-input]:checked')?.value
    if (!preset || !regions[preset]) return
    root.setAttribute('data-scl-preset', preset)
    root.setAttribute('data-scl-region', regions[preset])
    for (const panel of panels) panel.hidden = panel.getAttribute('data-scl-panel') !== preset
  }

  root.addEventListener('change', apply)
  apply()
  return () => root.removeEventListener('change', apply)
}

registerMdxContent(SCL_ROOT_SELECTOR, root => {
  const cleanup = setupScheduleClassifier(root)
  if (cleanup) window.addCleanup?.(cleanup)
})
