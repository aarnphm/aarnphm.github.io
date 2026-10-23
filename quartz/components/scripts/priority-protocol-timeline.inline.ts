import { registerMdxContent } from './mdx-content'

const PPT_ROOT_SELECTOR = '[data-priority-protocol-timeline]'

const setupPriorityProtocolTimeline = (root: HTMLElement): (() => void) | null => {
  const panels = Array.from(root.querySelectorAll<HTMLElement>('[data-ppt-panel]'))
  const presetPanels = Array.from(root.querySelectorAll<HTMLElement>('[data-ppt-preset-panel]'))
  const protocolPanels = Array.from(root.querySelectorAll<HTMLElement>('[data-ppt-protocol-panel]'))
  if (!panels.length) return null

  // Every state is server-rendered from the simulation; this only picks which one shows.
  const apply = () => {
    const preset = root.querySelector<HTMLInputElement>('[data-ppt-preset]:checked')?.value
    const protocol = root.querySelector<HTMLInputElement>('[data-ppt-protocol]:checked')?.value
    if (!preset || !protocol) return
    const key = `${preset}:${protocol}`
    if (!panels.some(p => p.getAttribute('data-ppt-panel') === key)) return
    root.setAttribute('data-ppt-key', key)
    for (const p of panels) p.hidden = p.getAttribute('data-ppt-panel') !== key
    for (const p of presetPanels) p.hidden = p.getAttribute('data-ppt-preset-panel') !== preset
    for (const p of protocolPanels)
      p.hidden = p.getAttribute('data-ppt-protocol-panel') !== protocol
  }

  root.addEventListener('change', apply)
  apply()
  return () => root.removeEventListener('change', apply)
}

registerMdxContent(PPT_ROOT_SELECTOR, root => {
  const cleanup = setupPriorityProtocolTimeline(root)
  if (cleanup) window.addCleanup?.(cleanup)
})
