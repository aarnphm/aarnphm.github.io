export {}

const KVC_ROOT_SELECTOR = '[data-kv-compress-blocks]'

const setupKVCompressBlocks = (root: HTMLElement): (() => void) => {
  const tabs = Array.from(root.querySelectorAll<HTMLButtonElement>('[data-kvc-tab]'))
  const panels = Array.from(root.querySelectorAll<HTMLElement>('[data-kvc-panel]'))
  const steps = Array.from(root.querySelectorAll<HTMLInputElement>('[data-kvc-step-input]'))
  const navs = Array.from(root.querySelectorAll<HTMLButtonElement>('[data-kvc-nav]'))

  const selectTab = (tab: HTMLButtonElement, focus: boolean) => {
    const view = tab.getAttribute('data-kvc-tab')
    if (!view) return
    for (const t of tabs) {
      const on = t === tab
      t.setAttribute('aria-selected', on ? 'true' : 'false')
      t.tabIndex = on ? 0 : -1
    }
    for (const p of panels) p.hidden = p.getAttribute('data-kvc-panel') !== view
    root.setAttribute('data-kvc-view', view)
    if (focus) tab.focus()
  }

  const applyStep = () => {
    const current = steps.findIndex(s => s.checked)
    if (current < 0) return
    root.setAttribute('data-kvc-step', String(current + 1))
    for (const nav of navs) {
      const delta = Number(nav.getAttribute('data-kvc-nav'))
      const target = current + delta
      nav.disabled = target < 0 || target >= steps.length
    }
  }

  const onClick = (event: MouseEvent) => {
    const target = event.target as Element | null
    const tab = target?.closest<HTMLButtonElement>('[data-kvc-tab]')
    if (tab && root.contains(tab)) return selectTab(tab, false)
    const nav = target?.closest<HTMLButtonElement>('[data-kvc-nav]')
    if (!nav || !root.contains(nav) || nav.disabled) return
    const current = steps.findIndex(s => s.checked)
    const next = steps[current + Number(nav.getAttribute('data-kvc-nav'))]
    if (!next) return
    next.checked = true
    applyStep()
    // A disabled button drops focus; keep keyboard users inside the stepper.
    if (nav.disabled) navs.find(other => other !== nav)?.focus()
  }

  const onKey = (event: KeyboardEvent) => {
    const tab = (event.target as Element | null)?.closest<HTMLButtonElement>('[data-kvc-tab]')
    if (!tab) return
    const i = tabs.indexOf(tab)
    const n = tabs.length
    const to =
      event.key === 'ArrowRight'
        ? (i + 1) % n
        : event.key === 'ArrowLeft'
          ? (i - 1 + n) % n
          : event.key === 'Home'
            ? 0
            : event.key === 'End'
              ? n - 1
              : -1
    if (to < 0) return
    event.preventDefault()
    selectTab(tabs[to], true)
  }

  root.addEventListener('click', onClick)
  root.addEventListener('keydown', onKey)
  root.addEventListener('change', applyStep)
  applyStep()
  return () => {
    root.removeEventListener('click', onClick)
    root.removeEventListener('keydown', onKey)
    root.removeEventListener('change', applyStep)
  }
}

document.addEventListener('nav', () => {
  for (const root of document.querySelectorAll<HTMLElement>(KVC_ROOT_SELECTOR)) {
    window.addCleanup?.(setupKVCompressBlocks(root))
  }
})
