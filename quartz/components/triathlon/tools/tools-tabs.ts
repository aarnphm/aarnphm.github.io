type ToolsTab = 'tools' | 'pace-chart'

const TOOLS_TABS: readonly ToolsTab[] = ['tools', 'pace-chart']

const isToolsTab = (value: string | undefined): value is ToolsTab =>
  TOOLS_TABS.some(tab => tab === value)

export const setupToolsTabs = (root: HTMLElement): (() => void) | null => {
  const tablist = root.querySelector<HTMLElement>('.tri-tools-tabs')
  const tabs = Array.from(tablist?.querySelectorAll<HTMLButtonElement>('[data-tools-tab]') ?? [])
  const panels = Array.from(root.querySelectorAll<HTMLElement>('.tri-tools [data-tools-panel]'))
  if (!tablist || tabs.length !== TOOLS_TABS.length || panels.length !== TOOLS_TABS.length)
    return null

  const select = (selected: ToolsTab, updateHash: boolean, focus: boolean): void => {
    for (const tab of tabs) {
      const active = tab.dataset.toolsTab === selected
      tab.setAttribute('aria-selected', String(active))
      tab.tabIndex = active ? 0 : -1
      if (active && focus) tab.focus({ preventScroll: true })
    }
    for (const panel of panels) panel.hidden = panel.dataset.toolsPanel !== selected
    if (!updateHash) return
    const url = new URL(window.location.href)
    url.hash = selected === 'tools' ? '' : selected
    window.history.replaceState(window.history.state, '', url)
  }

  const onClick = (event: MouseEvent): void => {
    const tab =
      event.target instanceof Element
        ? event.target.closest<HTMLButtonElement>('[data-tools-tab]')
        : null
    if (!tab || !tablist.contains(tab) || !isToolsTab(tab.dataset.toolsTab)) return
    select(tab.dataset.toolsTab, true, false)
  }

  const onKeyDown = (event: KeyboardEvent): void => {
    if (!(event.target instanceof HTMLButtonElement) || !tablist.contains(event.target)) return
    const current = tabs.indexOf(event.target)
    if (current < 0) return
    const next =
      event.key === 'Home'
        ? 0
        : event.key === 'End'
          ? tabs.length - 1
          : event.key === 'ArrowLeft'
            ? (current - 1 + tabs.length) % tabs.length
            : event.key === 'ArrowRight'
              ? (current + 1) % tabs.length
              : -1
    const nextValue = tabs[next]?.dataset.toolsTab
    if (!isToolsTab(nextValue)) return
    event.preventDefault()
    select(nextValue, true, true)
  }

  const onHashChange = (): void =>
    select(window.location.hash === '#pace-chart' ? 'pace-chart' : 'tools', false, false)
  tablist.addEventListener('click', onClick)
  tablist.addEventListener('keydown', onKeyDown)
  window.addEventListener('hashchange', onHashChange)
  onHashChange()
  return () => {
    tablist.removeEventListener('click', onClick)
    tablist.removeEventListener('keydown', onKeyDown)
    window.removeEventListener('hashchange', onHashChange)
  }
}
