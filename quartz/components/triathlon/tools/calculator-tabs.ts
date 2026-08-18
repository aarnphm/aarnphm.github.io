export type CalculatorTab = 'race' | 'gear-ratios' | 'tire-pressure'

const CALCULATOR_TABS: readonly CalculatorTab[] = ['race', 'gear-ratios', 'tire-pressure']

const isCalculatorTab = (value: string | undefined): value is CalculatorTab =>
  CALCULATOR_TABS.some(tab => tab === value)

export const calculatorTabFromHash = (hash: string): CalculatorTab => {
  const value = hash.startsWith('#') ? hash.slice(1) : hash
  return isCalculatorTab(value) ? value : 'race'
}

export const setupCalculatorTabs = (root: HTMLElement): (() => void) | null => {
  const tablist = root.querySelector<HTMLElement>('.tri-calc-tabs')
  const tabs = Array.from(tablist?.querySelectorAll<HTMLButtonElement>('[data-calc-tab]') ?? [])
  const panels = Array.from(root.querySelectorAll<HTMLElement>('.tri-calc--page [data-calc-panel]'))
  const controls = Array.from(
    root.querySelectorAll<HTMLElement>('.tri-calc--page [data-calc-tab-control]'),
  )
  if (
    !tablist ||
    tabs.length !== CALCULATOR_TABS.length ||
    panels.length !== CALCULATOR_TABS.length
  )
    return null

  const select = (selected: CalculatorTab, updateHash: boolean, focus: boolean): void => {
    for (const tab of tabs) {
      const active = tab.dataset.calcTab === selected
      tab.classList.toggle('tri-calc-tab--on', active)
      tab.setAttribute('aria-selected', String(active))
      tab.tabIndex = active ? 0 : -1
      if (active && focus) tab.focus({ preventScroll: true })
    }
    for (const panel of panels) panel.hidden = panel.dataset.calcPanel !== selected
    for (const control of controls) {
      if (control.dataset.calcTabControl === selected) {
        const previous = control.dataset.calcTabHidden
        if (previous !== undefined) control.hidden = previous === 'true'
        delete control.dataset.calcTabHidden
      } else {
        control.dataset.calcTabHidden ??= String(control.hidden)
        control.hidden = true
      }
    }
    if (!updateHash) return
    const url = new URL(window.location.href)
    url.hash = selected === 'race' ? '' : selected
    window.history.replaceState(window.history.state, '', url)
  }

  const onClick = (event: MouseEvent): void => {
    const tab =
      event.target instanceof Element ? event.target.closest<HTMLElement>('[data-calc-tab]') : null
    if (!tab || !tablist.contains(tab) || !isCalculatorTab(tab.dataset.calcTab)) return
    select(tab.dataset.calcTab, true, false)
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
    const nextValue = tabs[next]?.dataset.calcTab
    if (next < 0 || !isCalculatorTab(nextValue)) return
    event.preventDefault()
    select(nextValue, true, true)
  }

  const onHashChange = (): void => select(calculatorTabFromHash(window.location.hash), false, false)
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
