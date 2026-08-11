import { blurFocusedPanelSearch } from './command-palette'
import { isEditable } from './command-palette'
import { mapDetailMetricTabForKey } from './command-palette'
import { toggleSearchFocus } from './command-palette'

export const setupShortcuts = (root: HTMLElement): (() => void) => {
  let waitingForG = false
  let gTimeout: number | null = null

  const clearG = (): void => {
    waitingForG = false
    if (gTimeout) {
      clearTimeout(gTimeout)
      gTimeout = null
    }
  }
  const go = (path: string) => {
    const url = new URL(path, window.location.toString())
    if (window.spaNavigate) window.spaNavigate(url)
    else window.location.href = url.toString()
  }
  const subView = root.dataset.triView
  const subpageNav: Record<string, string> = {
    g: '/triathlon/tools',
    c: '/triathlon/calc',
    a: '/triathlon/analytics',
    m: '/triathlon/maps',
    t: '/triathlon/training',
    r: '/triathlon/training',
    f: '/triathlon/feed',
    o: '/triathlon/on',
    h: '/triathlon',
  }

  if (subView) {
    const keyboardScrollScope =
      root.querySelector<HTMLElement>('[data-keyboard-scroll-scope]') ??
      (root.matches('[data-keyboard-scroll-scope]') ? root : null)
    keyboardScrollScope?.focus({ preventScroll: true })
  }

  const listItems = (): HTMLElement[] => {
    if (subView === 'feed')
      return Array.from(root.querySelectorAll<HTMLButtonElement>('.tri-feed-head'))
    if (subView === 'on')
      return Array.from(root.querySelectorAll<HTMLAnchorElement>('.tri-tree-list a[href]'))
    return []
  }
  const focusedListItem = (items: HTMLElement[]): HTMLElement | null => {
    const active = document.activeElement
    return active instanceof HTMLElement && items.includes(active) ? active : null
  }
  const handleListKey = (event: KeyboardEvent): boolean => {
    if (event.isComposing || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey)
      return false
    const items = listItems()
    if (items.length === 0) return false
    const focused = focusedListItem(items)
    if (event.key === 'Enter' && focused && !event.repeat) {
      focused.click()
      return true
    }
    if (event.key === 'Escape' && focused && !event.repeat) {
      if (focused.matches('.tri-feed-head[aria-expanded="true"]')) focused.click()
      else focused.blur()
      return true
    }
    if (event.key !== 'j' && event.key !== 'k') return false
    const current = focused ? items.indexOf(focused) : event.key === 'j' ? -1 : items.length
    const nextIndex = Math.min(
      Math.max(current + (event.key === 'j' ? 1 : -1), 0),
      items.length - 1,
    )
    const next = items[nextIndex]
    next.focus({ preventScroll: true })
    next.scrollIntoView({ behavior: 'instant', block: 'nearest', inline: 'nearest' })
    return true
  }

  const modalChords: Record<string, { btn: string; openClass: string; close: string }> = {
    a: { btn: '.tri-analytics-btn', openClass: 'tri-analytics-open', close: '.tri-ana-close' },
    c: { btn: '.tri-calc-btn', openClass: 'tri-calc-open', close: '.tri-calc-close' },
    m: { btn: '.tri-map-btn', openClass: 'tri-map-open', close: '.tri-map-close' },
    t: { btn: '.tri-training-btn', openClass: 'tri-training-open', close: '.tri-training-close' },
  }
  const closeOpenModals = (except?: string) => {
    for (const k in modalChords) {
      if (k === except) continue
      const mc = modalChords[k]
      if (root.classList.contains(mc.openClass)) root.querySelector<HTMLElement>(mc.close)?.click()
    }
  }
  const toggleModal = (key: string) => {
    const mc = modalChords[key]
    if (root.classList.contains(mc.openClass)) {
      root.querySelector<HTMLElement>(mc.close)?.click()
      return
    }
    closeOpenModals(key)
    root.querySelector<HTMLElement>(mc.btn)?.click()
  }
  const runChord = (key: string): boolean => {
    if (subView) {
      const path = subpageNav[key]
      if (!path) return false
      go(path)
      return true
    }
    if (key === 'a' || key === 'c' || key === 'm' || key === 't') {
      toggleModal(key)
      return true
    }
    if (key === 'g') {
      closeOpenModals()
      root.querySelector<HTMLElement>('.tri-gear-btn')?.click()
      return true
    }
    if (key === 'p') {
      closeOpenModals()
      root.querySelector<HTMLElement>('.tri-pace-btn')?.click()
      return true
    }
    if (key === 's') {
      root.querySelector<HTMLElement>('.tri-total')?.click()
      return true
    }
    if (key === 'h') {
      go('/')
      return true
    }
    return false
  }
  const onKey = (e: KeyboardEvent) => {
    if (e.shiftKey && (e.ctrlKey || e.metaKey) && !e.altKey && e.key.toLowerCase() === 'g') {
      clearG()
      e.preventDefault()
      e.stopImmediatePropagation()
      go('/triathlon/tools')
      return
    }

    if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && e.key === '\\') {
      clearG()
      e.preventDefault()
      e.stopImmediatePropagation()
      runChord('h')
      return
    }

    const el = e.target instanceof HTMLElement ? e.target : null
    if (e.key === 'Escape' && blurFocusedPanelSearch(root)) {
      clearG()
      e.preventDefault()
      e.stopImmediatePropagation()
      return
    }
    if (el?.closest('.tri-map-tablist')) {
      clearG()
      return
    }
    if (
      e.key === '/' &&
      !e.shiftKey &&
      !e.ctrlKey &&
      !e.metaKey &&
      !e.altKey &&
      toggleSearchFocus(root, el)
    ) {
      clearG()
      e.preventDefault()
      e.stopImmediatePropagation()
      return
    }

    if (el && isEditable(el)) {
      clearG()
      return
    }

    if (handleListKey(e)) {
      clearG()
      e.preventDefault()
      e.stopImmediatePropagation()
      return
    }

    if (e.ctrlKey || e.metaKey || e.altKey) {
      clearG()
      return
    }

    const metricTab = e.isComposing || e.repeat ? null : mapDetailMetricTabForKey(root, e.key)
    if (metricTab) {
      clearG()
      e.preventDefault()
      e.stopImmediatePropagation()
      metricTab.click()
      metricTab.focus()
      return
    }

    if (waitingForG) {
      if (runChord(e.key.toLowerCase())) {
        e.preventDefault()
        e.stopImmediatePropagation()
      }
      clearG()
    } else if (e.key.toLowerCase() === 'g') {
      waitingForG = true
      gTimeout = window.setTimeout(() => {
        waitingForG = false
        gTimeout = null
      }, 1000)
    }
  }

  document.addEventListener('keydown', onKey, true)
  return () => {
    clearG()
    document.removeEventListener('keydown', onKey, true)
  }
}
