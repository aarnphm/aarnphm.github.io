import { rootNavSignal } from './root-lifecycle'

type QueryModule = typeof import('./base-query')
let modulePromise: Promise<QueryModule> | undefined

function loadQueryEditor(): Promise<QueryModule> {
  if (!modulePromise) {
    const source = new URL('base-query.js', import.meta.url).href
    modulePromise = import(source).catch(cause => {
      modulePromise = undefined
      throw cause
    })
  }
  return modulePromise
}

document.addEventListener('nav', () => {
  for (const root of document.querySelectorAll<HTMLElement>('[data-base-query]')) {
    const signal = rootNavSignal(root)
    const tabs = [...root.querySelectorAll<HTMLButtonElement>('[data-base-query-toggle]')]
    const panel = root.querySelector<HTMLElement>('#base-query-panel')
    let loading = false
    let requestedMode: 'filter' | 'sort' = 'filter'
    const dismiss = (restoreFocus = true): void => {
      if (panel) panel.hidden = true
      for (const tab of tabs) tab.setAttribute('aria-expanded', 'false')
      if (restoreFocus) tabs.find(tab => tab.dataset.baseQueryToggle === requestedMode)?.focus()
    }
    const open = async (event: MouseEvent): Promise<void> => {
      if (!(event.target instanceof Element) || root.dataset.baseQueryInitialized !== undefined)
        return
      if (event.target.closest('[data-base-query-close]')) {
        dismiss()
        return
      }
      const button = event.target.closest<HTMLButtonElement>('[data-base-query-toggle]')
      const mode = button?.dataset.baseQueryToggle
      if (mode !== 'filter' && mode !== 'sort') return
      if (loading && requestedMode === mode && !panel?.hidden) {
        dismiss()
        return
      }
      requestedMode = mode
      if (loading) return
      loading = true
      const status = root.querySelector<HTMLElement>('[data-base-query-status]')
      for (const tab of tabs) {
        const selected = tab === button
        tab.setAttribute('aria-selected', String(selected))
        tab.setAttribute('aria-expanded', String(selected))
        tab.tabIndex = selected ? 0 : -1
      }
      if (panel) {
        panel.hidden = false
        panel.setAttribute('aria-labelledby', `base-query-tab-${mode}`)
        panel.setAttribute('aria-busy', 'true')
      }
      if (status) status.textContent = 'Loading base editor…'
      try {
        const editor = await loadQueryEditor()
        if (signal.aborted || panel?.hidden) return
        root.removeEventListener('click', open)
        editor.mountBaseQuery(root, requestedMode)
      } catch (cause) {
        if (!signal.aborted && status) {
          status.textContent = `${cause instanceof Error ? cause.message : 'The editor could not load.'} Select a control to retry.`
        }
      } finally {
        loading = false
        panel?.removeAttribute('aria-busy')
      }
    }
    root.addEventListener('click', open, { signal })
    document.addEventListener(
      'pointerdown',
      event => {
        if (
          root.dataset.baseQueryInitialized === undefined &&
          !panel?.hidden &&
          event.target instanceof Node &&
          !root.contains(event.target)
        )
          dismiss(false)
      },
      { signal },
    )
    root.addEventListener(
      'keydown',
      event => {
        if (
          root.dataset.baseQueryInitialized === undefined &&
          event.key === 'Escape' &&
          !panel?.hidden
        ) {
          event.preventDefault()
          dismiss()
          return
        }
        if (!(event.target instanceof HTMLButtonElement) || !tabs.includes(event.target)) return
        const index = tabs.indexOf(event.target)
        const next =
          event.key === 'ArrowRight'
            ? tabs[(index + 1) % tabs.length]
            : event.key === 'ArrowLeft'
              ? tabs[(index + tabs.length - 1) % tabs.length]
              : event.key === 'Home'
                ? tabs[0]
                : event.key === 'End'
                  ? tabs.at(-1)
                  : undefined
        if (!next) return
        event.preventDefault()
        next.focus()
        if (next.getAttribute('aria-selected') !== 'true' || panel?.hidden) next.click()
      },
      { signal },
    )
  }
})

export default ''
