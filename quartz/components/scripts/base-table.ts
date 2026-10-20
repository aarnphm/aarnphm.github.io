import { rootNavSignal } from './root-lifecycle'

export function initializeBaseTables(): void {
  for (const wrapper of document.querySelectorAll<HTMLElement>('.base-table-wrapper')) {
    if (wrapper.dataset.tableInitialized !== undefined) continue
    wrapper.dataset.tableInitialized = ''
    const signal = rootNavSignal(wrapper)
    const updateDivider = () => {
      wrapper.toggleAttribute('data-base-table-scrolled', wrapper.scrollTop > 0)
    }
    const firstHeading = wrapper.querySelector('thead th')
    const resize = new ResizeObserver(() => {
      if (firstHeading) {
        wrapper.style.setProperty(
          '--base-frozen-column-width',
          `${firstHeading.getBoundingClientRect().width}px`,
        )
      }
    })
    if (firstHeading) resize.observe(firstHeading)

    wrapper.addEventListener('scroll', updateDivider, { passive: true, signal })
    updateDivider()
    signal.addEventListener(
      'abort',
      () => {
        resize.disconnect()
        delete wrapper.dataset.tableInitialized
        wrapper.removeAttribute('data-base-table-scrolled')
        wrapper.style.removeProperty('--base-frozen-column-width')
      },
      { once: true },
    )
  }
}
