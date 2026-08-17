import { TRI_ANALYTICS_BOOT_CLASS } from '../triathlon/analytics/boot'
import { mountTriathlon } from '../triathlon/runtime/mount'
import { currentNavSignal } from './nav-lifecycle'

document.addEventListener('nav', () => {
  const root = document.querySelector<HTMLElement>('.triathlon')
  document.documentElement.classList.toggle(
    TRI_ANALYTICS_BOOT_CLASS,
    root?.dataset.triView === 'analytics',
  )
  const runtime = mountTriathlon(currentNavSignal())
  window.quartzTriathlon = { dayCard: runtime.dayCard }
  window.addCleanup(runtime.cleanup)
})
