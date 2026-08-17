import { mountTriathlon } from '../triathlon/runtime/mount'
import { currentNavSignal } from './nav-lifecycle'

document.addEventListener('nav', () => {
  const root = document.querySelector<HTMLElement>('.triathlon')
  document.documentElement.classList.toggle(
    'tri-analytics-booting',
    root?.dataset.triView === 'analytics',
  )
  const runtime = mountTriathlon(currentNavSignal())
  window.quartzTriathlon = { dayCard: runtime.dayCard }
  window.addCleanup(runtime.cleanup)
})
