import { mountTriathlon } from '../triathlon/runtime/mount'
import { currentNavSignal } from './nav-lifecycle'

document.addEventListener('nav', () => {
  const runtime = mountTriathlon(currentNavSignal())
  window.quartzTriathlon = { dayCard: runtime.dayCard }
  window.addCleanup(runtime.cleanup)
})
