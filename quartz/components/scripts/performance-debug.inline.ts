import { setupSitePerformanceDebug } from './performance-debug'

document.addEventListener('nav', () => {
  window.addCleanup(setupSitePerformanceDebug(document))
})
