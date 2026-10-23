import type { SearchSession } from './state'

export function startSearchProgress(s: SearchSession, reason: 'search' | 'preview' = 'search') {
  const { progressBar, progressReasons } = s
  progressReasons.add(reason)
  progressBar.setAttribute(
    'aria-label',
    progressReasons.has('search') ? 'Loading search' : 'Loading preview',
  )
  window.clearTimeout(s.progressResetTimer)
  progressBar.setAttribute('aria-hidden', 'false')
  progressBar.style.opacity = '1'
  progressBar.style.width = '100%'
  progressBar.style.animation = 'semantic-progress-sweep 2.5s linear infinite'
}

export function completeSearchProgress(s: SearchSession, reason: 'search' | 'preview' = 'search') {
  const { progressBar, progressReasons } = s
  progressReasons.delete(reason)
  if (progressReasons.size > 0) {
    progressBar.setAttribute(
      'aria-label',
      progressReasons.has('search') ? 'Loading search' : 'Loading preview',
    )
    return
  }
  window.clearTimeout(s.progressResetTimer)
  progressBar.setAttribute('aria-hidden', 'true')
  progressBar.style.animation = 'none'
  progressBar.style.opacity = '0'
  s.progressResetTimer = window.setTimeout(() => {
    progressBar.style.width = '0'
    progressBar.style.backgroundPosition = '-100% 0'
  }, 300)
}

export function resetProgressBar(s: SearchSession) {
  const { progressBar, progressReasons } = s
  progressReasons.clear()
  window.clearTimeout(s.progressResetTimer)
  progressBar.setAttribute('aria-hidden', 'true')
  progressBar.style.animation = 'none'
  progressBar.style.opacity = '0'
  progressBar.style.width = '0'
  progressBar.style.backgroundPosition = '-100% 0'
}
