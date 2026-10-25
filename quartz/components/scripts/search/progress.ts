import type { SearchSession } from './state'
import { showSearchProgress } from '../search-progress'

export function startSearchProgress(
  s: SearchSession,
  reason: 'search' | 'preview' | 'index' | 'model' = 'search',
) {
  s.progressReasons.add(reason)
  showSearchProgress(
    s.progressBar,
    true,
    [...s.progressReasons].some(value => value !== 'preview') ? 'search' : 'preview',
  )
}

export function completeSearchProgress(
  s: SearchSession,
  reason: 'search' | 'preview' | 'index' | 'model' = 'search',
) {
  s.progressReasons.delete(reason)
  showSearchProgress(
    s.progressBar,
    s.progressReasons.size > 0,
    [...s.progressReasons].some(value => value !== 'preview') ? 'search' : 'preview',
  )
}

export function resetProgressBar(s: SearchSession) {
  s.progressReasons.clear()
  showSearchProgress(s.progressBar, false)
}
