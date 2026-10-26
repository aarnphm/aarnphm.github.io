import { isRecord } from '../../util/type-guards'
import { setStatusText } from './status-flap'

/** Keeps the count slot stable while asynchronous search owns its text. */
export function mountBaseSearchStatus(root: HTMLElement, count: () => number) {
  const element = document.createElement('div')
  element.className = 'base-search-status'
  element.setAttribute('role', 'status')
  element.setAttribute('aria-live', 'polite')
  const error = document.createElement('p')
  error.className = 'base-search-error'
  error.setAttribute('role', 'alert')
  error.hidden = true
  root.append(error, element)
  let visible: number | undefined
  const renderCount = (event?: Event) => {
    if (
      event instanceof CustomEvent &&
      isRecord(event.detail) &&
      typeof event.detail.visible === 'number'
    )
      visible = event.detail.visible
    if (element.dataset.state === 'loading') return
    const value = visible ?? count()
    setStatusText(element, `${value.toLocaleString()} ${value === 1 ? 'result' : 'results'}`)
  }
  const progress = (event: Event) => {
    if (
      element.dataset.state !== 'loading' ||
      !(event instanceof CustomEvent) ||
      !isRecord(event.detail) ||
      typeof event.detail.label !== 'string'
    )
      return
    setStatusText(element, event.detail.label)
  }
  window.addEventListener('quartz-semantic-progress', progress)
  document.addEventListener('base-search-results', renderCount)
  return {
    set(state: 'idle' | 'loading' | 'ready' | 'unavailable', notice: string) {
      element.dataset.state = state
      error.hidden = state !== 'unavailable'
      error.textContent =
        state === 'unavailable' ? notice || 'Semantic search unavailable. Reload to retry.' : ''
      if (state === 'loading') setStatusText(element, 'loading semantic results…')
      renderCount()
    },
    message(value: string) {
      if (value) setStatusText(element, value)
      else renderCount()
    },
    dispose() {
      window.removeEventListener('quartz-semantic-progress', progress)
      document.removeEventListener('base-search-results', renderCount)
      error.remove()
      element.remove()
    },
  }
}
