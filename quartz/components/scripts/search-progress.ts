import { isRecord } from '../../util/type-guards'

const eventName = 'quartz-semantic-progress'

export function reportSemanticProgress(label: string, loaded?: number, total?: number) {
  window.dispatchEvent(new CustomEvent(eventName, { detail: { label, loaded, total } }))
}

export function createSearchProgress(input: HTMLInputElement): HTMLDivElement {
  const progress = document.createElement('div')
  progress.className = 'semantic-search-progress'
  progress.setAttribute('role', 'progressbar')
  progress.setAttribute('aria-hidden', 'true')
  const panel = input.parentElement?.parentElement
  const helper = panel?.querySelector<HTMLUListElement>('ul#helper')
  const footer = document.createElement('div')
  footer.className = 'search-footer'
  if (helper) {
    helper.before(footer)
    footer.append(helper, progress)
  } else panel?.append(footer)

  const update = (event: Event) => {
    if (!(event instanceof CustomEvent) || progress.getAttribute('aria-hidden') !== 'false') return
    if (progress.dataset.reason !== 'search') return
    const detail: unknown = event.detail
    if (!isRecord(detail) || typeof detail.label !== 'string') return
    progress.textContent = detail.label
    progress.title = detail.label
    progress.setAttribute('aria-label', detail.label)
    if (typeof detail.loaded === 'number' && typeof detail.total === 'number' && detail.total > 0) {
      progress.setAttribute('aria-valuemin', '0')
      progress.setAttribute('aria-valuemax', String(detail.total))
      progress.setAttribute('aria-valuenow', String(detail.loaded))
    } else progress.removeAttribute('aria-valuenow')
  }
  window.addEventListener(eventName, update)
  window.addCleanup(() => {
    window.removeEventListener(eventName, update)
    if (helper) footer.before(helper)
    footer.remove()
  })
  return progress
}

export function showSearchProgress(progress: HTMLElement, loading: boolean, reason = 'search') {
  progress.dataset.reason = reason
  progress.setAttribute('aria-hidden', String(!loading))
  progress.removeAttribute('aria-valuenow')
  progress.textContent = reason === 'preview' ? 'loading preview…' : 'searching…'
  progress.title = progress.textContent
  progress.setAttribute('aria-label', progress.textContent)
  progress.style.opacity = loading ? '1' : '0'
}
