import { isRecord } from '../../util/type-guards'
import { setStatusText } from './status-flap'

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
  } else {
    footer.append(progress)
    panel?.append(footer)
  }

  const update = (event: Event) => {
    if (!(event instanceof CustomEvent) || progress.getAttribute('aria-hidden') !== 'false') return
    if (progress.dataset.reason !== 'search' || progress.dataset.source === 'local') return
    const detail: unknown = event.detail
    if (!isRecord(detail) || typeof detail.label !== 'string') return
    updateSearchProgress(
      progress,
      detail.label,
      typeof detail.loaded === 'number' ? detail.loaded : undefined,
      typeof detail.total === 'number' ? detail.total : undefined,
      'semantic',
    )
  }
  window.addEventListener(eventName, update)
  window.addCleanup(() => {
    window.removeEventListener(eventName, update)
    if (helper) footer.before(helper)
    footer.remove()
  })
  return progress
}

export function updateSearchProgress(
  progress: HTMLElement,
  label: string,
  loaded?: number,
  total?: number,
  source: 'local' | 'semantic' = 'local',
) {
  progress.dataset.source = source
  setStatusText(progress, label)
  progress.title = label
  progress.setAttribute('aria-label', label)
  for (const name of ['aria-valuemin', 'aria-valuemax', 'aria-valuenow'])
    progress.removeAttribute(name)
  if (loaded !== undefined && total !== undefined && total > 0) {
    progress.setAttribute('aria-valuemin', '0')
    progress.setAttribute('aria-valuemax', String(total))
    progress.setAttribute('aria-valuenow', String(loaded))
  }
}

export function showSearchProgress(progress: HTMLElement, loading: boolean, reason = 'search') {
  const unchanged =
    loading &&
    progress.getAttribute('aria-hidden') === 'false' &&
    progress.dataset.reason === reason
  progress.dataset.reason = reason
  progress.setAttribute('aria-hidden', String(!loading))
  if (!unchanged)
    updateSearchProgress(
      progress,
      reason === 'preview' ? 'loading preview…' : 'searching…',
      undefined,
      undefined,
      'semantic',
    )
  if (!loading) setStatusText(progress, '')
  progress.style.opacity = loading ? '1' : '0'
}
