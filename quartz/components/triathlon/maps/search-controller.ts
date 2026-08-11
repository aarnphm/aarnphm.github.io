import type { Analytics } from '../../../plugins/stores/analytics'
import type { DetailPayload } from '../activity/data'
import type { TriathlonContext } from '../runtime/context'
import { ROUTE_SPORTS, type ActivityKind } from '../../../plugins/stores/strava'
import { activityResultItems, setActivityResultSelection } from '../analytics/search'
import { buildMapSearchView } from './search'

export interface MapSearchController {
  clear(): void
  run(): void
  dispose(): void
}

export const createMapSearchController = ({
  context,
  panel,
  search,
  results,
  analytics,
  details,
  detailsReady,
  setSports,
  selectRoute,
}: {
  context: TriathlonContext
  panel: HTMLElement
  search: HTMLInputElement | null
  results: HTMLElement | null
  analytics: () => Analytics | null
  details: () => DetailPayload | null
  detailsReady: () => boolean
  setSports: (sports: ReadonlySet<ActivityKind>) => void
  selectRoute: (id: string) => void
}): MapSearchController => {
  let selectedIndex = -1
  const resultItems = (): HTMLElement[] => activityResultItems(results)
  const setSelection = (index: number): void => {
    selectedIndex = setActivityResultSelection(results, index)
  }
  const run = (): void => {
    if (!search || !results) return
    const query = search.value.trim().toLowerCase()
    results.replaceChildren()
    if (!query) {
      panel.classList.remove('tri-map--searching')
      results.setAttribute('aria-hidden', 'true')
      setSports(new Set(ROUTE_SPORTS))
      return
    }
    panel.classList.add('tri-map--searching')
    results.setAttribute('aria-hidden', 'false')
    const view = buildMapSearchView(
      context.presentation,
      query,
      analytics(),
      details(),
      detailsReady(),
    )
    setSports(view.sport ? new Set([view.sport]) : new Set(ROUTE_SPORTS))
    results.replaceChildren(...view.nodes)
    setSelection(0)
  }
  const activate = (item: HTMLElement | undefined): void => {
    if (!item) return
    if (item.dataset.id) {
      selectRoute(item.dataset.id)
      return
    }
    if (!item.dataset.insert || !search) return
    const tokens = search.value.trim().split(/\s+/)
    tokens[tokens.length - 1] = item.dataset.insert
    search.value = tokens.join(' ') + (item.dataset.insert.endsWith(':') ? '' : ' ')
    search.focus()
    run()
  }
  const onResultsClick = (event: MouseEvent): void => {
    activate(
      event.target instanceof Element
        ? (event.target.closest('.tri-ana-ritem') ?? undefined)
        : undefined,
    )
  }
  const onSearchKey = (event: KeyboardEvent): void => {
    if (!panel.classList.contains('tri-map--searching')) return
    if (event.key === 'ArrowDown' || (event.ctrlKey && (event.key === 'n' || event.key === 'N'))) {
      event.preventDefault()
      setSelection(selectedIndex + 1)
      return
    }
    if (event.key === 'ArrowUp' || (event.ctrlKey && (event.key === 'p' || event.key === 'P'))) {
      event.preventDefault()
      setSelection(selectedIndex - 1)
      return
    }
    if (event.key === 'Enter') {
      event.preventDefault()
      const items = resultItems()
      activate(items[selectedIndex] ?? items[0])
    }
  }
  const clear = (): void => {
    if (search) search.value = ''
    panel.classList.remove('tri-map--searching')
    results?.replaceChildren()
    results?.setAttribute('aria-hidden', 'true')
    selectedIndex = -1
    setSports(new Set(ROUTE_SPORTS))
  }
  search?.addEventListener('input', run)
  search?.addEventListener('focus', run)
  search?.addEventListener('keydown', onSearchKey)
  results?.addEventListener('click', onResultsClick)
  return {
    clear,
    run,
    dispose: () => {
      search?.removeEventListener('input', run)
      search?.removeEventListener('focus', run)
      search?.removeEventListener('keydown', onSearchKey)
      results?.removeEventListener('click', onResultsClick)
    },
  }
}
