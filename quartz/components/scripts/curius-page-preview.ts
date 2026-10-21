import { curiusDateFilter } from '../../util/curius-date'
import {
  curiusSavedRows,
  fetchCuriusSavedLinks,
  filterCuriusSavedLinks,
} from './curius-saved-links'
import { renderPagePreview, type PagePreview } from './page-preview-card'

export function curiusPagePreviewSource(link: HTMLAnchorElement) {
  const target = new URL(link.href)
  if (target.origin !== location.origin || !/^\/curius\/?$/.test(target.pathname)) return null
  const filter = curiusDateFilter(target.hash)
  return {
    popoverId: `popover-curius-${filter.kind === 'day' ? `${filter.key}-${encodeURIComponent(target.hash)}` : filter.kind}`,
    className: 'page-preview-popover',
    loadingLabel: 'loading saved links…',
    async load(signal: AbortSignal) {
      const preview = await loadCuriusPagePreview(target, signal)
      return preview ? (inner: HTMLDivElement) => renderPagePreview(preview, inner) : null
    },
  }
}

export async function loadCuriusPagePreview(
  target: URL,
  signal: AbortSignal,
): Promise<PagePreview | null> {
  const filter = curiusDateFilter(target.hash)

  const preview: PagePreview = {
    kind: 'curius',
    title: 'curius',
    href: target.href,
    description: 'Saved links, highlights, and reading trails.',
    meta: filter.kind === 'day' ? [filter.label] : [],
  }
  if (filter.kind === 'invalid') preview.description = 'invalid date · use MM-DD-YYYY or today'
  else {
    try {
      const links = filterCuriusSavedLinks(
        await fetchCuriusSavedLinks(signal, filter.kind === 'day'),
        filter,
      )
      preview.rows = curiusSavedRows(links).slice(0, 3)
      preview.meta?.push(
        filter.kind === 'day'
          ? `${links.length} saved ${links.length === 1 ? 'link' : 'links'}`
          : 'recently saved',
      )
      if (!links.length)
        preview.description =
          filter.kind === 'day' ? 'no links saved on this day' : 'no saved links available'
    } catch {
      if (signal.aborted) return null
      preview.meta?.push('could not load saved links · open curius to retry')
    }
  }
  return preview
}
