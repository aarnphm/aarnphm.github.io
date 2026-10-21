import { curiusDateFilter } from '../../util/curius-date'
import { bindCuriusPopover } from './curius-popover'
import {
  fetchCuriusSavedLinks,
  filterCuriusSavedLinks,
  type CuriusSavedLink,
} from './curius-saved-links'
import { currentNavSignal } from './nav-lifecycle'
import { previewText } from './page-preview-card'

document.addEventListener('nav', event => {
  const signal = currentNavSignal()
  let archive: Promise<CuriusSavedLink[]> | undefined
  const load = () =>
    (archive ??= fetchCuriusSavedLinks(signal).catch(error => {
      archive = undefined
      throw error
    }))
  const versions = new WeakMap<HTMLElement, number>()
  const render = async (root: HTMLElement) => {
    const version = (versions.get(root) ?? 0) + 1
    versions.set(root, version)
    const filter = curiusDateFilter(root.dataset.curiusDate ?? '')
    const heading = document.createElement('a')
    heading.className = 'internal curius-date-heading'
    heading.href = `/curius#${root.dataset.curiusDate ?? ''}`
    heading.textContent =
      root.dataset.curiusLabel || `curius · ${filter.kind === 'day' ? filter.label : 'saved links'}`
    heading.dataset.noPopover = 'true'
    const status = document.createElement('p')
    status.role = 'status'
    status.className = 'curius-date-status'
    root.replaceChildren(heading, status)
    if (root.dataset.curiusPage === 'true') {
      const all = document.createElement('a')
      all.href = '/curius'
      all.className = 'internal curius-date-reset'
      all.textContent = 'all saved links'
      root.insertBefore(all, status)
    }
    if (filter.kind === 'invalid') {
      root.setAttribute('aria-busy', 'false')
      status.textContent = 'invalid date · use MM-DD-YYYY or today'
      return
    }
    root.setAttribute('aria-busy', 'true')
    status.textContent = 'loading saved links…'
    const progress = document.createElement('progress')
    progress.setAttribute('aria-label', 'loading saved links')
    root.appendChild(progress)
    try {
      const links = filterCuriusSavedLinks(await load(), filter)
      if (signal.aborted || !root.isConnected || versions.get(root) !== version) return
      status.textContent = links.length
        ? `${links.length} saved ${links.length === 1 ? 'link' : 'links'}`
        : 'no links saved on this day'
      const list = document.createElement('ul')
      list.className = 'curius-date-list'
      for (const link of links) {
        const row = document.createElement('li')
        const anchor = document.createElement('a')
        anchor.href = link.link
        anchor.textContent = previewText(link.title || link.link, 200)
        anchor.target = '_blank'
        anchor.rel = 'noopener noreferrer'
        bindCuriusPopover(
          {
            link: link.link,
            title: link.title || link.link,
            snippet: link.snippet ?? '',
            createdDate: link.createdDate ?? undefined,
          },
          anchor,
        )
        row.appendChild(anchor)
        const source = document.createElement('span')
        source.textContent = new URL(link.link).hostname
        row.appendChild(source)
        list.appendChild(row)
      }
      root.appendChild(list)
    } catch {
      if (signal.aborted || versions.get(root) !== version) return
      status.textContent = 'could not load saved links'
      const retry = document.createElement('button')
      retry.type = 'button'
      retry.textContent = 'retry'
      retry.addEventListener('click', () => void render(root), { signal })
      root.appendChild(retry)
    } finally {
      if (!signal.aborted && versions.get(root) === version) {
        root.setAttribute('aria-busy', 'false')
        progress.remove()
      }
    }
  }
  for (const root of document.querySelectorAll<HTMLElement>('[data-curius-date]')) void render(root)
  if (event.detail.url !== 'curius') return
  const feed = document.querySelector<HTMLElement>('#curius-fragments')
  if (!feed) return
  const root = document.createElement('section')
  root.className = 'curius-date-view'
  root.dataset.curiusPage = 'true'
  root.hidden = true
  feed.before(root)
  const navigation = document.querySelector<HTMLElement>('#curius-pagination')
  const search = document.querySelector<HTMLElement>('.curius-search')
  const update = () => {
    const date = location.hash.slice(1)
    feed.hidden = Boolean(date)
    if (navigation) navigation.hidden = Boolean(date)
    if (search) search.hidden = Boolean(date)
    root.hidden = !date
    if (date) {
      root.dataset.curiusDate = date
      void render(root)
    } else {
      versions.set(root, (versions.get(root) ?? 0) + 1)
      root.removeAttribute('data-curius-date')
      root.removeAttribute('aria-busy')
      feed.dispatchEvent(new Event('scroll'))
    }
  }
  window.addEventListener('hashchange', update, { signal })
  update()
  window.addCleanup(() => root.remove())
})
