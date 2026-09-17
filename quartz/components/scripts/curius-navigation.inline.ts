import type { CuriusResponse, Link } from '../../types/curius'
import { createLinkEl, fetchLinksHeaders } from './curius'

declare global {
  interface Window {
    curiusState?: { currentPage: number; hasMore: boolean }
  }

  interface DocumentEventMap {
    'curius:links-ready': CustomEvent<CuriusResponse | undefined>
  }
}

export function updateNavigation(canGoPrevious = (window.curiusState?.currentPage ?? 0) > 0) {
  const navigation = document.getElementById('curius-pagination')
  const prevButton = document.getElementById('curius-prev')
  const nextButton = document.getElementById('curius-next')

  if (!(prevButton instanceof HTMLButtonElement) || !(nextButton instanceof HTMLButtonElement)) {
    return
  }

  const state = window.curiusState
  const isNavigating = navigation?.getAttribute('aria-busy') === 'true'
  prevButton.disabled = isNavigating || !state || !canGoPrevious
  nextButton.disabled = isNavigating || !state || !state.hasMore
  navigation?.setAttribute('aria-label', `Saved links, page ${(state?.currentPage ?? 0) + 1}`)
}

interface LoadedPage {
  page: number
  first: HTMLElement
}

const linkKey = (link: Link): string =>
  typeof link.id === 'number' ? `id:${link.id}` : `url:${link.link}`

document.addEventListener('nav', event => {
  if (event.detail.url !== 'curius') return

  const navigation = document.getElementById('curius-pagination')
  const fragment = document.getElementById('curius-fragments')
  const prevButton = document.getElementById('curius-prev')
  const nextButton = document.getElementById('curius-next')
  if (!navigation || !fragment || !prevButton || !nextButton) return

  const controller = new AbortController()
  const { signal } = controller
  const pages: LoadedPage[] = []
  const seen = new Set<string>()
  const sentinel = document.createElement('li')
  sentinel.className = 'curius-load-more'
  sentinel.setAttribute('aria-hidden', 'true')
  let currentIndex = 0
  let lastLoadedPage = 0
  let hasMore = false
  let ready = false
  let loading = false
  let automaticPaused = false
  let frame: number | undefined
  const loadAhead = 240

  const nearEnd = () =>
    fragment.scrollHeight - fragment.scrollTop - fragment.clientHeight <= loadAhead

  const publishState = () => {
    window.curiusState = {
      currentPage: pages[currentIndex]?.page ?? lastLoadedPage,
      hasMore: currentIndex < pages.length - 1 || hasMore,
    }
    updateNavigation(currentIndex > 0)
  }

  const updateVisiblePage = () => {
    if (!ready || pages.length === 0) return
    const top = fragment.getBoundingClientRect().top + fragment.clientTop + 1
    let index = 0
    for (const [candidate, page] of pages.entries()) {
      if (page.first.getBoundingClientRect().top > top) break
      index = candidate
    }
    if (fragment.scrollHeight - fragment.scrollTop - fragment.clientHeight <= 1)
      index = pages.length - 1
    currentIndex = index
    publishState()
  }

  const scrollToPage = (page: LoadedPage) => {
    const offset =
      page.first.getBoundingClientRect().top -
      fragment.getBoundingClientRect().top -
      fragment.clientTop
    fragment.scrollTo({ top: fragment.scrollTop + offset, behavior: 'instant' })
    currentIndex = pages.indexOf(page)
    publishState()
  }

  const queueEndCheck = () => {
    if (frame !== undefined) return
    frame = requestAnimationFrame(() => {
      frame = undefined
      if (ready && hasMore && !loading && !automaticPaused && nearEnd()) void appendNextPage()
    })
  }

  const observer = new IntersectionObserver(
    entries => {
      for (const entry of entries) {
        if (!entry.isIntersecting) automaticPaused = false
        else if (ready && !automaticPaused) queueEndCheck()
      }
    },
    { root: fragment, rootMargin: `0px 0px ${loadAhead}px 0px` },
  )

  const updateSentinel = () => {
    if (hasMore) {
      if (fragment.lastElementChild !== sentinel) fragment.appendChild(sentinel)
      observer.observe(sentinel)
    } else {
      observer.unobserve(sentinel)
      sentinel.remove()
    }
  }

  const setFeedStatus = (message: string) => {
    if (pages.length > 0) return
    const status = document.createElement('li')
    status.className = 'curius-list-status'
    status.setAttribute('role', 'status')
    status.textContent = message
    fragment.querySelector('.curius-list-status')?.remove()
    fragment.insertBefore(status, sentinel.isConnected ? sentinel : null)
  }

  const appendNextPage = async (moveToPage = false) => {
    if (!ready || loading || !hasMore || signal.aborted) return
    loading = true
    navigation.setAttribute('aria-busy', 'true')
    fragment.setAttribute('aria-busy', 'true')
    publishState()
    document.dispatchEvent(
      new CustomEvent('toast', { detail: { message: 'Récupération des liens curius…' } }),
    )

    try {
      while (hasMore && !signal.aborted) {
        const page = lastLoadedPage + 1
        const response = await fetch(`/api/curius?query=links&page=${page}`, {
          ...fetchLinksHeaders,
          signal: AbortSignal.any([signal, AbortSignal.timeout(20_000)]),
        })
        if (!response.ok) throw new Error('Failed to load Curius links')
        const data: CuriusResponse = await response.json()
        if (signal.aborted || !fragment.isConnected) return
        if (!Array.isArray(data.links)) throw new Error('Invalid Curius links response')

        const rawLinks = data.links
        const links = rawLinks.filter(link => {
          if (link.trails.length > 0 || seen.has(linkKey(link))) return false
          seen.add(linkKey(link))
          return true
        })
        lastLoadedPage = page
        hasMore = rawLinks.length > 0 && (data.hasMore ?? true)
        if (links.length === 0) continue

        const rows = links.map(link => {
          const row = createLinkEl(link)
          row.dataset.curiusPage = String(page)
          return row
        })
        fragment.querySelector('.curius-list-status')?.remove()
        const scrollTop = fragment.scrollTop
        sentinel.remove()
        fragment.append(...rows)
        const boundary = { page, first: rows[0] }
        pages.push(boundary)
        updateSentinel()
        fragment.scrollTop = scrollTop
        if (moveToPage) scrollToPage(boundary)
        document.dispatchEvent(
          new CustomEvent('toast', {
            detail: { message: `Page ${page + 1} chargée.`, durationMs: 1200 },
          }),
        )
        return
      }
      if (!signal.aborted) {
        setFeedStatus('Aucun lien disponible pour le moment.')
        document.dispatchEvent(
          new CustomEvent('toast', { detail: { message: "Pas d'autres liens pour le moment." } }),
        )
      }
    } catch (error) {
      if (signal.aborted) return
      automaticPaused = true
      console.error(error)
      setFeedStatus('Impossible de charger les liens curius.')
      document.dispatchEvent(
        new CustomEvent('toast', {
          detail: {
            message: 'Échec de la récupération des liens. Réessayez avec la flèche suivante.',
            durationMs: 5000,
          },
        }),
      )
    } finally {
      loading = false
      if (!signal.aborted) {
        navigation.removeAttribute('aria-busy')
        fragment.removeAttribute('aria-busy')
        updateSentinel()
        publishState()
        queueEndCheck()
      }
    }
  }

  document.addEventListener(
    'curius:links-ready',
    ({ detail }) => {
      if (ready) return
      const links = detail?.links ?? []
      const rows = Array.from(fragment.querySelectorAll<HTMLElement>(':scope > .curius-item'))
      lastLoadedPage = detail?.page ?? 0
      hasMore = links.length > 0 && (detail?.hasMore ?? true)
      for (const link of links) if (link.trails.length === 0) seen.add(linkKey(link))
      for (const row of rows) row.dataset.curiusPage = String(lastLoadedPage)
      if (rows[0]) pages.push({ page: lastLoadedPage, first: rows[0] })
      ready = true
      updateSentinel()
      publishState()
      queueEndCheck()
    },
    { signal },
  )
  fragment.addEventListener(
    'scroll',
    () => {
      updateVisiblePage()
      if (!nearEnd()) automaticPaused = false
      else queueEndCheck()
    },
    { signal, passive: true },
  )
  prevButton.addEventListener(
    'click',
    () => {
      if (!loading && pages[currentIndex - 1]) scrollToPage(pages[currentIndex - 1])
    },
    { signal },
  )
  nextButton.addEventListener(
    'click',
    () => {
      if (loading) return
      const next = pages[currentIndex + 1]
      if (next) scrollToPage(next)
      else {
        automaticPaused = false
        void appendNextPage(true)
      }
    },
    { signal },
  )
  updateNavigation()
  window.addCleanup(() => {
    controller.abort()
    observer.disconnect()
    if (frame !== undefined) cancelAnimationFrame(frame)
  })
})
