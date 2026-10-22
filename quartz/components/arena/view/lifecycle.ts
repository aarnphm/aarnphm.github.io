import type { ArenaEvent, SearchScope } from '../model'
import { mountArenaChannel } from '../channel'
import { createArenaExternalEmbeds } from '../external-embed'
import {
  cleanupMaps,
  cleanupPdfs,
  hydrateInternalHosts,
  hydrateMapboxMaps,
  hydrateSubstackEmbeds,
  mapInstances,
  mountPdfEmbeds,
} from './embeds'
import { closeModal } from './modal'
import { createModalDataFromJson } from './modal-markup'
import { fetchArenaSearchIndex, getSearchResults } from './search'
import { addCleanup, session } from './session-state'

export const mountArena = (dispatch: (event: ArenaEvent) => void) => {
  const cleanups: Array<() => void> = []
  const previousRegister = session.registerCleanup
  const register = (cleanup: () => void) => {
    cleanups.push(cleanup)
  }
  session.registerCleanup = register
  const embeds = createArenaExternalEmbeds()
  session.externalEmbeds = embeds
  addCleanup(() => {
    embeds.cleanup()
    if (session.externalEmbeds === embeds) session.externalEmbeds = null
  })

  const entryPage = document.querySelector<HTMLElement>('.arena-entry-page')
  if (entryPage) {
    hydrateSubstackEmbeds(entryPage)
    embeds.mount(entryPage)
    hydrateInternalHosts(entryPage)
    hydrateMapboxMaps(entryPage)
    mountPdfEmbeds(entryPage)
    addCleanup(() => {
      cleanupMaps(entryPage)
      cleanupPdfs(entryPage)
    })
  }

  session.totalBlocks = document.querySelectorAll('[data-block-id][data-block-index]').length

  const channelPage = document.querySelector<HTMLElement>('.arena-channel-page')
  session.channelViews = channelPage ? mountArenaChannel(channelPage) : null
  if (session.channelViews) session.totalBlocks = session.channelViews.blockIds.length
  session.arenaSearchData = null
  session.searchIndex = []
  session.searchRequest?.abort()
  session.searchRequest = new AbortController()
  addCleanup(() => {
    closeModal()
    session.channelViews?.destroy()
    session.channelViews = null
    session.arenaSearchData = null
    session.searchIndex = []
    session.searchRequest?.abort()
    session.searchRequest = null
  })
  const blockCollection = document.getElementById('arena-block-collection') as HTMLElement | null
  const viewToggleButtons = channelPage
    ? Array.from(channelPage.querySelectorAll<HTMLButtonElement>('.arena-view-toggle-button'))
    : []

  type ArenaViewMode = 'grid' | 'list'

  const applyViewMode = (mode: ArenaViewMode) => {
    if (!channelPage || !blockCollection) return
    const normalized: ArenaViewMode = mode === 'list' ? 'list' : 'grid'
    channelPage.dataset.viewMode = normalized
    blockCollection.dataset.viewMode = normalized
    session.channelViews?.refresh()
    viewToggleButtons.forEach(button => {
      const targetMode = (button.dataset.viewMode as ArenaViewMode) || 'grid'
      const isActive = targetMode === normalized
      button.classList.toggle('active', isActive)
      button.setAttribute('aria-pressed', isActive ? 'true' : 'false')
    })
  }

  if (channelPage && blockCollection && viewToggleButtons.length > 0) {
    const initialMode = (blockCollection.dataset.viewMode as ArenaViewMode | undefined) ?? 'grid'
    applyViewMode(initialMode)

    viewToggleButtons.forEach(button => {
      const desiredMode = (button.dataset.viewMode as ArenaViewMode) || 'grid'
      const onToggleClick = () => applyViewMode(desiredMode)
      button.addEventListener('click', onToggleClick)
      addCleanup(() => button.removeEventListener('click', onToggleClick))
    })
  }

  // Build search index
  session.searchInput = document.querySelector<HTMLInputElement>('.arena-search-input')
  session.activeResultIndex = null

  if (session.searchInput) {
    const input = session.searchInput
    const scope = input.getAttribute('data-search-scope') as SearchScope
    dispatch({ type: 'search.index.request', scope })

    if (session.searchDebounceTimer) {
      window.clearTimeout(session.searchDebounceTimer)
    }

    const onSearchInput = (e: Event) => {
      const target = e.target as HTMLInputElement
      const query = target.value

      window.clearTimeout(session.searchDebounceTimer)
      session.searchDebounceTimer = window.setTimeout(() => {
        dispatch({ type: 'ui.search.query', query, scope })
      }, 300)
    }

    input.addEventListener('input', onSearchInput)
    addCleanup(() => input.removeEventListener('input', onSearchInput))
    addCleanup(() => {
      if (session.searchDebounceTimer) {
        window.clearTimeout(session.searchDebounceTimer)
        session.searchDebounceTimer = undefined
      }
    })
  }

  const onClick = async (e: MouseEvent) => {
    const target = e.target as HTMLElement

    const internalLink = target.closest('.arena-modal-body a.internal') as HTMLAnchorElement | null
    if (internalLink) {
      if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) {
        return
      }

      e.preventDefault()

      // Check if this is a wikilink trail anchor - open in stacked notes
      const isWikilinkTrail = internalLink.classList.contains('arena-wikilink-trail-anchor')

      if (isWikilinkTrail && typeof window.stacked !== 'undefined') {
        try {
          const destination = new URL(internalLink.href)
          await window.stacked.navigate(destination)
          return
        } catch (err) {
          console.error('Failed to open in stacked notes:', err)
        }
      }

      // Regular internal link - close modal and navigate
      dispatch({ type: 'ui.modal.close' })
      try {
        const destination = new URL(internalLink.href)
        if (typeof window.spaNavigate === 'function') {
          window.spaNavigate(destination)
        } else {
          window.location.assign(destination)
        }
      } catch (err) {
        console.error(err)
        window.location.assign(internalLink.href)
      }
      return
    }

    const copyButton = target.closest('button.arena-url-copy-button') as HTMLElement | null
    if (copyButton) {
      e.preventDefault()
      e.stopPropagation()
      dispatch({ type: 'ui.copy', button: copyButton })
      return
    }

    // Click anywhere on a channel row should navigate to the header link
    const channelRow = target.closest('.arena-channel-row') as HTMLElement | null
    if (channelRow) {
      // If the user clicked a real interactive element, let it handle itself
      if (target.closest('a,button,[role=button],input,textarea,select,summary')) {
        return
      }
      const headerLink = channelRow.querySelector(
        '.arena-channel-row-header a[href]',
      ) as HTMLAnchorElement | null
      if (headerLink) {
        e.preventDefault()
        // prefer native navigation so SPA router (if any) can hook the event
        headerLink.click()
      }
      return
    }

    if (target.closest('button.arena-modal-prev')) {
      dispatch({ type: 'ui.block.navigate', direction: -1 })
      return
    }

    if (target.closest('button.arena-modal-next')) {
      dispatch({ type: 'ui.block.navigate', direction: 1 })
      return
    }

    if (target.closest('.arena-modal-collapse')) {
      const collapseBtn = target.closest('.arena-modal-collapse') as HTMLElement | null
      const host = collapseBtn?.closest('#arena-modal, .arena-entry-shell')
      const sidebar = host?.querySelector('.arena-modal-sidebar') as HTMLElement | null
      if (sidebar) {
        sidebar.classList.toggle('collapsed')
        collapseBtn?.classList.toggle('active')
        collapseBtn?.setAttribute('aria-expanded', String(!sidebar.classList.contains('collapsed')))

        // Resize maps after layout change - wait for CSS transition
        const modalBody = host?.querySelector('.arena-modal-body') as HTMLElement | null
        if (modalBody) {
          const resizeMaps = () => {
            modalBody
              .querySelectorAll<HTMLElement>(".arena-modal-map[data-map-initialized='1']")
              .forEach(mapNode => {
                const map = mapInstances.get(mapNode)
                if (map) {
                  try {
                    map.resize()
                  } catch (error) {
                    console.error(error)
                  }
                }
              })
          }

          const onTransitionEnd = (event: TransitionEvent) => {
            if (event.target === sidebar && event.propertyName === 'width') {
              sidebar.removeEventListener('transitionend', onTransitionEnd)
              requestAnimationFrame(resizeMaps)
            }
          }

          sidebar.addEventListener('transitionend', onTransitionEnd)
          setTimeout(() => {
            sidebar.removeEventListener('transitionend', onTransitionEnd)
            requestAnimationFrame(resizeMaps)
          }, 500)
        }
      }
      return
    }

    // Handle search result clicks with smart navigation based on hasModalInDom
    const searchResultItem = target.closest('.arena-search-result-item') as HTMLElement | null
    if (searchResultItem) {
      const blockId = searchResultItem.getAttribute('data-block-id')
      const channelSlug = searchResultItem.getAttribute('data-channel-slug')

      if (blockId) {
        e.preventDefault()
        dispatch({ type: 'ui.search.clear', blur: true })

        if (session.channelViews) {
          dispatch({ type: 'ui.modal.open', blockId })
          return
        }

        // Check if this is an arxiv block without notes and redirect instead of opening modal
        if (session.arenaSearchData) {
          const blockData = session.arenaSearchData.blocks.find(b => b.id === blockId)
          const isArxivUrl = blockData?.url
            ? /^https?:\/\/(?:ar5iv\.(?:labs\.)?)?arxiv\.org\//i.test(blockData.url)
            : false

          if (isArxivUrl && blockData?.url) {
            // Only redirect if the block has no notes (description or content)
            const hasNotes = !!blockData.content
            if (!hasNotes) {
              window.open(blockData.url, '_blank', 'noopener,noreferrer')
              return
            }
            // If it has notes, continue to show the modal (without embed)
          }
        }

        // Find the result in search index to check hasModalInDom flag
        const resultData = session.searchIndex.find(item => item.blockId === blockId)

        if (resultData?.hasModalInDom) {
          // Block has prerendered modal in DOM - show it instantly
          dispatch({ type: 'ui.modal.open', blockId })
        } else {
          // Block doesn't have modal in DOM - create it dynamically from JSON
          const existingModal = document.getElementById(`arena-modal-data-${blockId}`)

          if (!existingModal && channelSlug) {
            // Ensure search data is loaded
            if (!session.arenaSearchData) {
              await fetchArenaSearchIndex()
            }

            if (!session.arenaSearchData) {
              console.error('Failed to load arena search data')
              return
            }

            // Find full block data in search JSON
            const fullBlockData = session.arenaSearchData.blocks.find(b => b.id === blockId)

            if (fullBlockData) {
              // Create modal data element from JSON
              const modalDataEl = createModalDataFromJson(fullBlockData, channelSlug)

              // Append to body so getElementById can find it easily
              document.body.appendChild(modalDataEl)

              // Now show the modal
              dispatch({ type: 'ui.modal.open', blockId })
            } else if (channelSlug) {
              // Fallback: navigate to channel page if we can't find the data
              const currentSlug = document.body?.dataset.slug || ''
              const targetChannelSlug = `arena/${channelSlug}`

              if (currentSlug === targetChannelSlug) {
                dispatch({ type: 'ui.modal.open', blockId })
              } else {
                const targetUrl = new URL(
                  `/${['arena', ...channelSlug.split('/')].map(encodeURIComponent).join('/')}`,
                  window.location.origin,
                )
                targetUrl.hash = blockId
                window.location.assign(targetUrl)
              }
            } else {
              // Last resort: try to show modal anyway
              dispatch({ type: 'ui.modal.open', blockId })
            }
          } else {
            // Modal data already exists, just show it
            dispatch({ type: 'ui.modal.open', blockId })
          }
        }
      }
      return
    }

    if (target.closest('.arena-modal-close') || target.classList.contains('arena-block-modal')) {
      dispatch({ type: 'ui.modal.close' })
    }
  }

  const onKey = (e: KeyboardEvent) => {
    const key = e.key.toLowerCase()
    if ((e.metaKey || e.ctrlKey) && key === 'k') {
      e.preventDefault()
      if (e.shiftKey) {
        dispatch({ type: 'ui.search.focus', prefill: '#' })
      } else {
        dispatch({ type: 'ui.search.focus' })
      }
      return
    }

    const results = getSearchResults()
    const searchContainer = document.getElementById('arena-search-container')
    const searchOpen = Boolean(searchContainer && searchContainer.classList.contains('active'))
    const resultFocused = document.activeElement?.classList.contains('arena-search-result-item')
    const inputFocused = document.activeElement === session.searchInput

    if (searchOpen && results.length > 0) {
      if (
        key === 'ArrowDown' ||
        (!e.shiftKey && key === 'Tab') ||
        (key === 'n' && (e.ctrlKey || e.metaKey))
      ) {
        e.preventDefault()
        if (
          !e.shiftKey &&
          key === 'Tab' &&
          resultFocused &&
          session.activeResultIndex === results.length - 1
        ) {
          dispatch({
            type: 'ui.search.result.activate',
            index: null,
            options: { focus: true, scroll: false },
          })
        } else if (inputFocused || session.activeResultIndex === null) {
          dispatch({ type: 'ui.search.result.activate', index: 0 })
        } else {
          const nextIndex = Math.min((session.activeResultIndex ?? -1) + 1, results.length - 1)
          dispatch({ type: 'ui.search.result.activate', index: nextIndex })
        }
        return
      }

      if (
        key === 'ArrowUp' ||
        (e.shiftKey && key === 'Tab') ||
        (key === 'p' && (e.ctrlKey || e.metaKey))
      ) {
        e.preventDefault()
        if (!resultFocused || session.activeResultIndex === null) {
          dispatch({ type: 'ui.search.result.activate', index: results.length - 1 })
        } else if (session.activeResultIndex <= 0) {
          dispatch({
            type: 'ui.search.result.activate',
            index: null,
            options: { focus: true, scroll: false },
          })
        } else {
          dispatch({ type: 'ui.search.result.activate', index: session.activeResultIndex - 1 })
        }
        return
      }

      if (key === 'enter') {
        e.preventDefault()
        if (resultFocused) {
          document.activeElement?.dispatchEvent(new MouseEvent('click', { bubbles: true }))
        } else if (inputFocused && results.length > 0) {
          // Open first result if Enter pressed in input
          results[0]?.dispatchEvent(new MouseEvent('click', { bubbles: true }))
        }
        return
      }
    }

    if (key === 'escape') {
      if (searchContainer && searchContainer.classList.contains('active')) {
        dispatch({ type: 'ui.search.close' })
      } else {
        dispatch({ type: 'ui.modal.close' })
      }
      return
    }

    // Handle PDF page navigation with Shift+Arrow
    if (e.shiftKey && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
      const modal = document.getElementById('arena-modal')
      const modalBody = modal?.querySelector('.arena-modal-body')
      const pdfViewer = modalBody?.querySelector(".pdf-embed[data-pdf-status='loaded']")

      if (pdfViewer) {
        e.preventDefault()
        const controls = pdfViewer.querySelector('.pdf-embed-toolbar')
        if (controls) {
          if (e.key === 'ArrowLeft') {
            const prevBtn = controls.querySelector('.pdf-embed-prev') as HTMLButtonElement
            if (prevBtn && !prevBtn.disabled) {
              prevBtn.click()
            }
          } else {
            const nextBtn = controls.querySelector('.pdf-embed-next') as HTMLButtonElement
            if (nextBtn && !nextBtn.disabled) {
              nextBtn.click()
            }
          }
        }
        return
      }
    }

    if (e.key === 'ArrowLeft') {
      dispatch({ type: 'ui.block.navigate', direction: -1 })
    } else if (e.key === 'ArrowRight') {
      dispatch({ type: 'ui.block.navigate', direction: 1 })
    }
  }

  document.addEventListener('click', onClick)
  document.addEventListener('keydown', onKey)
  addCleanup(() => document.removeEventListener('click', onClick))
  addCleanup(() => document.removeEventListener('keydown', onKey))

  return () => {
    const pending = cleanups.slice()
    cleanups.length = 0
    for (const cleanup of pending) {
      cleanup()
    }
    if (session.registerCleanup === register) {
      session.registerCleanup = previousRegister
    }
    session.searchInput = null
    session.activeResultIndex = null
  }
}
