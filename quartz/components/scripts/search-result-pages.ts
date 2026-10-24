const PAGE_SIZE = 60

export type ResultPages = ReturnType<typeof mountResultPages>

export function mountResultPages<T>(
  container: HTMLElement,
  entries: readonly T[],
  append: (page: readonly T[], signal: AbortSignal) => HTMLElement | null,
) {
  let offset = 0
  let frame = 0
  let disposed = false
  const events = new AbortController()
  const more = document.createElement('button')
  more.type = 'button'
  more.className = 'search-load-more'
  more.textContent = 'load more'
  more.setAttribute('aria-label', 'load more search results')

  const observer = new IntersectionObserver(
    observations => {
      if (
        document.activeElement !== more &&
        observations.some(observation => observation.isIntersecting)
      )
        loadMore()
    },
    { root: container, rootMargin: '0px 0px 320px', threshold: 0 },
  )

  function observeMore() {
    if (!disposed && offset < entries.length && document.activeElement !== more)
      observer.observe(more)
  }

  function loadMore() {
    if (disposed || offset >= entries.length) return
    observer.disconnect()
    cancelAnimationFrame(frame)
    const focusNext = document.activeElement === more
    const page = entries.slice(offset, offset + PAGE_SIZE)
    offset += page.length
    const first = append(page, events.signal)
    if (focusNext) first?.focus()
    if (offset >= entries.length) {
      more.remove()
      return
    }
    container.append(more)
    // Let the image grid pack appended cards before measuring the end of the results.
    frame = requestAnimationFrame(() => {
      frame = requestAnimationFrame(() => {
        frame = 0
        observeMore()
      })
    })
  }

  more.addEventListener('click', loadMore)
  // Keep native activation stable when focus scrolls the button into the prefetch margin.
  more.addEventListener('focus', () => observer.disconnect(), { signal: events.signal })
  more.addEventListener('blur', observeMore, { signal: events.signal })
  loadMore()

  return {
    dispose() {
      disposed = true
      events.abort()
      cancelAnimationFrame(frame)
      observer.disconnect()
      more.removeEventListener('click', loadMore)
      more.remove()
    },
  }
}
