import { rootNavSignal } from './root-lifecycle'

document.addEventListener('nav', () => {
  for (const board of document.querySelectorAll<HTMLElement>('.base-kanban')) {
    if (board.dataset.kanbanInitialized !== undefined) continue
    board.dataset.kanbanInitialized = ''
    const signal = rootNavSignal(board)
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')
    const motions = new Map<HTMLElement, Animation>()
    const titles = board.querySelectorAll<HTMLElement>(
      '.base-kanban-column-title, .base-card-title',
    )

    const stop = (title: HTMLElement) => {
      motions.get(title)?.cancel()
      motions.delete(title)
      delete title.dataset.kanbanScrolling
    }
    const stopAll = () => {
      for (const title of titles) stop(title)
    }

    for (const title of titles) {
      title.title = title.textContent ?? ''
      if (title.classList.contains('base-kanban-column-title')) continue

      let text = title.querySelector<HTMLElement>(':scope > .base-kanban-title-text')
      if (!text) {
        text = document.createElement('span')
        text.className = 'base-kanban-title-text'
        text.append(...title.childNodes)
        title.append(text)
      }
      const titleText = text
      const focusTarget = title.closest<HTMLAnchorElement>('a') ?? title

      const start = () => {
        if (reducedMotion.matches || motions.has(title)) return
        title.dataset.kanbanScrolling = 'true'
        const distance = titleText.getBoundingClientRect().width - title.clientWidth
        if (distance <= 1) {
          delete title.dataset.kanbanScrolling
          return
        }

        const duration = (distance / 40) * 1000 + 1200
        const end = `translateX(${-distance}px)`
        motions.set(
          title,
          titleText.animate(
            [
              { transform: 'translateX(0)', offset: 0 },
              { transform: 'translateX(0)', offset: 600 / duration },
              { transform: end, offset: 1 - 600 / duration },
              { transform: end, offset: 1 },
            ],
            { duration, easing: 'linear', fill: 'forwards' },
          ),
        )
      }

      title.addEventListener('pointerenter', start, { signal })
      title.addEventListener(
        'pointerleave',
        () => {
          if (!focusTarget.matches(':focus-visible')) stop(title)
        },
        { signal },
      )
      focusTarget.addEventListener(
        'focus',
        () => {
          if (focusTarget.matches(':focus-visible')) start()
        },
        { signal },
      )
      focusTarget.addEventListener(
        'blur',
        () => {
          if (!title.matches(':hover')) stop(title)
        },
        { signal },
      )
    }

    const updateHeadingFocus = () => {
      if (signal.aborted) return
      for (const title of titles) {
        if (title.classList.contains('base-kanban-column-title')) {
          if (title.scrollWidth > title.clientWidth) title.tabIndex = 0
          else title.removeAttribute('tabindex')
        }
      }
    }
    const resize = new ResizeObserver(updateHeadingFocus)
    resize.observe(board)
    updateHeadingFocus()
    void document.fonts.ready.then(updateHeadingFocus)
    reducedMotion.addEventListener('change', stopAll, { signal })

    const cleanup = () => {
      stopAll()
      resize.disconnect()
      delete board.dataset.kanbanInitialized
    }
    signal.addEventListener('abort', cleanup, { once: true })
    window.addCleanup(cleanup)
  }
})

export default ''
