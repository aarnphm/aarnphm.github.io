document.addEventListener('nav', () => {
  const deck = document.querySelector<HTMLDivElement>('.slides-deck')
  const slides = Array.from(document.querySelectorAll<HTMLElement>('.slide'))
  const prev = document.querySelector<HTMLButtonElement>('.slides-controls .prev')
  const next = document.querySelector<HTMLButtonElement>('.slides-controls .next')
  const status = document.querySelector<HTMLSpanElement>('.slides-controls .status')
  const toc = document.querySelector<HTMLElement>('.slides-toc')
  const tocItems = Array.from(toc?.querySelectorAll<HTMLAnchorElement>('.slides-toc-item') ?? [])
  const tocList = toc?.querySelector<HTMLOListElement>('.slides-toc-list')
  const tocScroller = toc?.querySelector<HTMLElement>('.slides-toc-list-scroll')
  const progress = document.querySelector<HTMLDivElement>('.slides-controls .slides-progress')
  const progressBar = document.querySelector<HTMLDivElement>(
    '.slides-controls .slides-progress-bar',
  )
  if (!deck || slides.length === 0 || !prev || !next || !status) return

  let idx = 0
  const clamp = (v: number) => Math.max(0, Math.min(slides.length - 1, v))

  const parseHash = () => {
    const h = window.location.hash
    const m = h.match(/slide-(\d+)/)
    if (m) {
      const n = parseInt(m[1], 10)
      if (!Number.isNaN(n)) idx = clamp(n)
    }
  }

  const update = (scroll: boolean = true) => {
    const activePopups = document.querySelectorAll('#mermaid-container.active')
    activePopups.forEach(popup => popup.classList.remove('active'))

    slides.forEach((el, i) => {
      el.classList.toggle('active', i === idx)
      el.setAttribute('aria-hidden', i === idx ? 'false' : 'true')
    })
    status.textContent = `${idx + 1} / ${slides.length}`
    const pct = ((idx + 1) / slides.length) * 100
    if (progressBar) progressBar.style.width = `${pct}%`
    if (progress) progress.setAttribute('aria-valuenow', String(idx + 1))
    const tocProgress = slides.length > 1 ? idx / (slides.length - 1) : 1
    if (tocList) tocList.style.setProperty('--slides-toc-progress', String(tocProgress))
    tocItems.forEach((item, i) => {
      const active = i === idx
      item.classList.toggle('is-active', active)
      item.classList.toggle('is-complete', i <= idx)
      if (active) {
        item.setAttribute('aria-current', 'step')
      } else {
        item.removeAttribute('aria-current')
      }
    })
    const activeTocItem = tocItems[idx]
    if (tocScroller && activeTocItem) {
      const tocRect = tocScroller.getBoundingClientRect()
      const itemRect = activeTocItem.getBoundingClientRect()
      if (itemRect.top < tocRect.top) {
        tocScroller.scrollTop -= tocRect.top - itemRect.top + 12
      } else if (itemRect.bottom > tocRect.bottom) {
        tocScroller.scrollTop += itemRect.bottom - tocRect.bottom + 12
      }
    }
    deck.scrollTop = 0
    if (scroll) {
      const target = document.getElementById(`slide-${idx}`)
      if (target) target.scrollIntoView({ behavior: 'auto', block: 'start' })
    }
    history.replaceState(null, '', `#slide-${idx}`)
    requestAnimationFrame(() => {
      deck.classList.toggle('gradient-active', deck.scrollHeight > deck.clientHeight)
    })

    document.dispatchEvent(new CustomEvent('slidechange', { detail: {} }))
  }

  const goPrev = () => {
    if (idx > 0) idx -= 1
    update()
  }
  const goNext = () => {
    if (idx < slides.length - 1) idx += 1
    update()
  }

  const expandAllCallouts = () => {
    const callouts = deck.querySelectorAll<HTMLElement>('blockquote.callout, .callout')
    for (const el of Array.from(callouts)) {
      el.classList.remove('is-collapsed')
      if (el.style && typeof el.style.maxHeight !== 'undefined') el.style.maxHeight = ''
      const descendants = el.querySelectorAll<HTMLElement>("[style*='max-height']")
      descendants.forEach(child => (child.style.maxHeight = ''))
    }
  }

  const expandAllTranscludes = () => {
    const transcludes = deck.querySelectorAll<HTMLElement>('.transclude-collapsible')
    for (const el of Array.from(transcludes)) {
      el.classList.remove('is-collapsed')
      const content = el.querySelector<HTMLElement>('.transclude-content')
      if (content && content.style) {
        content.style.gridTemplateRows = '1fr'
      }
      const descendants = el.querySelectorAll<HTMLElement>('.transclude-content')
      descendants.forEach(child => {
        if (child.style) child.style.gridTemplateRows = '1fr'
      })
    }
  }

  parseHash()
  expandAllCallouts()
  expandAllTranscludes()
  update(false)

  const keyEvent = (e: KeyboardEvent) => {
    const target = e.target
    if (
      target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      target instanceof HTMLSelectElement ||
      (target instanceof HTMLElement && target.isContentEditable)
    ) {
      return
    }

    if (e.key === 'ArrowLeft') {
      e.preventDefault()
      goPrev()
    }
    if (e.key === 'ArrowRight' || e.key === ' ') {
      e.preventDefault()
      goNext()
    }
  }

  const tocClick = (event: MouseEvent) => {
    if (!(event.target instanceof Element)) return

    const item = event.target.closest<HTMLAnchorElement>('.slides-toc-item')
    if (!item) return

    const nextIdx = Number(item.dataset.slideTarget)
    if (!Number.isInteger(nextIdx)) return

    event.preventDefault()
    idx = clamp(nextIdx)
    update()
  }

  const onDeckScroll = () => {
    const atBottom = deck.scrollTop + deck.clientHeight >= deck.scrollHeight - 4
    deck.classList.toggle('gradient-active', !atBottom)
  }

  prev.addEventListener('click', goPrev)
  next.addEventListener('click', goNext)
  toc?.addEventListener('click', tocClick)
  window.addEventListener('keydown', keyEvent)
  deck.addEventListener('scroll', onDeckScroll)
  window.addCleanup(() => {
    prev.removeEventListener('click', goPrev)
    next.removeEventListener('click', goNext)
    toc?.removeEventListener('click', tocClick)
    window.removeEventListener('keydown', keyEvent)
    deck.removeEventListener('scroll', onDeckScroll)
  })
})
