import { registerEscapeHandler } from './escape-handler'

const SLIDE_SHORTCUTS: [string, string][] = [
  ['← →', 'slide'],
  ['o', 'overview'],
  ['p', 'present'],
  ['t', 'timer'],
  ['1–9', 'go to slide'],
  ['Home End', 'first, last'],
]

document.addEventListener('nav', () => {
  const root = document.querySelector<HTMLElement>('.slides-root')
  const deck = document.querySelector<HTMLDivElement>('.slides-deck')
  const slides = Array.from(document.querySelectorAll<HTMLElement>('.slide'))
  const prev = document.querySelector<HTMLAnchorElement>('.slides-controls .prev')
  const next = document.querySelector<HTMLAnchorElement>('.slides-controls .next')
  const status = document.querySelector<HTMLSpanElement>('.slides-controls .status')
  const timerEl = document.querySelector<HTMLSpanElement>('.slides-controls .slides-timer')
  const overviewBtn = document.querySelector<HTMLButtonElement>('.slides-controls .overview')
  const presentBtn = document.querySelector<HTMLButtonElement>('.slides-controls .present')
  const toc = document.querySelector<HTMLElement>('.slides-toc')
  const railToggle = toc?.querySelector<HTMLButtonElement>('.slides-toc-toggle')
  const tocEntries = Array.from(toc?.querySelectorAll<HTMLLIElement>('.slides-toc-entry') ?? [])
  const tocItems = Array.from(toc?.querySelectorAll<HTMLAnchorElement>('.slides-toc-item') ?? [])
  const tocList = toc?.querySelector<HTMLOListElement>('.slides-toc-list')
  const tocScroller = toc?.querySelector<HTMLElement>('.slides-toc-list-scroll')
  const progress = document.querySelector<HTMLDivElement>('.slides-controls .slides-progress')
  const segments = Array.from(
    progress?.querySelectorAll<HTMLElement>('.slides-progress-segment') ?? [],
  )
  if (!root || !deck || slides.length === 0 || !prev || !next || !status) return

  // the page renders its own slide; the others arrive as fragments
  let idx = Math.max(
    0,
    slides.findIndex(slide => slide.classList.contains('active')),
  )
  let overview = false
  let presenting = false
  const clamp = (v: number) => Math.max(0, Math.min(slides.length - 1, v))
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const canPresent =
    typeof root.requestFullscreen === 'function' && document.fullscreenEnabled === true

  const updateTocOverflow = () => {
    if (!tocScroller) return

    const scrollable = tocScroller.scrollHeight > tocScroller.clientHeight + 1
    const atStart = tocScroller.scrollTop <= 1
    const atEnd = tocScroller.scrollTop + tocScroller.clientHeight >= tocScroller.scrollHeight - 1

    tocScroller.classList.toggle('is-scrollable', scrollable)
    tocScroller.classList.toggle('at-start', scrollable && atStart)
    tocScroller.classList.toggle('at-end', scrollable && atEnd)
  }

  const railHidden = () => root.classList.contains('is-rail-collapsed')
  const sublists = tocEntries.map(entry =>
    entry.querySelector<HTMLElement>(':scope > .slides-toc-sublist'),
  )

  // Open the current slide's sublist and fold the rest. Heights are explicit px so
  // they transition together. Returns how far each entry moves once they settle,
  // because the scroll target needs that geometry, not this frame's.
  // An entry's min-height absorbs part of a short sublist, so predict entry heights.
  const settleSublists = (): { shift: number; height: number }[] => {
    if (railHidden()) return []
    const rows = tocEntries.map((entry, i) => {
      const sub = sublists[i]
      const now = entry.getBoundingClientRect().height
      const to = sub && i === idx ? sub.scrollHeight : 0
      const floor = parseFloat(getComputedStyle(entry).minHeight) || 0
      const next = sub ? Math.max(floor, tocItems[i].offsetHeight + to) : now
      return { sub, to, next, delta: next - now }
    })
    let shift = 0
    return rows.map(({ sub, to, next, delta }) => {
      const entryShift = shift
      if (sub) sub.style.height = `${to}px`
      shift += delta
      return { shift: entryShift, height: next }
    })
  }

  // The rail scroll rides the sublists' curve (cubic-bezier(0.23, 1, 0.32, 1) over
  // 220ms, $panel-ease and $panel-duration), so entries and scroll move as one.
  // Smooth scrollBy runs on the browser's own curve and visibly lags the lists.
  const panelEase = (t: number) => {
    const bezier = (a: number, b: number, u: number) =>
      3 * a * u * (1 - u) ** 2 + 3 * b * u * u * (1 - u) + u ** 3
    let lo = 0
    let hi = 1
    let u = t
    for (let i = 0; i < 20; i++) {
      u = (lo + hi) / 2
      if (bezier(0.23, 0.32, u) < t) lo = u
      else hi = u
    }
    return bezier(1, 1, u)
  }
  let tocTween = 0
  const scrollRail = (delta: number, animate: boolean) => {
    cancelAnimationFrame(tocTween)
    if (!tocScroller || Math.abs(delta) < 1) return
    const from = tocScroller.scrollTop
    if (!animate) {
      tocScroller.scrollTop = from + delta
      return
    }
    const start = performance.now()
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / 220)
      tocScroller.scrollTop = from + delta * panelEase(t)
      if (t < 1) tocTween = requestAnimationFrame(step)
      else updateTocOverflow()
    }
    tocTween = requestAnimationFrame(step)
  }

  const activeSubItems = () =>
    Array.from(tocEntries[idx]?.querySelectorAll<HTMLAnchorElement>('.slides-toc-subitem') ?? [])

  // A heading's scroll offset is its distance from the slide's top edge. At that
  // offset it rests where the slide heading rests at scrollTop 0. Both rects carry
  // the enter animation's translate, so the difference ignores it.
  const headingOffset = (heading: HTMLElement) =>
    heading.getBoundingClientRect().top - slides[idx].getBoundingClientRect().top

  // scroll-spy: the last subheading at or above its resting position is current
  // a rail square's centre, from the ::before that draws it
  const nodeCenter = (el: HTMLElement) => {
    const square = getComputedStyle(el, '::before')
    return el.getBoundingClientRect().top + parseFloat(square.top) + parseFloat(square.height) / 2
  }

  // The ink runs down the current slide's rail segment as the deck scrolls. It
  // reaches a subsection's square as that subsection becomes current, so the
  // position moves continuously between squares. The segment ends at the next
  // slide's square, at the entry's settled height.
  const setSubProgress = (anchors: [number, number][], atEnd: boolean) => {
    const entry = tocEntries[idx]
    const sub = sublists[idx]
    if (!entry || railHidden()) return
    let y = 0
    if (atEnd && anchors.length > 0) {
      y = anchors[anchors.length - 1][1]
    } else {
      const s = deck.scrollTop
      for (let i = 0; i < anchors.length; i++) {
        const [x1, y1] = anchors[i]
        const [x0, y0] = i === 0 ? [0, 0] : anchors[i - 1]
        if (s >= x1) {
          y = y1
          continue
        }
        y = x1 > x0 ? y0 + ((s - x0) / (x1 - x0)) * (y1 - y0) : y0
        break
      }
    }
    const floor = parseFloat(getComputedStyle(entry).minHeight) || 0
    const height = Math.max(floor, tocItems[idx].offsetHeight + (sub?.scrollHeight ?? 0))
    const progress = height > 0 ? Math.min(1, Math.max(0, y / height)) : 0
    entry.style.setProperty('--slides-toc-sub-progress', progress.toFixed(4))
  }

  const updateSubSpy = () => {
    const subs = activeSubItems()
    const scrollable = deck.scrollHeight > deck.clientHeight
    const atEnd = scrollable && deck.scrollTop + deck.clientHeight >= deck.scrollHeight - 4
    const maxScroll = Math.max(0, deck.scrollHeight - deck.clientHeight)
    const itemCenter = subs.length > 0 ? nodeCenter(tocItems[idx]) : 0
    const anchors: [number, number][] = []
    let current = -1
    subs.forEach((sub, i) => {
      const heading = sub.dataset.headingId ? document.getElementById(sub.dataset.headingId) : null
      if (!heading) return
      const at = Math.max(0, headingOffset(heading) - 24)
      if (at <= deck.scrollTop) current = i
      anchors.push([Math.min(at, maxScroll), nodeCenter(sub) - itemCenter])
    })
    setSubProgress(anchors, atEnd)
    if (subs.length === 0) return
    if (atEnd) current = subs.length - 1
    subs.forEach((sub, i) => {
      const active = i === current
      sub.classList.toggle('is-active', active)
      if (active) {
        sub.setAttribute('aria-current', 'location')
      } else {
        sub.removeAttribute('aria-current')
      }
    })
  }

  // a new slide lands on the heading at once; within the current slide it scrolls
  const scrollDeckToHeading = (headingId: string, smooth: boolean) => {
    const heading = document.getElementById(headingId)
    if (!heading) return
    deck.scrollTo({
      top: Math.max(0, headingOffset(heading)),
      behavior: smooth && !reducedMotion ? 'smooth' : 'auto',
    })
  }

  // Sibling slide pages share a directory, so a slide's URL is ./<n>. Relative
  // links in the page and in fragments resolve the same from every slide.
  const slideUrl = (i: number) => new URL(`./${i + 1}`, window.location.href).href
  const setPager = (link: HTMLAnchorElement, target: number, enabled: boolean) => {
    if (enabled) {
      link.setAttribute('href', `./${target + 1}`)
      link.removeAttribute('aria-disabled')
      link.removeAttribute('role')
    } else {
      link.removeAttribute('href')
      link.setAttribute('aria-disabled', 'true')
      link.setAttribute('role', 'link')
    }
  }
  const plainClick = (e: MouseEvent) =>
    e.button === 0 && !e.metaKey && !e.ctrlKey && !e.shiftKey && !e.altKey

  // announce: tell listeners such as mermaid the visible slide changed. A freshly
  // mounted slide is announced as new content instead, so it initialises once.
  const update = (announce = true) => {
    const activePopups = document.querySelectorAll('#mermaid-container.active')
    activePopups.forEach(popup => popup.classList.remove('active'))

    slides.forEach((el, i) => {
      el.classList.toggle('active', i === idx)
      el.setAttribute('aria-hidden', overview || i === idx ? 'false' : 'true')
    })
    status.textContent = `${idx + 1} / ${slides.length}`
    const atStart = idx === 0
    const atEnd = idx === slides.length - 1
    const focused = document.activeElement
    setPager(prev, idx - 1, !atStart)
    setPager(next, idx + 1, !atEnd)
    // a bound disables the focused button; keep keyboard focus on the controls
    if (atStart && !atEnd && focused === prev) next.focus()
    if (atEnd && !atStart && focused === next) prev.focus()
    segments.forEach((segment, i) => segment.classList.toggle('is-complete', i <= idx))
    if (progress) progress.setAttribute('aria-valuenow', String(idx + 1))
    tocEntries.forEach((entry, i) => {
      entry.classList.toggle('is-active', i === idx)
      // a slide that is no longer current is full (passed) or empty, never partial
      if (i !== idx) entry.style.removeProperty('--slides-toc-sub-progress')
    })
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
    const settled = settleSublists()
    const activeTocItem = tocItems[idx]
    if (tocScroller && activeTocItem && settled[idx]) {
      // bring the settled entry, open sublist included, into view; its title wins
      const tocRect = tocScroller.getBoundingClientRect()
      const top = activeTocItem.getBoundingClientRect().top + settled[idx].shift
      const bottom = top + settled[idx].height
      const animate = !reducedMotion && !!tocList?.classList.contains('is-settled')
      if (top < tocRect.top + 12) {
        scrollRail(top - tocRect.top - 12, animate)
      } else if (bottom > tocRect.bottom - 12) {
        scrollRail(Math.min(bottom - tocRect.bottom + 12, top - tocRect.top - 12), animate)
      }
      updateTocOverflow()
    }
    // Only the active slide is displayed, so scrollTop 0 is its top. scrollIntoView
    // would align the section box, past the heading's collapsed margin and mid-way
    // through the enter translate, so each slide would rest at a different height.
    if (overview) {
      slides[idx].scrollIntoView({ block: 'nearest' })
    } else {
      deck.scrollTop = 0
    }
    const url = slideUrl(idx)
    if (url !== window.location.href) history.replaceState(null, '', url)
    requestAnimationFrame(() => {
      deck.classList.toggle('gradient-active', deck.scrollHeight > deck.clientHeight)
      updateTocOverflow()
      updateSubSpy()
    })

    if (announce) document.dispatchEvent(new CustomEvent('slidechange', { detail: {} }))
  }

  // Fragment text is fetched ahead for the neighbours but parsed into the deck only
  // when shown, so scripts that measure, like mermaid, run on a visible slide.
  const fragments = new Map<number, Promise<string>>()
  const fetchFragment = (i: number) => {
    let text = fragments.get(i)
    if (text) return text
    const src = slides[i].dataset.src
    // the worker answers an agent user agent's Accept: */* with markdown
    text = src
      ? fetch(src, { headers: { Accept: 'text/html' } }).then(response => {
          if (!response.ok) throw new Error(`slide fragment ${response.status}`)
          return response.text()
        })
      : Promise.reject(new Error('slide without fragment'))
    fragments.set(i, text)
    text.catch(() => fragments.delete(i))
    return text
  }
  const isMounted = (i: number) => 'loaded' in slides[i].dataset
  const mount = (i: number, html: string) => {
    const section = slides[i]
    const body = section.querySelector<HTMLElement>('.slide-body')
    if (isMounted(i) || !body) return null
    const template = document.createElement('template')
    template.innerHTML = html
    body.replaceChildren(template.content)
    section.dataset.loaded = ''
    expandAllCallouts(body)
    expandAllTranscludes(body)
    return body
  }
  const announceContent = (i: number, body: HTMLElement) =>
    document.dispatchEvent(
      new CustomEvent('contentdecrypted', {
        detail: { article: slides[i], content: body, slug: document.body.dataset.slug },
      }),
    )
  const prefetchAround = () => {
    for (const i of [idx + 1, idx - 1]) {
      if (i >= 0 && i < slides.length && !isMounted(i)) void fetchFragment(i)
    }
  }
  const mountAll = () =>
    Promise.all(
      slides.map(async (_, i) => {
        if (isMounted(i)) return
        const html = await fetchFragment(i).catch(() => null)
        const body = html === null ? null : mount(i, html)
        if (body) announceContent(i, body)
      }),
    )

  // The latest turn wins: a slow fragment never overrides a later key press. A
  // fragment that fails to load falls back to the slide's own page.
  let turn = 0
  const goTo = async (n: number): Promise<boolean> => {
    const target = clamp(n)
    const ticket = ++turn
    let fresh: HTMLElement | null = null
    if (!isMounted(target)) {
      let html: string
      try {
        html = await fetchFragment(target)
      } catch {
        if (ticket === turn) window.location.assign(slideUrl(target))
        return false
      }
      if (ticket !== turn) return false
      fresh = mount(target, html)
    }
    idx = target
    update(fresh === null)
    if (fresh) announceContent(target, fresh)
    prefetchAround()
    return true
  }
  const goPrev = () => void goTo(idx - 1)
  const goNext = () => void goTo(idx + 1)

  // Overview grid: the fewest columns whose rows still fit the deck, so the cards
  // are as large as the space allows; past that the grid scrolls. Each card shows
  // the slide at the scale that maps the slide's width onto the card.
  let slideWidth = 0
  const px = (v: string) => parseFloat(v) || 0
  const layoutOverview = () => {
    const cs = getComputedStyle(deck)
    const gap = px(cs.columnGap)
    const width = deck.clientWidth - px(cs.paddingLeft) - px(cs.paddingRight)
    const height = deck.clientHeight - px(cs.paddingTop) - px(cs.paddingBottom)
    const count = slides.length
    const minCard = 14 * px(getComputedStyle(document.documentElement).fontSize)
    const floorCols = Math.max(1, Math.floor((width + gap) / (minCard + gap)))
    let cols = floorCols
    for (let c = 1; c <= floorCols; c++) {
      const cardW = (width - gap * (c - 1)) / c
      const rows = Math.ceil(count / c)
      const total = rows * (cardW * 0.75) + gap * (rows - 1)
      if (total <= height) {
        cols = c
        break
      }
    }
    cols = Math.max(1, Math.min(cols, count))
    const cardW = (width - gap * (cols - 1)) / cols
    const pad = px(getComputedStyle(slides[0]).paddingLeft)
    const scale = slideWidth > 0 ? (cardW - 2 * pad) / slideWidth : 0.45
    deck.style.setProperty('--slides-overview-cols', String(cols))
    deck.style.setProperty('--slides-overview-scale', String(Math.min(1, Math.max(0.2, scale))))
  }

  const setOverview = (on: boolean) => {
    if (overview === on) return
    if (on) slideWidth = slides[idx].getBoundingClientRect().width
    overview = on
    root.classList.toggle('is-overview', on)
    overviewBtn?.setAttribute('aria-pressed', String(on))
    if (on) {
      layoutOverview()
      void mountAll()
    }
    update()
  }

  const setPresenting = (on: boolean) => {
    if (presenting === on) return
    presenting = on
    root.classList.toggle('is-presenting', on)
    presentBtn?.setAttribute('aria-pressed', String(on))
    if (on) setOverview(false)
    update()
  }
  const togglePresent = () => {
    if (!canPresent) return
    if (document.fullscreenElement === root) {
      void document.exitFullscreen()
    } else {
      root.requestFullscreen({ navigationUI: 'hide' }).catch(() => {})
    }
  }
  const onFullscreenChange = () => setPresenting(document.fullscreenElement === root)

  let timerStart = 0
  let timerId: number | undefined
  const tickTimer = () => {
    if (!timerEl) return
    const total = Math.floor((Date.now() - timerStart) / 1000)
    const mm = String(Math.floor(total / 60)).padStart(2, '0')
    const ss = String(total % 60).padStart(2, '0')
    timerEl.textContent = `${mm}:${ss}`
  }
  const toggleTimer = () => {
    if (!timerEl) return
    if (timerId !== undefined) {
      window.clearInterval(timerId)
      timerId = undefined
      timerEl.hidden = true
      return
    }
    timerStart = Date.now()
    tickTimer()
    timerEl.hidden = false
    timerId = window.setInterval(tickTimer, 1000)
  }

  const expandAllCallouts = (scope: HTMLElement) => {
    const callouts = scope.querySelectorAll<HTMLElement>('blockquote.callout, .callout')
    for (const el of Array.from(callouts)) {
      el.classList.remove('is-collapsed')
      if (el.style && typeof el.style.maxHeight !== 'undefined') el.style.maxHeight = ''
      const descendants = el.querySelectorAll<HTMLElement>("[style*='max-height']")
      descendants.forEach(child => (child.style.maxHeight = ''))
    }
  }

  const expandAllTranscludes = (scope: HTMLElement) => {
    const transcludes = scope.querySelectorAll<HTMLElement>('.transclude-collapsible')
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

  // the rail folds to its two controls; the choice survives reloads
  const RAIL_KEY = 'slides-rail-collapsed'
  const setRailCollapsed = (collapsed: boolean) => {
    root.classList.toggle('is-rail-collapsed', collapsed)
    railToggle?.setAttribute('aria-expanded', String(!collapsed))
    railToggle?.setAttribute('aria-label', collapsed ? 'show rail' : 'hide rail')
    try {
      localStorage.setItem(RAIL_KEY, collapsed ? '1' : '0')
    } catch {}
    if (!collapsed) update()
  }
  const onRailToggle = () => setRailCollapsed(!root.classList.contains('is-rail-collapsed'))
  let railCollapsed = false
  try {
    railCollapsed = localStorage.getItem(RAIL_KEY) === '1'
  } catch {}
  if (railCollapsed) setRailCollapsed(true)

  if (presentBtn) presentBtn.hidden = !canPresent
  // a link such as ./3#heading lands on that heading; update() rewrites the URL
  const landing = document.getElementById(decodeURIComponent(window.location.hash.slice(1)))
  expandAllCallouts(deck)
  expandAllTranscludes(deck)
  update()
  if (landing && landing !== slides[idx] && slides[idx].contains(landing)) {
    scrollDeckToHeading(landing.id, false)
  }
  const idle = window.requestIdleCallback ?? ((run: () => void) => window.setTimeout(run, 200))
  idle(prefetchAround)
  // the first layout lands in place; later changes animate from it
  if (tocList) {
    void tocList.offsetHeight
    tocList.classList.add('is-settled')
  }

  // overview grid: vertical arrows move by one row
  const overviewColumns = () =>
    Math.max(1, getComputedStyle(deck).gridTemplateColumns.split(' ').length)

  let digitBuffer = ''
  let digitTimer: number | undefined
  const commitDigits = () => {
    const n = parseInt(digitBuffer, 10)
    digitBuffer = ''
    digitTimer = undefined
    if (n >= 1) void goTo(n - 1)
  }

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
    if (e.metaKey || e.ctrlKey || e.altKey || e.isComposing) return

    switch (e.key) {
      case 'ArrowLeft':
        e.preventDefault()
        goPrev()
        return
      case 'ArrowRight':
        e.preventDefault()
        goNext()
        return
      case ' ':
        e.preventDefault()
        if (e.shiftKey) goPrev()
        else goNext()
        return
      case 'ArrowUp':
      case 'ArrowDown':
        if (!overview) return
        e.preventDefault()
        void goTo(idx + (e.key === 'ArrowDown' ? 1 : -1) * overviewColumns())
        return
      case 'Home':
        e.preventDefault()
        void goTo(0)
        return
      case 'End':
        e.preventDefault()
        void goTo(slides.length - 1)
        return
      case 'Enter':
        if (!overview) return
        e.preventDefault()
        setOverview(false)
        return
      case 'o':
        e.preventDefault()
        setOverview(!overview)
        return
      case 'p':
        if (!canPresent) return
        e.preventDefault()
        togglePresent()
        return
      case 't':
        e.preventDefault()
        toggleTimer()
        return
    }
    if (/^\d$/.test(e.key)) {
      e.preventDefault()
      digitBuffer += e.key
      if (digitTimer !== undefined) window.clearTimeout(digitTimer)
      digitTimer = window.setTimeout(commitDigits, 700)
    }
  }

  const tocClick = (event: MouseEvent) => {
    if (!(event.target instanceof Element)) return

    const item = event.target.closest<HTMLAnchorElement>('.slides-toc-item, .slides-toc-subitem')
    if (!item) return

    const nextIdx = Number(item.dataset.slideTarget)
    if (!Number.isInteger(nextIdx) || !plainClick(event)) return

    event.preventDefault()
    const sameSlide = !overview && clamp(nextIdx) === idx
    const headingId = item.dataset.headingId
    setOverview(false)
    if (sameSlide && headingId) {
      scrollDeckToHeading(headingId, true)
    } else if (sameSlide) {
      deck.scrollTo({ top: 0, behavior: reducedMotion ? 'auto' : 'smooth' })
    } else {
      void goTo(nextIdx).then(turned => {
        if (turned && headingId) scrollDeckToHeading(headingId, false)
      })
    }
  }

  const pagerClick = (step: number) => (event: MouseEvent) => {
    if (!plainClick(event)) return
    event.preventDefault()
    const target = idx + step
    if (target >= 0 && target < slides.length) void goTo(target)
  }
  const onPrevClick = pagerClick(-1)
  const onNextClick = pagerClick(1)

  const deckClick = (event: MouseEvent) => {
    if (!overview || !(event.target instanceof Element)) return
    const card = event.target.closest<HTMLElement>('.slide')
    if (!card) return
    event.preventDefault()
    const n = Number(card.dataset.index)
    if (!Number.isInteger(n)) return
    // a card still loading opens once its fragment lands
    if (isMounted(clamp(n))) {
      idx = clamp(n)
      setOverview(false)
    } else {
      setOverview(false)
      void goTo(n)
    }
  }

  const onDeckScroll = () => {
    const atBottom = deck.scrollTop + deck.clientHeight >= deck.scrollHeight - 4
    deck.classList.toggle('gradient-active', !atBottom)
    if (!overview) updateSubSpy()
  }

  // touch: a horizontal swipe on the deck turns the slide; elements that scroll
  // sideways (tables, code) keep their own gesture
  let swipe: { id: number; x: number; y: number } | null = null
  const onPointerDown = (e: PointerEvent) => {
    if (e.pointerType !== 'touch' || overview) return
    let el = e.target instanceof HTMLElement ? e.target : null
    while (el && el !== deck) {
      if (el.scrollWidth > el.clientWidth + 1 && /auto|scroll/.test(getComputedStyle(el).overflowX))
        return
      el = el.parentElement
    }
    swipe = { id: e.pointerId, x: e.clientX, y: e.clientY }
  }
  const onPointerUp = (e: PointerEvent) => {
    if (!swipe || e.pointerId !== swipe.id) return
    const dx = e.clientX - swipe.x
    const dy = e.clientY - swipe.y
    swipe = null
    if (Math.abs(dx) < 48 || Math.abs(dx) < Math.abs(dy) * 2) return
    if (dx < 0) goNext()
    else goPrev()
  }
  const onPointerCancel = () => {
    swipe = null
  }

  const onResize = () => {
    updateTocOverflow()
    settleSublists()
    if (overview) layoutOverview()
  }

  const leaveMode = () => {
    if (overview) setOverview(false)
    else if (document.fullscreenElement === root) void document.exitFullscreen()
  }
  registerEscapeHandler(root, leaveMode, () => overview || presenting)

  // list the deck's keys in the site shortcut panel while this page is open
  const shortcutList = document.querySelector<HTMLUListElement>('#shortcut-list')
  const shortcutRows = SLIDE_SHORTCUTS.map(([key, label]) => {
    const li = document.createElement('li')
    const row = document.createElement('div')
    row.id = 'shortcuts'
    row.className = 'slides-shortcut'
    row.dataset.key = key
    row.dataset.value = label
    const kbd = document.createElement('kbd')
    kbd.textContent = key
    const span = document.createElement('span')
    span.textContent = label
    row.append(kbd, span)
    li.append(row)
    return li
  })
  shortcutList?.append(...shortcutRows)

  updateTocOverflow()

  const onOverviewClick = () => setOverview(!overview)
  prev.addEventListener('click', onPrevClick)
  next.addEventListener('click', onNextClick)
  railToggle?.addEventListener('click', onRailToggle)
  overviewBtn?.addEventListener('click', onOverviewClick)
  presentBtn?.addEventListener('click', togglePresent)
  toc?.addEventListener('click', tocClick)
  deck.addEventListener('click', deckClick)
  window.addEventListener('keydown', keyEvent)
  window.addEventListener('resize', onResize)
  document.addEventListener('fullscreenchange', onFullscreenChange)
  deck.addEventListener('scroll', onDeckScroll, { passive: true })
  deck.addEventListener('pointerdown', onPointerDown, { passive: true })
  deck.addEventListener('pointerup', onPointerUp)
  deck.addEventListener('pointercancel', onPointerCancel)
  tocScroller?.addEventListener('scroll', updateTocOverflow, { passive: true })
  window.addCleanup(() => {
    prev.removeEventListener('click', onPrevClick)
    next.removeEventListener('click', onNextClick)
    railToggle?.removeEventListener('click', onRailToggle)
    overviewBtn?.removeEventListener('click', onOverviewClick)
    presentBtn?.removeEventListener('click', togglePresent)
    toc?.removeEventListener('click', tocClick)
    deck.removeEventListener('click', deckClick)
    window.removeEventListener('keydown', keyEvent)
    window.removeEventListener('resize', onResize)
    document.removeEventListener('fullscreenchange', onFullscreenChange)
    deck.removeEventListener('scroll', onDeckScroll)
    deck.removeEventListener('pointerdown', onPointerDown)
    deck.removeEventListener('pointerup', onPointerUp)
    deck.removeEventListener('pointercancel', onPointerCancel)
    tocScroller?.removeEventListener('scroll', updateTocOverflow)
    cancelAnimationFrame(tocTween)
    if (timerId !== undefined) window.clearInterval(timerId)
    if (digitTimer !== undefined) window.clearTimeout(digitTimer)
    shortcutRows.forEach(row => row.remove())
    if (document.fullscreenElement === root) void document.exitFullscreen()
  })
})
