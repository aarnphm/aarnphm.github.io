import { rootNavSignal } from './root-lifecycle'

const SVG_NS = 'http://www.w3.org/2000/svg'
const PRELOAD = 3

type Row = { index: number; el: HTMLLIElement; centre: number }

function initializeBookcase(root: HTMLElement): void {
  if (root.dataset.bookcaseReady !== undefined) return
  const scene = root.querySelector<SVGSVGElement>('svg.bookcase-scene')
  const world = root.querySelector<HTMLElement>('.bookcase-world')
  const viewport = root.querySelector<HTMLElement>('.bookcase-viewport')
  const stage = root.querySelector<HTMLElement>('.bookcase-stage')
  if (!scene || !world || !viewport || !stage) return
  root.dataset.bookcaseReady = ''
  const signal = rootNavSignal(root)
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')

  const entries = [...root.querySelectorAll<HTMLLIElement>('.bookcase-entry')]
  const books = new Map<number, SVGGElement>()
  for (const book of scene.querySelectorAll<SVGGElement>('.bc-book'))
    books.set(Number(book.dataset.bookIndex), book)
  const [originX, originY] = (scene.dataset.origin ?? '0 0').split(' ').map(Number)
  const box = scene.viewBox.baseVal
  const fit = Number(scene.dataset.fit) || box.width
  const step = Number(root.dataset.scrollStep) || 56

  const page = {
    cover: root.querySelector<HTMLImageElement>('.bookcase-page-cover img'),
    title: root.querySelector<HTMLAnchorElement>('.bookcase-page-title'),
    byline: root.querySelector<HTMLElement>('.bookcase-page-byline'),
    detail: root.querySelector<HTMLElement>('.bookcase-page-detail'),
  }
  const readout = root.querySelector<HTMLElement>('[data-bookcase-readout]')
  const shelfLabel = root.querySelector<HTMLElement>('[data-bookcase-shelf]')

  let scale = 1
  let narrow = false
  let rows: Row[] = []
  let focus = -1
  let frame = 0
  const lifts = new Map<number, SVGSVGElement>()
  const preloaded = new Map<string, HTMLImageElement>()

  const total = books.size
  const pad = (n: number) => String(n).padStart(String(total).length, '0')

  const measure = () => {
    const scrollY = window.scrollY
    rows = []
    for (const el of entries) {
      if (el.offsetParent === null) continue
      const rect = el.getBoundingClientRect()
      rows.push({
        index: Number(el.dataset.bookIndex),
        el,
        centre: rect.top + scrollY + rect.height / 2,
      })
    }
  }

  const resize = () => {
    narrow = root.getBoundingClientRect().width < 880
    scale = Math.min(1, Math.max(0.55, viewport.clientWidth / fit))
    root.style.setProperty('--bc-scene-width', `${box.width * scale}px`)
    root.style.setProperty('--bc-scene-height', `${box.height * scale}px`)
    for (const [index, lift] of lifts) {
      lift.remove()
      lifts.delete(index)
      books.get(index)?.removeAttribute('data-lifted')
    }
    if (focus >= 0) lift(focus, false)
    measure()
    schedule()
  }

  const readingLine = () =>
    narrow ? stage.getBoundingClientRect().bottom + step : window.innerHeight * 0.4

  const centreOf = (index: number): [number, number] => {
    const book = books.get(index)
    return [Number(book?.dataset.cx ?? 0), Number(book?.dataset.cy ?? 0)]
  }

  const moveCamera = (a: number, b: number, t: number) => {
    const [ax, ay] = centreOf(a)
    const [bx, by] = centreOf(b)
    const x = ax + (bx - ax) * t
    const y = ay + (by - ay) * t
    const anchorX = viewport.clientWidth * (narrow ? 0.5 : 0.38)
    const anchorY = viewport.clientHeight * (narrow ? 0.45 : 0.52)
    world.style.transform = `translate3d(${anchorX - (x - originX) * scale}px, ${anchorY - (y - originY) * scale}px, 0)`
  }

  const preload = (index: number) => {
    const urls = new Set<string>()
    for (let i = index - PRELOAD; i <= index + PRELOAD; i++) {
      const url = books.get(i)?.dataset.cover
      if (url) urls.add(url)
    }
    for (const url of preloaded.keys()) if (!urls.has(url)) preloaded.delete(url)
    for (const url of urls) {
      if (preloaded.has(url)) continue
      const image = new Image()
      image.decoding = 'async'
      image.src = url
      preloaded.set(url, image)
    }
  }

  // The pulled book leaves the case entirely, so it is drawn in its own layer above the case.
  const lift = (index: number, animate: boolean) => {
    const book = books.get(index)
    if (!book || lifts.has(index)) return
    const bbox = book.getBBox()
    const margin = 4
    const svg = document.createElementNS(SVG_NS, 'svg')
    svg.classList.add('bc-lift')
    svg.setAttribute('aria-hidden', 'true')
    svg.setAttribute(
      'viewBox',
      `${bbox.x - margin} ${bbox.y - margin} ${bbox.width + margin * 2} ${bbox.height + margin * 2}`,
    )
    svg.style.left = `${(bbox.x - margin - originX) * scale}px`
    svg.style.top = `${(bbox.y - margin - originY) * scale}px`
    svg.style.width = `${(bbox.width + margin * 2) * scale}px`
    svg.style.height = `${(bbox.height + margin * 2) * scale}px`
    const clone = book.cloneNode(true) as SVGGElement
    clone.removeAttribute('data-lifted')
    clone.removeAttribute('data-dim')
    const cover = book.dataset.cover
    const side = clone.querySelector('.bc-side')
    if (cover && side) {
      const [width, height] = (book.dataset.coverSize ?? '0 0').split(' ')
      const image = document.createElementNS(SVG_NS, 'image')
      image.setAttribute('href', cover)
      image.setAttribute('width', width)
      image.setAttribute('height', height)
      image.setAttribute('preserveAspectRatio', 'xMidYMid slice')
      image.setAttribute('transform', book.dataset.coverTransform ?? '')
      side.after(image)
    }
    svg.append(clone)
    world.append(svg)
    book.setAttribute('data-lifted', '')
    lifts.set(index, svg)

    const style = getComputedStyle(book)
    const pull = `translate(${parseFloat(style.getPropertyValue('--pull-x')) * scale}px, ${parseFloat(style.getPropertyValue('--pull-y')) * scale}px)`
    if (!animate || reducedMotion.matches) {
      svg.style.transform = pull
      return
    }
    requestAnimationFrame(() => requestAnimationFrame(() => (svg.style.transform = pull)))
  }

  const drop = (index: number) => {
    const svg = lifts.get(index)
    if (!svg) return
    lifts.delete(index)
    const done = () => {
      svg.remove()
      if (!lifts.has(index)) books.get(index)?.removeAttribute('data-lifted')
    }
    if (reducedMotion.matches) return done()
    svg.style.transform = 'none'
    svg.addEventListener('transitionend', done, { once: true })
    window.setTimeout(done, 260)
  }

  const showPage = (row: HTMLLIElement) => {
    const link = row.querySelector<HTMLAnchorElement>('.bookcase-entry-link')
    if (page.title && link) {
      page.title.href = link.href
      page.title.dataset.slug = link.dataset.slug ?? ''
      page.title.textContent = row.querySelector('.bookcase-entry-title')?.textContent ?? ''
    }
    if (page.byline)
      page.byline.textContent = row.querySelector('.bookcase-entry-byline')?.textContent ?? ''
    if (page.detail) {
      const detail = row.querySelector('.bookcase-entry-detail')
      page.detail.replaceChildren(
        ...[...(detail?.childNodes ?? [])].map(node => node.cloneNode(true)),
      )
    }
    if (page.cover) {
      const cover = row.dataset.cover
      page.cover.hidden = !cover
      if (cover && page.cover.getAttribute('src') !== cover) page.cover.src = cover
    }
    if (shelfLabel) {
      const header = row
        .closest('.bookcase-index-group')
        ?.querySelector('.base-list-group-header > span:first-child')
      shelfLabel.textContent = header?.textContent ?? ''
    }
  }

  const setFocus = (index: number) => {
    if (index === focus) return
    const previous = focus
    focus = index
    for (const row of rows) {
      const current = row.index === index
      row.el.toggleAttribute('data-current', current)
      if (current) row.el.setAttribute('aria-current', 'true')
      else row.el.removeAttribute('aria-current')
    }
    if (previous >= 0) drop(previous)
    lift(index, true)
    const row = rows.find(row => row.index === index)
    if (row) showPage(row.el)
    if (readout) {
      const position = rows.findIndex(row => row.index === index) + 1
      readout.textContent = `${pad(position)} / ${rows.length}`
    }
    preload(index)
  }

  const update = () => {
    frame = 0
    if (signal.aborted || rows.length === 0) return
    const line = window.scrollY + readingLine()
    let lo = 0
    let hi = rows.length - 1
    if (line <= rows[0].centre) hi = 0
    else if (line >= rows[hi].centre) lo = hi
    else {
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1
        if (rows[mid].centre <= line) lo = mid
        else hi = mid
      }
    }
    const a = rows[lo]
    const b = rows[hi]
    const t = a === b ? 0 : (line - a.centre) / (b.centre - a.centre)
    moveCamera(a.index, b.index, t)
    setFocus((t < 0.5 ? a : b).index)
  }

  function schedule() {
    if (!frame) frame = requestAnimationFrame(update)
  }

  const scrollToIndex = (index: number) => {
    const row = rows.find(row => row.index === index)
    if (!row) return
    window.scrollTo({
      top: row.centre - readingLine(),
      behavior: reducedMotion.matches ? 'auto' : 'smooth',
    })
  }

  scene.addEventListener(
    'click',
    event => {
      const book = (event.target as Element).closest<SVGGElement>('.bc-book')
      if (book) scrollToIndex(Number(book.dataset.bookIndex))
    },
    { signal },
  )
  world.addEventListener(
    'click',
    event => {
      if (!(event.target as Element).closest('.bc-lift')) return
      rows
        .find(row => row.index === focus)
        ?.el.querySelector<HTMLAnchorElement>('.bookcase-entry-link')
        ?.click()
    },
    { signal },
  )
  root.addEventListener(
    'focusin',
    event => {
      const row = (event.target as Element).closest<HTMLLIElement>('.bookcase-entry')
      if (row && Number(row.dataset.bookIndex) !== focus)
        scrollToIndex(Number(row.dataset.bookIndex))
    },
    { signal },
  )
  window.addEventListener('scroll', schedule, { passive: true, signal })
  document.addEventListener(
    'base-search-results',
    () => {
      for (const el of entries) {
        const book = books.get(Number(el.dataset.bookIndex))
        book?.toggleAttribute('data-dim', el.style.display === 'none')
      }
      measure()
      schedule()
    },
    { signal },
  )

  const observer = new ResizeObserver(resize)
  observer.observe(root)
  void document.fonts.ready.then(() => {
    if (!signal.aborted) {
      measure()
      schedule()
    }
  })

  signal.addEventListener(
    'abort',
    () => {
      observer.disconnect()
      if (frame) cancelAnimationFrame(frame)
      for (const svg of lifts.values()) svg.remove()
      lifts.clear()
      preloaded.clear()
      delete root.dataset.bookcaseReady
    },
    { once: true },
  )
}

function initializeBookcases(): void {
  for (const root of document.querySelectorAll<HTMLElement>('.bookcase')) initializeBookcase(root)
}

document.addEventListener('nav', () => {
  initializeBookcases()
  document.addEventListener('base-content-changed', initializeBookcases)
  window.addCleanup(() => {
    document.removeEventListener('base-content-changed', initializeBookcases)
  })
})

export default ''
