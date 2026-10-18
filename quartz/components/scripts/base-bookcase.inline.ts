import { createHardcoverPreview } from './base-bookcase-preview'
import { registerEscapeHandler } from './escape-handler'
import { rootNavSignal } from './root-lifecycle'

const SVG_NS = 'http://www.w3.org/2000/svg'
const PRELOAD = 3

type Camera = { x: number; y: number; scale: number }
type Drag = { id: number; x: number; y: number; camera: Camera; book: number; moved: boolean }

function addCover(book: SVGGElement): void {
  const cover = book.dataset.cover
  const side = book.querySelector('.bc-side')
  if (!cover || !side || book.querySelector('.bc-cover')) return
  const [width, height] = (book.dataset.coverSize ?? '0 0').split(' ')
  const image = document.createElementNS(SVG_NS, 'image')
  image.classList.add('bc-cover')
  image.setAttribute('href', cover)
  image.setAttribute('width', width)
  image.setAttribute('height', height)
  image.setAttribute('preserveAspectRatio', 'xMidYMid slice')
  image.setAttribute('transform', book.dataset.coverTransform ?? '')
  side.after(image)
}

function initializeBookcase(root: HTMLElement): void {
  if (root.dataset.bookcaseReady !== undefined) return
  const scene = root.querySelector<SVGSVGElement>('svg.bookcase-scene')
  const world = root.querySelector<HTMLElement>('.bookcase-world')
  const viewport = root.querySelector<HTMLElement>('.bookcase-viewport')
  const indexPane = root.querySelector<HTMLElement>('.bookcase-index')
  const dialog = root.querySelector<HTMLDialogElement>('.bookcase-preview')
  if (!scene || !world || !viewport || !indexPane || !dialog) return
  root.dataset.bookcaseReady = ''
  const signal = rootNavSignal(root)
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')
  const entries = [...root.querySelectorAll<HTMLLIElement>('.bookcase-entry')]
  const groups = [...root.querySelectorAll<HTMLDetailsElement>('.bookcase-index-group')]
  const books = new Map<number, SVGGElement>()
  const bookBounds = new Map<number, DOMRect>()
  const entryByIndex = new Map(entries.map(entry => [Number(entry.dataset.bookIndex), entry]))
  for (const book of scene.querySelectorAll<SVGGElement>('.bc-book'))
    books.set(Number(book.dataset.bookIndex), book)
  for (const shelf of scene.querySelectorAll('.bc-shelf')) {
    const end = [...shelf.querySelectorAll<SVGGElement>('.bc-book')].at(-1)
    if (end) addCover(end)
  }
  const [originX, originY] = (scene.dataset.origin ?? '0 0').split(' ').map(Number)
  const { x, y, width, height } = scene.viewBox.baseVal
  const box = { x, y, width, height }
  for (const [index, book] of books) bookBounds.set(index, book.getBBox())
  const shelves = [...scene.querySelectorAll<SVGGElement>('.bc-shelf')].map(element => {
    const anchor = document.createComment('')
    element.after(anchor)
    return { element, anchor, bounds: element.getBBox() }
  })

  const page = {
    cover: root.querySelector<HTMLImageElement>('.bookcase-page-cover img'),
    title: root.querySelector<HTMLAnchorElement>('.bookcase-page-title'),
    byline: root.querySelector<HTMLElement>('.bookcase-page-byline'),
    detail: root.querySelector<HTMLElement>('.bookcase-page-detail'),
  }
  const readout = root.querySelector<HTMLElement>('[data-bookcase-readout]')
  const shelfLabel = root.querySelector<HTMLElement>('[data-bookcase-shelf]')
  const previewArt = dialog.querySelector<HTMLElement>('.bookcase-preview-art')
  const hardcover = previewArt ? createHardcoverPreview(previewArt, signal) : undefined
  const previousButton = dialog.querySelector<HTMLButtonElement>('[data-preview="previous"]')
  const nextButton = dialog.querySelector<HTMLButtonElement>('[data-preview="next"]')
  const zoomOut = root.querySelector<HTMLButtonElement>('[data-camera="out"]')
  const zoomIn = root.querySelector<HTMLButtonElement>('[data-camera="in"]')
  const sidebarToggle = root.querySelector<HTMLButtonElement>('.bookcase-sidebar-toggle')
  indexPane.id ||= `bookcase-index-${crypto.randomUUID()}`
  sidebarToggle?.setAttribute('aria-controls', indexPane.id)
  const preloaded = new Map<string, HTMLImageElement>()
  let camera: Camera = { x: 0, y: 0, scale: 1 }
  let viewportWidth = viewport.clientWidth
  let viewportHeight = viewport.clientHeight
  let fitScale = 1
  let fitted = true
  let focus = -1
  let lifted: SVGSVGElement | undefined
  let liftGeometry:
    | { x: number; y: number; width: number; height: number; px: number; py: number }
    | undefined
  let drag: Drag | undefined
  let frame = 0
  let indexFrame = 0
  let followIndexScroll = false
  let browsePosition = 0
  const scrollStep = Math.max(1, Number(root.dataset.scrollStep) || 56)
  let previewWheelDelta = 0
  let previewWheelAt = 0
  let previewWheelStep = -Infinity
  let returnFocus: HTMLElement | undefined
  let searchOpen: Map<HTMLDetailsElement, boolean> | undefined

  const matches = () =>
    [...indexPane.querySelectorAll<HTMLLIElement>('.bookcase-entry')].filter(
      entry => entry.style.display !== 'none' && !entry.closest('.bookcase-index-group[hidden]'),
    )
  const maxScale = () => Math.max(2, fitScale * 4)
  const clamp = (n: number, min: number, max: number) => Math.min(max, Math.max(min, n))

  const paint = () => {
    frame = 0
    if (signal.aborted) return
    const insetX = Math.min(viewportWidth, box.width * camera.scale) * 0.3
    const insetY = Math.min(viewportHeight, box.height * camera.scale) * 0.3
    camera.x = clamp(camera.x, insetX - box.width * camera.scale, viewportWidth - insetX)
    camera.y = clamp(camera.y, insetY - box.height * camera.scale, viewportHeight - insetY)
    // Keep the raster surface bounded by the viewport at every camera scale.
    scene.setAttribute(
      'viewBox',
      `${box.x - camera.x / camera.scale} ${box.y - camera.y / camera.scale} ${viewportWidth / camera.scale} ${viewportHeight / camera.scale}`,
    )
    const top = box.y - (camera.y + viewportHeight / 2) / camera.scale
    const bottom = top + (viewportHeight * 2) / camera.scale
    for (const { element, anchor, bounds } of shelves) {
      const visible = bounds.y + bounds.height >= top && bounds.y <= bottom
      if (visible && !element.isConnected) anchor.before(element)
      else if (!visible && element.isConnected) element.remove()
    }
    placeLift()
    if (zoomOut) zoomOut.disabled = camera.scale <= fitScale + 0.001
    if (zoomIn) zoomIn.disabled = camera.scale >= maxScale() - 0.001
  }
  const schedule = () => {
    if (!frame) frame = requestAnimationFrame(paint)
  }
  const fit = () => {
    fitted = true
    camera = {
      x: (viewportWidth - box.width * fitScale) / 2,
      y: (viewportHeight - box.height * fitScale) / 2,
      scale: fitScale,
    }
    schedule()
  }
  const zoom = (factor: number, x = viewportWidth / 2, y = viewportHeight / 2) => {
    const scale = clamp(camera.scale * factor, fitScale, maxScale())
    const ratio = scale / camera.scale
    camera = { x: x - (x - camera.x) * ratio, y: y - (y - camera.y) * ratio, scale }
    fitted = false
    schedule()
  }

  const cloneBook = (index: number): SVGSVGElement | undefined => {
    const book = books.get(index)
    if (!book) return
    const bbox = bookBounds.get(index)
    if (!bbox) return
    const svg = document.createElementNS(SVG_NS, 'svg')
    svg.classList.add('bc-lift')
    svg.setAttribute('aria-hidden', 'true')
    svg.setAttribute('viewBox', `${bbox.x - 4} ${bbox.y - 4} ${bbox.width + 8} ${bbox.height + 8}`)
    const clone = book.cloneNode(true)
    if (!(clone instanceof SVGGElement)) return
    clone.removeAttribute('data-lifted')
    clone.removeAttribute('data-dim')
    addCover(clone)
    svg.append(clone)
    return svg
  }

  const lift = (index: number) => {
    lifted?.remove()
    for (const book of books.values()) book.removeAttribute('data-lifted')
    const book = books.get(index)
    const svg = cloneBook(index)
    if (!book || !svg) return
    world.append(svg)
    book.setAttribute('data-lifted', '')
    lifted = svg
    const bounds = svg.viewBox.baseVal
    const [dx, dy] = (book.dataset.offset ?? '0 0').split(' ').map(Number)
    const style = book.style
    liftGeometry = {
      x: bounds.x + dx - originX,
      y: bounds.y + dy - originY,
      width: bounds.width,
      height: bounds.height,
      px: parseFloat(style.getPropertyValue('--pull-x')),
      py: parseFloat(style.getPropertyValue('--pull-y')),
    }
    placeLift()
  }

  function placeLift(): void {
    if (!lifted || !liftGeometry) return
    const { x, y, width, height, px, py } = liftGeometry
    lifted.style.left = `${camera.x + x * camera.scale}px`
    lifted.style.top = `${camera.y + y * camera.scale}px`
    lifted.style.width = `${width * camera.scale}px`
    lifted.style.height = `${height * camera.scale}px`
    lifted.style.transform = `translate(${px * camera.scale}px, ${py * camera.scale}px)`
  }

  const copyDetail = (row: HTMLLIElement, target: Element | null) => {
    const detail = row.querySelector('.bookcase-entry-detail')
    target?.replaceChildren(...[...(detail?.childNodes ?? [])].map(node => node.cloneNode(true)))
  }

  const select = (index: number, reveal = false) => {
    const row = entryByIndex.get(index)
    const book = books.get(index)
    if (!row || !book) return
    let revealTop: number | undefined
    if (reveal && !indexPane.hidden) {
      const group = row.closest('details')
      if (group instanceof HTMLDetailsElement) group.open = true
      const bounds = row.getBoundingClientRect()
      const pane = indexPane.getBoundingClientRect()
      revealTop = bounds.top - pane.top - pane.height / 2 + bounds.height / 2
    }
    const previous = entryByIndex.get(focus)
    if (previous !== row) {
      previous?.removeAttribute('data-current')
      previous?.querySelector('.bookcase-entry-link')?.removeAttribute('aria-current')
    }
    focus = index
    row.setAttribute('data-current', '')
    row.querySelector('.bookcase-entry-link')?.setAttribute('aria-current', 'true')
    const link = row.querySelector<HTMLAnchorElement>('.bookcase-entry-link')
    const title = row.querySelector('.bookcase-entry-title')?.textContent ?? ''
    const byline = row.querySelector('.bookcase-entry-byline')?.textContent ?? ''
    if (page.title && link) {
      page.title.href = link.href
      page.title.dataset.slug = link.dataset.slug ?? ''
      page.title.textContent = title
    }
    if (page.byline) page.byline.textContent = byline
    copyDetail(row, page.detail)
    if (page.cover) {
      page.cover.hidden = !row.dataset.cover
      page.cover.alt = title
      if (row.dataset.cover && page.cover.getAttribute('src') !== row.dataset.cover)
        page.cover.src = row.dataset.cover
    }
    if (shelfLabel)
      shelfLabel.textContent =
        row.closest('.bookcase-index-group')?.querySelector('.bookcase-index-label')?.textContent ??
        ''
    const visible = matches()
    browsePosition = visible.indexOf(row)
    if (readout)
      readout.textContent = `${String(visible.indexOf(row) + 1).padStart(3, '0')} / ${visible.length}`
    if (
      !lifted ||
      lifted.querySelector<SVGGElement>('.bc-book')?.dataset.bookIndex !== String(index)
    )
      lift(index)
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
    if (revealTop !== undefined) {
      followIndexScroll = false
      // Scroll only the index pane; scrollIntoView would also move the workspace.
      indexPane.scrollBy({ top: revealTop, behavior: reducedMotion.matches ? 'instant' : 'smooth' })
    }
  }

  const focusCamera = (index: number, towards = index, progress = 0) => {
    const book = books.get(index)
    if (!book) return
    const scale = Math.max(
      fitScale,
      Math.min(1.25, viewportHeight / 600, viewportWidth / (box.width + 80)),
    )
    const startY = Number(book.dataset.cy)
    const endY = Number(books.get(towards)?.dataset.cy ?? startY)
    const y = startY + (endY - startY) * progress - originY
    camera = {
      x: (viewportWidth - box.width * scale) / 2,
      y: viewportHeight / 2 - y * scale,
      scale,
    }
    fitted = false
    schedule()
  }

  const renderPreview = () => {
    const row = entryByIndex.get(focus)
    if (!row) return
    const visible = matches()
    const position = visible.indexOf(row)
    const link = row.querySelector<HTMLAnchorElement>('.bookcase-entry-link')
    const previewLink = dialog.querySelector<HTMLAnchorElement>('.bookcase-preview-link')
    const title = row.querySelector('.bookcase-entry-title')?.textContent ?? ''
    dialog.setAttribute('aria-label', `Preview: ${title}`)
    const heading = dialog.querySelector('.bookcase-preview-title')
    if (heading) heading.textContent = title
    const byline = dialog.querySelector('.bookcase-preview-byline')
    if (byline) byline.textContent = row.querySelector('.bookcase-entry-byline')?.textContent ?? ''
    const positionLabel = dialog.querySelector('.bookcase-preview-position')
    if (positionLabel) positionLabel.textContent = `${position + 1} / ${visible.length}`
    if (previousButton) previousButton.disabled = position <= 0
    if (nextButton) nextButton.disabled = position < 0 || position >= visible.length - 1
    if (previewLink && link) {
      previewLink.href = link.href
      previewLink.dataset.slug = link.dataset.slug ?? ''
    }
    copyDetail(row, dialog.querySelector('.bookcase-preview-detail'))
    const book = books.get(focus)
    if (book) {
      const style = book.style
      hardcover?.show({
        cover: book.dataset.fullCover ?? book.dataset.cover,
        spine: style.getPropertyValue('--spine').trim(),
        ink: style.getPropertyValue('--spine-ink').trim(),
        title,
        author: row.querySelector('.bookcase-entry-byline')?.textContent ?? '',
      })
    }
  }

  const openPreview = (index: number) => {
    select(index, true)
    renderPreview()
    if (!dialog.open) {
      const active = document.activeElement
      returnFocus = active instanceof HTMLElement && active !== document.body ? active : undefined
      dialog.showModal()
    }
  }
  const browsePreview = (delta: number) => {
    const visible = matches()
    const position = visible.findIndex(entry => Number(entry.dataset.bookIndex) === focus)
    const row = visible[position + delta]
    if (!row) return
    select(Number(row.dataset.bookIndex), true)
    focusCamera(focus)
    renderPreview()
  }

  const wheelUnit = (event: WheelEvent, height: number) =>
    event.deltaMode === WheelEvent.DOM_DELTA_LINE
      ? 16
      : event.deltaMode === WheelEvent.DOM_DELTA_PAGE
        ? height
        : 1

  viewport.addEventListener(
    'pointerdown',
    event => {
      if (!event.isPrimary || event.button !== 0) return
      const target = event.target
      const book = target instanceof Element ? target.closest<SVGGElement>('.bc-book') : null
      drag = {
        id: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        camera: { ...camera },
        book: book ? Number(book.dataset.bookIndex) : -1,
        moved: false,
      }
      viewport.setPointerCapture(event.pointerId)
    },
    { signal },
  )
  viewport.addEventListener(
    'pointermove',
    event => {
      if (!drag || event.pointerId !== drag.id) return
      const dx = event.clientX - drag.x
      const dy = event.clientY - drag.y
      if (!drag.moved && Math.hypot(dx, dy) < 5) return
      drag.moved = true
      fitted = false
      viewport.dataset.dragging = ''
      camera = { ...drag.camera, x: drag.camera.x + dx, y: drag.camera.y + dy }
      schedule()
    },
    { signal },
  )
  viewport.addEventListener(
    'pointerup',
    event => {
      if (!drag || event.pointerId !== drag.id) return
      const gesture = drag
      drag = undefined
      delete viewport.dataset.dragging
      if (viewport.hasPointerCapture(event.pointerId))
        viewport.releasePointerCapture(event.pointerId)
      if (!gesture.moved && gesture.book >= 0 && !books.get(gesture.book)?.hasAttribute('data-dim'))
        openPreview(gesture.book)
    },
    { signal },
  )
  const cancelDrag = () => {
    drag = undefined
    delete viewport.dataset.dragging
  }
  viewport.addEventListener('pointercancel', cancelDrag, { signal })
  viewport.addEventListener('lostpointercapture', cancelDrag, { signal })
  viewport.addEventListener(
    'keydown',
    event => {
      if (event.target !== viewport) return
      if (event.key === 'Home') fit()
      else if (event.key === '+' || event.key === '=') zoom(1.35)
      else if (event.key === '-') zoom(1 / 1.35)
      else if (event.key === 'Enter' && focus >= 0) openPreview(focus)
      else if (event.key.startsWith('Arrow')) {
        if (event.key === 'ArrowLeft') camera.x += 48
        if (event.key === 'ArrowRight') camera.x -= 48
        if (event.key === 'ArrowUp') camera.y += 48
        if (event.key === 'ArrowDown') camera.y -= 48
        fitted = false
        schedule()
      } else return
      event.preventDefault()
    },
    { signal },
  )
  viewport.addEventListener(
    'wheel',
    event => {
      event.preventDefault()
      if (event.ctrlKey || event.metaKey) {
        const bounds = viewport.getBoundingClientRect()
        zoom(
          Math.exp(-event.deltaY * 0.01),
          event.clientX - bounds.left,
          event.clientY - bounds.top,
        )
      } else {
        const unit = wheelUnit(event, viewport.clientHeight)
        if (Math.abs(event.deltaX) > Math.abs(event.deltaY) || event.shiftKey) {
          camera.x -= (event.deltaX || event.deltaY) * unit
          fitted = false
          schedule()
          return
        }
        const visible = matches()
        if (!visible.length) return
        const position = clamp(
          browsePosition + (event.deltaY * unit) / scrollStep,
          0,
          visible.length - 1,
        )
        const selected = Number(visible[Math.round(position)].dataset.bookIndex)
        if (selected !== focus) select(selected, true)
        browsePosition = position
        const start = Math.floor(position)
        focusCamera(
          Number(visible[start].dataset.bookIndex),
          Number(visible[Math.min(start + 1, visible.length - 1)].dataset.bookIndex),
          position - start,
        )
      }
    },
    { passive: false, signal },
  )

  root.addEventListener(
    'click',
    event => {
      if (!(event.target instanceof Element)) return
      const cameraControl = event.target.closest<HTMLElement>('[data-camera]')
      if (cameraControl?.dataset.camera === 'fit') fit()
      else if (cameraControl) zoom(cameraControl.dataset.camera === 'in' ? 1.35 : 1 / 1.35)
      if (event.target.closest('.bookcase-sidebar-toggle')) {
        const collapsed = !indexPane.hidden
        indexPane.hidden = collapsed
        root.toggleAttribute('data-index-collapsed', collapsed)
        sidebarToggle?.setAttribute('aria-expanded', String(!collapsed))
        const label = collapsed ? 'Show list' : 'Hide list'
        sidebarToggle?.setAttribute('aria-label', label)
        sidebarToggle?.setAttribute('title', label)
      }
      if (event.target.closest('.bookcase-preview-open') && focus >= 0) openPreview(focus)
      const action = event.target.closest<HTMLElement>('[data-preview]')?.dataset.preview
      if (action === 'close') dialog.close()
      else if (action === 'previous') browsePreview(-1)
      else if (action === 'next') browsePreview(1)
    },
    { signal },
  )
  const selectFromIndex = (event: Event) => {
    if (!(event.target instanceof Element)) return
    const row = event.target.closest<HTMLLIElement>('.bookcase-entry')
    if (!row || Number(row.dataset.bookIndex) === focus) return
    select(Number(row.dataset.bookIndex))
    focusCamera(focus)
  }
  indexPane.addEventListener('pointermove', selectFromIndex, { signal })
  indexPane.addEventListener(
    'focusin',
    event => {
      followIndexScroll = false
      selectFromIndex(event)
    },
    { signal },
  )
  const enableIndexScroll = () => {
    followIndexScroll = true
  }
  indexPane.addEventListener('wheel', enableIndexScroll, { passive: true, signal })
  indexPane.addEventListener('pointerdown', enableIndexScroll, { signal })
  indexPane.addEventListener(
    'keydown',
    event => {
      if (['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', ' '].includes(event.key))
        enableIndexScroll()
    },
    { signal },
  )
  indexPane.addEventListener(
    'scroll',
    () => {
      if (!followIndexScroll || dialog.open || indexFrame) return
      indexFrame = requestAnimationFrame(() => {
        indexFrame = 0
        if (signal.aborted || dialog.open) return
        const visible = matches().filter(row => row.offsetParent !== null)
        if (!visible.length) return
        const pane = indexPane.getBoundingClientRect()
        const line = pane.top + pane.height * 0.4
        let nearest = visible[0]
        let distance = Infinity
        for (const row of visible) {
          const bounds = row.getBoundingClientRect()
          const delta = Math.abs(bounds.top + bounds.height / 2 - line)
          if (delta < distance) {
            nearest = row
            distance = delta
          }
        }
        if (indexPane.scrollTop <= 0) nearest = visible[0]
        else if (indexPane.scrollTop + indexPane.clientHeight >= indexPane.scrollHeight - 1)
          nearest = visible[visible.length - 1]
        const index = Number(nearest.dataset.bookIndex)
        if (index === focus) return
        select(index)
        focusCamera(index)
      })
    },
    { passive: true, signal },
  )
  const disposeEscape = registerEscapeHandler(dialog, () => dialog.close())
  dialog.addEventListener(
    'click',
    event => {
      if (event.target !== dialog) return
      const bounds = dialog.getBoundingClientRect()
      if (
        event.clientX < bounds.left ||
        event.clientX > bounds.right ||
        event.clientY < bounds.top ||
        event.clientY > bounds.bottom
      )
        dialog.close()
    },
    { signal },
  )
  dialog.addEventListener(
    'close',
    () => {
      hardcover?.clear()
      const fallback = entryByIndex
        .get(focus)
        ?.querySelector<HTMLAnchorElement>('.bookcase-entry-link')
      const target = returnFocus?.isConnected ? returnFocus : fallback
      target?.focus({ preventScroll: true })
    },
    { signal },
  )
  dialog.addEventListener(
    'wheel',
    event => {
      if (event.ctrlKey || event.metaKey || !dialog.open) return
      const delta = Math.abs(event.deltaY) >= Math.abs(event.deltaX) ? event.deltaY : event.deltaX
      if (!delta) return
      const body =
        event.target instanceof Element ? event.target.closest('.bookcase-preview-body') : null
      if (
        body instanceof HTMLElement &&
        ((delta > 0 && body.scrollTop + body.clientHeight < body.scrollHeight - 1) ||
          (delta < 0 && body.scrollTop > 0))
      )
        return
      event.preventDefault()
      const now = performance.now()
      if (now - previewWheelStep < 220) return
      if (now - previewWheelAt > 160 || Math.sign(delta) !== Math.sign(previewWheelDelta))
        previewWheelDelta = 0
      previewWheelAt = now
      previewWheelDelta += delta * wheelUnit(event, dialog.clientHeight)
      if (Math.abs(previewWheelDelta) < 64) return
      browsePreview(Math.sign(previewWheelDelta))
      previewWheelDelta = 0
      previewWheelStep = now
    },
    { passive: false, signal },
  )
  dialog.addEventListener(
    'keydown',
    event => {
      if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
        event.preventDefault()
        browsePreview(event.key === 'ArrowLeft' ? -1 : 1)
      }
    },
    { signal },
  )

  document.addEventListener(
    'base-search-results',
    () => {
      const searching = Boolean(
        document.querySelector<HTMLInputElement>('#base-search-input')?.value.trim(),
      )
      if (searching && !searchOpen) searchOpen = new Map(groups.map(group => [group, group.open]))
      for (const group of groups) {
        const shown = [...group.querySelectorAll<HTMLLIElement>('.bookcase-entry')].filter(
          entry => entry.style.display !== 'none',
        )
        const count = group.querySelector('.bookcase-index-count')
        if (count) count.textContent = String(shown.length)
        if (searching && shown.length) group.open = true
        else if (!searching && searchOpen) group.open = searchOpen.get(group) ?? true
      }
      if (!searching) searchOpen = undefined
      for (const entry of entries)
        books
          .get(Number(entry.dataset.bookIndex))
          ?.toggleAttribute('data-dim', entry.style.display === 'none')
      const visible = matches()
      const selected =
        visible.find(entry => Number(entry.dataset.bookIndex) === focus) ?? visible[0]
      root.toggleAttribute('data-empty', !selected)
      if (selected) {
        const index = Number(selected.dataset.bookIndex)
        const changed = index !== focus
        select(index)
        if (changed) focusCamera(index)
      } else {
        focus = -1
        lifted?.remove()
        lifted = undefined
        liftGeometry = undefined
        for (const book of books.values()) book.removeAttribute('data-lifted')
        if (readout) readout.textContent = '0 / 0'
        for (const entry of entries) {
          entry.removeAttribute('data-current')
          entry.querySelector('.bookcase-entry-link')?.removeAttribute('aria-current')
        }
      }
      if (dialog.open) {
        if (selected) renderPreview()
        else dialog.close()
      }
    },
    { signal },
  )

  const resize = () => {
    camera.x += (viewport.clientWidth - viewportWidth) / 2
    camera.y += (viewport.clientHeight - viewportHeight) / 2
    viewportWidth = viewport.clientWidth
    viewportHeight = viewport.clientHeight
    const previousFit = fitScale
    fitScale = Math.max(
      0.01,
      Math.min((viewport.clientWidth - 32) / box.width, (viewport.clientHeight - 72) / box.height),
    )
    if (fitted || camera.scale < fitScale || previousFit === 1) fit()
    else schedule()
  }
  const observer = new ResizeObserver(resize)
  observer.observe(root)
  observer.observe(viewport)
  window.addEventListener('resize', resize, { signal })
  select(Number(entries[0]?.dataset.bookIndex ?? -1))
  resize()
  if (focus >= 0) focusCamera(focus)
  signal.addEventListener(
    'abort',
    () => {
      observer.disconnect()
      disposeEscape()
      if (frame) cancelAnimationFrame(frame)
      if (indexFrame) cancelAnimationFrame(indexFrame)
      if (dialog.open) dialog.close()
      lifted?.remove()
      for (const { element, anchor } of shelves) anchor.replaceWith(element)
      scene.setAttribute('viewBox', `${box.x} ${box.y} ${box.width} ${box.height}`)
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
  window.addCleanup(() => document.removeEventListener('base-content-changed', initializeBookcases))
})

export default ''
