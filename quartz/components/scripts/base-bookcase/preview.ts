export interface HardcoverBook {
  cover?: string
  thumbnail?: string
  spine: string
  ink: string
  title: string
  author: string
  spineArt?: string
  spineArtDetail?: string
  spineArtDetailWidth?: number
  spineArtDetailHeight?: number
}

function spineImageSource(
  value: string | undefined,
  width: number | undefined,
  height: number | undefined,
): string | undefined {
  if (!value) return undefined
  const url = new URL(value, document.baseURI)
  if (url.pathname.endsWith('.svg') && width && height) {
    // SVG deliveries use native, zero-origin bounds. Safari needs both view parameters.
    url.hash = `svgView(viewBox(0,0,${width},${height});preserveAspectRatio(none))`
  }
  return url.href
}

export function createHardcoverPreview(root: HTMLElement, signal: AbortSignal) {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')
  const mount = document.createElement('span')
  mount.className = 'bookcase-hardcover-mount'
  mount.setAttribute('aria-hidden', 'true')
  const book = document.createElement('span')
  book.className = 'bookcase-hardcover'
  const face = (name: string) => {
    const element = document.createElement('span')
    element.className = `bookcase-hardcover-${name}`
    book.append(element)
    return element
  }
  face('back')
  for (const side of ['front', 'back']) {
    const board = face('board')
    board.dataset.board = side
    board.dataset.edge = 'inner'
  }
  const front = face('front')
  const image = document.createElement('img')
  image.alt = ''
  image.decoding = 'async'
  image.draggable = false
  image.dataset.noPopover = ''
  const fallback = document.createElement('span')
  front.append(image, fallback)
  const spine = face('spine')
  const spineImage = document.createElement('img')
  spineImage.alt = ''
  spineImage.decoding = 'async'
  spineImage.draggable = false
  spineImage.hidden = true
  const title = document.createElement('span')
  const author = document.createElement('span')
  spine.append(spineImage, title, author)
  for (const name of ['top', 'bottom', 'fore-edge']) {
    const edge = face(name)
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
    svg.setAttribute('viewBox', '0 0 100 100')
    svg.setAttribute('preserveAspectRatio', 'none')
    const lines = document.createElementNS('http://www.w3.org/2000/svg', 'path')
    lines.setAttribute(
      'd',
      Array.from({ length: name === 'fore-edge' ? 35 : 7 }, (_, i) => {
        const position = ((i + 1) * 100) / (name === 'fore-edge' ? 36 : 8)
        return name === 'fore-edge' ? `M${position} 0V100` : `M0 ${position}H100`
      }).join(''),
    )
    svg.append(lines)
    edge.append(svg)
  }
  for (const side of ['front', 'back']) {
    for (const edge of ['top', 'bottom', 'fore-edge']) {
      const board = face('board')
      board.dataset.board = side
      board.dataset.edge = edge
    }
  }
  mount.append(book)
  root.replaceChildren(mount)

  let aspect = 1.53
  const restingPose = () => ({ x: -18, y: 20 })
  let rotation = restingPose()
  let drag: { id: number; x: number; y: number; rx: number; ry: number } | undefined
  let frame = 0
  let revision = 0
  let originalDecoder: HTMLImageElement | undefined
  let spineDecoder: HTMLImageElement | undefined
  let upgradeTimer: number | undefined
  let pendingCoverSrc: string | undefined
  let pendingSpineSrc: string | undefined
  let upgradingCover = false
  let lowSpine: string | undefined
  let detailSpine: string | undefined
  const cancelDecoders = () => {
    if (upgradeTimer !== undefined) clearTimeout(upgradeTimer)
    upgradeTimer = undefined
    pendingCoverSrc = undefined
    pendingSpineSrc = undefined
    upgradingCover = false
    originalDecoder?.removeAttribute('src')
    spineDecoder?.removeAttribute('src')
    originalDecoder = undefined
    spineDecoder = undefined
  }
  let size = { width: 0, height: 0 }
  let availableSize = { width: 0, height: 0 }
  let motion: Animation[] = []
  const pose = (x: number, y: number) => `rotateX(${x}deg) rotateY(${y}deg) rotateZ(-2deg)`
  const cancelMotion = () => {
    for (const animation of motion) animation.cancel()
    motion = []
    delete root.dataset.bookcaseFlying
  }
  const flushUpgrades = () => {
    if (motion.length > 0 || drag || signal.aborted) return
    if (pendingCoverSrc) {
      const nextSrc = pendingCoverSrc
      pendingCoverSrc = undefined
      upgradingCover = true
      image.src = nextSrc
    }
    if (pendingSpineSrc) {
      const nextSpine = pendingSpineSrc
      pendingSpineSrc = undefined
      spineImage.src = nextSpine
      showLoadedSpine()
    }
  }
  const paint = () => {
    frame = 0
    if (signal.aborted) return
    book.style.transform = pose(rotation.x, rotation.y)
  }
  const schedule = () => {
    if (!frame) frame = requestAnimationFrame(paint)
  }
  const resize = () => {
    if (availableSize.width <= 0 || availableSize.height <= 0) return
    const width = Math.max(
      1,
      Math.min(400, availableSize.width * 0.76, (availableSize.height * 0.76) / aspect),
    )
    const height = width * aspect
    if (Math.abs(size.width - width) < 0.1 && Math.abs(size.height - height) < 0.1) return
    cancelMotion()
    size = { width, height }
    mount.style.width = `${width}px`
    mount.style.height = `${height}px`
    book.style.setProperty('--bc-book-depth', `${Math.max(12, width * 0.14)}px`)
    book.style.setProperty('--bc-book-type', `${Math.min(12, width * 0.04)}px`)
  }
  image.addEventListener(
    'load',
    () => {
      if (upgradingCover) {
        upgradingCover = false
        image.hidden = false
        fallback.hidden = true
        return
      }
      const nextAspect = image.naturalWidth > 0 ? image.naturalHeight / image.naturalWidth : aspect
      const changed = Math.abs(nextAspect - aspect) > 0.01
      aspect = nextAspect
      image.hidden = false
      fallback.hidden = true
      if (changed) resize()
    },
    { signal },
  )
  image.addEventListener(
    'error',
    () => {
      image.hidden = true
      fallback.hidden = false
    },
    { signal },
  )
  const showLoadedSpine = () => {
    if (!spineImage.complete || spineImage.naturalWidth === 0) return
    spineImage.hidden = false
    spineImage.dataset.spineResolution = spineImage.src === detailSpine ? 'detail' : 'low'
    spine.dataset.artReady = ''
  }
  spineImage.addEventListener('load', showLoadedSpine, { signal })
  spineImage.addEventListener(
    'error',
    () => {
      spineImage.hidden = true
      delete spine.dataset.artReady
      delete spineImage.dataset.spineResolution
      if (lowSpine && lowSpine !== detailSpine && spineImage.src === detailSpine)
        spineImage.src = lowSpine
    },
    { signal },
  )
  root.addEventListener(
    'pointerdown',
    event => {
      if (!event.isPrimary || event.button !== 0) return
      cancelMotion()
      drag = {
        id: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        rx: rotation.x,
        ry: rotation.y,
      }
      root.dataset.turning = ''
      root.setPointerCapture(event.pointerId)
    },
    { signal },
  )
  root.addEventListener(
    'pointermove',
    event => {
      if (!drag || event.pointerId !== drag.id) return
      rotation = {
        x: Math.max(-25, Math.min(25, drag.rx - (event.clientY - drag.y) * 0.25)),
        y: Math.max(-65, Math.min(65, drag.ry + (event.clientX - drag.x) * 0.35)),
      }
      schedule()
    },
    { signal },
  )
  const stop = () => {
    drag = undefined
    delete root.dataset.turning
    flushUpgrades()
  }
  root.addEventListener('pointerup', stop, { signal })
  root.addEventListener('pointercancel', stop, { signal })
  root.addEventListener('lostpointercapture', stop, { signal })
  const observer = new ResizeObserver(() => {
    availableSize = { width: root.clientWidth, height: root.clientHeight }
    resize()
  })
  observer.observe(root)
  reducedMotion.addEventListener('change', cancelMotion, { signal })
  signal.addEventListener(
    'abort',
    () => {
      observer.disconnect()
      revision++
      cancelDecoders()
      image.removeAttribute('src')
      spineImage.removeAttribute('src')
      cancelMotion()
      if (frame) cancelAnimationFrame(frame)
    },
    { once: true },
  )

  return {
    async show(value: HardcoverBook): Promise<boolean> {
      const current = ++revision
      cancelDecoders()
      cancelMotion()
      lowSpine = spineImageSource(
        value.spineArt,
        value.spineArtDetailWidth,
        value.spineArtDetailHeight,
      )
      detailSpine = spineImageSource(
        value.spineArtDetail,
        value.spineArtDetailWidth,
        value.spineArtDetailHeight,
      )
      const url = value.thumbnail ?? value.cover
      const sameCover = Boolean(url && image.getAttribute('src') === url)
      // Keep the current geometry until the next cover supplies its aspect, avoiding a reset
      // to the fallback dimensions between books.
      if (sameCover && image.naturalWidth > 0) aspect = image.naturalHeight / image.naturalWidth
      rotation = restingPose()
      book.style.setProperty('--bc-book-cloth', value.spine)
      book.style.setProperty('--bc-book-ink', value.ink)
      title.textContent = value.title
      author.textContent = value.author
      fallback.textContent = value.title
      const loaded = sameCover && image.complete && image.naturalWidth > 0
      fallback.hidden = loaded
      image.hidden = !loaded
      spineImage.hidden = true
      delete spine.dataset.artReady
      delete spineImage.dataset.spineResolution
      if (!availableSize.width || !availableSize.height)
        availableSize = { width: root.clientWidth, height: root.clientHeight }
      resize()
      paint()
      if (lowSpine) spineImage.src = lowSpine
      else spineImage.removeAttribute('src')
      if (url) {
        if (!sameCover) image.src = url
      } else image.removeAttribute('src')
      // Decode cover and spine derivatives in parallel before revealing.
      const decodes: Promise<unknown>[] = []
      if (url) decodes.push(image.decode().catch(() => undefined))
      if (lowSpine) decodes.push(spineImage.decode().catch(() => undefined))
      if (decodes.length) await Promise.all(decodes)
      if (current !== revision || signal.aborted) return false
      const spineAvailable = Boolean(lowSpine && spineImage.complete && spineImage.naturalWidth > 0)
      spineImage.hidden = !spineAvailable
      if (spineAvailable)
        spineImage.dataset.spineResolution = spineImage.src === detailSpine ? 'detail' : 'low'
      spine.toggleAttribute('data-art-ready', spineAvailable)
      const available = Boolean(url && image.complete && image.naturalWidth > 0)
      image.hidden = !available
      fallback.hidden = available
      const nextAspect = available ? image.naturalHeight / image.naturalWidth : 1.53
      if (Math.abs(nextAspect - aspect) > 0.01) {
        aspect = nextAspect
        resize()
      }
      paint()
      // Wheel browsing can replace a book every 220 ms; upgrade once that input has settled.
      upgradeTimer = window.setTimeout(() => {
        upgradeTimer = undefined
        if (current !== revision || signal.aborted) return
        if (value.cover && value.cover !== url) {
          const original = new Image()
          originalDecoder = original
          original.decoding = 'async'
          original.src = value.cover
          void original.decode().then(
            () => {
              if (originalDecoder !== original) return
              originalDecoder = undefined
              if (current === revision && !signal.aborted) {
                pendingCoverSrc = original.src
                flushUpgrades()
              }
              original.removeAttribute('src')
            },
            () => {
              if (originalDecoder === original) originalDecoder = undefined
              original.removeAttribute('src')
            },
          )
        }
        if (detailSpine && detailSpine !== lowSpine) {
          const detail = new Image()
          spineDecoder = detail
          detail.decoding = 'async'
          detail.src = detailSpine
          void detail.decode().then(
            () => {
              if (spineDecoder !== detail) return
              spineDecoder = undefined
              if (current === revision && !signal.aborted) {
                pendingSpineSrc = detail.src
                flushUpgrades()
              }
              detail.removeAttribute('src')
            },
            () => {
              if (spineDecoder === detail) spineDecoder = undefined
              detail.removeAttribute('src')
            },
          )
        }
      }, 300)
      return true
    },
    reveal() {
      cancelMotion()
      if (reducedMotion.matches || signal.aborted) return
      if (!size.width || !size.height) return
      const duration = 240
      root.dataset.bookcaseFlying = ''
      const movement = mount.animate(
        [
          { transform: 'translate(0, 12px) scale(0.96)' },
          { transform: 'translate(0, -2.16px) scale(1)', offset: 0.22 },
          { transform: 'translate(0, 0) scale(1)' },
        ],
        { duration, easing: 'cubic-bezier(0.23, 1, 0.32, 1)' },
      )
      const turn = book.animate(
        [
          { transform: pose(-12, 65) },
          { transform: pose(-18, 62), offset: 0.22 },
          { transform: pose(rotation.x, rotation.y) },
        ],
        { duration, easing: 'cubic-bezier(0.23, 1, 0.32, 1)' },
      )
      motion = [movement, turn]
      movement.onfinish = () => {
        motion = []
        delete root.dataset.bookcaseFlying
        flushUpgrades()
      }
    },
    clear() {
      revision++
      cancelDecoders()
      cancelMotion()
      aspect = 1.53
      lowSpine = undefined
      detailSpine = undefined
      image.removeAttribute('src')
      spineImage.removeAttribute('src')
      delete spineImage.dataset.spineResolution
      delete spine.dataset.artReady
    },
  }
}
