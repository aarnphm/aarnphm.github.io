export interface HardcoverBook {
  cover?: string
  thumbnail?: string
  spine: string
  ink: string
  title: string
  author: string
  spineArt?: string
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
  let size = { width: 0, height: 0 }
  let availableSize = { width: 0, height: 0 }
  let motion: Animation[] = []
  const pose = (x: number, y: number) => `rotateX(${x}deg) rotateY(${y}deg) rotateZ(-2deg)`
  const cancelMotion = () => {
    for (const animation of motion) animation.cancel()
    motion = []
    delete root.dataset.bookcaseFlying
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
      cancelMotion()
      if (frame) cancelAnimationFrame(frame)
    },
    { once: true },
  )

  return {
    async show(value: HardcoverBook): Promise<boolean> {
      const current = ++revision
      cancelMotion()
      const url = value.thumbnail ?? value.cover
      const sameCover = Boolean(url && image.getAttribute('src') === url)
      aspect = sameCover && image.naturalWidth > 0 ? image.naturalHeight / image.naturalWidth : 1.53
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
      if (value.spineArt) spineImage.src = value.spineArt
      else spineImage.removeAttribute('src')
      if (url) {
        if (!sameCover) image.src = url
        // Decode the derivative before a turn can expose the front board.
        try {
          await image.decode()
        } catch {
          // A failed cover keeps the title on the cloth board.
        }
      } else image.removeAttribute('src')
      if (value.spineArt) {
        try {
          await spineImage.decode()
        } catch {
          // The title remains readable if an edition texture cannot load.
        }
      }
      if (current !== revision || signal.aborted) return false
      const spineAvailable = Boolean(
        value.spineArt && spineImage.complete && spineImage.naturalWidth > 0,
      )
      spineImage.hidden = !spineAvailable
      spine.toggleAttribute('data-art-ready', spineAvailable)
      const available = Boolean(url && image.complete && image.naturalWidth > 0)
      image.hidden = !available
      fallback.hidden = available
      aspect = available ? image.naturalHeight / image.naturalWidth : 1.53
      if (!availableSize.width || !availableSize.height)
        availableSize = { width: root.clientWidth, height: root.clientHeight }
      resize()
      paint()
      if (value.cover && value.cover !== url) {
        const original = new Image()
        original.src = value.cover
        void original.decode().then(
          () => {
            if (current === revision && !signal.aborted) image.src = original.src
          },
          () => {},
        )
      }
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
      }
    },
    clear() {
      revision++
      cancelMotion()
      image.removeAttribute('src')
      spineImage.removeAttribute('src')
      delete spine.dataset.artReady
    },
  }
}
