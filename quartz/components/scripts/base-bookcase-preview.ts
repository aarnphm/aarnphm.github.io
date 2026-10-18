export interface HardcoverBook {
  cover?: string
  spine: string
  ink: string
  title: string
  author: string
}

export function createHardcoverPreview(root: HTMLElement, signal: AbortSignal) {
  const book = document.createElement('div')
  book.className = 'bookcase-hardcover'
  const face = (name: string) => {
    const element = document.createElement('div')
    element.className = `bookcase-hardcover-${name}`
    book.append(element)
    return element
  }
  face('back')
  const front = face('front')
  const image = document.createElement('img')
  image.alt = ''
  image.decoding = 'async'
  image.draggable = false
  image.dataset.noPopover = ''
  const fallback = document.createElement('span')
  front.append(image, fallback)
  const spine = face('spine')
  const title = document.createElement('span')
  const author = document.createElement('span')
  spine.append(title, author)
  for (const name of ['top', 'bottom', 'fore-edge']) {
    const edge = face(name)
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
    svg.setAttribute('viewBox', '0 0 100 100')
    svg.setAttribute('preserveAspectRatio', 'none')
    const lines = document.createElementNS('http://www.w3.org/2000/svg', 'path')
    lines.setAttribute(
      'd',
      Array.from({ length: name === 'fore-edge' ? 35 : 7 }, (_, i) => {
        const y = ((i + 1) * 100) / (name === 'fore-edge' ? 36 : 8)
        return `M0 ${y}H100`
      }).join(''),
    )
    svg.append(lines)
    edge.append(svg)
  }
  root.replaceChildren(book)

  let aspect = 1.53
  let rotation = { x: -12, y: 32 }
  let drag: { id: number; x: number; y: number; rx: number; ry: number } | undefined
  let frame = 0
  const paint = () => {
    frame = 0
    if (signal.aborted) return
    book.style.transform = `rotateX(${rotation.x}deg) rotateY(${rotation.y}deg) rotateZ(-2deg)`
  }
  const schedule = () => {
    if (!frame) frame = requestAnimationFrame(paint)
  }
  const resize = () => {
    const width = Math.max(
      1,
      Math.min(400, root.clientWidth * 0.76, (root.clientHeight * 0.76) / aspect),
    )
    book.style.width = `${width}px`
    book.style.height = `${width * aspect}px`
    book.style.setProperty('--bc-book-depth', `${Math.max(12, width * 0.14)}px`)
  }
  image.addEventListener(
    'load',
    () => {
      if (image.naturalWidth > 0) aspect = image.naturalHeight / image.naturalWidth
      image.hidden = false
      fallback.hidden = true
      resize()
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
  const observer = new ResizeObserver(resize)
  observer.observe(root)
  signal.addEventListener(
    'abort',
    () => {
      observer.disconnect()
      if (frame) cancelAnimationFrame(frame)
    },
    { once: true },
  )

  return {
    show(value: HardcoverBook) {
      const sameCover = Boolean(value.cover && image.getAttribute('src') === value.cover)
      aspect = sameCover && image.naturalWidth > 0 ? image.naturalHeight / image.naturalWidth : 1.53
      rotation = { x: -12, y: 32 }
      book.style.setProperty('--bc-book-cloth', value.spine)
      book.style.setProperty('--bc-book-ink', value.ink)
      title.textContent = value.title
      author.textContent = value.author
      fallback.textContent = value.title
      const loaded = sameCover && image.complete && image.naturalWidth > 0
      fallback.hidden = loaded
      image.hidden = !loaded
      if (value.cover) {
        if (!sameCover) image.src = value.cover
      } else image.removeAttribute('src')
      resize()
      schedule()
    },
    clear() {
      image.removeAttribute('src')
    },
  }
}
