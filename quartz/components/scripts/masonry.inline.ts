import { createMasonryPacking } from './masonry-packing'

type MasonryMode = 'tidy' | 'chaos'
const modeStorageKey = 'masonry-mode'

function readMode(): MasonryMode {
  try {
    return localStorage.getItem(modeStorageKey) === 'chaos' ? 'chaos' : 'tidy'
  } catch {
    return 'tidy'
  }
}

function rememberMode(mode: MasonryMode) {
  try {
    localStorage.setItem(modeStorageKey, mode)
  } catch {
    // The gallery remains usable when storage is unavailable.
  }
}

function setupCaptionModal(img: HTMLImageElement, modal: HTMLElement) {
  const onMouseMove = (event: MouseEvent) => {
    const { width, height } = modal.getBoundingClientRect()
    modal.style.left = `${Math.max(0, Math.min(event.clientX, window.innerWidth - width))}px`
    modal.style.top = `${Math.max(0, Math.min(event.clientY + 20, window.innerHeight - height))}px`
  }

  const onMouseEnter = (event: MouseEvent) => {
    modal.textContent = img.alt
    modal.classList.add('visible')
    onMouseMove(event)
  }

  const onMouseLeave = () => modal.classList.remove('visible')
  img.addEventListener('mouseenter', onMouseEnter)
  img.addEventListener('mousemove', onMouseMove)
  img.addEventListener('mouseleave', onMouseLeave)
  window.addCleanup(() => {
    img.removeEventListener('mouseenter', onMouseEnter)
    img.removeEventListener('mousemove', onMouseMove)
    img.removeEventListener('mouseleave', onMouseLeave)
  })
}

function shuffleArray<T>(array: T[]): T[] {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[array[i], array[j]] = [array[j], array[i]]
  }
  return array
}

document.addEventListener('nav', () => {
  const gridElement = document.getElementById('masonry-grid')
  const captionElement = document.getElementById('masonry-caption-modal')
  const toggleElement = document.getElementById('masonry-mode-toggle')
  if (!gridElement || !captionElement || !(toggleElement instanceof HTMLButtonElement)) return
  const container = gridElement
  const modal = captionElement
  const toggle = toggleElement

  const images = shuffleArray(Array.from(container.querySelectorAll<HTMLImageElement>('img')))
  const sizes = images.map(img => ({
    width: Number(img.getAttribute('width')),
    height: Number(img.getAttribute('height')),
  }))
  const fragment = document.createDocumentFragment()
  for (const img of images) {
    fragment.appendChild(img)
    if (img.alt) setupCaptionModal(img, modal)
  }
  container.appendChild(fragment)

  const motion = window.matchMedia('(prefers-reduced-motion: reduce)')
  let mode = readMode()
  let timer: number | undefined
  let frame: number | undefined
  let resumePacking: (() => void) | undefined
  let disposed = false
  let observedWidth = container.getBoundingClientRect().width

  function stopPacking() {
    if (timer !== undefined) window.clearTimeout(timer)
    if (frame !== undefined) window.cancelAnimationFrame(frame)
    timer = undefined
    frame = undefined
    resumePacking = undefined
  }

  function clearPositions() {
    for (const image of images) {
      delete image.dataset.positioned
      image.style.removeProperty('left')
      image.style.removeProperty('top')
      image.style.removeProperty('width')
      image.style.removeProperty('height')
    }
  }

  function startChaos(progressive: boolean) {
    stopPacking()
    const width = container.getBoundingClientRect().width
    if (width < 1) return
    // Keep the scrollbar while the first tiles arrive, so it cannot trigger a repack.
    const retainedHeight = progressive
      ? window.innerHeight + 1
      : container.getBoundingClientRect().height
    const gap = parseFloat(getComputedStyle(container).columnGap)
    const packing = createMasonryPacking(sizes, width, window.innerHeight, gap)
    clearPositions()
    container.dataset.masonryStatus = 'filling'
    container.dataset.placedCount = '0'
    container.dataset.backfilledCount = '0'
    container.setAttribute('aria-busy', 'true')
    container.style.height = `${retainedHeight}px`
    let placedCount = 0

    const step = () => {
      frame = undefined
      if (disposed || mode !== 'chaos') return
      const started = performance.now()
      const limit = progressive && !motion.matches ? 4 : 32
      for (let count = 0; count < limit && performance.now() - started < 6; count++) {
        const placement = packing.next()
        if (!placement) break
        const image = images[placement.index]
        image.style.left = `${placement.x}px`
        image.style.top = `${placement.y}px`
        image.style.width = `${placement.width}px`
        image.style.height = `${placement.height}px`
        image.dataset.positioned = ''
        placedCount++
      }
      container.style.height = `${Math.max(packing.height, retainedHeight)}px`
      container.dataset.placedCount = `${placedCount}`
      container.dataset.backfilledCount = `${packing.backfilled}`
      if (packing.remaining === 0) {
        resumePacking = undefined
        container.style.height = `${Math.max(packing.height, window.innerHeight)}px`
        container.dataset.masonryStatus = 'ready'
        container.setAttribute('aria-busy', 'false')
        return
      }
      if (progressive && !motion.matches) {
        timer = window.setTimeout(() => {
          timer = undefined
          frame = window.requestAnimationFrame(step)
        }, 80)
      } else {
        frame = window.requestAnimationFrame(step)
      }
    }
    resumePacking = step
    frame = window.requestAnimationFrame(step)
  }

  function applyMode() {
    stopPacking()
    modal.classList.remove('visible')
    container.dataset.masonryMode = mode
    toggle.setAttribute('aria-pressed', `${mode === 'chaos'}`)
    toggle.title = mode === 'chaos' ? 'Switch to tidy mode' : 'Switch to chaos mode'
    if (mode === 'chaos') {
      startChaos(true)
    } else {
      clearPositions()
      container.style.removeProperty('height')
      container.dataset.masonryStatus = 'ready'
      delete container.dataset.placedCount
      delete container.dataset.backfilledCount
      container.setAttribute('aria-busy', 'false')
    }
  }

  const onToggle = () => {
    mode = mode === 'tidy' ? 'chaos' : 'tidy'
    rememberMode(mode)
    applyMode()
  }
  const onMotionChange = () => {
    const resume = resumePacking
    if (mode !== 'chaos' || !resume) return
    stopPacking()
    resumePacking = resume
    frame = window.requestAnimationFrame(resume)
  }
  const observer = new ResizeObserver(entries => {
    const width = entries[0]?.contentRect.width
    if (width === undefined || Math.abs(width - observedWidth) < 1) return
    observedWidth = width
    if (mode === 'chaos') startChaos(false)
  })
  observer.observe(container)
  toggle.addEventListener('click', onToggle)
  motion.addEventListener('change', onMotionChange)
  applyMode()
  toggle.hidden = false

  window.addCleanup(() => {
    disposed = true
    stopPacking()
    observer.disconnect()
    toggle.removeEventListener('click', onToggle)
    motion.removeEventListener('change', onMotionChange)
    modal.classList.remove('visible')
  })
})
