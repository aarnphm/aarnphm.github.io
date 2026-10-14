function setupCaptionModal(img: HTMLImageElement, modal: HTMLElement) {
  const onMouseEnter = (event: MouseEvent) => {
    modal.textContent = img.alt
    modal.classList.add('visible')
    onMouseMove(event)
  }

  const onMouseMove = (event: MouseEvent) => {
    const { width, height } = modal.getBoundingClientRect()
    modal.style.left = `${Math.max(0, Math.min(event.clientX, window.innerWidth - width))}px`
    modal.style.top = `${Math.max(0, Math.min(event.clientY + 20, window.innerHeight - height))}px`
  }

  const onMouseLeave = () => {
    modal.classList.remove('visible')
  }

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
  const container = document.getElementById('masonry-grid')
  const modal = document.getElementById('masonry-caption-modal')
  if (!container || !modal) return

  const images = shuffleArray(Array.from(container.querySelectorAll<HTMLImageElement>('img')))
  const fragment = document.createDocumentFragment()
  for (const img of images) {
    fragment.appendChild(img)
    if (img.alt) setupCaptionModal(img, modal)
  }
  container.appendChild(fragment)

  window.addCleanup(() => modal.classList.remove('visible'))
})
