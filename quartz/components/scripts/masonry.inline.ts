interface ImagePosition {
  x: number
  y: number
  width: number
  height: number
  element: HTMLImageElement
}

interface ImageDimensions {
  width: number
  height: number
}

function checkCollision(pos: ImagePosition, positioned: ImagePosition[]): boolean {
  for (let i = 0; i < positioned.length; i++) {
    const other = positioned[i]
    if (
      pos.x < other.x + other.width &&
      pos.x + pos.width > other.x &&
      pos.y < other.y + other.height &&
      pos.y + pos.height > other.y
    ) {
      return true
    }
  }
  return false
}

function calculateTargetDimensions(
  naturalWidth: number,
  naturalHeight: number,
  containerWidth: number,
): ImageDimensions {
  const aspectRatio = naturalWidth / naturalHeight
  const pixels = naturalWidth * naturalHeight

  let targetPixels: number
  if (aspectRatio > 8) {
    // ultra-wide
    targetPixels = 40000
  } else if (aspectRatio > 4) {
    // wide
    targetPixels = 60000
  } else {
    // standard
    targetPixels = 150000
  }

  const adjustmentFactor = Math.sqrt(targetPixels / pixels)
  let width = Math.floor(naturalWidth * adjustmentFactor)
  let height = Math.floor(naturalHeight * adjustmentFactor)

  // cap width at container width minus margins
  const maxWidth = containerWidth - 20
  if (width > maxWidth) {
    const scale = maxWidth / width
    width = maxWidth
    height = Math.floor(height * scale)
  }

  return { width, height }
}

function loadImage(img: HTMLImageElement): Promise<void> {
  return new Promise((resolve, reject) => {
    if (img.complete && img.naturalWidth > 0) {
      resolve()
      return
    }

    const onLoad = () => {
      cleanup()
      resolve()
    }

    const onError = () => {
      cleanup()
      reject(new Error(`Failed to load image: ${img.src}`))
    }

    function cleanup() {
      img.removeEventListener("load", onLoad)
      img.removeEventListener("error", onError)
    }

    img.addEventListener("load", onLoad)
    img.addEventListener("error", onError)
  })
}

function shuffleArray<T>(array: T[]): T[] {
  const shuffled = [...array]
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
  }
  return shuffled
}

function positionImages(
  images: HTMLImageElement[],
  container: HTMLElement,
  containerWidth: number,
): { cancel: () => void } {
  const positioned: ImagePosition[] = []
  let containerHeight = 600
  let intervalId: ReturnType<typeof setInterval> | undefined

  // shuffle images for random placement order
  const shuffledImages = shuffleArray(images)

  // trigger loading for all images first
  for (let i = 0; i < shuffledImages.length; i++) {
    const img = shuffledImages[i]
    const src = img.dataset.src
    if (!src) continue

    if (!img.src) {
      img.src = src
    }
  }

  function cancel() {
    if (intervalId !== undefined) {
      clearInterval(intervalId)
      intervalId = undefined
    }
  }

  // wait for all images to load
  const loadPromises = shuffledImages.map((img) => loadImage(img).catch(() => {}))

  Promise.all(loadPromises).then(() => {
    let i = 0

    // place images incrementally with visual feedback
    intervalId = setInterval(() => {
      if (i >= shuffledImages.length) {
        clearInterval(intervalId!)
        intervalId = undefined

        // set final container height
        let maxBottom = 0
        for (let j = 0; j < positioned.length; j++) {
          const bottom = positioned[j].y + positioned[j].height
          if (bottom > maxBottom) {
            maxBottom = bottom
          }
        }
        container.style.height = `${maxBottom + 20}px`
        return
      }

      const img = shuffledImages[i]
      const naturalWidth = img.naturalWidth
      const naturalHeight = img.naturalHeight

      if (naturalWidth === 0 || naturalHeight === 0) {
        img.style.display = "none"
        i++
        return
      }

      const dims = calculateTargetDimensions(naturalWidth, naturalHeight, containerWidth)

      let positioned_successfully = false

      while (!positioned_successfully) {
        const maxAttempts = 50
        const wMax = containerWidth - dims.width
        const hMax = containerHeight - dims.height

        for (let attempt = 0; attempt < maxAttempts; attempt++) {
          const x = Math.floor(Math.random() * wMax)
          const y = Math.floor(Math.random() * hMax)

          const pos: ImagePosition = {
            x,
            y,
            width: dims.width,
            height: dims.height,
            element: img,
          }

          if (!checkCollision(pos, positioned)) {
            positioned.push(pos)
            positioned_successfully = true

            // apply position to image
            img.style.position = "absolute"
            img.style.left = `${pos.x}px`
            img.style.top = `${pos.y}px`
            img.style.width = `${pos.width}px`
            img.style.height = `${pos.height}px`
            img.classList.add("positioned")
            break
          }
        }

        // if couldn't position after max attempts, expand container height
        if (!positioned_successfully) {
          containerHeight += 50
        }
      }

      i++
    }, 100)
  })

  return { cancel }
}

function setupCaptionModal(images: HTMLImageElement[], modal: HTMLElement) {
  for (let i = 0; i < images.length; i++) {
    const img = images[i]
    const caption = img.dataset.caption

    if (!caption) continue

    const onMouseEnter = () => {
      modal.textContent = caption
      modal.classList.add("visible")
    }

    const onMouseMove = (e: MouseEvent) => {
      modal.style.left = `${e.clientX}px`
      modal.style.top = `${e.clientY + 20}px`
    }

    const onMouseLeave = () => {
      modal.classList.remove("visible")
    }

    img.addEventListener("mouseenter", onMouseEnter)
    img.addEventListener("mousemove", onMouseMove)
    img.addEventListener("mouseleave", onMouseLeave)

    window.addCleanup(() => {
      img.removeEventListener("mouseenter", onMouseEnter)
      img.removeEventListener("mousemove", onMouseMove)
      img.removeEventListener("mouseleave", onMouseLeave)
    })
  }
}

function initMasonry() {
  const container = document.getElementById("masonry-grid")
  if (!container) return

  const modal = document.getElementById("masonry-caption-modal")
  if (!modal) return

  const images = Array.from(container.querySelectorAll<HTMLImageElement>(".masonry-image"))
  if (images.length === 0) return

  const containerWidth = container.getBoundingClientRect().width

  let currentPositioning = positionImages(images, container, containerWidth)
  setupCaptionModal(images, modal)

  // handle resize with ResizeObserver for better detection
  let resizeTimeout: number | undefined
  let lastWidth = containerWidth

  const resizeObserver = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const newWidth = entry.contentRect.width

      // only recalculate if width actually changed
      if (Math.abs(newWidth - lastWidth) < 1) continue

      lastWidth = newWidth

      // cancel any ongoing positioning
      if (currentPositioning) {
        currentPositioning.cancel()
      }

      if (resizeTimeout !== undefined) {
        window.clearTimeout(resizeTimeout)
      }

      resizeTimeout = window.setTimeout(() => {
        // reset positions
        images.forEach((img) => {
          img.classList.remove("positioned")
          img.style.opacity = "0"
        })

        currentPositioning = positionImages(images, container, newWidth)
      }, 150)
    }
  })

  resizeObserver.observe(container)
  window.addCleanup(() => resizeObserver.disconnect())
}

document.addEventListener("nav", initMasonry)
document.addEventListener("content-decrypted", initMasonry)
