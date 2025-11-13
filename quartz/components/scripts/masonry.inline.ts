interface ImageData {
  src: string
  alt: string
}

interface ImagePosition {
  x: number
  y: number
  width: number
  height: number
  element: HTMLElement
}

interface ImageDimensions {
  width: number
  height: number
}

interface CachedMasonry {
  positions: Map<number, ImagePosition>
  containerHeight: number
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

function findPosition(
  dims: ImageDimensions,
  positioned: ImagePosition[],
  containerWidth: number,
  containerHeight: number,
): { x: number; y: number; newHeight: number } | null {
  const maxAttempts = 50
  let currentHeight = containerHeight

  while (true) {
    const wMax = containerWidth - dims.width
    const hMax = currentHeight - dims.height

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const x = Math.floor(Math.random() * wMax)
      const y = Math.floor(Math.random() * hMax)

      const pos: ImagePosition = {
        x,
        y,
        width: dims.width,
        height: dims.height,
        element: null as any,
      }

      if (!checkCollision(pos, positioned)) {
        return { x, y, newHeight: currentHeight }
      }
    }

    // if couldn't position after max attempts, expand container height
    currentHeight += 50
  }
}

function positionImage(
  img: HTMLImageElement,
  idx: number,
  positioned: ImagePosition[],
  containerWidth: number,
  containerHeight: number,
): { position: ImagePosition; newHeight: number } {
  const naturalWidth = img.naturalWidth
  const naturalHeight = img.naturalHeight

  if (naturalWidth === 0 || naturalHeight === 0) {
    img.style.display = "none"
    const pos: ImagePosition = { x: 0, y: 0, width: 0, height: 0, element: img }
    return { position: pos, newHeight: containerHeight }
  }

  const dims = calculateTargetDimensions(naturalWidth, naturalHeight, containerWidth)
  const result = findPosition(dims, positioned, containerWidth, containerHeight)

  if (!result) {
    img.style.display = "none"
    const pos: ImagePosition = { x: 0, y: 0, width: 0, height: 0, element: img }
    return { position: pos, newHeight: containerHeight }
  }

  const pos: ImagePosition = {
    x: result.x,
    y: result.y,
    width: dims.width,
    height: dims.height,
    element: img,
  }

  // apply position to image instantly
  img.style.position = "absolute"
  img.style.left = `${pos.x}px`
  img.style.top = `${pos.y}px`
  img.style.width = `${pos.width}px`
  img.style.height = `${pos.height}px`
  img.classList.add("positioned")

  return { position: pos, newHeight: result.newHeight }
}

function setupCaptionModal(img: HTMLImageElement, modal: HTMLElement, caption: string) {
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

function getCacheKey(): string {
  const slug = document.body.dataset.slug || ""
  return `masonry:${slug}`
}

function saveCache(positions: ImagePosition[], containerHeight: number) {
  const cacheKey = getCacheKey()
  const posMap = new Map<number, ImagePosition>()

  positions.forEach((pos, idx) => {
    posMap.set(idx, {
      x: pos.x,
      y: pos.y,
      width: pos.width,
      height: pos.height,
      element: null as any,
    })
  })

  const cache: CachedMasonry = {
    positions: posMap,
    containerHeight,
  }

  try {
    sessionStorage.setItem(cacheKey, JSON.stringify(Array.from(posMap.entries())))
    sessionStorage.setItem(`${cacheKey}:height`, String(containerHeight))
  } catch (e) {
    // ignore quota errors
  }
}

function loadCache(): CachedMasonry | null {
  const cacheKey = getCacheKey()

  try {
    const posData = sessionStorage.getItem(cacheKey)
    const heightData = sessionStorage.getItem(`${cacheKey}:height`)

    if (!posData || !heightData) return null

    const entries = JSON.parse(posData) as Array<[number, ImagePosition]>
    const positions = new Map<number, ImagePosition>(entries)
    const containerHeight = Number(heightData)

    return { positions, containerHeight }
  } catch (e) {
    return null
  }
}

async function initMasonry() {
  const container = document.getElementById("masonry-grid")
  if (!container) return

  const modal = document.getElementById("masonry-caption-modal")
  if (!modal) return

  const jsonPath = container.dataset.jsonPath
  if (!jsonPath) return

  let imageData: ImageData[]
  try {
    const response = await fetch(jsonPath)
    if (!response.ok) {
      console.error("Failed to fetch masonry data", response.statusText)
      return
    }
    imageData = await response.json()
  } catch (e) {
    console.error("Failed to load masonry data", e)
    return
  }

  if (imageData.length === 0) return

  const containerWidth = container.getBoundingClientRect().width
  const cache = loadCache()

  const positioned: ImagePosition[] = []
  let containerHeight = cache?.containerHeight || 600
  const loadedImages = new Set<number>()

  // create img elements upfront without loading them
  const images: HTMLImageElement[] = []
  for (let i = 0; i < imageData.length; i++) {
    const data = imageData[i]
    const img = document.createElement("img")
    img.alt = data.alt
    img.className = "masonry-image"
    img.dataset.src = data.src
    img.dataset.caption = data.alt
    img.dataset.index = String(i)
    img.style.position = "absolute"
    container.appendChild(img)
    images.push(img)
  }

  // set initial container height
  container.style.height = `${containerHeight}px`

  // setup intersection observer
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          const img = entry.target as HTMLImageElement
          const idx = Number(img.dataset.index)

          if (loadedImages.has(idx)) continue
          loadedImages.add(idx)

          // set src to trigger loading
          img.src = img.dataset.src!

          loadImage(img)
            .then(() => {
              const result = positionImage(img, idx, positioned, containerWidth, containerHeight)
              positioned.push(result.position)
              containerHeight = result.newHeight

              // update container height
              let maxBottom = 0
              for (let j = 0; j < positioned.length; j++) {
                const bottom = positioned[j].y + positioned[j].height
                if (bottom > maxBottom) {
                  maxBottom = bottom
                }
              }
              container.style.height = `${maxBottom + 20}px`

              // setup caption
              if (img.dataset.caption) {
                setupCaptionModal(img, modal, img.dataset.caption)
              }

              // save cache periodically
              if (positioned.length % 10 === 0) {
                saveCache(positioned, containerHeight)
              }
            })
            .catch(() => {
              img.style.display = "none"
            })

          observer.unobserve(img)
        }
      }
    },
    { rootMargin: "200px" },
  )

  // observe all images
  images.forEach((img) => observer.observe(img))

  // cleanup
  window.addCleanup(() => {
    observer.disconnect()
    saveCache(positioned, containerHeight)
  })

  // handle resize with ResizeObserver
  let resizeTimeout: number | undefined
  let lastWidth = containerWidth

  const resizeObserver = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const newWidth = entry.contentRect.width

      // only recalculate if width actually changed
      if (Math.abs(newWidth - lastWidth) < 1) continue

      lastWidth = newWidth

      if (resizeTimeout !== undefined) {
        window.clearTimeout(resizeTimeout)
      }

      resizeTimeout = window.setTimeout(() => {
        // clear cache on resize
        const cacheKey = getCacheKey()
        sessionStorage.removeItem(cacheKey)
        sessionStorage.removeItem(`${cacheKey}:height`)

        // reset and reinitialize
        container.innerHTML = ""
        positioned.length = 0
        loadedImages.clear()

        // re-run init
        initMasonry()
      }, 150)
    }
  })

  resizeObserver.observe(container)
  window.addCleanup(() => resizeObserver.disconnect())
}

document.addEventListener("nav", initMasonry)
document.addEventListener("content-decrypted", initMasonry)
