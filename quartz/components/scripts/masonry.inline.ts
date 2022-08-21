interface ImageData {
  src: string
  alt: string
}

interface ImagePosition {
  x: number
  y: number
  width: number
  height: number
  element: HTMLElement | null
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
  const aspectRatio = Math.max(naturalWidth / naturalHeight, naturalHeight / naturalWidth)
  const pixels = naturalWidth * naturalHeight

  let goalPixels: number
  if (aspectRatio > 8) {
    goalPixels = 200 * 200
  } else if (aspectRatio > 4) {
    goalPixels = 300 * 200
  } else {
    goalPixels = 500 * 300
  }

  const adjustmentFactor = Math.sqrt(pixels / goalPixels)
  let width = Math.floor(naturalWidth / adjustmentFactor)
  let height = Math.floor(naturalHeight / adjustmentFactor)

  // cap width at container width minus margins
  const maxWidth = containerWidth - 10
  if (width + 10 > containerWidth) {
    width = containerWidth - 10
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

      const pos: ImagePosition = { x, y, ...dims, element: null }

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

function shuffleArray<T>(array: T[]): T[] {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[array[i], array[j]] = [array[j], array[i]]
  }
  return array
}

async function initMasonry() {
  const container = document.getElementById("masonry-grid") as HTMLElement
  if (!container) return

  const modal = document.getElementById("masonry-caption-modal") as HTMLElement
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

  const shuffledData = shuffleArray([...imageData])
  const containerWidth = container.getBoundingClientRect().width
  const positioned: ImagePosition[] = []
  let containerHeight = 600

  container.style.height = `${containerHeight}px`

  // load and position images incrementally
  let loadedCount = 0
  const batchSize = 10

  async function loadAndPositionBatch(startIdx: number) {
    const endIdx = Math.min(startIdx + batchSize, shuffledData.length)

    for (let i = startIdx; i < endIdx; i++) {
      const data = shuffledData[i]
      const tempImg = new Image()
      tempImg.src = data.src

      let dims: ImageDimensions
      try {
        await loadImage(tempImg)
        dims = calculateTargetDimensions(
          tempImg.naturalWidth,
          tempImg.naturalHeight,
          containerWidth,
        )
      } catch {
        dims = { width: 200, height: 200 }
      }

      // position this image
      let placed = false
      while (!placed) {
        const wMax = containerWidth - dims.width
        const hMax = containerHeight - dims.height

        for (let attempt = 0; attempt < 50; attempt++) {
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
            const img = document.createElement("img")
            img.alt = data.alt
            img.className = "masonry-image"
            img.src = data.src
            img.dataset.caption = data.alt
            img.style.position = "absolute"
            img.style.left = `${x}px`
            img.style.top = `${y}px`
            img.style.width = `${dims.width}px`
            img.style.height = `${dims.height}px`
            img.classList.add("positioned")

            pos.element = img
            positioned.push(pos)
            container.appendChild(img)

            if (data.alt) {
              setupCaptionModal(img, modal, data.alt)
            }

            placed = true
            break
          }
        }

        if (!placed) {
          containerHeight += 50
        }
      }

      loadedCount++

      // update container height after each image
      let maxBottom = 0
      for (let j = 0; j < positioned.length; j++) {
        const bottom = positioned[j].y + positioned[j].height
        if (bottom > maxBottom) {
          maxBottom = bottom
        }
      }
      container.style.height = `${maxBottom + 20}px`
    }

    // continue with next batch
    if (endIdx < shuffledData.length) {
      requestAnimationFrame(() => loadAndPositionBatch(endIdx))
    } else {
      // all images loaded, save cache
      saveCache(positioned, containerHeight)
    }
  }

  // start loading batches
  loadAndPositionBatch(0)

  // cleanup
  window.addCleanup(() => {
    saveCache(positioned, containerHeight)
  })

  // handle resize with ResizeObserver
  let resizeTimeout: number | undefined
  let lastWidth = containerWidth

  const resizeObserver = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const newWidth = entry.contentRect.width

      if (Math.abs(newWidth - lastWidth) < 1) continue

      lastWidth = newWidth

      if (resizeTimeout !== undefined) {
        window.clearTimeout(resizeTimeout)
      }

      resizeTimeout = window.setTimeout(() => {
        const cacheKey = getCacheKey()
        sessionStorage.removeItem(cacheKey)
        sessionStorage.removeItem(`${cacheKey}:height`)

        container.innerHTML = ""

        initMasonry()
      }, 150)
    }
  })

  resizeObserver.observe(container)
  window.addCleanup(() => resizeObserver.disconnect())
}

document.addEventListener("nav", initMasonry)
document.addEventListener("content-decrypted", initMasonry)
