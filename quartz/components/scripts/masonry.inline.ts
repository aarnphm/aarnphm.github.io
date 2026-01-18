interface ImageMetadata {
  src: string
  alt: string
  width: number
  height: number
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
  const goalPixels = 500 * 300
  const actualPixels = naturalWidth * naturalHeight

  let width = naturalWidth
  let height = naturalHeight

  if (actualPixels / goalPixels > 16) {
    width = Math.floor(naturalWidth / 8)
    height = Math.floor(naturalHeight / 8)
  } else if (actualPixels / goalPixels > 4) {
    width = Math.floor(naturalWidth / 4)
    height = Math.floor(naturalHeight / 4)
  } else {
    width = Math.floor(naturalWidth / 2)
    height = Math.floor(naturalHeight / 2)
  }

  if (width + 10 > containerWidth) {
    width = containerWidth - 10
  }

  return { width, height }
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

  fetch(jsonPath)
    .then((f) => {
      if (f.status == 404) {
        alert(`unable to find ${jsonPath}`)
      }
      return f.json()
    })
    .then((imageData) => {
      const shuffledData: ImageMetadata[] = shuffleArray(imageData)
      const containerWidth = container.getBoundingClientRect().width
      const positioned: ImagePosition[] = []
      let containerHeight = 600

      // calculate dimensions from JSON data
      const imageDimensions: ImageDimensions[] = shuffledData.map((data) =>
        calculateTargetDimensions(data.width, data.height, containerWidth),
      )

      let i = 0
      const intervalId = setInterval(() => {
        const dims = imageDimensions[i]
        const data = shuffledData[i]

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
              img.loading = "lazy"
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

        i++
        if (i === shuffledData.length) {
          clearInterval(intervalId)
        }
      }, 100)
    })
}

document.addEventListener("nav", initMasonry)
document.addEventListener("contentdecrypted", initMasonry)
