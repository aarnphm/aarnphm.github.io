import { getFullSlug } from "../../util/path"
import { loadMapbox, applyMonochromeMapPalette } from "./mapbox-client"

function initScrollMask() {
  const root = document.querySelector("[data-slug='lyd']") as HTMLElement
  if (!root) return

  function updateMaskOpacity() {
    const scrollY = window.scrollY
    const documentHeight = document.documentElement.scrollHeight
    const viewportHeight = window.innerHeight

    // Calculate scroll progress (0 to 1)
    const maxScroll = documentHeight - viewportHeight
    const scrollProgress = Math.min(scrollY / maxScroll, 1)

    // Calculate opacity: start at 1, fade to 0.3 as user scrolls
    // This reveals more of the background image
    const opacity = Math.max(0.3, 1 - scrollProgress * 0.7)

    // Update the CSS custom property
    root.style.setProperty("--mask-opacity", opacity.toString())
  }

  // Initial check
  updateMaskOpacity()

  // Update on scroll with throttling for better performance
  let ticking = false
  window.addEventListener(
    "scroll",
    () => {
      if (!ticking) {
        window.requestAnimationFrame(() => {
          updateMaskOpacity()
          ticking = false
        })
        ticking = true
      }
    },
    { passive: true },
  )
}

function initHoverModal() {
  const root = document.querySelector("[data-slug='lyd']") as HTMLElement
  if (!root) return

  // Create modal element
  let modal = document.querySelector(".lydia-modal") as HTMLElement
  if (!modal) {
    modal = document.createElement("div")
    modal.className = "lydia-modal"
    root.appendChild(modal)
  }

  // Find all timeline items with hover text
  const timelineItems = root.querySelectorAll(".timeline-item[data-hover-text]")

  timelineItems.forEach((item) => {
    const hoverText = item.getAttribute("data-hover-text")
    if (!hoverText) return

    const content = item.querySelector(".timeline-content") as HTMLElement
    if (!content) return

    const onMouseEnter = () => {
      modal.textContent = hoverText
      modal.classList.add("active")
    }

    const onMouseLeave = () => {
      modal.classList.remove("active")
    }

    const onMouseMove = (e: MouseEvent) => {
      // Position modal near cursor with offset
      modal.style.left = `${e.pageX + 15}px`
      modal.style.top = `${e.pageY + 15}px`
    }

    content.addEventListener("mouseenter", onMouseEnter)
    content.addEventListener("mouseleave", onMouseLeave)
    content.addEventListener("mousemove", onMouseMove)
  })
}

function initPictureFrameDecorations() {
  const root = document.querySelector("[data-slug='lyd']") as HTMLElement
  if (!root) return

  const historyItems = root.querySelectorAll(".timeline-item[data-type='history']")

  historyItems.forEach((item, index) => {
    const content = item.querySelector<HTMLElement>(".timeline-content") as HTMLElement
    if (!content) return

    // Check if decorations already exist
    if (content.querySelector(".frame-corner")) return

    // Create corner decorations (4 corners)
    for (let i = 0; i < 4; i++) {
      const corner = document.createElement("div")
      corner.className = `frame-corner frame-corner-${i + 1}`
      corner.innerHTML = generateCornerOrnament(i, index)
      content.appendChild(corner)
    }

    // Create edge decorations (top, right, bottom, left)
    for (let i = 0; i < 4; i++) {
      const edge = document.createElement("div")
      edge.className = `frame-edge frame-edge-${i + 1}`
      edge.innerHTML = generateEdgePattern(i, index)
      content.appendChild(edge)
    }

    // Create particle shower container
    const showerContainer = document.createElement("div")
    showerContainer.className = "particle-shower"
    content.appendChild(showerContainer)

    // Add hover effect to trigger shower
    content.addEventListener("mouseenter", () => {
      showerContainer.innerHTML = ""
      for (let i = 0; i < 15; i++) {
        const particle = document.createElement("div")
        particle.className = "particle"
        particle.style.left = `${Math.random() * 100}%`
        particle.style.animationDelay = `${Math.random() * 0.6}s`
        particle.style.animationDuration = `${1.2 + Math.random() * 1.2}s`

        // Vary particle shapes
        if (Math.random() > 0.7) {
          particle.classList.add("particle-star")
        } else if (Math.random() > 0.5) {
          particle.classList.add("particle-dot")
        }

        showerContainer.appendChild(particle)
      }
    })

    content.addEventListener("animationend", (e) => {
      if ((e.target as HTMLElement).classList.contains("particle")) {
        // @ts-ignore
        e.target?.remove()
      }
    })
  })
}

function generateCornerOrnament(corner: number, frameIndex: number): string {
  const patterns = [
    // Pattern set 1: floral corners
    [
      `<svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
        <path d="M5,35 Q8,30 15,28 Q10,25 8,18 Q12,22 18,20 Q15,15 15,8" stroke="currentColor" fill="none" stroke-width="2"/>
        <circle cx="18" cy="8" r="3" fill="currentColor" opacity="0.6"/>
        <circle cx="8" cy="18" r="2.5" fill="currentColor" opacity="0.5"/>
      </svg>`,
      `<svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
        <path d="M35,35 Q32,30 25,28 Q30,25 32,18 Q28,22 22,20 Q25,15 25,8" stroke="currentColor" fill="none" stroke-width="2"/>
        <circle cx="22" cy="8" r="3" fill="currentColor" opacity="0.6"/>
        <circle cx="32" cy="18" r="2.5" fill="currentColor" opacity="0.5"/>
      </svg>`,
      `<svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
        <path d="M35,5 Q32,10 25,12 Q30,15 32,22 Q28,18 22,20 Q25,25 25,32" stroke="currentColor" fill="none" stroke-width="2"/>
        <circle cx="22" cy="32" r="3" fill="currentColor" opacity="0.6"/>
        <circle cx="32" cy="22" r="2.5" fill="currentColor" opacity="0.5"/>
      </svg>`,
      `<svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
        <path d="M5,5 Q8,10 15,12 Q10,15 8,22 Q12,18 18,20 Q15,25 15,32" stroke="currentColor" fill="none" stroke-width="2"/>
        <circle cx="18" cy="32" r="3" fill="currentColor" opacity="0.6"/>
        <circle cx="8" cy="22" r="2.5" fill="currentColor" opacity="0.5"/>
      </svg>`,
    ],
    // Pattern set 2: geometric corners
    [
      `<svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
        <path d="M2,32 L8,32 L8,26 M2,20 L14,20 L14,8" stroke="currentColor" fill="none" stroke-width="2"/>
        <circle cx="14" cy="8" r="2" fill="currentColor"/>
      </svg>`,
      `<svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
        <path d="M38,32 L32,32 L32,26 M38,20 L26,20 L26,8" stroke="currentColor" fill="none" stroke-width="2"/>
        <circle cx="26" cy="8" r="2" fill="currentColor"/>
      </svg>`,
      `<svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
        <path d="M38,8 L32,8 L32,14 M38,20 L26,20 L26,32" stroke="currentColor" fill="none" stroke-width="2"/>
        <circle cx="26" cy="32" r="2" fill="currentColor"/>
      </svg>`,
      `<svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
        <path d="M2,8 L8,8 L8,14 M2,20 L14,20 L14,32" stroke="currentColor" fill="none" stroke-width="2"/>
        <circle cx="14" cy="32" r="2" fill="currentColor"/>
      </svg>`,
    ],
    // Pattern set 3: wavy corners
    [
      `<svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
        <path d="M0,30 Q5,28 10,30 T20,30 M0,20 Q8,18 16,20 Q20,18 24,12 Q22,8 20,4" stroke="currentColor" fill="none" stroke-width="2"/>
        <circle cx="20" cy="4" r="2.5" fill="currentColor" opacity="0.7"/>
      </svg>`,
      `<svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
        <path d="M40,30 Q35,28 30,30 T20,30 M40,20 Q32,18 24,20 Q20,18 16,12 Q18,8 20,4" stroke="currentColor" fill="none" stroke-width="2"/>
        <circle cx="20" cy="4" r="2.5" fill="currentColor" opacity="0.7"/>
      </svg>`,
      `<svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
        <path d="M40,10 Q35,12 30,10 T20,10 M40,20 Q32,22 24,20 Q20,22 16,28 Q18,32 20,36" stroke="currentColor" fill="none" stroke-width="2"/>
        <circle cx="20" cy="36" r="2.5" fill="currentColor" opacity="0.7"/>
      </svg>`,
      `<svg viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
        <path d="M0,10 Q5,12 10,10 T20,10 M0,20 Q8,22 16,20 Q20,22 24,28 Q22,32 20,36" stroke="currentColor" fill="none" stroke-width="2"/>
        <circle cx="20" cy="36" r="2.5" fill="currentColor" opacity="0.7"/>
      </svg>`,
    ],
  ]

  const patternSet = patterns[frameIndex % patterns.length]
  return patternSet[corner]
}

function generateEdgePattern(edge: number, frameIndex: number): string {
  const patterns = [
    // Pattern set 1: dotted edges
    [
      `<svg viewBox="0 0 200 20" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
        <circle cx="50" cy="10" r="2" fill="currentColor" opacity="0.4"/>
        <circle cx="100" cy="10" r="2" fill="currentColor" opacity="0.4"/>
        <circle cx="150" cy="10" r="2" fill="currentColor" opacity="0.4"/>
      </svg>`,
      `<svg viewBox="0 0 20 200" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
        <circle cx="10" cy="50" r="2" fill="currentColor" opacity="0.4"/>
        <circle cx="10" cy="100" r="2" fill="currentColor" opacity="0.4"/>
        <circle cx="10" cy="150" r="2" fill="currentColor" opacity="0.4"/>
      </svg>`,
      `<svg viewBox="0 0 200 20" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
        <circle cx="50" cy="10" r="2" fill="currentColor" opacity="0.4"/>
        <circle cx="100" cy="10" r="2" fill="currentColor" opacity="0.4"/>
        <circle cx="150" cy="10" r="2" fill="currentColor" opacity="0.4"/>
      </svg>`,
      `<svg viewBox="0 0 20 200" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
        <circle cx="10" cy="50" r="2" fill="currentColor" opacity="0.4"/>
        <circle cx="10" cy="100" r="2" fill="currentColor" opacity="0.4"/>
        <circle cx="10" cy="150" r="2" fill="currentColor" opacity="0.4"/>
      </svg>`,
    ],
    // Pattern set 2: wavy edges
    [
      `<svg viewBox="0 0 200 20" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
        <path d="M0,10 Q50,5 100,10 T200,10" stroke="currentColor" fill="none" stroke-width="1.5" opacity="0.5"/>
      </svg>`,
      `<svg viewBox="0 0 20 200" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
        <path d="M10,0 Q5,50 10,100 T10,200" stroke="currentColor" fill="none" stroke-width="1.5" opacity="0.5"/>
      </svg>`,
      `<svg viewBox="0 0 200 20" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
        <path d="M0,10 Q50,15 100,10 T200,10" stroke="currentColor" fill="none" stroke-width="1.5" opacity="0.5"/>
      </svg>`,
      `<svg viewBox="0 0 20 200" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
        <path d="M10,0 Q15,50 10,100 T10,200" stroke="currentColor" fill="none" stroke-width="1.5" opacity="0.5"/>
      </svg>`,
    ],
    // Pattern set 3: dashed edges
    [
      `<svg viewBox="0 0 200 20" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
        <line x1="30" y1="10" x2="70" y2="10" stroke="currentColor" stroke-width="2" opacity="0.5"/>
        <line x1="130" y1="10" x2="170" y2="10" stroke="currentColor" stroke-width="2" opacity="0.5"/>
      </svg>`,
      `<svg viewBox="0 0 20 200" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
        <line x1="10" y1="30" x2="10" y2="70" stroke="currentColor" stroke-width="2" opacity="0.5"/>
        <line x1="10" y1="130" x2="10" y2="170" stroke="currentColor" stroke-width="2" opacity="0.5"/>
      </svg>`,
      `<svg viewBox="0 0 200 20" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
        <line x1="30" y1="10" x2="70" y2="10" stroke="currentColor" stroke-width="2" opacity="0.5"/>
        <line x1="130" y1="10" x2="170" y2="10" stroke="currentColor" stroke-width="2" opacity="0.5"/>
      </svg>`,
      `<svg viewBox="0 0 20 200" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
        <line x1="10" y1="30" x2="10" y2="70" stroke="currentColor" stroke-width="2" opacity="0.5"/>
        <line x1="10" y1="130" x2="10" y2="170" stroke="currentColor" stroke-width="2" opacity="0.5"/>
      </svg>`,
    ],
  ]

  const patternSet = patterns[frameIndex % patterns.length]
  return patternSet[edge]
}

function initCharacter() {
  const timelineContainer = document.querySelector(".timeline-container") as HTMLElement
  if (!timelineContainer) return

  // Check if character already exists
  if (document.querySelector(".lydia-character")) return

  // Create character container
  const characterDiv = document.createElement("div")
  characterDiv.className = "lydia-character"

  // Random variant selection
  const variants = [drawAlienVariant1, drawAlienVariant2]
  const randomVariant = variants[Math.floor(Math.random() * variants.length)]

  characterDiv.innerHTML = randomVariant()

  timelineContainer.appendChild(characterDiv)

  // Add cursor interaction
  const alien = characterDiv.querySelector("svg") as SVGElement
  if (!alien) return

  const eyes = alien.querySelectorAll(".alien-eye")
  if (!eyes.length) return

  const lookRange = 3

  const handlePointerMove = (event: PointerEvent) => {
    if (!document.body.contains(characterDiv)) {
      cleanupEyeTracking()
      return
    }

    const rect = characterDiv.getBoundingClientRect()
    const centerX = rect.left + rect.width / 2
    const centerY = rect.top + rect.height / 2

    const angle = Math.atan2(event.clientY - centerY, event.clientX - centerX)
    const rotation = (angle * 180) / Math.PI / 20

    alien.style.transform = `rotate(${rotation}deg)`

    const offsetX = Math.cos(angle) * lookRange
    const offsetY = Math.sin(angle) * lookRange

    eyes.forEach((eye) => {
      eye.setAttribute("transform", `translate(${offsetX}, ${offsetY})`)
    })
  }

  const resetEyes = () => {
    alien.style.transform = "rotate(0deg)"
    eyes.forEach((eye) => eye.setAttribute("transform", "translate(0, 0)"))
  }

  const handleNav = (event: CustomEventMap["nav"]) => {
    if (event.detail?.url !== "lyd") {
      cleanupEyeTracking()
    }
  }

  function cleanupEyeTracking() {
    document.body.removeEventListener("pointermove", handlePointerMove)
    document.body.removeEventListener("pointerleave", resetEyes)
    document.removeEventListener("nav", handleNav)
  }

  document.body.addEventListener("pointermove", handlePointerMove)
  document.body.addEventListener("pointerleave", resetEyes)
  document.addEventListener("nav", handleNav)
}

function readCoordinates(node: HTMLElement): { lat: number; lon: number } | null {
  const lat = Number.parseFloat(node.getAttribute("data-lat") ?? "")
  const lon = Number.parseFloat(node.getAttribute("data-lon") ?? "")
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    return null
  }
  return { lat, lon }
}

function formatCoordinate(value: number, axis: "lat" | "lon"): string {
  const direction = axis === "lat" ? (value >= 0 ? "N" : "S") : value >= 0 ? "E" : "W"
  return `${Math.abs(value).toFixed(4)} ${direction}`
}

function initTimelineLocationPreview() {
  const root = document.querySelector("[data-slug='lyd']") as HTMLElement | null
  if (!root) return

  if (root.dataset.locationMapReady === "1") {
    return
  }

  const locationItems = Array.from(
    root.querySelectorAll<HTMLElement>(
      ".timeline-item[data-type='place'], .timeline-item[data-type='food']",
    ),
  )

  if (locationItems.length === 0) return
  root.dataset.locationMapReady = "1"

  const tooltip = document.createElement("div")
  tooltip.className = "lydia-map-tooltip"
  tooltip.setAttribute("aria-hidden", "true")
  tooltip.innerHTML = `
    <p class="lydia-map-title">toronto</p>
    <div class="lydia-map-meta">
      <span class="lydia-map-label">lat <strong data-role="lat">—</strong></span>
      <span class="lydia-map-label">lon <strong data-role="lon">—</strong></span>
    </div>
    <p class="lydia-map-status">hover over a marker</p>
    <div class="lydia-map-canvas" role="presentation"></div>
  `
  root.appendChild(tooltip)

  const titleEl = tooltip.querySelector(".lydia-map-title") as HTMLElement
  const latEl = tooltip.querySelector("[data-role='lat']") as HTMLElement
  const lonEl = tooltip.querySelector("[data-role='lon']") as HTMLElement
  const statusEl = tooltip.querySelector(".lydia-map-status") as HTMLElement
  const mapCanvas = tooltip.querySelector(".lydia-map-canvas") as HTMLElement

  let previewToken = 0
  let mapBundlePromise: Promise<{ map: any; marker: any } | null> | null = null

  const setTooltipVisible = (visible: boolean) => {
    tooltip.classList.toggle("active", visible)
    tooltip.setAttribute("aria-hidden", visible ? "false" : "true")
  }

  const updateTooltipPosition = (point: { x: number; y: number }) => {
    const tooltipRect = tooltip.getBoundingClientRect()
    const offset = 18
    const maxX = window.innerWidth - tooltipRect.width - 12
    const maxY = window.innerHeight - tooltipRect.height - 12
    const x = Math.max(12, Math.min(point.x + offset, maxX))
    const y = Math.max(12, Math.min(point.y + offset, maxY))
    tooltip.style.left = `${x}px`
    tooltip.style.top = `${y}px`
  }

  const ensureMapBundle = () => {
    if (!mapBundlePromise) {
      mapBundlePromise = loadMapbox()
        .then((mapboxgl) => {
          if (!mapboxgl) {
            return null
          }

          const map = new mapboxgl.Map({
            container: mapCanvas,
            style: "mapbox://styles/mapbox/light-v11",
            center: [-79.3832, 43.6532],
            zoom: 12.5,
            attributionControl: false,
            interactive: false,
            pitch: 0,
            bearing: 0,
          })

          const marker = new mapboxgl.Marker({ color: "#2b2418" })
            .setLngLat([-79.3832, 43.6532])
            .addTo(map)

          if (map.scrollZoom) map.scrollZoom.disable()
          if (map.boxZoom) map.boxZoom.disable()
          if (map.dragRotate) map.dragRotate.disable()
          if (map.dragPan) map.dragPan.disable()
          if (map.doubleClickZoom) map.doubleClickZoom.disable()
          if (map.touchZoomRotate) map.touchZoomRotate.disable()

          map.once("load", () => applyMonochromeMapPalette(map))

          return { map, marker }
        })
        .catch((error) => {
          console.error(error)
          return null
        })
    }

    return mapBundlePromise
  }

  const showPreview = (item: HTMLElement, point: { x: number; y: number }) => {
    const requestId = ++previewToken
    setTooltipVisible(true)
    updateTooltipPosition(point)

    const heading = item.querySelector("h3")?.textContent?.trim() ?? "toronto"
    titleEl.textContent = heading
    latEl.textContent = "…"
    lonEl.textContent = "…"
    statusEl.textContent = "looking up location…"

    const coords = readCoordinates(item)

    if (!coords) {
      statusEl.textContent = "location unavailable"
      latEl.textContent = "—"
      lonEl.textContent = "—"
      return
    }

    latEl.textContent = formatCoordinate(coords.lat, "lat")
    lonEl.textContent = formatCoordinate(coords.lon, "lon")
    statusEl.textContent = ""

    ensureMapBundle().then((bundle) => {
      if (requestId !== previewToken) return
      if (!bundle) {
        statusEl.textContent = `• map offline`
        return
      }

      bundle.marker.setLngLat([coords.lon, coords.lat])
      bundle.map.jumpTo({ center: [coords.lon, coords.lat], zoom: 13.3, bearing: 0, pitch: 0 })
      statusEl.textContent = "• map online"

      window.requestAnimationFrame(() => {
        try {
          bundle.map.resize()
        } catch (error) {
          console.error(error)
        }
      })
    })
  }

  const hidePreview = () => {
    previewToken += 1
    setTooltipVisible(false)
  }

  const centroidFromMarker = (marker: HTMLElement) => {
    const rect = marker.getBoundingClientRect()
    return {
      x: rect.left + rect.width / 2,
      y: rect.top + rect.height / 2,
    }
  }

  locationItems.forEach((item) => {
    const marker = item.querySelector(".timeline-marker") as HTMLElement | null
    if (!marker) return

    if (!marker.getAttribute("tabindex")) {
      marker.setAttribute("tabindex", "0")
    }

    marker.addEventListener("pointerenter", (event) => {
      showPreview(item, { x: event.clientX, y: event.clientY })
    })

    marker.addEventListener("pointermove", (event) => {
      if (!tooltip.classList.contains("active")) return
      updateTooltipPosition({ x: event.clientX, y: event.clientY })
    })

    marker.addEventListener("pointerleave", hidePreview)
    marker.addEventListener("pointercancel", hidePreview)

    marker.addEventListener("focus", () => {
      showPreview(item, centroidFromMarker(marker))
    })

    marker.addEventListener("blur", hidePreview)
  })

  const handleNav = (event: CustomEventMap["nav"]) => {
    if (event.detail?.url === "lyd") return
    if (mapBundlePromise) {
      mapBundlePromise.then((bundle) => {
        const mapInstance = bundle?.map
        if (mapInstance && typeof mapInstance.remove === "function") {
          try {
            mapInstance.remove()
          } catch (error) {
            console.error(error)
          }
        }
      })
      mapBundlePromise = null
    }
    document.removeEventListener("nav", handleNav)
  }

  document.addEventListener("nav", handleNav)
}

function drawAlienVariant1(): string {
  return `
    <svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
      <!-- Pixel art classic alien -->
      <!-- Body -->
      <rect x="16" y="16" width="32" height="8" fill="#FFD4A3"/>
      <rect x="12" y="24" width="40" height="16" fill="#FFD4A3"/>

      <!-- Eyes -->
      <g class="alien-eye">
        <rect x="20" y="28" width="8" height="8" fill="#4A3A2C"/>
        <rect x="22" y="30" width="4" height="4" fill="#FFF"/>
      </g>
      <g class="alien-eye">
        <rect x="36" y="28" width="8" height="8" fill="#4A3A2C"/>
        <rect x="38" y="30" width="4" height="4" fill="#FFF"/>
      </g>

      <!-- Antennae -->
      <rect x="20" y="8" width="4" height="8" fill="#E8A474"/>
      <rect x="18" y="4" width="8" height="4" fill="#FFB8B8"/>
      <rect x="40" y="8" width="4" height="8" fill="#E8A474"/>
      <rect x="38" y="4" width="8" height="4" fill="#FFB8B8"/>

      <!-- Legs (animated walk) -->
      <rect x="16" y="40" width="6" height="4" fill="#FFD4A3"/>
      <rect x="14" y="44" width="6" height="8" fill="#E8A474"/>

      <rect x="29" y="40" width="6" height="4" fill="#FFD4A3"/>
      <rect x="29" y="44" width="6" height="8" fill="#E8A474"/>

      <rect x="42" y="40" width="6" height="4" fill="#FFD4A3"/>
      <rect x="44" y="44" width="6" height="8" fill="#E8A474"/>
    </svg>
  `
}

function drawAlienVariant2(): string {
  return `
    <svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
      <!-- Pixel art squid alien (cute version) -->
      <!-- Head -->
      <rect x="20" y="12" width="24" height="4" fill="#F5D5C8"/>
      <rect x="16" y="16" width="32" height="16" fill="#F5D5C8"/>
      <rect x="20" y="32" width="24" height="4" fill="#F5D5C8"/>

      <!-- Eyes (bigger and rounder for cuteness) -->
      <g class="alien-eye">
        <rect x="19" y="19" width="8" height="10" fill="#5C4A3D"/>
        <rect x="21" y="21" width="4" height="6" fill="#FFF"/>
        <rect x="23" y="22" width="2" height="3" fill="#E8F4FF"/>
      </g>
      <g class="alien-eye">
        <rect x="37" y="19" width="8" height="10" fill="#5C4A3D"/>
        <rect x="39" y="21" width="4" height="6" fill="#FFF"/>
        <rect x="41" y="22" width="2" height="3" fill="#E8F4FF"/>
      </g>

      <!-- Spots that follow eyes -->
      <g class="alien-eye">
        <rect x="28" y="24" width="3" height="3" fill="#FFD4E0"/>
        <rect x="33" y="24" width="3" height="3" fill="#FFD4E0"/>
      </g>

      <!-- Blush marks (static, for coziness) -->
      <rect x="13" y="24" width="4" height="4" fill="#FFCCD5" opacity="0.6"/>
      <rect x="47" y="24" width="4" height="4" fill="#FFCCD5" opacity="0.6"/>

      <!-- Tentacles (softer colors) -->
      <rect x="16" y="36" width="4" height="8" fill="#F5D5C8"/>
      <rect x="14" y="44" width="4" height="8" fill="#E8C4B8"/>

      <rect x="24" y="36" width="4" height="12" fill="#F5D5C8"/>
      <rect x="24" y="48" width="4" height="6" fill="#E8C4B8"/>

      <rect x="30" y="36" width="4" height="16" fill="#F5D5C8"/>

      <rect x="36" y="36" width="4" height="12" fill="#F5D5C8"/>
      <rect x="36" y="48" width="4" height="6" fill="#E8C4B8"/>

      <rect x="44" y="36" width="4" height="8" fill="#F5D5C8"/>
      <rect x="46" y="44" width="4" height="8" fill="#E8C4B8"/>

      <!-- Little heart detail on head -->
      <rect x="30" y="14" width="4" height="2" fill="#FFB8C8"/>
    </svg>
  `
}

// Initialize when DOM is ready
document.addEventListener("nav", () => {
  const slug = getFullSlug(window)
  if (slug === "lyd") {
    initScrollMask()
    initHoverModal()
    initPictureFrameDecorations()
    initCharacter()
    initTimelineLocationPreview()
  }
})

// Also run on initial page load
if (getFullSlug(window) === "lyd") {
  initScrollMask()
  initHoverModal()
  initPictureFrameDecorations()
  initCharacter()
  initTimelineLocationPreview()
}
