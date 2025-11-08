import { getFullSlug } from "../../util/path"

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

function initCharacter() {
  const timelineContainer = document.querySelector(".timeline-container") as HTMLElement
  if (!timelineContainer) return

  // Check if character already exists
  if (document.querySelector(".lydia-character")) return

  // Create character container
  const characterDiv = document.createElement("div")
  characterDiv.className = "lydia-character"

  // Random variant selection
  const variants = [drawAlienVariant1, drawAlienVariant2, drawAlienVariant3]
  const randomVariant = variants[Math.floor(Math.random() * variants.length)]

  characterDiv.innerHTML = randomVariant()

  timelineContainer.appendChild(characterDiv)

  // Add cursor interaction
  const alien = characterDiv.querySelector("svg") as SVGElement
  if (!alien) return

  const eyes = alien.querySelectorAll(".alien-eye")

  timelineContainer.addEventListener("mousemove", (e: MouseEvent) => {
    const rect = characterDiv.getBoundingClientRect()
    const centerX = rect.left + rect.width / 2
    const centerY = rect.top + rect.height / 2

    const angle = Math.atan2(e.clientY - centerY, e.clientX - centerX)
    const distance = Math.min(
      Math.sqrt(Math.pow(e.clientX - centerX, 2) + Math.pow(e.clientY - centerY, 2)),
      100,
    )

    // Rotate alien slightly toward cursor
    const rotation = (angle * 180) / Math.PI
    const tilt = Math.min(distance / 10, 8)

    alien.style.transform = `rotate(${rotation / 20}deg)`

    // Move eyes to look at cursor
    eyes.forEach((eye) => {
      const offsetX = Math.cos(angle) * 2
      const offsetY = Math.sin(angle) * 2
      eye.setAttribute("transform", `translate(${offsetX}, ${offsetY})`)
    })
  })

  timelineContainer.addEventListener("mouseleave", () => {
    alien.style.transform = "rotate(0deg)"
    eyes.forEach((eye) => {
      eye.setAttribute("transform", "translate(0, 0)")
    })
  })
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
      <!-- Pixel art squid alien -->
      <!-- Head -->
      <rect x="20" y="12" width="24" height="4" fill="#E8A474"/>
      <rect x="16" y="16" width="32" height="16" fill="#E8A474"/>
      <rect x="20" y="32" width="24" height="4" fill="#E8A474"/>

      <!-- Eyes -->
      <g class="alien-eye">
        <rect x="20" y="20" width="6" height="8" fill="#4A3A2C"/>
        <rect x="22" y="22" width="2" height="4" fill="#FFF"/>
      </g>
      <g class="alien-eye">
        <rect x="38" y="20" width="6" height="8" fill="#4A3A2C"/>
        <rect x="40" y="22" width="2" height="4" fill="#FFF"/>
      </g>

      <!-- Tentacles (blocky) -->
      <rect x="16" y="36" width="4" height="8" fill="#E8A474"/>
      <rect x="14" y="44" width="4" height="8" fill="#E8A474"/>

      <rect x="24" y="36" width="4" height="12" fill="#E8A474"/>
      <rect x="24" y="48" width="4" height="6" fill="#E8A474"/>

      <rect x="30" y="36" width="4" height="16" fill="#E8A474"/>

      <rect x="36" y="36" width="4" height="12" fill="#E8A474"/>
      <rect x="36" y="48" width="4" height="6" fill="#E8A474"/>

      <rect x="44" y="36" width="4" height="8" fill="#E8A474"/>
      <rect x="46" y="44" width="4" height="8" fill="#E8A474"/>

      <!-- Spots -->
      <rect x="24" y="24" width="4" height="4" fill="#FFB8B8"/>
      <rect x="36" y="24" width="4" height="4" fill="#FFB8B8"/>
    </svg>
  `
}

function drawAlienVariant3(): string {
  return `
    <svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
      <!-- Pixel art blob alien -->
      <!-- Organic boxy shape -->
      <rect x="24" y="12" width="16" height="4" fill="#FFE0B8"/>
      <rect x="20" y="16" width="24" height="4" fill="#FFE0B8"/>
      <rect x="16" y="20" width="32" height="20" fill="#FFE0B8"/>
      <rect x="20" y="40" width="24" height="4" fill="#FFE0B8"/>
      <rect x="24" y="44" width="16" height="4" fill="#FFE0B8"/>

      <!-- Three eyes in a row -->
      <g class="alien-eye">
        <rect x="20" y="26" width="6" height="6" fill="#4A3A2C"/>
        <rect x="22" y="28" width="2" height="2" fill="#FFF"/>
      </g>
      <g class="alien-eye">
        <rect x="29" y="24" width="6" height="8" fill="#4A3A2C"/>
        <rect x="31" y="26" width="2" height="4" fill="#FFF"/>
      </g>
      <g class="alien-eye">
        <rect x="38" y="26" width="6" height="6" fill="#4A3A2C"/>
        <rect x="40" y="28" width="2" height="2" fill="#FFF"/>
      </g>

      <!-- Smile -->
      <rect x="24" y="36" width="4" height="2" fill="#4A3A2C"/>
      <rect x="28" y="38" width="8" height="2" fill="#4A3A2C"/>
      <rect x="36" y="36" width="4" height="2" fill="#4A3A2C"/>

      <!-- Floating pixels -->
      <rect x="12" y="18" width="2" height="2" fill="#FFB8B8" opacity="0.6"/>
      <rect x="50" y="20" width="2" height="2" fill="#FFB8B8" opacity="0.6"/>
      <rect x="14" y="38" width="2" height="2" fill="#FFB8B8" opacity="0.6"/>
    </svg>
  `
}

// Initialize when DOM is ready
document.addEventListener("nav", () => {
  const slug = getFullSlug(window)
  if (slug === "lyd") {
    initScrollMask()
    initHoverModal()
    initCharacter()
  }
})

// Also run on initial page load
if (getFullSlug(window) === "lyd") {
  initScrollMask()
  initHoverModal()
  initCharacter()
}
