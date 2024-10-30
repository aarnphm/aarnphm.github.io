import { removeAllChildren } from "./util"

const ARTICLE_CONTENT_SELECTOR = ".center"
const FOOTNOTE_SECTION_SELECTOR = "section[data-footnotes] > ol"
const INDIVIDUAL_FOOTNOTE_SELECTOR = "li[id^='user-content-fn-']"

// Computes an offset such that setting `top` on elemToAlign will put it
// in vertical alignment with targetAlignment.
function computeOffsetForAlignment(elemToAlign: HTMLElement, targetAlignment: HTMLElement) {
  const offsetParentTop = elemToAlign.offsetParent!.getBoundingClientRect().top
  return targetAlignment.getBoundingClientRect().top - offsetParentTop
}

// Clamp value between min and max
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(value, max))
}

function isInViewport(element: HTMLElement, buffer: number = 100) {
  const rect = element.getBoundingClientRect()
  return (
    rect.top >= -buffer &&
    rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) + buffer
  )
}

// Get bounds for the sidenote positioning
function getSidenoteBounds(
  sideContainer: HTMLElement,
  sidenote: HTMLElement,
): { min: number; max: number } {
  const containerRect = sideContainer.getBoundingClientRect()
  const sidenoteRect = sidenote.getBoundingClientRect()
  const parentRect = sideContainer.offsetParent?.getBoundingClientRect() || containerRect

  return {
    min: 0, // Minimum is top of container
    max: containerRect.height - sidenoteRect.height, // Maximum keeps footnote within container
  }
}

function updateSidenotes(
  articleContent: HTMLElement,
  sideContainer: HTMLElement,
  footnoteElements: NodeListOf<HTMLElement>,
) {
  const containerRect = sideContainer.getBoundingClientRect()

  footnoteElements.forEach((sidenote) => {
    const sideId = sidenote.id.replace("sidebar-", "")
    const intextLink = articleContent.querySelector(`a[href="#${sideId}"]`) as HTMLElement
    if (!intextLink) return

    // Calculate ideal position
    let referencePosition = computeOffsetForAlignment(sidenote, intextLink)

    // Get bounds for this sidenote
    const bounds = getSidenoteBounds(sideContainer, sidenote)

    // Clamp the position within bounds
    referencePosition = clamp(referencePosition, bounds.min, bounds.max)

    // Apply position
    sidenote.style.top = `${referencePosition}px`

    // Update visibility state
    if (isInViewport(intextLink)) {
      sidenote.classList.add("in-view")
      intextLink.classList.add("active")
    } else {
      sidenote.classList.remove("in-view")
      intextLink.classList.remove("active")
    }

    // Add class if note is at bounds
    if (referencePosition === bounds.min) {
      sidenote.classList.add("at-top-bound")
    } else if (referencePosition === bounds.max) {
      sidenote.classList.add("at-bottom-bound")
    } else {
      sidenote.classList.remove("at-top-bound", "at-bottom-bound")
    }
  })
}

function debounce(fn: Function, delay: number) {
  let timeoutId: ReturnType<typeof setTimeout>
  return (...args: any[]) => {
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => fn(...args), delay)
  }
}

document.addEventListener("nav", () => {
  const articleContent = document.querySelector(ARTICLE_CONTENT_SELECTOR) as HTMLElement
  const footnoteSection = document.querySelector(FOOTNOTE_SECTION_SELECTOR)
  if (!footnoteSection || !articleContent) return

  const sideContainer = document.querySelector(".sidenotes") as HTMLElement
  if (!sideContainer) return

  removeAllChildren(sideContainer)

  // Set container height to match article content
  const articleRect = articleContent.getBoundingClientRect()
  sideContainer.style.height = `${articleRect.height}px`
  sideContainer.style.top = `0px`

  const ol = document.createElement("ol")
  sideContainer.appendChild(ol)

  const footnotes = footnoteSection.querySelectorAll(
    INDIVIDUAL_FOOTNOTE_SELECTOR,
  ) as NodeListOf<HTMLLIElement>

  footnotes.forEach((footnote) => {
    const footnoteId = footnote.id
    const intextLink = articleContent.querySelector(`a[href="#${footnoteId}"]`) as HTMLElement
    if (!intextLink) return

    const sidenote = document.createElement("li")
    sidenote.classList.add("sidenote-element")
    sidenote.style.position = "absolute"
    sidenote.id = `sidebar-${footnoteId}`
    const cloned = footnote.cloneNode(true) as HTMLElement
    sidenote.append(...cloned.children)
    ol.appendChild(sidenote)
  })

  // Get all sidenotes for updates
  const sidenotes = sideContainer.querySelectorAll(".sidenote-element") as NodeListOf<HTMLElement>

  // Initial position update
  updateSidenotes(articleContent, sideContainer, sidenotes)

  // Update on scroll with debouncing
  const debouncedUpdate = debounce(
    () => updateSidenotes(articleContent, sideContainer, sidenotes),
    16, // ~60fps
  )

  // Add scroll listener
  document.addEventListener("scroll", debouncedUpdate, { passive: true })

  // Add resize listener
  window.addEventListener("resize", debouncedUpdate, { passive: true })

  // Cleanup
  window.addCleanup(() => {
    document.removeEventListener("scroll", debouncedUpdate)
    window.removeEventListener("resize", debouncedUpdate)
  })
})
