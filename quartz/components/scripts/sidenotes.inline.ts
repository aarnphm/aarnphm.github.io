import { debounce } from "./util"

// viewport calculation constants
const CONTENT_WIDTH = 35 // rem
const SIDENOTE_WIDTH = 20 // rem
const SPACING = 1 // rem
const GAP = 1 // rem

// convert rem to pixels
function remToPx(rem: number): number {
  return rem * parseFloat(getComputedStyle(document.documentElement).fontSize)
}

// calculate viewport thresholds
function getViewportThresholds() {
  const contentWidth = remToPx(CONTENT_WIDTH)
  const sidenoteWidth = remToPx(SIDENOTE_WIDTH)
  const spacing = remToPx(SPACING)

  return {
    ultraWide: contentWidth + 2 * (sidenoteWidth + 2 * spacing),
    medium: contentWidth + sidenoteWidth + 2 * spacing,
  }
}

type LayoutMode = "double-sided" | "single-sided" | "inline"

function getLayoutMode(): LayoutMode {
  const thresholds = getViewportThresholds()
  const windowWidth = window.innerWidth

  if (windowWidth > thresholds.ultraWide) {
    return "double-sided"
  } else if (windowWidth > thresholds.medium) {
    return "single-sided"
  } else {
    return "inline"
  }
}

interface SidenoteState {
  span: HTMLElement
  label: HTMLElement
  content: HTMLElement
  side?: "left" | "right"
  clickHandler?: (e: MouseEvent) => void
  keyHandler?: (e: KeyboardEvent) => void
  inlineExpanded?: boolean
}

class SidenoteManager {
  private sidenotes: SidenoteState[] = []
  private lastBottomLeft = 0
  private lastBottomRight = 0
  private layoutMode: LayoutMode = "inline"

  constructor() {
    this.initialize()
  }

  private measureContentHeight(content: HTMLElement): number {
    // temporarily display offscreen to measure
    const originalDisplay = content.style.display
    const originalVisibility = content.style.visibility
    const originalPosition = content.style.position
    const originalLeft = content.style.left

    content.style.display = "block"
    content.style.visibility = "hidden"
    content.style.position = "absolute"
    content.style.left = "-9999px"

    const height = content.getBoundingClientRect().height

    // restore
    content.style.display = originalDisplay
    content.style.visibility = originalVisibility
    content.style.position = originalPosition
    content.style.left = originalLeft

    return height
  }

  private initialize() {
    const sidenoteSpans = document.querySelectorAll<HTMLSpanElement>(".sidenote")

    sidenoteSpans.forEach((span) => {
      const label = span.querySelector<HTMLSpanElement>(".sidenote-label")
      if (!label) return

      const content = span.nextElementSibling as HTMLElement | null
      if (!content || !content.classList.contains("sidenote-content")) return

      content.style.display = "none"
      content.setAttribute("aria-hidden", "true")

      if (!label.hasAttribute("aria-controls") && content.id) {
        label.setAttribute("aria-controls", content.id)
      }

      this.sidenotes.push({ span, label, content })
    })
  }

  private resetPositions() {
    this.lastBottomLeft = 0
    this.lastBottomRight = 0

    this.sidenotes.forEach((state) => {
      const { span, label, content } = state

      const currentExpanded = label.getAttribute("aria-expanded") === "true" || content.style.display === "block"
      state.inlineExpanded = currentExpanded

      // remove inline mode attributes and handlers
      if (state.clickHandler) {
        label.removeEventListener("click", state.clickHandler)
        state.clickHandler = undefined
      }
      if (state.keyHandler) {
        label.removeEventListener("keydown", state.keyHandler)
        state.keyHandler = undefined
      }

      label.removeAttribute("role")
      label.removeAttribute("tabindex")
      label.removeAttribute("aria-expanded")
      label.removeAttribute("aria-haspopup")
      label.removeAttribute("data-inline")

      // remove active classes
      span.classList.remove("active")
      label.classList.remove("active")

      // reset content
      content.style.display = "none"
      content.style.position = ""
      content.classList.remove("sidenote-left", "sidenote-right", "sidenote-inline")
      content.setAttribute("aria-hidden", "true")
    })
  }

  private positionSidenoteSideBySide(
    state: SidenoteState,
    mode: "double-sided" | "single-sided",
  ): boolean {
    const { span, label, content } = state
    const labelRect = label.getBoundingClientRect()

    // measure content height accurately
    const contentHeight = this.measureContentHeight(content)

    // calculate vertical position
    const scrollTop = window.scrollY || document.documentElement.scrollTop
    const topPosition = labelRect.top + scrollTop

    // check footer collision
    const footer = document.querySelector("footer")
    const footerTop = footer ? footer.getBoundingClientRect().top + scrollTop : Infinity

    const wouldCollideWithFooter = topPosition + contentHeight > footerTop

    if (wouldCollideWithFooter) {
      return false // fall back to inline
    }

    // check if left side is allowed
    const allowLeft = span.getAttribute("data-allow-left") !== "false"

    // determine side
    let side: "left" | "right" = "right"

    if (mode === "double-sided" && allowLeft) {
      // alternate sides, but prefer side with more space
      const rightSpace = topPosition - this.lastBottomRight
      const leftSpace = topPosition - this.lastBottomLeft

      if (rightSpace >= contentHeight + remToPx(GAP)) {
        side = "right"
      } else if (leftSpace >= contentHeight + remToPx(GAP)) {
        side = "left"
      } else {
        return false // not enough space on either side
      }
    } else {
      // single-sided mode or left not allowed: only use right
      const rightSpace = topPosition - this.lastBottomRight
      if (rightSpace < contentHeight + remToPx(GAP)) {
        return false // not enough space
      }
      side = "right"
    }

    // apply positioning
    content.classList.add(`sidenote-${side}`)
    content.style.display = "block"
    content.setAttribute("aria-hidden", "false")

    // update tracking
    const bottomPosition = topPosition + contentHeight
    if (side === "left") {
      this.lastBottomLeft = bottomPosition
    } else {
      this.lastBottomRight = bottomPosition
    }

    state.side = side
    state.inlineExpanded = false
    return true
  }

  private positionSidenoteInline(state: SidenoteState) {
    const { span, label, content } = state

    content.classList.add("sidenote-inline")
    content.style.display = "none"
    content.style.position = "static"

    // set up interactivity
    label.setAttribute("role", "button")
    label.setAttribute("tabindex", "0")
    label.setAttribute("aria-haspopup", "true")
    label.setAttribute("data-inline", "")

    const toggle = () => {
      const isExpanded = label.getAttribute("aria-expanded") === "true"
      const newExpandedState = !isExpanded

      content.style.display = newExpandedState ? "block" : "none"
      label.setAttribute("aria-expanded", newExpandedState.toString())
      content.setAttribute("aria-hidden", (!newExpandedState).toString())

      // toggle active class based on expanded state
      if (newExpandedState) {
        span.classList.add("active")
        label.classList.add("active")
      } else {
        span.classList.remove("active")
        label.classList.remove("active")
      }

      state.inlineExpanded = newExpandedState
    }

    const onClick = (e: MouseEvent) => {
      e.preventDefault()
      e.stopPropagation()
      toggle()
    }

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault()
        toggle()
      }
    }

    // store handlers for cleanup
    state.clickHandler = onClick
    state.keyHandler = onKeyDown

    label.addEventListener("click", onClick)
    label.addEventListener("keydown", onKeyDown)

    const shouldExpand = state.inlineExpanded ?? false
    if (shouldExpand) {
      label.setAttribute("aria-expanded", "true")
      content.style.display = "block"
      content.setAttribute("aria-hidden", "false")
      span.classList.add("active")
      label.classList.add("active")
    } else {
      label.setAttribute("aria-expanded", "false")
      content.style.display = "none"
      content.setAttribute("aria-hidden", "true")
      span.classList.remove("active")
      label.classList.remove("active")
    }
  }

  public layout() {
    this.layoutMode = getLayoutMode()
    this.resetPositions()

    this.sidenotes.forEach((state) => {
      // check if sidenote is forced to be inline
      const forceInline = state.span.getAttribute("data-force-inline") === "true"

      if (this.layoutMode === "inline" || forceInline) {
        this.positionSidenoteInline(state)
      } else {
        const success = this.positionSidenoteSideBySide(state, this.layoutMode)
        if (!success) {
          this.positionSidenoteInline(state)
        }
      }
    })
  }
}

document.addEventListener("nav", () => {
  const manager = new SidenoteManager()
  manager.layout()

  const debouncedLayout = debounce(() => manager.layout(), 100)

  window.addEventListener("resize", debouncedLayout, { passive: true })
  document.addEventListener("scroll", debouncedLayout, { passive: true })

  window.addCleanup(() => {
    window.removeEventListener("resize", debouncedLayout)
    document.removeEventListener("scroll", debouncedLayout)
  })
})
