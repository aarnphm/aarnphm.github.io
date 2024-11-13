import { getFullSlug } from "../../util/path"

export function registerEscapeHandler(outsideContainer: HTMLElement | null, cb: () => void) {
  if (!outsideContainer) return
  function click(this: HTMLElement, e: HTMLElementEventMap["click"]) {
    if (e.target !== this) return
    e.preventDefault()
    e.stopPropagation()
    cb()
  }

  function esc(e: HTMLElementEventMap["keydown"]) {
    if (!e.key.startsWith("Esc")) return
    e.preventDefault()
    cb()
  }

  outsideContainer?.addEventListener("click", click)
  window.addCleanup(() => outsideContainer?.removeEventListener("click", click))
  document.addEventListener("keydown", esc)
  window.addCleanup(() => document.removeEventListener("keydown", esc))
}

export function removeAllChildren(node: HTMLElement) {
  while (node.firstChild) {
    node.removeChild(node.firstChild)
  }
}

export function registerMouseHover(el: HTMLElement, ...classList: string[]) {
  const onMouseEnter = () => el.classList.add(...classList)
  const onMouseLeave = () => el.classList.remove(...classList)

  registerEvents(el, ["mouseenter", onMouseEnter], ["mouseleave", onMouseLeave])
}

type EventType = HTMLElementEventMap[keyof HTMLElementEventMap]
type EventHandlers<E extends EventType> = (evt: E) => any

export function registerEvents<
  T extends Document | HTMLElement | null,
  E extends keyof HTMLElementEventMap,
>(element: T, ...events: [E, EventHandlers<HTMLElementEventMap[E]>][]) {
  if (!element) return

  events.forEach(([event, cb]) => {
    const listener: EventListener = (evt) => cb(evt as HTMLElementEventMap[E])
    element.addEventListener(event, listener)
    window.addCleanup(() => element.removeEventListener(event, listener))
  })
}

export function decodeString(el: HTMLSpanElement, targetString: string, duration: number = 1000) {
  const start = performance.now()
  const end = start + duration

  function update() {
    const current = performance.now()
    const progress = (current - start) / duration
    const currentIndex = Math.floor(progress * targetString.length)

    if (current < end) {
      let decodingString =
        targetString.substring(0, currentIndex) +
        getRandomString(targetString.length - currentIndex)
      el.textContent = decodingString
      requestAnimationFrame(update)
    } else {
      el.textContent = targetString
    }
  }

  requestAnimationFrame(update)
}

export function getRandomString(length: number) {
  const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
  let result = ""
  for (let i = 0; i < length; i++) {
    result += characters.charAt(Math.floor(Math.random() * characters.length))
  }
  return result
}

export function isInViewport(element: HTMLElement, buffer: number = 100) {
  const rect = element.getBoundingClientRect()
  return (
    rect.top >= -buffer &&
    rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) + buffer
  )
}

// Computes an offset such that setting `top` on elemToAlign will put it
// in vertical alignment with targetAlignment.
function computeOffsetForAlignment(elemToAlign: HTMLElement, targetAlignment: HTMLElement) {
  const elemRect = elemToAlign.getBoundingClientRect()
  const targetRect = targetAlignment.getBoundingClientRect()
  const parentRect = elemToAlign.parentElement?.getBoundingClientRect() || elemRect
  return targetRect.top - parentRect.top
}

// Get bounds for the sidenote positioning
function getBounds(parent: HTMLElement, child: HTMLElement): { min: number; max: number } {
  const containerRect = parent.getBoundingClientRect()
  const sidenoteRect = child.getBoundingClientRect()

  return {
    min: 0,
    max: containerRect.height - sidenoteRect.height,
  }
}

export function updatePosition(ref: HTMLElement, child: HTMLElement, parent: HTMLElement) {
  // Calculate ideal position
  let referencePosition = computeOffsetForAlignment(child, ref)

  // Get bounds for this sidenote
  const bounds = getBounds(parent, child)

  // Clamp the position within bounds
  referencePosition = Math.max(referencePosition, Math.min(bounds.min, bounds.max))

  // Apply position
  child.style.top = `${referencePosition}px`
}

function updateSidenoteState(content: HTMLElement, isCollapsed: boolean) {
  // handle sidenotes state
  const sidenoteRefs = content.querySelectorAll("a[data-footnote-ref]") as NodeListOf<HTMLElement>
  const sideContainer = document.querySelector(".sidenotes") as HTMLElement | null
  if (!sideContainer) return
  for (const ref of sidenoteRefs) {
    const sideId = ref.getAttribute("href")?.replace("#", "sidebar-")
    const sidenote = sideContainer.querySelector(
      `.sidenote-element[id="${sideId}"]`,
    ) as HTMLElement | null
    if (!sidenote) continue

    if (isCollapsed) {
      sidenote.classList.remove("in-view")
      ref.classList.remove("active")
      sidenote.classList.add("collapsed")
    } else if (isInViewport(ref)) {
      sidenote.classList.add("in-view")
      ref.classList.add("active")
      sidenote.classList.remove("collapsed")
      updatePosition(ref, sidenote, sideContainer!)
    } else {
      sidenote.classList.remove("collapsed")
    }
  }
}

export type CollapsedState = "true" | "false"

const collapseId = (win: Window, id: string): string => `${getFullSlug(win)}-${id}`

export function getCollapsedState(win: Window, id: string): CollapsedState {
  return (localStorage.getItem(collapseId(win, id)) ?? "false") as CollapsedState
}
export function setCollapsedState(win: Window, id: string, state: CollapsedState) {
  localStorage.setItem(collapseId(win, id), state)
}

export function setHeaderState(button: HTMLElement, content: HTMLElement, collapsed: boolean) {
  button.setAttribute("aria-expanded", collapsed ? "false" : "true")
  button.classList.toggle("collapsed", collapsed)
  content.style.maxHeight = collapsed ? "0px" : "inherit"
  content.classList.toggle("collapsed", collapsed)
  button.parentElement!.parentElement!.classList.toggle("collapsed", collapsed)
  updateSidenoteState(content, collapsed)
}

export function closeReader(readerView: HTMLElement | null) {
  if (!readerView) return
  readerView.classList.remove("active")
  const toolbar = document.querySelector(".toolbar") as HTMLElement
  const allHr = document.querySelectorAll("hr")
  const quartz = document.getElementById("quartz-root")
  if (!toolbar) return
  if (!allHr) return
  if (!quartz) return
  const readerButton = toolbar.querySelector("#reader-button")
  readerButton?.setAttribute("data-active", "false")
  allHr.forEach((hr) => (hr.style.visibility = "show"))
  quartz.style.overflow = ""
  quartz.style.maxHeight = ""
}

export function updateContainerHeights() {
  const articleContent = document.querySelector(".center") as HTMLElement
  const sideContainer = document.querySelector(".sidenotes") as HTMLElement
  if (!articleContent || !sideContainer) return

  // Set sidenotes container height to match article content
  const articleRect = articleContent.getBoundingClientRect()
  const viewportHeight = window.innerHeight
  const topSpacing = 96 // 6rem

  sideContainer.style.height = "auto"
  if (articleRect.height > viewportHeight - topSpacing) {
    sideContainer.style.minHeight = `${articleRect.height}px`
  } else {
    sideContainer.style.minHeight = ""
  }

  // Recalculate sidenote positions
  const sidenotes = sideContainer.querySelectorAll(".sidenote-element") as NodeListOf<HTMLElement>
  const inViewSidenotes = Array.from(sidenotes).filter((note) => note.classList.contains("in-view"))

  for (const sidenote of inViewSidenotes) {
    const sideId = sidenote.id.replace("sidebar-", "")
    const intextLink = articleContent.querySelector(`a[href="#${sideId}"]`) as HTMLElement
    if (intextLink) {
      updatePosition(intextLink, sidenote, sideContainer)
    }
  }
}

export function debounceUpdateHeight(delay: number = 150) {
  let timeoutId: ReturnType<typeof setTimeout>
  return () => {
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => {
      updateContainerHeights()
    }, delay)
  }
}
