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

export function updateSidenoteState(content: HTMLElement, isCollapsed: boolean) {
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

export function getCollapsedState(win: Window, id: string): CollapsedState | null {
  return localStorage.getItem(collapseId(win, id)) as CollapsedState | null
}
export function setCollapsedState(win: Window, id: string, state: CollapsedState) {
  localStorage.setItem(collapseId(win, id), state)
}

export function setHeaderState(
  button: HTMLElement,
  content: HTMLElement,
  wrapper: HTMLElement,
  collapsed: boolean,
) {
  button.setAttribute("aria-expanded", collapsed ? "false" : "true")
  button.classList.toggle("collapsed", collapsed)
  content.classList.toggle("collapsed", collapsed)
  wrapper.classList.toggle("collapsed", collapsed)
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

  // First ensure article content height includes all elements
  let totalHeight = 0
  const contentElements = articleContent.children
  Array.from(contentElements).forEach((element) => {
    const rect = (element as HTMLElement).getBoundingClientRect()
    totalHeight += rect.height
  })

  // Account for margins and padding
  const style = window.getComputedStyle(articleContent)
  totalHeight +=
    parseFloat(style.paddingTop) +
    parseFloat(style.paddingBottom) +
    parseFloat(style.marginTop) +
    parseFloat(style.marginBottom)

  // Set heights
  articleContent.style.minHeight = `${totalHeight}px`
  sideContainer.style.height = `${totalHeight}px`

  // Force a reflow to ensure scrollHeight is updated
  void sideContainer.offsetHeight

  // Recalculate sidenote positions with slight delay to ensure DOM updates
  requestAnimationFrame(() => {
    const sidenotes = sideContainer.querySelectorAll(".sidenote-element") as NodeListOf<HTMLElement>
    const inViewSidenotes = Array.from(sidenotes).filter((note) =>
      note.classList.contains("in-view"),
    )

    for (const sidenote of inViewSidenotes) {
      const sideId = sidenote.id.replace("sidebar-", "")
      const intextLink = articleContent.querySelector(`a[href="#${sideId}"]`) as HTMLElement
      if (intextLink) {
        updatePosition(intextLink, sidenote, sideContainer)
      }
    }
  })
}

export function debounce(fn: Function, delay: number) {
  let timeoutId: ReturnType<typeof setTimeout>
  return (...args: any[]) => {
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => fn(...args), delay)
  }
}

interface DagNode {
  slug: string
  title: string
  contents: HTMLElement[]
  note: HTMLElement
  anchor?: HTMLElement | null
  hash?: string
}

export class Dag {
  private nodes: Map<string, DagNode>
  private order: string[] // Maintain order of nodes

  constructor() {
    this.nodes = new Map()
    this.order = []
  }

  addNode(node: DagNode) {
    const { slug } = node
    if (!this.nodes.has(slug)) {
      this.nodes.set(slug, node)
      this.order.push(slug)
    }
    return this.nodes.get(slug)!
  }

  getOrderedNodes(): DagNode[] {
    return this.order.map((slug) => this.nodes.get(slug)!).filter(Boolean)
  }

  truncateAfter(slug: string) {
    const idx = this.order.indexOf(slug)
    if (idx === -1) return

    // Remove all nodes after idx from both order and nodes map
    const removed = this.order.splice(idx + 1)
    removed.forEach((slug) => this.nodes.delete(slug))
  }

  clear() {
    this.nodes.clear()
    this.order = []
  }

  has(slug: string): boolean {
    return this.nodes.has(slug)
  }

  get(slug: string): DagNode | undefined {
    return this.nodes.get(slug)
  }

  getTail(): DagNode | undefined {
    const lastSlug = this.order[this.order.length - 1]
    return lastSlug ? this.nodes.get(lastSlug) : undefined
  }
}
