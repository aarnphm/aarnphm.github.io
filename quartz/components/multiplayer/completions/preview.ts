import { selectedCompletion, completionStatus } from "@codemirror/autocomplete"
import { EditorView, ViewUpdate } from "@codemirror/view"
import { normalizeRelativeURLs } from "../../../util/path"
import { fetchCanonical } from "../../scripts/util"

const parser = new DOMParser()

let popover: HTMLElement | null = null
let inner: HTMLElement | null = null
let rafId: number | null = null
let abort: AbortController | null = null
let currentSlug: string | null = null

function ensurePopover() {
  if (popover) return
  popover = document.createElement("div")
  popover.classList.add("popover", "completion-preview")
  inner = document.createElement("div")
  inner.className = "popover-inner"
  popover.appendChild(inner)
  document.body.appendChild(popover)
}

function showSkeleton() {
  if (!inner) return
  inner.innerHTML = `
    <div class="preview-skeleton">
      <div class="skeleton-title"></div>
      <div class="skeleton-meta"></div>
      <div class="skeleton-line"></div>
      <div class="skeleton-line short"></div>
      <div class="skeleton-line"></div>
    </div>
  `
  inner.classList.remove("grid")
}

async function render(slug: string) {
  if (!inner) return
  showSkeleton()

  abort?.abort()
  abort = new AbortController()

  try {
    const url = new URL(slug, window.location.origin)
    const res = await fetchCanonical(url, { signal: abort.signal })
    if (!res.ok) throw new Error()

    const doc = parser.parseFromString(await res.text(), "text/html")
    normalizeRelativeURLs(doc, url)

    const hints = Array.from(doc.getElementsByClassName("popover-hint")) as HTMLElement[]
    if (!hints.length) throw new Error()

    inner.innerHTML = ""
    inner.classList.add("grid")

    for (const hint of hints) {
      hint
        .querySelectorAll(
          "section[data-references], section[data-footnotes], [data-skip-preview], .telescopic-container",
        )
        .forEach((e) => e.remove())
      hint.querySelectorAll("[id]").forEach((el) => {
        el.id = `preview-${el.id}`
      })
      inner.appendChild(hint)
    }
  } catch (e: any) {
    if (e.name === "AbortError") return
    inner.innerHTML = '<p class="preview-empty">No preview available</p>'
  }
}

let scrollContainer: HTMLElement | null = null
let scrollHandler: (() => void) | null = null

function positionSync() {
  const tooltip = document.querySelector(".cm-tooltip-autocomplete")
  if (!tooltip || !popover) return

  const rect = tooltip.getBoundingClientRect()
  const popoverRect = popover.getBoundingClientRect()

  let x = rect.left - popoverRect.width - 8
  let y = rect.top

  if (x < 8) {
    x = rect.right + 8
  }

  if (y + popoverRect.height > window.innerHeight - 8) {
    y = window.innerHeight - popoverRect.height - 8
  }
  if (y < 8) y = 8

  popover.style.left = `${x}px`
  popover.style.top = `${y}px`
}

function startTracking() {
  scrollContainer = document.querySelector(".center") ?? document.body
  scrollHandler = () => positionSync()

  scrollContainer.addEventListener("scroll", scrollHandler, { passive: true })
  window.addEventListener("scroll", scrollHandler, { passive: true })

  if (rafId !== null) return
  const loop = () => {
    positionSync()
    rafId = requestAnimationFrame(loop)
  }
  rafId = requestAnimationFrame(loop)
}

function stopTracking() {
  if (scrollHandler) {
    scrollContainer?.removeEventListener("scroll", scrollHandler)
    window.removeEventListener("scroll", scrollHandler)
    scrollHandler = null
    scrollContainer = null
  }
  if (rafId !== null) {
    cancelAnimationFrame(rafId)
    rafId = null
  }
}

async function updateView(view: EditorView) {
  const completion = selectedCompletion(view.state)
  if (!completion?.detail) return

  const slug = completion.detail.split(" ")[0]
  if (slug !== currentSlug) {
    currentSlug = slug
    await render(slug)
  }
}

export function showPreview(view: EditorView): void {
  ensurePopover()
  positionSync()
  popover!.classList.add("active-popover")
  startTracking()
  updateView(view)
}

export function hidePreview(): void {
  stopTracking()
  abort?.abort()
  popover?.classList.remove("active-popover")
  currentSlug = null
}

export function togglePreview(view: EditorView): boolean {
  if (isPreviewVisible()) {
    hidePreview()
  } else {
    showPreview(view)
  }
  return true
}

export function isPreviewVisible(): boolean {
  return popover?.classList.contains("active-popover") ?? false
}

export function onEditorUpdate(update: ViewUpdate): void {
  if (!isPreviewVisible()) return
  if (completionStatus(update.state) !== "active") return hidePreview()
  updateView(update.view)
}

export function cleanupPreview(): void {
  hidePreview()
  popover?.remove()
  popover = null
  inner = null
}
