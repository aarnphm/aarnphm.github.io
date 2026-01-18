import { annotate } from "rough-notation"
import type { RoughAnnotation } from "rough-notation/lib/model"

let ag: RoughAnnotation | null = null
const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    const slug = entry.target.id
    const tocEntryElement = document.querySelector(`button[data-for="${slug}"]`)
    if (!tocEntryElement) continue

    const toc = document.querySelector<HTMLDivElement>(".toc")
    if (!toc) continue

    const windowHeight = entry.rootBounds?.height
    if (!windowHeight) continue

    const layout = toc.dataset.layout
    if (layout === "minimal") {
      tocEntryElement.classList.toggle("in-view", entry.boundingClientRect.y < windowHeight)
    } else {
      const parentLi = tocEntryElement.parentElement as HTMLLIElement
      const inView = entry.boundingClientRect.y < windowHeight
      tocEntryElement.classList.toggle("in-view", inView)
      parentLi.classList.toggle("in-view", inView)
    }
  }
})

function onClick(evt: MouseEvent) {
  const indicator = evt.target as HTMLDivElement
  const button = indicator.parentElement as HTMLButtonElement

  const href = button.dataset.href
  if (!href?.startsWith("#")) return

  evt.preventDefault()
  scrollToElement(href)

  document.body.classList.remove("toc-hover")
  const buttons = document.querySelectorAll<HTMLButtonElement>("#toc button[data-for]")
  buttons.forEach((button) => {
    const fill = button.querySelector(".fill") as HTMLElement
    if (fill) {
      fill.style.transform = "scaleX(1)"
      fill.style.opacity = ""
    }
  })

  if (window.location.hash) {
    setTimeout(() => {
      scrollToElement(window.location.hash)
    }, 10)
  }
}

function scrollToElement(hash: string) {
  const elementId = hash.slice(1)
  const element = document.getElementById(elementId)
  if (!element) return

  const collapsibleParent = element.closest(".collapsible-header-content")
  if (collapsibleParent) {
    const wrapper = collapsibleParent.closest(".collapsible-header")
    const button = wrapper?.querySelector(".toggle-button") as HTMLButtonElement
    if (button?.getAttribute("aria-expanded") === "false") {
      button.click()
    }
  }

  if (ag) ag.hide()

  const highlight = element.querySelector("span.highlight-span") as HTMLElement
  if (highlight) {
    ag = annotate(highlight, {
      type: "bracket",
      color: "rgba(234, 157, 52, 0.45)",
      animate: false,
      multiline: true,
      brackets: ["left", "right"],
    })

    setTimeout(() => ag!.show(), 50)
    window.setTimeout(() => ag?.hide(), 2500)
  }

  const rect = element.getBoundingClientRect()
  const absoluteTop = window.scrollY + rect.top

  window.scrollTo({
    top: absoluteTop - 100,
    behavior: "smooth",
  })

  history.pushState(null, "", hash)
}

document.addEventListener("nav", (ev: CustomEventMap["nav"]) => {
  if (ev.detail.url) {
    const url = new URL(ev.detail.url, window.location.origin)
    if (url.hash) {
      scrollToElement(decodeURIComponent(url.hash))
    }
  }
})

function setupToc() {
  const toc = document.getElementById("toc")
  if (!toc) return

  if (toc.dataset.layout === "minimal") {
    const nav = toc.querySelector("#toc-vertical") as HTMLElement
    if (!nav) return

    const buttons = toc.querySelectorAll("button[data-for]") as NodeListOf<HTMLButtonElement>
    for (const button of buttons) {
      button.addEventListener("click", onClick)
      window.addCleanup(() => button.removeEventListener("click", onClick))
    }

    const onMouseEnter = () => document.body.classList.add("toc-hover")

    const onMouseLeave = () => {
      document.body.classList.remove("toc-hover")
      buttons.forEach((button) => {
        const fill = button.querySelector(".fill") as HTMLElement
        if (fill) {
          fill.style.transform = "scaleX(1)"
          fill.style.opacity = ""
        }
      })
    }

    const onMouseMove = (evt: MouseEvent) => {
      const navRect = nav.getBoundingClientRect()
      const mouseY = evt.clientY - navRect.top

      buttons.forEach((button) => {
        const buttonRect = button.getBoundingClientRect()
        const buttonY = buttonRect.top + buttonRect.height / 2 - navRect.top
        const styles = getComputedStyle(button)
        const distance = mouseY - buttonY
        const sigma = 42
        const maxScale = parseFloat(styles.getPropertyValue("--indicator-position"))
        const isButton = Math.abs(distance) < buttonRect.height / 2

        const fill = button.querySelector(".fill") as HTMLElement
        if (!fill) return

        const minScale = parseInt(styles.getPropertyValue("--fill-width"))
        fill.style.animation = "unset !important"
        fill.style.opacity = isButton ? "1" : "0.35"
        fill.style.transform = `scaleX(${isButton ? maxScale : minScale + (maxScale - minScale) * Math.exp(-Math.pow(distance, 2) / (2 * Math.pow(sigma, 2)))})`
      })
    }

    toc.addEventListener("mouseenter", onMouseEnter)
    toc.addEventListener("mouseleave", onMouseLeave)
    nav.addEventListener("mousemove", onMouseMove)

    window.addCleanup(() => {
      toc.removeEventListener("mouseenter", onMouseEnter)
      toc.removeEventListener("mouseleave", onMouseLeave)
      nav.removeEventListener("mousemove", onMouseMove)
    })
  }
}

window.addEventListener("resize", setupToc)
document.addEventListener("nav", () => {
  setupToc()
  observer.disconnect()
  document
    .querySelectorAll("h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]")
    .forEach((header) => observer.observe(header))
})
