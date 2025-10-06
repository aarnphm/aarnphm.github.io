import { registerEscapeHandler, closeReader } from "./util"

function cleanupArticleContent(clonedArticle: HTMLElement): void {
  // Remove popovers and simplify links
  clonedArticle.querySelectorAll("a.internal").forEach((link) => {
    link.className = "internal"
    link.setAttribute("data-no-popover", "true")
  })

  // Simplify collapsible callouts
  clonedArticle.querySelectorAll("blockquote.is-collapsible").forEach((callout) => {
    const dataCallout = callout.getAttribute("data-callout")
    callout.className = `callout ${dataCallout}`
    callout.removeAttribute("style")
  })

  // Remove callout icons
  clonedArticle.querySelectorAll(".callout-title").forEach((title) => {
    title.querySelectorAll(".callout-icon, .fold-callout-icon").forEach((icon) => icon.remove())
  })

  // Remove singleton elements
  clonedArticle.querySelectorAll("[data-singleton]").forEach((el) => el.remove())

  // Cleanup code block themes
  clonedArticle.querySelectorAll("code[data-language], pre").forEach((el) => {
    el.removeAttribute("data-theme")
    if (el.hasAttribute("style")) {
      const style = el.getAttribute("style") || ""
      const cleanedStyle = style
        .split(";")
        .filter((s) => s.trim() && !s.includes("--shiki"))
        .join(";")
      if (cleanedStyle) {
        el.setAttribute("style", cleanedStyle)
      } else {
        el.removeAttribute("style")
      }
    }
  })

  // Simplify mermaid blocks
  clonedArticle.querySelectorAll("code.mermaid").forEach((code) => {
    const dataClipboard = code.getAttribute("data-clipboard")
    if (dataClipboard) {
      code.textContent = dataClipboard
    }
    // Remove expand button
    code.parentElement?.querySelectorAll(".expand-button").forEach((btn) => btn.remove())
  })
}

document.addEventListener("nav", () => {
  const readerView = document.querySelector(".reader") as HTMLElement
  if (!readerView) return

  const closeFunctor = () => {
    closeReader(readerView)
    const headers = document.querySelector<HTMLDivElement>('section[class~="header"]')
    if (headers) {
      headers.style.display = "grid"
    }
    // Clear reader content on close
    const readerContent = readerView.querySelector(".reader-content")
    if (readerContent) {
      readerContent.innerHTML = ""
    }
  }

  // Register escape handler for the reader view
  registerEscapeHandler(readerView, closeFunctor)

  // Register close button handler
  const closeBtn = readerView.querySelector(".reader-close")
  if (closeBtn) {
    closeBtn.addEventListener("click", closeFunctor)
    window.addCleanup(() => closeBtn.removeEventListener("click", closeFunctor))
  }

  function showReader() {
    // Find the main article element
    const article = document.querySelector("article.popover-hint")
    if (!article) return

    // Clone the article content
    const clonedArticle = article.cloneNode(true) as HTMLElement

    // Apply cleanup transformations
    cleanupArticleContent(clonedArticle)

    // Get the title
    const title = document.querySelector("h1.article-title")?.textContent || ""

    // Populate reader content
    const readerContent = readerView.querySelector(".reader-content")
    if (readerContent) {
      readerContent.innerHTML = ""
      if (title) {
        const titleEl = document.createElement("h1")
        titleEl.className = "reader-title"
        titleEl.textContent = title
        readerContent.appendChild(titleEl)
      }
      readerContent.appendChild(clonedArticle)
    }

    // Show the reader view
    readerView.classList.add("active")
    const quartz = document.getElementById("quartz-root") as HTMLDivElement
    quartz.style.overflow = "hidden"
    quartz.style.maxHeight = "0px"

    const headers = document.querySelector<HTMLDivElement>('section[class~="header"]')
    if (headers) {
      headers.style.display = "none"
    }
  }

  async function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (e.key === "b" && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
      e.preventDefault()
      const readerViewOpen = readerView.classList.contains("active")
      readerViewOpen ? closeFunctor() : showReader()
    }
  }

  document.addEventListener("keydown", shortcutHandler)
  window.addCleanup(() => document.removeEventListener("keydown", shortcutHandler))
})
