import { removeAllChildren, isInViewport, updatePosition, debounce } from "./util"

function checkSidenoteSpacing(current: HTMLElement, allSidenotes: NodeListOf<HTMLElement>) {
  const currentRect = current.getBoundingClientRect()
  const currentBottom = currentRect.top + currentRect.height

  const sortedSidenotes = Array.from(allSidenotes).sort((a, b) => {
    const aRect = a.getBoundingClientRect()
    const bRect = b.getBoundingClientRect()
    return aRect.top - bRect.top
  })

  const currentIndex = sortedSidenotes.indexOf(current)
  const nextSidenote = sortedSidenotes[currentIndex + 1]

  if (!nextSidenote) {
    // No next sidenote, can expand
    const inner = current.querySelector(".sidenote-inner") as HTMLElement
    if (inner) inner.style.maxHeight = "unset"
    return
  }

  const nextRect = nextSidenote.getBoundingClientRect()
  const spacing = nextRect.top - currentBottom

  const inner = current.querySelector(".sidenote-inner") as HTMLElement
  if (inner && spacing > 30) {
    inner.style.maxHeight = "unset"
  }
}

function updateSidenotes() {
  const articleContent = document.querySelector(".center") as HTMLElement
  const sideContainer = document.querySelector(".sidenotes") as HTMLElement
  if (!articleContent || !sideContainer) return

  const sidenotes = sideContainer.querySelectorAll(".sidenote-element") as NodeListOf<HTMLElement>

  for (const sidenote of sidenotes) {
    const sideId = sidenote.id.replace("sidebar-", "")
    const intextLink = articleContent.querySelector(`a[href="#${sideId}"]`) as HTMLElement
    if (!intextLink) return

    let currentElement: HTMLElement | null = intextLink
    let collapsedContent = null
    while (currentElement && !collapsedContent) {
      if (currentElement.classList.contains("collapsible-header-content")) {
        collapsedContent = currentElement
      } else if (
        currentElement.tagName === "article" &&
        currentElement.classList.contains("popover-hint")
      ) {
        break
      }
      currentElement = currentElement.parentElement
    }

    if (
      (collapsedContent && collapsedContent.classList.contains("collapsed")) ||
      !isInViewport(intextLink)
    ) {
      sidenote.classList.remove("in-view")
      intextLink.classList.remove("active")
    } else {
      sidenote.classList.add("in-view")
      intextLink.classList.add("active")
      updatePosition(intextLink, sidenote, sideContainer)
      checkSidenoteSpacing(sidenote, sidenotes)
    }
  }
}

function createSidenote(footnote: HTMLElement, footnoteId: string): HTMLLIElement {
  const sidenote = document.createElement("li")
  sidenote.classList.add("sidenote-element")
  sidenote.style.position = "absolute"
  sidenote.id = `sidebar-${footnoteId}`
  sidenote.append(...(footnote.cloneNode(true) as HTMLElement).children)

  // create inner child container
  let innerContainer = sidenote.querySelector(".sidenote-inner")
  if (!innerContainer) {
    innerContainer = document.createElement("div") as HTMLDivElement
    innerContainer.className = "sidenote-inner"
    while (sidenote.firstChild) {
      innerContainer.appendChild(sidenote.firstChild)
    }
    sidenote.appendChild(innerContainer)
  }

  return sidenote
}

document.addEventListener("nav", () => {
  const articleContent = document.querySelector(".center") as HTMLElement
  const footnoteSectionList = Array.from(
    articleContent.querySelectorAll("section[data-footnotes] > ol"),
  ) as HTMLOListElement[]
  if (!articleContent) return

  const sideContainer = document.querySelector(".left .sidenotes") as HTMLElement
  if (!sideContainer) return

  removeAllChildren(sideContainer)

  const ol = document.createElement("ol")
  sideContainer.appendChild(ol)

  // If no footnote sections or we disable sidenotes in frontmatter, we still want the dashed lines
  if (footnoteSectionList.length === 0 || sideContainer.dataset.disableNotes === "true") {
    updateSidenotes()
    return
  }

  const footnoteItems = footnoteSectionList.flatMap((ol) =>
    Array.from(ol.querySelectorAll("li[id^='user-content-fn-']")),
  ) as HTMLLIElement[]

  for (const [i, footnote] of footnoteItems.entries()) {
    const footnoteId = footnote.id
    const intextLink = articleContent.querySelector(`a[href="#${footnoteId}"]`) as HTMLElement
    if (!intextLink) continue
    const sidenote = createSidenote(footnote, footnoteId)
    sidenote.style.position = "absolute"
    sidenote.dataset.count = `${i + 1}`
    intextLink.innerHTML = `<span class="indicator-hook"></span>${i + 1}`
    ol.appendChild(sidenote)
  }

  updateSidenotes()

  // Update on scroll with debouncing
  const debouncedUpdate = debounce(updateSidenotes, 2)

  document.addEventListener("scroll", debouncedUpdate, { passive: true })
  window.addEventListener("resize", debouncedUpdate, { passive: true })

  // Cleanup
  window.addCleanup(() => {
    document.removeEventListener("scroll", debouncedUpdate)
    window.removeEventListener("resize", debouncedUpdate)
  })
})
