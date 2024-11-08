import { removeAllChildren, isInViewport, updatePosition } from "./util"

const ARTICLE_CONTENT_SELECTOR = ".center"
const FOOTNOTE_SECTION_SELECTOR = "section[data-footnotes] > ol"
const INDIVIDUAL_FOOTNOTE_SELECTOR = "li[id^='user-content-fn-']"
const SPACING_THRESHOLD = 30 // pixels
const GROUP_MAX_HEIGHT = 1000 // pixels

interface SidenoteEntry {
  footnoteId: string
  intextLink: HTMLElement
  sidenote: HTMLLIElement
  footnote: HTMLLIElement
  verticalPosition: number
}

interface SidenoteGroup {
  elements: SidenoteEntry[]
  top: number
  bottom: number
}

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
  if (inner && spacing > SPACING_THRESHOLD) {
    inner.style.maxHeight = "unset"
  }
}

function getVerticalPosition(element: HTMLElement): number {
  const rect = element.getBoundingClientRect()
  const scrollTop = window.scrollY || document.documentElement.scrollTop
  return rect.top + scrollTop
}

function updateSidenotes() {
  const articleContent = document.querySelector(ARTICLE_CONTENT_SELECTOR) as HTMLElement
  const sideContainer = document.querySelector(".sidenotes") as HTMLElement
  if (!articleContent || !sideContainer) return

  const sidenotes = sideContainer.querySelectorAll(".sidenote-element") as NodeListOf<HTMLElement>

  // If no sidenotes, ensure the container still has proper height for dashed line
  if (sidenotes.length === 0) {
    const articleRect = articleContent.getBoundingClientRect()
    sideContainer.style.height = `${articleRect.height}px`
    return
  }

  for (const sidenote of sidenotes) {
    const sideId = sidenote.id.replace("sidebar-", "")
    const intextLink = articleContent.querySelector(`a[href="#${sideId}"]`) as HTMLElement
    if (!intextLink) return

    if (isInViewport(intextLink)) {
      sidenote.classList.add("in-view")
      intextLink.classList.add("active")
      updatePosition(intextLink, sidenote, sideContainer)
      checkSidenoteSpacing(sidenote, sidenotes)
    } else {
      sidenote.classList.remove("in-view")
      intextLink.classList.remove("active")
    }
  }
}

function debounce(fn: Function, delay: number) {
  let timeoutId: ReturnType<typeof setTimeout>
  return (...args: any[]) => {
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => fn(...args), delay)
  }
}

function createSidenote(
  footnote: HTMLElement,
  footnoteId: string,
  sideContainer: HTMLElement,
): HTMLLIElement {
  const sidenote = document.createElement("li")
  sidenote.classList.add("sidenote-element")
  sidenote.style.position = "absolute"
  const rootFontSize = parseFloat(getComputedStyle(document.documentElement).fontSize)
  sidenote.style.minWidth = `${sideContainer.offsetWidth - rootFontSize}px`
  sidenote.style.maxWidth = `${sideContainer.offsetWidth - rootFontSize}px`
  sidenote.id = `sidebar-${footnoteId}`
  const cloned = footnote.cloneNode(true) as HTMLElement
  const backref = cloned.querySelector("a[data-footnote-backref]")
  backref?.remove()
  sidenote.append(...cloned.children)

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
  const articleContent = document.querySelector(ARTICLE_CONTENT_SELECTOR) as HTMLElement
  const sections = Array.from(document.querySelectorAll("section[data-footnotes]")) as HTMLElement[]
  const footnoteSectionList = Array.from(
    document.querySelectorAll(FOOTNOTE_SECTION_SELECTOR),
  ) as HTMLOListElement[]
  if (!articleContent) return

  const sideContainer = document.querySelector(".sidenotes") as HTMLElement
  if (!sideContainer) return

  removeAllChildren(sideContainer)

  // Set container height to match article content
  const articleRect = articleContent.getBoundingClientRect()
  sideContainer.style.height = `${articleRect.height}px`
  sideContainer.style.top = "0px"

  const ol = document.createElement("ol")
  sideContainer.appendChild(ol)

  // If no footnote sections or we disable sidenotes in frontmatter, we still want the dashed lines
  if (footnoteSectionList.length === 0 || sideContainer.dataset.disableNotes === "true") {
    updateSidenotes()
    return
  }

  const footnoteItems = footnoteSectionList.flatMap((ol) =>
    Array.from(ol.querySelectorAll(INDIVIDUAL_FOOTNOTE_SELECTOR)),
  ) as HTMLLIElement[]

  // Create array of sidenote entries with position information
  const sidenoteEntries: SidenoteEntry[] = []

  for (const footnote of footnoteItems) {
    const footnoteId = footnote.id
    const intextLink = articleContent.querySelector(`a[href="#${footnoteId}"]`) as HTMLElement
    if (!intextLink) continue

    const sidenote = createSidenote(footnote, footnoteId, sideContainer)
    const verticalPosition = getVerticalPosition(intextLink)
    sidenoteEntries.push({ footnoteId, intextLink, sidenote, footnote, verticalPosition })
  }

  sidenoteEntries.sort((a, b) => a.verticalPosition - b.verticalPosition)

  // update the index accordingly with transclude in consideration
  for (const [index, entry] of sidenoteEntries.entries()) {
    const counter = index + 1
    entry.sidenote.dataset.count = `${counter}`
    const linkContent = Array.from(entry.intextLink.childNodes)
    const textNode = linkContent.find((node) => node.nodeType === Node.TEXT_NODE)
    if (textNode) {
      textNode.textContent = `${counter}`
    }

    ol.appendChild(entry.sidenote)
  }

  if (sections.length !== 1) {
    const lastSection = sections.pop()
    sections.map((section) => section.remove())
    const olList = lastSection?.getElementsByTagName("ol")[0] as HTMLOListElement
    removeAllChildren(olList)
    for (const entry of sidenoteEntries) {
      olList.appendChild(entry.footnote)
    }
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
