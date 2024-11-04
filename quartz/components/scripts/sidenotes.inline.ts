import { removeAllChildren, isInViewport, updatePosition } from "./util"

const ARTICLE_CONTENT_SELECTOR = ".center"
const FOOTNOTE_SECTION_SELECTOR = "section[data-footnotes] > ol"
const INDIVIDUAL_FOOTNOTE_SELECTOR = "li[id^='user-content-fn-']"

function updateSidenotes() {
  const articleContent = document.querySelector(ARTICLE_CONTENT_SELECTOR) as HTMLElement
  const sideContainer = document.querySelector(".sidenotes") as HTMLElement
  if (!articleContent || !sideContainer) return

  const sidenotes = sideContainer.querySelectorAll(".sidenote-element") as NodeListOf<HTMLElement>
  for (const sidenote of sidenotes) {
    const sideId = sidenote.id.replace("sidebar-", "")
    const intextLink = articleContent.querySelector(`a[href="#${sideId}"]`) as HTMLElement
    if (!intextLink) return

    if (isInViewport(intextLink)) {
      sidenote.classList.add("in-view")
      intextLink.classList.add("active")
      updatePosition(intextLink, sidenote, sideContainer)
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

document.addEventListener("nav", () => {
  const articleContent = document.querySelector(ARTICLE_CONTENT_SELECTOR) as HTMLElement
  const footnoteSections = Array.from(document.querySelectorAll(FOOTNOTE_SECTION_SELECTOR))
  if (footnoteSections.length == 0 || !articleContent) return

  const lastIdx = footnoteSections.length - 1
  const footnoteSection = footnoteSections[lastIdx] as HTMLElement
  console.log(footnoteSection)

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
    const rootFontSize = parseFloat(getComputedStyle(document.documentElement).fontSize)
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

    ol.appendChild(sidenote)
  })

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
