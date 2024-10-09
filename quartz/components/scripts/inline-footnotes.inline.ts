import { removeAllChildren } from "./util"
import { computePosition, flip, shift, offset } from "@floating-ui/dom"

const ARTICLE_CONTENT_SELECTOR = ".center"
const FOOTNOTE_SECTION_SELECTOR = "section[data-footnotes] > ol"
const INDIVIDUAL_FOOTNOTE_SELECTOR = "li[id^='user-content-fn-']"

const observer = new IntersectionObserver((entries) => {
  const inlineFootnotesContainer = document.getElementsByClassName("inline-footnotes")[0]
  if (!inlineFootnotesContainer) return

  for (const entry of entries) {
    const footnoteLink = entry.target as HTMLElement
    console.log(entry.boundingClientRect.y)
    const footnoteId = footnoteLink.getAttribute("href")?.slice(1)
    const footnoteCopy = inlineFootnotesContainer.querySelector(
      `[data-footnote-id="${footnoteId}"]`,
    ) as HTMLElement

    if (footnoteCopy) {
      if (entry.isIntersecting) {
        footnoteCopy.classList.add("visible")
        positionFootnote(footnoteCopy, footnoteLink)
      } else {
        footnoteCopy.classList.remove("visible")
      }
    }
  }
})

async function updateFootnotePosition(footnoteCopy: HTMLElement, footnoteLink: HTMLElement) {
  const { x, y } = await computePosition(footnoteLink, footnoteCopy, {
    placement: "right-start",
    middleware: [
      offset(10), // Adds some space between the footnote link and the footnote
      flip(),
      shift({ padding: 5 }),
    ],
  })

  Object.assign(footnoteCopy.style, {
    left: `${x}px`,
    top: `${y}px`,
  })
}

document.addEventListener("nav", async () => {
  const articleContent = document.querySelector(ARTICLE_CONTENT_SELECTOR)
  const footnoteSection = document.querySelector(FOOTNOTE_SECTION_SELECTOR)
  if (!footnoteSection || !articleContent) return

  const inlineFootnotesContainer = document.getElementsByClassName("inline-footnotes")[0]
  if (!inlineFootnotesContainer) return
  removeAllChildren(inlineFootnotesContainer)

  const footnotes = footnoteSection.querySelectorAll(INDIVIDUAL_FOOTNOTE_SELECTOR)

  footnotes.forEach((footnote) => {
    const footnoteId = footnote.id
    const footnoteLink = articleContent.querySelector(`a[href="#${footnoteId}"]`) as HTMLElement
    if (footnoteLink) {
      const footnoteCopy = document.createElement("div")
      footnoteCopy.innerHTML = footnote.innerHTML
      footnoteCopy.classList.add("inline-footnote")
      footnoteCopy.dataset.footnoteId = footnoteId
      inlineFootnotesContainer.appendChild(footnoteCopy)

      observer.observe(footnoteLink)
    }
  })

  const updateAllPositions = async () => {
    const visibleFootnotes = inlineFootnotesContainer.querySelectorAll(".inline-footnote.visible")
    visibleFootnotes.forEach((footnoteCopy) => {
      const footnoteId = footnoteCopy.getAttribute("data-footnote-id")
      const footnoteLink = articleContent.querySelector(`a[href="#${footnoteId}"]`) as HTMLElement
      if (footnoteLink) {
        updateFootnotePosition(footnoteCopy as HTMLElement, footnoteLink)
      }
    })
  }

  window.addEventListener("resize", updateAllPositions)
  window.addCleanup(() => {
    window.removeEventListener("resize", updateAllPositions)
    observer.disconnect()
  })
})
