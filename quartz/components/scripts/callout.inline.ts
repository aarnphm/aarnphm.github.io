function toggleCallout(this: HTMLElement) {
  const outerBlock = this.parentElement!
  outerBlock.classList.toggle("is-collapsed")
  const collapsed = outerBlock.classList.contains("is-collapsed")
  const height = collapsed ? this.scrollHeight : outerBlock.scrollHeight
  outerBlock.style.maxHeight = height + "px"

  // walk and adjust height of all parents
  let current = outerBlock
  let parent = outerBlock.parentElement
  while (parent) {
    if (!parent.classList.contains("callout")) {
      return
    }

    const collapsed = parent.classList.contains("is-collapsed")
    const height = collapsed ? parent.scrollHeight : parent.scrollHeight + current.scrollHeight
    parent.style.maxHeight = height + "px"

    current = parent
    parent = parent.parentElement
  }
}

function wouldBreakAcrossPages(element: HTMLElement): boolean {
  const rect = element.getBoundingClientRect()
  const pageHeight = 1056 // Standard US Letter size in pixels (11 inches * 96 DPI)
  const elementHeight = rect.height
  const elementTop = rect.top

  // Calculate the position relative to the page
  const positionOnPage = elementTop % pageHeight

  // Check if element would cross page boundary
  return positionOnPage + elementHeight > pageHeight
}

function setupCallout() {
  const collapsible = document.getElementsByClassName(
    `callout is-collapsible`,
  ) as HTMLCollectionOf<HTMLDivElement>
  for (const div of collapsible) {
    const title = div.firstElementChild
    if (!title) continue

    title.addEventListener("click", toggleCallout)
    window.addCleanup(() => title.removeEventListener("click", toggleCallout))

    const collapsed = div.classList.contains("is-collapsed")
    const height = collapsed ? title.scrollHeight : div.scrollHeight
    div.style.maxHeight = height + "px"
  }

  const callouts = document.getElementsByClassName("callout") as HTMLCollectionOf<HTMLElement>
  for (const element of callouts) {
    if (wouldBreakAcrossPages(element)) {
      element.style.pageBreakBefore = "always"
      // Also add CSS class for stylesheet handling
      element.classList.add("force-page-break")
    }
  }
}

document.addEventListener("nav", setupCallout)
