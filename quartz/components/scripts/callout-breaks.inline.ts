import { debounce } from "./util"

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

// Function to handle page breaks for callouts
function handleCalloutPageBreaks() {
  const callouts = document.querySelectorAll(".callout")

  callouts.forEach((callout) => {
    const element = callout as HTMLElement
    if (wouldBreakAcrossPages(element)) {
      element.style.pageBreakBefore = "always"
      // Also add CSS class for stylesheet handling
      element.classList.add("force-page-break")
    }
  })
}

// Add event listeners
document.addEventListener("nav", () => {
  // Let the content render first
  setTimeout(handleCalloutPageBreaks, 100)

  // Handle window resize
  window.addEventListener("resize", debounce(handleCalloutPageBreaks, 150))
  window.addCleanup(() => window.removeEventListener("resize", handleCalloutPageBreaks))

  // Handle print media change
  const mediaQueryList = window.matchMedia("print")
  mediaQueryList.addEventListener("change", handleCalloutPageBreaks)
  window.addCleanup(() => mediaQueryList.removeEventListener("change", handleCalloutPageBreaks))
})
