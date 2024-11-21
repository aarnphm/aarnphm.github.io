const bufferPx = 150
const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    const slug = entry.target.id
    const tocEntryElement = document.querySelector(`a[data-for="${slug}"]`)
    const windowHeight = entry.rootBounds?.height
    if (windowHeight && tocEntryElement) {
      const parentLi = tocEntryElement.parentElement as HTMLLIElement
      if (entry.boundingClientRect.y < windowHeight) {
        tocEntryElement.classList.add("in-view")
        parentLi.classList.add("in-view")
      } else {
        tocEntryElement.classList.remove("in-view")
        parentLi.classList.remove("in-view")
      }
    }
  }
})

function setupToc() {
  const toc = document.getElementById("toc")
  if (toc) {
    const collapsed = toc.classList.contains("collapsed")
    const content = toc.nextElementSibling as HTMLElement | undefined
    if (!content) return
    content.style.maxHeight = collapsed ? "0px" : content.scrollHeight + "px"
  }
}

window.addEventListener("resize", setupToc)
document.addEventListener("nav", () => {
  setupToc()

  // update toc entry highlighting
  observer.disconnect()
  const headers = document.querySelectorAll("h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]")
  headers.forEach((header) => observer.observe(header))
})
