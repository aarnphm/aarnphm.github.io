const bufferPx = 150
const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    const slug = entry.target.id
    const tocEntryElement = document.querySelector(`a[data-for="${slug}"]`)
    const layout = (document.querySelector(".toc") as HTMLDivElement).dataset.layout
    const windowHeight = entry.rootBounds?.height
    if (windowHeight && tocEntryElement) {
      if (layout === "minimal") {
        if (entry.boundingClientRect.y < windowHeight) {
          tocEntryElement.classList.add("in-view")
        } else {
          tocEntryElement.classList.remove("in-view")
        }
      } else {
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
  }
})

document.addEventListener("nav", () => {
  // update toc entry highlighting
  observer.disconnect()
  const headers = document.querySelectorAll("h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]")
  headers.forEach((header) => observer.observe(header))
})
