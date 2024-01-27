const timeout = 500

const filterPage = (slug: string): boolean => {
  const dataSlug = document.querySelector("body")?.getAttribute("data-slug")
  return !dataSlug?.startsWith(slug)
}

document.addEventListener("nav", () => {
  const progress = document.getElementById("progress")
  let hideTimeout: ReturnType<typeof setTimeout>

  const hide = () => {
    if (!progress) return
    progress.style.backgroundColor = "transparent"
  }
  const show = () => {
    if (!progress) return
    progress.style.backgroundColor = "var(--secondary)"
    clearTimeout(hideTimeout)
    hideTimeout = setTimeout(hide, timeout)
  }

  if (filterPage("tags")) show()

  window.addEventListener("scroll", () => {
    if (!progress) return
    if (filterPage("tags")) {
      show()
      const totalHeight =
        document.documentElement.scrollHeight - document.documentElement.clientHeight
      const scrollPosition = window.scrollY
      const width = (scrollPosition / totalHeight) * 100
      progress.style.width = width + "%"
    }
  })

  if (filterPage("tags")) hideTimeout = setTimeout(hide, timeout)
})
