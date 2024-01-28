const timeout = 500

let prevShortcutHandler: ((e: HTMLElementEventMap["keydown"]) => void) | undefined = undefined

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

  show()

  window.addEventListener("scroll", () => {
    if (!progress) return
    show()
    const totalHeight =
      document.documentElement.scrollHeight - document.documentElement.clientHeight
    const scrollPosition = window.scrollY
    const width = (scrollPosition / totalHeight) * 100
    progress.style.width = width + "%"
  })

  hideTimeout = setTimeout(hide, timeout)
})

document.addEventListener("nav", () => {
  const img = document.querySelectorAll("img")
  const windowHeight = window.innerHeight / 1.5
  const checkImgPosition = () => {
    img.forEach((el) => {
      const position = el.getBoundingClientRect().top
      position - windowHeight <= 0 ? el.classList.add("visible") : el.classList.remove("visible")
    })
  }
  window.addEventListener("scroll", checkImgPosition)
  checkImgPosition()
})

// keybind shortcut
document.addEventListener("nav", () => {
  function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (e.key === "/" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      window.location.pathname = "/"
    }
  }

  if (prevShortcutHandler) {
    document.removeEventListener("keydown", prevShortcutHandler)
  }

  document.addEventListener("keydown", shortcutHandler)
  prevShortcutHandler = shortcutHandler
})
