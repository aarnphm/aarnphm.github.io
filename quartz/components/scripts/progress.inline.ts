import { getFullSlug } from "../../util/path"

document.addEventListener("nav", (e: CustomEventMap["nav"]) => {
  const timeout = 500
  let hideTimeout: ReturnType<typeof setTimeout>
  const slug = e.detail.url
  if (slug === "index") return

  const hide = () => {
    const progress = document.getElementById("progress")
    if (!progress) return
    if (document.activeElement?.classList.contains("active")) return
    progress.style.backgroundColor = "transparent"
  }
  const show = () => {
    const progress = document.getElementById("progress")
    if (!progress) return
    progress.style.backgroundColor = "var(--dark)"
    clearTimeout(hideTimeout)
    hideTimeout = setTimeout(hide, timeout)
  }

  window.addEventListener("scroll", () => {
    const progress = document.getElementById("progress")
    if (!progress) return
    if (document.querySelectorAll(".active").length > 0) {
      hide()
      return
    } else {
      show()
    }
    var winScroll = document.body.scrollTop || document.documentElement.scrollTop
    var height = document.documentElement.scrollHeight - document.documentElement.clientHeight
    progress.style.height = (winScroll / height) * 100 + "%"
    return
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
  window.onscroll = checkImgPosition
  checkImgPosition()
})
