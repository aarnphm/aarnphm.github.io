import { getFullSlug } from "../../util/path"

const timeout = 500

document.addEventListener("nav", (e: CustomEventMap["nav"]) => {
  let hideTimeout: ReturnType<typeof setTimeout>
  const slug = e.detail.url
  if (slug === "index") return

  const hide = () => {
    const progress = document.getElementById("progress")
    if (!progress) return
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
    show()
    var winScroll = document.body.scrollTop || document.documentElement.scrollTop
    var height = document.documentElement.scrollHeight - document.documentElement.clientHeight
    progress.style.height = (winScroll / height) * 100 + "%"
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
