document.addEventListener("nav", (e) => {
  if (e.detail.url !== "music") return

  const playlist = document.getElementsByClassName("playlists")[0] as HTMLDivElement | null
  const prev = document.querySelector(".previous") as HTMLButtonElement | null
  const next = document.querySelector(".next") as HTMLButtonElement | null
  const iframes = playlist?.querySelectorAll("iframe")
  console.log(iframes)
  if (!playlist || !iframes) return

  // index
  let current = 0
  const gap = 20

  const scroll = (idx: number) => {
    let position = 0
    for (let i = 0; i < idx; i++) {
      position += iframes[i].offsetWidth + gap
    }

    const selected = iframes[idx]
    if (!selected) return
    console.log(selected, idx)
    position -= (playlist.offsetWidth - selected.offsetWidth) / 2

    playlist.scrollTo({ left: position, behavior: "smooth" })
  }

  const prevClick = () => {
    if (current > 0) {
      current--
      scroll(current)
    }
  }

  const nextClick = () => {
    if (current < iframes.length - 1) {
      current++
      scroll(current)
    }
  }

  prev?.addEventListener("click", prevClick)
  window.addCleanup(() => prev?.removeEventListener("click", prevClick))

  next?.addEventListener("click", nextClick)
  window.addCleanup(() => next?.removeEventListener("click", nextClick))
})
