document.addEventListener("nav", () => {
  const deck = document.querySelector<HTMLDivElement>(".slides-deck")
  const slides = Array.from(document.querySelectorAll<HTMLElement>(".slide"))
  const prev = document.querySelector<HTMLButtonElement>(".slides-controls .prev")
  const next = document.querySelector<HTMLButtonElement>(".slides-controls .next")
  const status = document.querySelector<HTMLSpanElement>(".slides-controls .status")
  if (!deck || slides.length === 0 || !prev || !next || !status) return

  let idx = 0
  const clamp = (v: number) => Math.max(0, Math.min(slides.length - 1, v))

  const parseHash = () => {
    const h = window.location.hash
    const m = h.match(/slide-(\d+)/)
    if (m) {
      const n = parseInt(m[1], 10)
      if (!Number.isNaN(n)) idx = clamp(n)
    }
  }

  const update = () => {
    // close any open mermaid popups before switching slides
    const activePopups = document.querySelectorAll("#mermaid-container.active")
    activePopups.forEach((popup) => popup.classList.remove("active"))

    slides.forEach((el, i) => {
      el.classList.toggle("active", i === idx)
      el.setAttribute("aria-hidden", i === idx ? "false" : "true")
    })
    status.textContent = `${idx + 1} / ${slides.length}`
    const target = document.getElementById(`slide-${idx}`)
    if (target) target.scrollIntoView({ behavior: "auto", block: "start" })
    history.replaceState(null, "", `#slide-${idx}`)

    // trigger mermaid re-rendering for the new active slide
    document.dispatchEvent(new CustomEvent("slide-change"))
  }

  const goPrev = () => {
    if (idx > 0) idx -= 1
    update()
  }
  const goNext = () => {
    if (idx < slides.length - 1) idx += 1
    update()
  }

  // Expand all callouts by default for slides
  const expandAllCallouts = () => {
    const callouts = deck.querySelectorAll<HTMLElement>("blockquote.callout, .callout")
    for (const el of Array.from(callouts)) {
      el.classList.remove("is-collapsed")
      if (el.style && typeof el.style.maxHeight !== "undefined") el.style.maxHeight = ""
      // clear any descendant inline max-heights
      const descendants = el.querySelectorAll<HTMLElement>("[style*='max-height']")
      descendants.forEach((child) => (child.style.maxHeight = ""))
    }
  }

  // Expand all transcludes by default for slides
  const expandAllTranscludes = () => {
    const transcludes = deck.querySelectorAll<HTMLElement>(".transclude-collapsible")
    for (const el of Array.from(transcludes)) {
      el.classList.remove("is-collapsed")
      const content = el.querySelector<HTMLElement>(".transclude-content")
      if (content && content.style) {
        content.style.gridTemplateRows = "1fr"
      }
      // clear any descendant inline grid-template-rows
      const descendants = el.querySelectorAll<HTMLElement>(".transclude-content")
      descendants.forEach((child) => {
        if (child.style) child.style.gridTemplateRows = "1fr"
      })
    }
  }

  parseHash()
  expandAllCallouts()
  expandAllTranscludes()
  update()

  const keyEvent = (e: any) => {
    if (e.key === "ArrowLeft") goPrev()
    if (e.key === "ArrowRight" || e.key === " ") goNext()
  }

  prev.addEventListener("click", goPrev)
  next.addEventListener("click", goNext)
  window.addEventListener("keydown", keyEvent)
  window.addCleanup(() => {
    prev.removeEventListener("click", goPrev)
    next.removeEventListener("click", goNext)
    window.removeEventListener("keydown", keyEvent)
  })
})
