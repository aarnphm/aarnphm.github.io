function setupNoteStack(root: HTMLElement): () => void {
  const viewport = root.querySelector<HTMLElement>(':scope > .note-stack-viewport')
  const toolbar = root.querySelector<HTMLElement>(':scope > .note-stack-toolbar')
  const slides = Array.from(
    viewport?.querySelectorAll<HTMLElement>(':scope > .note-stack-slide') ?? [],
  )
  const steps = Array.from(toolbar?.querySelectorAll<HTMLButtonElement>('[data-note-step]') ?? [])
  const previous = toolbar?.querySelector<HTMLButtonElement>('[data-note-previous]')
  const next = toolbar?.querySelector<HTMLButtonElement>('[data-note-next]')
  const position = toolbar?.querySelector<HTMLElement>('[data-note-position]')
  const status = root.querySelector<HTMLElement>(':scope > .note-stack-status')
  if (!viewport || !previous || !next || !position || slides.length === 0) return () => {}
  const controller = new AbortController()
  const { signal } = controller
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')
  let active = 0
  let frame = 0
  let announce = false

  const update = (index: number) => {
    active = index
    for (const [i, slide] of slides.entries()) {
      const current = i === index
      if (!current && slide.contains(document.activeElement))
        steps[index]?.focus({ preventScroll: true })
      slide.inert = !current
      slide.tabIndex = current ? 0 : -1
      if (current) steps[i]?.setAttribute('aria-current', 'step')
      else steps[i]?.removeAttribute('aria-current')
    }
    previous.setAttribute('aria-disabled', String(index === 0))
    next.setAttribute('aria-disabled', String(index === slides.length - 1))
    position.textContent = `${index + 1} / ${slides.length}`
  }
  const move = (index: number) => {
    if (index < 0 || index >= slides.length) return
    announce = true
    update(index)
    const left = index * viewport.clientWidth
    if (Math.abs(viewport.scrollLeft - left) < 1) {
      settle()
      return
    }
    viewport.scrollTo({ left, behavior: reducedMotion.matches ? 'instant' : 'smooth' })
    if (reducedMotion.matches) settle()
  }
  const settle = () => {
    if (viewport.clientWidth === 0) return
    const index = Math.max(
      0,
      Math.min(slides.length - 1, Math.round(viewport.scrollLeft / viewport.clientWidth)),
    )
    update(index)
    if (announce && status) status.textContent = slides[index].getAttribute('aria-label') ?? ''
    announce = false
    const step = steps[index]
    if (step?.parentElement) {
      const rail = step.parentElement
      rail.scrollTo({
        left: step.offsetLeft - rail.offsetLeft - (rail.clientWidth - step.clientWidth) / 2,
        behavior: 'instant',
      })
    }
  }
  previous.addEventListener('click', () => move(active - 1), { signal })
  next.addEventListener('click', () => move(active + 1), { signal })
  for (const [index, step] of steps.entries()) {
    step.addEventListener('click', () => move(index), { signal })
    step.addEventListener(
      'keydown',
      event => {
        const target =
          event.key === 'ArrowRight'
            ? index + 1
            : event.key === 'ArrowLeft'
              ? index - 1
              : event.key === 'Home'
                ? 0
                : event.key === 'End'
                  ? steps.length - 1
                  : -1
        if (target < 0 || target >= steps.length) return
        event.preventDefault()
        steps[target].focus({ preventScroll: true })
        move(target)
      },
      { signal },
    )
  }
  viewport.addEventListener('scrollend', settle, { signal })
  const resize = new ResizeObserver(() => {
    cancelAnimationFrame(frame)
    frame = requestAnimationFrame(() => {
      if (viewport.clientWidth > 0)
        viewport.scrollTo({ left: active * viewport.clientWidth, behavior: 'instant' })
    })
  })
  root.dataset.noteStackReady = 'true'
  position.style.minWidth = `${String(slides.length).length * 2 + 3}ch`
  update(0)
  resize.observe(viewport)
  return () => {
    controller.abort()
    resize.disconnect()
    cancelAnimationFrame(frame)
    delete root.dataset.noteStackReady
    for (const slide of slides) {
      slide.inert = false
      slide.removeAttribute('tabindex')
    }
  }
}

document.addEventListener('nav', () => {
  const cleanups = Array.from(document.querySelectorAll<HTMLElement>('.note-stack'), setupNoteStack)
  window.addCleanup(() => cleanups.forEach(cleanup => cleanup()))
})
