const SOURCE = 'https://initialis.azlen.me/lib/initialis.js'

const mountProceduralInitials = () => {
  const nodes = document.querySelectorAll('canvas.gv-initial[data-settings]')
  if (nodes.length === 0) return
  document.dispatchEvent(new CustomEvent<void>('proceduralinitialstop'))
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const script = document.createElement('script')
  script.type = 'module'
  script.textContent = `
    import { grow, draw } from ${JSON.stringify(SOURCE)}
    const reduced = ${reduced ? 'true' : 'false'}
    const stop = new AbortController()
    document.addEventListener('proceduralinitialstop', () => stop.abort(), { once: true })
    const paint = (canvas) => {
      const raw = canvas.dataset.settings
      if (!raw || stop.signal.aborted) return
      const done = reduced
        ? draw(canvas, JSON.parse(raw), { paper: true })
        : grow(canvas, JSON.parse(raw), { paper: true, seconds: 2.4 })
      done.then(() => canvas.closest('.mag-initial')?.classList.add('is-drawn'))
    }
    const seen = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting || stop.signal.aborted) continue
        seen.unobserve(entry.target)
        if (entry.target instanceof HTMLCanvasElement) paint(entry.target)
      }
    })
    stop.signal.addEventListener('abort', () => seen.disconnect())
    for (const canvas of document.querySelectorAll('canvas.gv-initial[data-settings]')) seen.observe(canvas)
  `
  document.head.append(script)
  window.addCleanup(() => {
    document.dispatchEvent(new CustomEvent<void>('proceduralinitialstop'))
    script.remove()
  })
}

document.addEventListener('nav', mountProceduralInitials)
mountProceduralInitials()
