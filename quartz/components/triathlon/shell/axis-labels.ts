export const setupAxisLabels = (root: HTMLElement): (() => void) | null => {
  const axis = root.querySelector<HTMLElement>('.tri-axis')
  if (!axis) return null
  const labels = [...axis.querySelectorAll<HTMLElement>('.tri-axis-year')]
  if (labels.length === 0) return null
  const visible = new Map<Element, boolean>()
  let frame: number | null = null
  const apply = () => {
    frame = null
    const viewportBottom = window.innerHeight - 1
    const clippedByRect = labels.some(label => {
      const r = label.getBoundingClientRect()
      return r.top < 0 || r.bottom > viewportBottom
    })
    root.classList.toggle(
      'tri-axis-labels-hidden',
      clippedByRect || [...visible.values()].some(ok => !ok),
    )
  }
  const schedule = () => {
    if (frame == null) frame = window.requestAnimationFrame(apply)
  }
  const observer = new IntersectionObserver(
    entries => {
      for (const entry of entries)
        visible.set(entry.target, entry.isIntersecting && entry.intersectionRatio >= 0.98)
      schedule()
    },
    { threshold: [0, 0.98, 1] },
  )
  for (const label of labels) {
    visible.set(label, true)
    observer.observe(label)
  }
  const resize = new ResizeObserver(schedule)
  resize.observe(root)
  resize.observe(axis)
  window.addEventListener('resize', schedule, { passive: true })
  schedule()
  return () => {
    observer.disconnect()
    resize.disconnect()
    window.removeEventListener('resize', schedule)
    if (frame != null) window.cancelAnimationFrame(frame)
    root.classList.remove('tri-axis-labels-hidden')
  }
}
