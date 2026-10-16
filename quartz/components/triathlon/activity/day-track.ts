import {
  isTriActivityLayout,
  readTriActivityLayout,
  setTriActivityLayout,
  TRI_ACTIVITY_LAYOUT_EVENT,
  type TriActivityLayout,
} from '../../../util/triathlon-activity-layout'

const reducedMotion = (): boolean => window.matchMedia('(prefers-reduced-motion: reduce)').matches

export const mountDayTrack = (card: HTMLElement): (() => void) => {
  const track = card.querySelector<HTMLElement>(':scope > .tri-day-track')
  const pager = card.querySelector<HTMLElement>(':scope > .tri-day-pager')
  const embed = card.closest<HTMLElement>('.tri-day-embed')
  if (!track || !pager || !embed) return () => {}
  const activities = Array.from(track.querySelectorAll<HTMLElement>(':scope > .tri-act'))
  const count = activities.length
  const counter = pager.querySelector<HTMLElement>('.tri-day-pager-count')
  const steps = Array.from(pager.querySelectorAll<HTMLButtonElement>('.tri-day-pager-step'))
  const modes = Array.from(pager.querySelectorAll<HTMLButtonElement>('.tri-day-pager-mode'))
  let frame = 0

  // Pages come from layout geometry, so the stylesheet alone decides how many activities fit.
  const geometry = () => {
    const first = activities[0]
    const gap = Number.parseFloat(window.getComputedStyle(track).columnGap) || 0
    const pitch = (first?.getBoundingClientRect().width ?? 0) + gap
    const perPage = pitch > 0 ? Math.max(1, Math.round((track.clientWidth + gap) / pitch)) : count
    // A chart bleeding past its activity widens scrollWidth, so overflow alone cannot mark paging.
    const paged = embed.dataset.triActivityLayout !== 'all' && count > perPage
    const index = paged && pitch > 0 ? Math.round(track.scrollLeft / pitch) : 0
    return { pitch, paged, perPage, index: Math.min(index, Math.max(0, count - perPage)) }
  }

  const sync = (): void => {
    frame = 0
    const { paged, perPage, index } = geometry()
    pager.dataset.paged = String(paged)
    embed.dataset.triDayIndex = String(index)
    if (counter) {
      const last = Math.min(count, index + perPage)
      const text = paged
        ? `${last - index > 1 ? `${index + 1}–${last}` : index + 1} / ${count}`
        : `${count}`
      if (counter.textContent !== text) counter.textContent = text
    }
    for (const step of steps) {
      const direction = Number(step.dataset.step)
      step.disabled = !paged || (direction < 0 ? index <= 0 : index + perPage >= count)
    }
  }
  const schedule = (): void => {
    if (frame === 0) frame = window.requestAnimationFrame(sync)
  }

  const scrollToIndex = (index: number, smooth: boolean): void => {
    const { pitch, perPage } = geometry()
    const clamped = Math.max(0, Math.min(index, count - perPage))
    track.scrollTo({
      left: clamped * pitch,
      behavior: smooth && !reducedMotion() ? 'smooth' : 'auto',
    })
    schedule()
  }

  const applyLayout = (layout: TriActivityLayout): void => {
    const index = Number(embed.dataset.triDayIndex) || 0
    embed.dataset.triActivityLayout = layout
    for (const mode of modes)
      mode.setAttribute('aria-pressed', String(mode.dataset.layout === layout))
    scrollToIndex(index, false)
  }

  const onClick = (event: MouseEvent): void => {
    if (!(event.target instanceof Element)) return
    const mode = event.target.closest<HTMLButtonElement>('.tri-day-pager-mode')
    if (mode && isTriActivityLayout(mode.dataset.layout)) {
      setTriActivityLayout(mode.dataset.layout)
      return
    }
    const step = event.target.closest<HTMLButtonElement>('.tri-day-pager-step')
    if (!step) return
    const { index, perPage } = geometry()
    scrollToIndex(index + Number(step.dataset.step) * perPage, true)
  }
  const onLayout = (event: Event): void => {
    const layout = (event as CustomEvent<{ layout?: unknown }>).detail?.layout
    if (isTriActivityLayout(layout)) applyLayout(layout)
  }

  pager.addEventListener('click', onClick)
  track.addEventListener('scroll', schedule, { passive: true })
  window.addEventListener(TRI_ACTIVITY_LAYOUT_EVENT, onLayout)
  const resize = new ResizeObserver(schedule)
  resize.observe(track)
  applyLayout(readTriActivityLayout())

  return () => {
    if (frame !== 0) window.cancelAnimationFrame(frame)
    pager.removeEventListener('click', onClick)
    track.removeEventListener('scroll', schedule)
    window.removeEventListener(TRI_ACTIVITY_LAYOUT_EVENT, onLayout)
    resize.disconnect()
  }
}
