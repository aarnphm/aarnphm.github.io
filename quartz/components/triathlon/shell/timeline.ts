import type { DetailPayload } from '../activity/data'
import type { TriathlonContext } from '../runtime/context'
import type { TriathlonFormatter } from '../runtime/formatter'
import { setActivityExpanded } from '../activity/comparison'
import { buildDayCard } from '../activity/embeds'
import { el } from '../runtime/dom'
import { TRI_POWER_FILTER_EVENT } from '../runtime/preferences'

export const setup = (root: HTMLElement, context: TriathlonContext): (() => void) | null => {
  const barsEl = root.querySelector<HTMLElement>('.tri-bars')
  const pop = root.querySelector<HTMLElement>('.tri-pop')
  const timeline = root.querySelector<HTMLElement>('.tri-scroll')
  const timelineShell = root.querySelector<HTMLElement>('.tri-scroll-shell')
  const timelinePinnedYear = root.querySelector<HTMLElement>('.tri-axis-pinned')
  const timelineYears = Array.from(root.querySelectorAll<HTMLElement>('.tri-axis-year'))
  const bars = Array.from(root.querySelectorAll<HTMLElement>('.tri-bar'))
  if (!barsEl || !pop || bars.length === 0) return null

  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const location = root.dataset.location ?? 'Toronto'
  let active: HTMLElement | null = null
  let activeIdx = -1
  let payload: DetailPayload | null = null
  let pinned = false
  let locked = false
  let hideTimer = 0
  let timelineFrame = 0

  const updateTimeline = () => {
    if (!timeline || !timelineShell) return
    const maxScroll = Math.max(0, timeline.scrollWidth - timeline.clientWidth)
    const scrollable = maxScroll > 1
    timelineShell.dataset.scrollable = String(scrollable)
    timelineShell.dataset.scrollEnd = String(!scrollable || timeline.scrollLeft >= maxScroll - 1)
    let activeYear = timelineYears[0]
    for (const year of timelineYears) {
      if (year.offsetLeft > timeline.scrollLeft + 1) break
      activeYear = year
    }
    for (const year of timelineYears) year.dataset.current = String(year === activeYear)
    if (timelinePinnedYear && activeYear)
      timelinePinnedYear.textContent = activeYear.dataset.year ?? activeYear.textContent
  }
  const scheduleTimelineUpdate = () => {
    if (timelineFrame !== 0) return
    timelineFrame = window.requestAnimationFrame(() => {
      timelineFrame = 0
      updateTimeline()
    })
  }
  const timelineResize = new ResizeObserver(scheduleTimelineUpdate)
  if (timeline) {
    timeline.addEventListener('scroll', scheduleTimelineUpdate, { passive: true })
    timelineResize.observe(timeline)
    const track = timeline.querySelector<HTMLElement>('.tri-track')
    if (track) timelineResize.observe(track)
  }
  scheduleTimelineUpdate()

  const scroller = el('div', 'tri-pop-scroll')
  pop.appendChild(scroller)
  const updateOverflow = () => {
    pop.classList.toggle('tri-pop--top', scroller.scrollTop > 4)
    pop.classList.toggle(
      'tri-pop--more',
      scroller.scrollHeight - scroller.clientHeight - scroller.scrollTop > 4,
    )
  }
  const setLocked = (on: boolean) => {
    locked = on
    barsEl.classList.toggle('tri-bars--locked', on)
  }

  let audio: AudioContext | null = null
  let lastDrop = 0
  try {
    audio = new AudioContext()
    if (navigator.userActivation?.hasBeenActive) void audio.resume()
  } catch {
    audio = null
  }
  const armAudio = () => {
    if (audio && audio.state === 'suspended') void audio.resume()
  }
  const raindrop = (idx: number) => {
    if (!audio || audio.state !== 'running') return
    const t = audio.currentTime
    if (t - lastDrop < 0.05) return
    lastDrop = t
    const base = 560 + (idx % 8) * 28
    const osc = audio.createOscillator()
    const gain = audio.createGain()
    osc.type = 'sine'
    osc.frequency.setValueAtTime(base * 1.8, t)
    osc.frequency.exponentialRampToValueAtTime(base, t + 0.08)
    gain.gain.setValueAtTime(0.0001, t)
    gain.gain.exponentialRampToValueAtTime(0.05, t + 0.004)
    gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.13)
    osc.connect(gain)
    gain.connect(audio.destination)
    osc.start(t)
    osc.stop(t + 0.15)
  }
  window.addEventListener('pointerdown', armAudio)
  window.addEventListener('keydown', armAudio)

  const buildCard = (bar: HTMLElement) =>
    buildDayCard(context.presentation, bar.dataset.dateIso ?? '', payload, {
      location,
      event: bar.dataset.event,
    })
  let cardCleanup: (() => void) | null = null
  const replaceCard = (bar: HTMLElement): HTMLElement => {
    cardCleanup?.()
    const view = buildCard(bar)
    scroller.replaceChildren(view.element)
    cardCleanup = view.mount()
    return view.element
  }

  const place = (cx: number, cy: number) => {
    const r = pop.getBoundingClientRect()
    const gap = 18
    let left = cx + gap
    if (left + r.width > window.innerWidth - 8) left = cx - gap - r.width
    left = Math.max(8, left)
    let top = cy - r.height / 2
    top = Math.max(8, Math.min(top, window.innerHeight - r.height - 8))
    pop.style.left = `${left}px`
    pop.style.top = `${top}px`
  }

  const nearest = (clientX: number): number => {
    let best = Infinity
    let found = -1
    bars.forEach((bar, i) => {
      const r = bar.getBoundingClientRect()
      const d = Math.abs(r.left + r.width / 2 - clientX)
      if (d < best) {
        best = d
        found = i
      }
    })
    return found
  }

  const showFor = (idx: number, cx: number, cy: number) => {
    const bar = bars[idx]
    if (bar !== active) {
      const dir = activeIdx === -1 ? 0 : Math.sign(idx - activeIdx)
      if (active) active.classList.remove('tri-bar--active')
      active = bar
      activeIdx = idx
      bar.classList.add('tri-bar--active')
      raindrop(idx)
      const card = replaceCard(bar)
      scroller.scrollTop = 0
      updateOverflow()
      if (!reduce)
        card.animate(
          [
            { opacity: 0, transform: `translateX(${dir * 12}px)` },
            { opacity: 1, transform: 'none' },
          ],
          { duration: 200, easing: 'cubic-bezier(0.22, 1, 0.36, 1)' },
        )
    }
    place(cx, cy)
    root.classList.add('tri-hovering')
  }

  const setExpanded = (on: boolean) => {
    for (const activity of pop.querySelectorAll<HTMLElement>('.tri-act'))
      setActivityExpanded(activity, on)
    updateOverflow()
  }
  const hide = () => {
    if (active) active.classList.remove('tri-bar--active')
    active = null
    activeIdx = -1
    pinned = false
    setLocked(false)
    root.classList.remove('tri-hovering')
  }

  const onMove = (event: MouseEvent) => {
    if (pinned || locked) return
    window.clearTimeout(hideTimer)
    const idx = nearest(event.clientX)
    if (idx >= 0) showFor(idx, event.clientX, event.clientY)
  }
  const onBarsLeave = () => {
    if (!pinned && !locked) hideTimer = window.setTimeout(hide, 140)
  }
  const onPopEnter = () => {
    window.clearTimeout(hideTimer)
    pinned = true
  }
  const onPopLeave = () => {
    pinned = false
    if (!locked) hideTimer = window.setTimeout(hide, 140)
  }
  const onBarsClick = (event: MouseEvent) => {
    const idx = nearest(event.clientX)
    if (idx < 0) return
    if (locked && bars[idx] === active) {
      setLocked(false)
      setExpanded(false)
    } else {
      showFor(idx, event.clientX, event.clientY)
      setLocked(true)
      setExpanded(true)
    }
  }
  const dismiss = () => {
    if (!locked) return
    setLocked(false)
    setExpanded(false)
    hide()
  }
  const onDocClick = (event: MouseEvent) => {
    const t = event.target as Node
    if (locked && !barsEl.contains(t) && !pop.contains(t)) dismiss()
  }
  const onKey = (event: KeyboardEvent) => {
    if (event.key === 'Escape') dismiss()
  }

  const path = root.dataset.detailPath
  let live = true
  if (path)
    void context.resources.detail.load(path).then(result => {
      if (!live || result.status !== 'ready') return
      payload = result.value
      if (active) {
        replaceCard(active)
        if (locked) setExpanded(true)
        updateOverflow()
      }
    })

  const onFocusDay = (event: Event) => {
    const date = (event as CustomEvent<{ date?: string }>).detail?.date
    if (!date) return
    const idx = bars.findIndex(b => b.dataset.dateIso === date)
    if (idx < 0) return
    bars[idx].scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' })
    const r = bars[idx].getBoundingClientRect()
    showFor(idx, r.left + r.width / 2, r.top + r.height / 2)
    setLocked(true)
    setExpanded(true)
  }

  const onUnit = () => {
    if (!active) return
    replaceCard(active)
    if (locked) setExpanded(true)
    updateOverflow()
  }

  barsEl.addEventListener('mousemove', onMove)
  barsEl.addEventListener('mouseleave', onBarsLeave)
  barsEl.addEventListener('click', onBarsClick)
  pop.addEventListener('mouseenter', onPopEnter)
  pop.addEventListener('mouseleave', onPopLeave)
  scroller.addEventListener('scroll', updateOverflow, { passive: true })
  document.addEventListener('click', onDocClick)
  document.addEventListener('keydown', onKey)
  window.addEventListener('tri:focus-day', onFocusDay)
  window.addEventListener('tri:unit', onUnit)
  window.addEventListener(TRI_POWER_FILTER_EVENT, onUnit)

  return () => {
    live = false
    window.clearTimeout(hideTimer)
    window.cancelAnimationFrame(timelineFrame)
    timeline?.removeEventListener('scroll', scheduleTimelineUpdate)
    timelineResize.disconnect()
    for (const year of timelineYears) delete year.dataset.current
    barsEl.removeEventListener('mousemove', onMove)
    barsEl.removeEventListener('mouseleave', onBarsLeave)
    barsEl.removeEventListener('click', onBarsClick)
    pop.removeEventListener('mouseenter', onPopEnter)
    pop.removeEventListener('mouseleave', onPopLeave)
    scroller.removeEventListener('scroll', updateOverflow)
    document.removeEventListener('click', onDocClick)
    document.removeEventListener('keydown', onKey)
    window.removeEventListener('pointerdown', armAudio)
    window.removeEventListener('keydown', armAudio)
    window.removeEventListener('tri:focus-day', onFocusDay)
    window.removeEventListener('tri:unit', onUnit)
    window.removeEventListener(TRI_POWER_FILTER_EVENT, onUnit)
    cardCleanup?.()
    void audio?.close()
  }
}

export const wireEmbedCopy = (
  formatter: TriathlonFormatter,
  button: HTMLElement | null,
  source: () => string | null,
): (() => void) => {
  if (!button) return () => {}
  let timer = 0
  const reset = () => {
    timer = 0
    button.classList.remove('check')
    button.setAttribute('aria-label', formatter.text('Copy embed link'))
    button.setAttribute('title', formatter.text('Copy embed link'))
  }
  const onCopy = () => {
    const text = source()
    if (!text) return
    const write = navigator.clipboard?.writeText(text)
    if (!write) return
    void write.then(
      () => {
        button.classList.add('check')
        button.setAttribute('aria-label', formatter.text('copied'))
        button.setAttribute('title', formatter.text('copied'))
        window.clearTimeout(timer)
        timer = window.setTimeout(reset, 2000)
      },
      () => {},
    )
  }
  button.addEventListener('click', onCopy)
  return () => {
    button.removeEventListener('click', onCopy)
    window.clearTimeout(timer)
  }
}
