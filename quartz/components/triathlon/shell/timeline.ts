import type { DetailPayload } from '../activity/data'
import type { TriathlonContext } from '../runtime/context'
import type { TriathlonFormatter } from '../runtime/formatter'
import {
  beginSitePerformanceSample,
  endSitePerformanceSample,
} from '../../scripts/performance-sample'
import { setActivityExpanded } from '../activity/comparison'
import { buildDayCard } from '../activity/embeds'
import { el } from '../runtime/dom'
import { TRI_POWER_FILTER_EVENT } from '../runtime/preferences'

const OPEN_PANEL_SELECTOR = '.tri-calc-open, .tri-map-open, .tri-training-open, .tri-analytics-open'

export const setup = (root: HTMLElement, context: TriathlonContext): (() => void) | null => {
  const barsEl = root.querySelector<HTMLElement>('.tri-bars')
  const pop = root.querySelector<HTMLElement>('.tri-pop')
  const timeline = root.querySelector<HTMLElement>('.tri-scroll')
  const timelineShell = root.querySelector<HTMLElement>('.tri-scroll-shell')
  const timelinePinnedYear = root.querySelector<HTMLElement>('.tri-axis-pinned')
  const timelineYears = Array.from(root.querySelectorAll<HTMLElement>('.tri-axis-year'))
  const bars = Array.from(root.querySelectorAll<HTMLElement>('.tri-bar'))
  if (!barsEl || !pop || bars.length === 0) return null

  const location = root.dataset.location ?? 'Toronto'
  let active: HTMLElement | null = null
  let payload: DetailPayload | null = null
  let pinned = false
  let locked = false
  let hideTimer = 0
  let timelineFrame = 0
  let hoverFrame = 0
  let pendingClientX: number | null = null
  let geometryDirty = true
  let barsLeft = 0
  let barCenters: number[] = []
  let repositionActive: (() => void) | null = null

  const updateTimeline = () => {
    const startedAt = beginSitePerformanceSample()
    const barsRect = barsEl.getBoundingClientRect()
    barsLeft = barsRect.left
    if (geometryDirty) {
      barCenters = bars.map(bar => {
        const rect = bar.getBoundingClientRect()
        return rect.left - barsRect.left + rect.width / 2
      })
      geometryDirty = false
    }
    if (!timeline || !timelineShell) {
      repositionActive?.()
      endSitePerformanceSample('timeline', startedAt)
      return
    }
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
    repositionActive?.()
    endSitePerformanceSample('timeline', startedAt)
  }
  const scheduleTimelineUpdate = () => {
    if (timelineFrame !== 0) return
    timelineFrame = window.requestAnimationFrame(() => {
      timelineFrame = 0
      updateTimeline()
    })
  }
  const timelineResize = new ResizeObserver(() => {
    geometryDirty = true
    scheduleTimelineUpdate()
  })
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
    if (on && hoverFrame !== 0) {
      window.cancelAnimationFrame(hoverFrame)
      hoverFrame = 0
      pendingClientX = null
    }
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
  const replaceCard = (bar: HTMLElement): void => {
    cardCleanup?.()
    const view = buildCard(bar)
    scroller.replaceChildren(view.element)
    cardCleanup = view.mount()
  }

  const place = (bar: HTMLElement) => {
    const barRect = bar.getBoundingClientRect()
    const timelineRect = timeline?.getBoundingClientRect() ?? barsEl.getBoundingClientRect()
    const r = pop.getBoundingClientRect()
    const gap = 20
    const inset = 8
    const cx = barRect.left + barRect.width / 2
    const left = Math.max(inset, Math.min(cx - r.width / 2, window.innerWidth - r.width - inset))
    const below = timelineRect.bottom + gap
    const above = timelineRect.top - gap - r.height
    const top = below + r.height <= window.innerHeight - inset ? below : Math.max(inset, above)
    pop.style.left = `${left}px`
    pop.style.top = `${top}px`
  }
  repositionActive = () => {
    if (active && (locked || pinned)) place(active)
  }
  const onViewportResize = () => {
    geometryDirty = true
    scheduleTimelineUpdate()
    repositionActive?.()
  }
  const onViewportScroll = () => {
    scheduleTimelineUpdate()
    repositionActive?.()
  }

  const nearest = (clientX: number): number => {
    if (geometryDirty || barCenters.length !== bars.length) updateTimeline()
    if (barCenters.length === 0) return -1
    const contentX = clientX - barsLeft
    let low = 0
    let high = barCenters.length - 1
    while (low < high) {
      const mid = Math.floor((low + high) / 2)
      if (barCenters[mid] < contentX) low = mid + 1
      else high = mid
    }
    const right = low
    const left = Math.max(0, right - 1)
    return Math.abs(barCenters[left] - contentX) <= Math.abs(barCenters[right] - contentX)
      ? left
      : right
  }

  const showFor = (idx: number) => {
    const startedAt = beginSitePerformanceSample()
    const bar = bars[idx]
    if (bar !== active) {
      if (active) active.classList.remove('tri-bar--active')
      active = bar
      bar.classList.add('tri-bar--active')
      raindrop(idx)
      replaceCard(bar)
      scroller.scrollTop = 0
      updateOverflow()
    }
    place(bar)
    root.classList.add('tri-hovering')
    endSitePerformanceSample('popover', startedAt)
  }

  const setExpanded = (on: boolean) => {
    for (const activity of pop.querySelectorAll<HTMLElement>('.tri-act'))
      setActivityExpanded(activity, on)
    updateOverflow()
  }
  const hide = () => {
    if (active) active.classList.remove('tri-bar--active')
    active = null
    pinned = false
    setLocked(false)
    root.classList.remove('tri-hovering')
  }

  const panelOpen = () => root.matches(OPEN_PANEL_SELECTOR)
  const stopHover = () => {
    window.clearTimeout(hideTimer)
    if (hoverFrame !== 0) window.cancelAnimationFrame(hoverFrame)
    hoverFrame = 0
    pendingClientX = null
    setExpanded(false)
    hide()
  }

  const flushHover = () => {
    hoverFrame = 0
    const clientX = pendingClientX
    pendingClientX = null
    if (clientX === null || pinned || locked || panelOpen()) return
    window.clearTimeout(hideTimer)
    const idx = nearest(clientX)
    if (idx >= 0) showFor(idx)
  }
  const onMove = (event: MouseEvent) => {
    if (pinned || locked || panelOpen()) return
    pendingClientX = event.clientX
    if (hoverFrame === 0) hoverFrame = window.requestAnimationFrame(flushHover)
  }
  const onBarsLeave = () => {
    if (hoverFrame !== 0) window.cancelAnimationFrame(hoverFrame)
    hoverFrame = 0
    pendingClientX = null
    if (!pinned && !locked) hideTimer = window.setTimeout(hide, 140)
  }
  const onPopEnter = () => {
    if (panelOpen()) return
    window.clearTimeout(hideTimer)
    pinned = true
  }
  const onPopLeave = () => {
    pinned = false
    if (!locked) hideTimer = window.setTimeout(hide, 140)
  }
  const onBarsClick = (event: MouseEvent) => {
    if (panelOpen()) return
    if (hoverFrame !== 0) window.cancelAnimationFrame(hoverFrame)
    hoverFrame = 0
    pendingClientX = null
    const idx = nearest(event.clientX)
    if (idx < 0) return
    if (locked && bars[idx] === active) {
      setLocked(false)
      setExpanded(false)
    } else {
      showFor(idx)
      setLocked(true)
      setExpanded(true)
      if (active) place(active)
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
    if (panelOpen()) return
    const date = (event as CustomEvent<{ date?: string }>).detail?.date
    if (!date) return
    const idx = bars.findIndex(b => b.dataset.dateIso === date)
    if (idx < 0) return
    bars[idx].scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' })
    showFor(idx)
    setLocked(true)
    setExpanded(true)
    if (active) place(active)
  }

  const onUnit = () => {
    if (!active) return
    replaceCard(active)
    if (locked) setExpanded(true)
    updateOverflow()
  }

  let wasPanelOpen = panelOpen()
  const panelObserver = new MutationObserver(() => {
    const open = panelOpen()
    if (open === wasPanelOpen) return
    wasPanelOpen = open
    if (open) stopHover()
  })
  panelObserver.observe(root, { attributes: true, attributeFilter: ['class'] })

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
  window.addEventListener('resize', onViewportResize)
  window.addEventListener('scroll', onViewportScroll, { passive: true })

  return () => {
    live = false
    window.clearTimeout(hideTimer)
    window.cancelAnimationFrame(timelineFrame)
    window.cancelAnimationFrame(hoverFrame)
    timeline?.removeEventListener('scroll', scheduleTimelineUpdate)
    timelineResize.disconnect()
    panelObserver.disconnect()
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
    window.removeEventListener('resize', onViewportResize)
    window.removeEventListener('scroll', onViewportScroll)
    repositionActive = null
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
