import type { DetailPayload } from '../activity/data'
import type { TriathlonContext } from '../runtime/context'
import type { TriathlonFormatter } from '../runtime/formatter'
import { buildTimelineDayCard } from '../../../util/triathlon-card'
import {
  beginSitePerformanceSample,
  endSitePerformanceSample,
} from '../../scripts/performance-sample'
import { createDomFactory, el } from '../runtime/dom'
import { TRI_POWER_FILTER_EVENT } from '../runtime/preferences'

const OPEN_PANEL_SELECTOR = '.tri-calc-open, .tri-map-open, .tri-training-open, .tri-analytics-open'
const DROP_TONE_COUNT = 8

export const setup = (root: HTMLElement, context: TriathlonContext): (() => void) | null => {
  const barsEl = root.querySelector<HTMLElement>('.tri-bars')
  const pop = root.querySelector<HTMLElement>('.tri-pop')
  const timeline = root.querySelector<HTMLElement>('.tri-scroll')
  const timelineShell = root.querySelector<HTMLElement>('.tri-scroll-shell')
  const timelinePinnedYear = root.querySelector<HTMLElement>('.tri-axis-pinned')
  const timelineYears = Array.from(root.querySelectorAll<HTMLElement>('.tri-axis-year'))
  const bars = Array.from(root.querySelectorAll<HTMLElement>('.tri-bar'))
  if (!barsEl || !pop || bars.length === 0) return null

  let active: HTMLElement | null = null
  let payload: DetailPayload | null = null
  let timelineFrame = 0
  let hoverFrame = 0
  let pendingClientX: number | null = null
  let geometryDirty = true
  let barsLeft = 0
  let barCenters: number[] = []
  let yearOffsets: number[] = []
  let repositionActive: (() => void) | null = null

  const setData = (element: HTMLElement, key: string, value: string) => {
    if (element.dataset[key] !== value) element.dataset[key] = value
  }
  const setStyle = (element: HTMLElement, key: 'left' | 'maxHeight' | 'top', value: string) => {
    if (element.style[key] !== value) element.style[key] = value
  }

  const updateTimeline = () => {
    const startedAt = beginSitePerformanceSample()
    const barsRect = barsEl.getBoundingClientRect()
    barsLeft = barsRect.left
    if (geometryDirty) {
      barCenters = bars.map(bar => {
        const rect = bar.getBoundingClientRect()
        return rect.left - barsRect.left + rect.width / 2
      })
      yearOffsets = timelineYears.map(year => year.offsetLeft)
      geometryDirty = false
    }
    if (!timeline || !timelineShell) {
      repositionActive?.()
      endSitePerformanceSample('timeline', startedAt)
      return
    }
    const scrollLeft = timeline.scrollLeft
    const maxScroll = Math.max(0, timeline.scrollWidth - timeline.clientWidth)
    const scrollable = maxScroll > 1
    let activeIndex = 0
    for (let index = 0; index < yearOffsets.length; index += 1) {
      if (yearOffsets[index] > scrollLeft + 1) break
      activeIndex = index
    }
    setData(timelineShell, 'scrollable', String(scrollable))
    setData(timelineShell, 'scrollEnd', String(!scrollable || scrollLeft >= maxScroll - 1))
    for (let index = 0; index < timelineYears.length; index += 1)
      setData(timelineYears[index], 'current', String(index === activeIndex))
    const activeYear = timelineYears[activeIndex]
    if (timelinePinnedYear && activeYear) {
      const label = activeYear.dataset.year ?? activeYear.textContent ?? ''
      if (timelinePinnedYear.textContent !== label) timelinePinnedYear.textContent = label
    }
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
  const applyOverflow = (scrollTop: number, contentHeight: number, viewHeight: number) => {
    pop.classList.toggle('tri-pop--top', scrollTop > 4)
    pop.classList.toggle('tri-pop--more', contentHeight - viewHeight - scrollTop > 4)
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
    const base = 560 + (idx % DROP_TONE_COUNT) * 28
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

  const domF = createDomFactory(context.presentation)
  const buildCard = (bar: HTMLElement) =>
    buildTimelineDayCard(domF, bar.dataset.dateIso ?? '', payload)
  const replaceCard = (bar: HTMLElement): void => {
    scroller.replaceChildren(buildCard(bar))
  }

  const place = (bar: HTMLElement) => {
    const gap = 10
    const inset = 8
    const barRect = bar.getBoundingClientRect()
    const timelineRect = timeline?.getBoundingClientRect() ?? barsEl.getBoundingClientRect()
    const popRect = pop.getBoundingClientRect()
    const viewHeight = scroller.clientHeight
    const contentHeight = scroller.scrollHeight
    const scrollTop = scroller.scrollTop
    const frameHeight = Math.max(0, pop.offsetHeight - viewHeight)
    const naturalHeight = contentHeight + frameHeight
    const below = timelineRect.bottom + gap
    const availableBelow = Math.max(0, window.innerHeight - inset - below)
    const availableAbove = Math.max(0, timelineRect.top - gap - inset)
    const placeBelow =
      naturalHeight > availableAbove &&
      (naturalHeight <= availableBelow || availableBelow >= availableAbove)
    const availableHeight = placeBelow ? availableBelow : availableAbove
    const maxContentHeight = Math.max(0, Math.floor(availableHeight - frameHeight))
    const height = Math.min(naturalHeight, maxContentHeight + frameHeight)
    const cx = barRect.left + barRect.width / 2
    const left = Math.max(
      inset,
      Math.min(cx - popRect.width / 2, window.innerWidth - popRect.width - inset),
    )
    const top = placeBelow ? below : Math.max(inset, timelineRect.top - gap - height)
    setStyle(scroller, 'maxHeight', `${maxContentHeight}px`)
    setStyle(pop, 'left', `${left}px`)
    setStyle(pop, 'top', `${top}px`)
    setData(pop, 'placement', placeBelow ? 'below' : 'above')
    applyOverflow(scrollTop, contentHeight, Math.min(contentHeight, maxContentHeight))
  }
  repositionActive = () => {
    if (active) place(active)
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
    }
    place(bar)
    root.classList.add('tri-hovering')
    pop.setAttribute('aria-hidden', 'false')
    endSitePerformanceSample('popover', startedAt)
  }
  const hide = () => {
    if (active) active.classList.remove('tri-bar--active')
    active = null
    root.classList.remove('tri-hovering')
    pop.setAttribute('aria-hidden', 'true')
  }

  const panelOpen = () => root.matches(OPEN_PANEL_SELECTOR)
  const stopHover = () => {
    if (hoverFrame !== 0) window.cancelAnimationFrame(hoverFrame)
    hoverFrame = 0
    pendingClientX = null
    hide()
  }

  const flushHover = () => {
    hoverFrame = 0
    const clientX = pendingClientX
    pendingClientX = null
    if (clientX === null || panelOpen()) return
    const idx = nearest(clientX)
    if (idx >= 0) showFor(idx)
  }
  const onMove = (event: MouseEvent) => {
    if (panelOpen()) return
    pendingClientX = event.clientX
    if (hoverFrame === 0) hoverFrame = window.requestAnimationFrame(flushHover)
  }
  const onBarsLeave = () => {
    if (hoverFrame !== 0) window.cancelAnimationFrame(hoverFrame)
    hoverFrame = 0
    pendingClientX = null
    hide()
  }
  const onBarsClick = (event: MouseEvent) => {
    if (event.defaultPrevented || event.button !== 0) return
    if (event.target instanceof Element && event.target.closest('.tri-bar')) return
    const idx = nearest(event.clientX)
    const bar = bars[idx]
    if (!bar?.hasAttribute('href')) return
    event.preventDefault()
    bar.click()
  }
  const path = root.dataset.detailPath
  let live = true
  if (path)
    void context.resources.detail.load(path).then(result => {
      if (!live || result.status !== 'ready') return
      payload = result.value
      if (active) {
        replaceCard(active)
        place(active)
      }
    })

  const onFocusDay = (event: Event) => {
    if (panelOpen()) return
    const date = (event as CustomEvent<{ date?: string }>).detail?.date
    if (!date) return
    const idx = bars.findIndex(b => b.dataset.dateIso === date)
    if (idx < 0) return
    bars[idx].scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' })
  }

  const onUnit = () => {
    if (!active) return
    replaceCard(active)
    place(active)
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
  window.addEventListener('tri:focus-day', onFocusDay)
  window.addEventListener('tri:unit', onUnit)
  window.addEventListener(TRI_POWER_FILTER_EVENT, onUnit)
  window.addEventListener('resize', onViewportResize)
  window.addEventListener('scroll', onViewportScroll, { passive: true })

  return () => {
    live = false
    window.cancelAnimationFrame(timelineFrame)
    window.cancelAnimationFrame(hoverFrame)
    timeline?.removeEventListener('scroll', scheduleTimelineUpdate)
    timelineResize.disconnect()
    panelObserver.disconnect()
    for (const year of timelineYears) delete year.dataset.current
    barsEl.removeEventListener('mousemove', onMove)
    barsEl.removeEventListener('mouseleave', onBarsLeave)
    barsEl.removeEventListener('click', onBarsClick)
    window.removeEventListener('pointerdown', armAudio)
    window.removeEventListener('keydown', armAudio)
    window.removeEventListener('tri:focus-day', onFocusDay)
    window.removeEventListener('tri:unit', onUnit)
    window.removeEventListener(TRI_POWER_FILTER_EVENT, onUnit)
    window.removeEventListener('resize', onViewportResize)
    window.removeEventListener('scroll', onViewportScroll)
    repositionActive = null
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
