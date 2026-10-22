import type { TriathlonContext } from '../runtime/context'
import type { TriathlonFormatter } from '../runtime/formatter'
import { SPORT_ICON, type ActivityKind } from '../../../plugins/stores/strava'
import { buildBattery, buildIcon } from '../../../util/triathlon-card/activity-icons'
import { dist, dur, prettyDate } from '../../../util/triathlon-card/metric-text'
import { triText } from '../../../util/triathlon-i18n'
import { readTimelineActivities } from '../../../util/triathlon-timeline'
import {
  beginSitePerformanceSample,
  endSitePerformanceSample,
} from '../../scripts/performance-sample'
import { createDomFactory, el, setText, svg } from '../runtime/dom'
import { TRI_POWER_FILTER_EVENT } from '../runtime/preferences'

const OPEN_PANEL_SELECTOR = '.tri-calc-open, .tri-map-open, .tri-analytics-open, .tri-calendar-open'
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
  let activeIdx = -1
  const activities = new Map(
    bars.map(bar => [
      bar,
      readTimelineActivities(bar.dataset.timelineActivities).toSorted(
        (a, b) => b.distanceKm - a.distanceKm,
      ),
    ]),
  )
  let timelineFrame = 0
  let hoverFrame = 0
  let pendingClientX: number | null = null
  let geometryDirty = true
  let barsLeft = 0
  let timelineTop = 0
  let timelineBottom = 0
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
    const tlRect = timeline ? timeline.getBoundingClientRect() : barsRect
    barsLeft = barsRect.left
    timelineTop = tlRect.top
    timelineBottom = tlRect.bottom
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
      setText(timelinePinnedYear, label)
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
  const maxDayActivities = Math.max(1, ...Array.from(activities.values(), list => list.length))
  const cardEl = el('div', 'tri-timeline-card')
  const dateEl = el('span', 'tri-pop-date')
  const restEl = el('span', 'tri-timeline-row tri-timeline-rest', undefined, { role: 'group' })
  const restValEl = el('span', 'tri-timeline-value', undefined, { 'data-i18n': 'rest' })
  restEl.append(buildBattery(domF), restValEl)
  const listEl = el('ul', 'tri-timeline-list')
  // Slots are reused across sports, so each holds as many paths as the largest icon.
  const iconPathSlots = Math.max(...Object.values(SPORT_ICON).map(paths => paths.length))
  const itemSlots = Array.from({ length: maxDayActivities }, () => {
    const item = el('li', 'tri-timeline-item')
    const entry = el('span', 'tri-timeline-row tri-timeline-activity', undefined, { role: 'group' })
    const icon = svg('svg', { class: 'tri-ico', viewBox: '0 0 24 24', fill: 'none' })
    const paths = Array.from({ length: iconPathSlots }, () => {
      const p = svg('path', { d: '' })
      icon.appendChild(p)
      return p
    })
    const val = el('span', 'tri-timeline-value')
    entry.append(icon, val)
    item.appendChild(entry)
    listEl.appendChild(item)
    return { item, entry, icon, paths, val }
  })
  cardEl.append(dateEl, restEl, listEl)
  scroller.appendChild(cardEl)

  const iconSpecs = new Map<ActivityKind, { cls: string; paths: string[] }>()
  const getIconSpec = (sport: ActivityKind): { cls: string; paths: string[] } => {
    let spec = iconSpecs.get(sport)
    if (!spec) {
      const node = buildIcon(domF, sport)
      spec = {
        cls: node.getAttribute('class') ?? 'tri-ico',
        paths: Array.from(node.querySelectorAll('path')).map(p => p.getAttribute('d') ?? ''),
      }
      iconSpecs.set(sport, spec)
    }
    return spec
  }

  const replaceCard = (bar: HTMLElement): void => {
    setText(dateEl, prettyDate(bar.dataset.dateIso ?? ''))
    const day = activities.get(bar) ?? []
    if (day.length === 0) {
      const label = triText(context.presentation.locale, 'rest')
      restEl.setAttribute('aria-label', label)
      setText(restValEl, label)
      restEl.style.display = ''
      listEl.style.display = 'none'
      for (const slot of itemSlots) {
        slot.item.style.display = 'none'
        setText(slot.val, '')
      }
      return
    }
    restEl.style.display = 'none'
    setText(restValEl, '')
    listEl.style.display = ''
    for (let i = 0; i < itemSlots.length; i++) {
      const slot = itemSlots[i]
      const activity = day[i]
      if (!activity) {
        slot.item.style.display = 'none'
        setText(slot.val, '')
        continue
      }
      slot.item.style.display = ''
      const value =
        activity.distanceKm > 0
          ? dist(context.presentation, activity.distanceKm, activity.sport)
          : dur(activity.movingTimeS)
      const aria = `${triText(context.presentation.locale, activity.sport)} · ${activity.name} · ${value}`
      slot.entry.setAttribute('aria-label', aria)
      slot.entry.setAttribute('title', activity.name)
      const spec = getIconSpec(activity.sport)
      slot.icon.setAttribute('class', spec.cls)
      for (let k = 0; k < slot.paths.length; k++) {
        const d = spec.paths[k]
        if (d) {
          slot.paths[k].setAttribute('d', d)
          slot.paths[k].removeAttribute('display')
        } else {
          slot.paths[k].setAttribute('display', 'none')
        }
      }
      setText(slot.val, value)
    }
  }

  const place = (bar: HTMLElement, idx = activeIdx) => {
    const gap = 10
    const inset = 8
    const cx =
      idx >= 0 && idx < barCenters.length
        ? barsLeft + barCenters[idx]
        : (() => {
            const r = bar.getBoundingClientRect()
            return r.left + r.width / 2
          })()
    if (timelineTop === 0 && timelineBottom === 0) {
      const tlRect = timeline?.getBoundingClientRect() ?? barsEl.getBoundingClientRect()
      timelineTop = tlRect.top
      timelineBottom = tlRect.bottom
    }
    const popWidth = pop.offsetWidth
    const popHeight = pop.offsetHeight
    const viewHeight = scroller.clientHeight
    const contentHeight = scroller.scrollHeight
    const scrollTop = scroller.scrollTop
    const frameHeight = Math.max(0, popHeight - viewHeight)
    const naturalHeight = contentHeight + frameHeight
    const below = timelineBottom + gap
    const availableBelow = Math.max(0, window.innerHeight - inset - below)
    const availableAbove = Math.max(0, timelineTop - gap - inset)
    const placeBelow =
      naturalHeight > availableAbove &&
      (naturalHeight <= availableBelow || availableBelow >= availableAbove)
    const availableHeight = placeBelow ? availableBelow : availableAbove
    const maxContentHeight = Math.max(0, Math.floor(availableHeight - frameHeight))
    const height = Math.min(naturalHeight, maxContentHeight + frameHeight)
    const left = Math.max(inset, Math.min(cx - popWidth / 2, window.innerWidth - popWidth - inset))
    const top = placeBelow ? below : Math.max(inset, timelineTop - gap - height)
    setStyle(scroller, 'maxHeight', `${maxContentHeight}px`)
    setStyle(pop, 'left', `${left}px`)
    setStyle(pop, 'top', `${top}px`)
    setData(pop, 'placement', placeBelow ? 'below' : 'above')
    applyOverflow(scrollTop, contentHeight, Math.min(contentHeight, maxContentHeight))
  }
  repositionActive = () => {
    if (active) place(active, activeIdx)
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
    const bar = bars[idx]
    if (bar === active && root.classList.contains('tri-hovering')) return
    const startedAt = beginSitePerformanceSample()
    if (bar !== active) {
      if (active) active.classList.remove('tri-bar--active')
      active = bar
      activeIdx = idx
      bar.classList.add('tri-bar--active')
      raindrop(idx)
      replaceCard(bar)
      scroller.scrollTop = 0
    }
    root.classList.add('tri-hovering')
    pop.setAttribute('aria-hidden', 'false')
    place(bar, idx)
    endSitePerformanceSample('popover', startedAt)
  }
  const hide = () => {
    if (active) active.classList.remove('tri-bar--active')
    active = null
    activeIdx = -1
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
    place(active, activeIdx)
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
