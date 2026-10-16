import type { ActivityKind } from '../../../plugins/stores/strava'
import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import {
  isTriActivityLayout,
  readTriActivityLayout,
  setTriActivityLayout,
  TRI_ACTIVITY_LAYOUT_EVENT,
  type TriActivityLayout,
} from '../../../util/triathlon-activity-layout'
import { buildDayPageTabs, dayTrackPages } from '../../../util/triathlon-card'
import { createDomFactory } from '../runtime/dom'
import { nextMapMetricShortcutIndex } from '../shell/command-palette'

const reducedMotion = (): boolean => window.matchMedia('(prefers-reduced-motion: reduce)').matches

export const mountDayTrack = (
  card: HTMLElement,
  presentation: TriathlonPresentation,
): (() => void) => {
  const track = card.querySelector<HTMLElement>(':scope > .tri-day-track')
  const pager = card.querySelector<HTMLElement>(':scope > .tri-day-pager')
  const tablist = pager?.querySelector<HTMLElement>('.tri-day-pages')
  const embed = card.closest<HTMLElement>('.tri-day-embed')
  if (!track || !pager || !tablist || !embed) return () => {}
  const activities = Array.from(track.querySelectorAll<HTMLElement>(':scope > .tri-act'))
  const sports = activities.map(activity => (activity.dataset.sport ?? 'other') as ActivityKind)
  const count = activities.length
  const modes = Array.from(pager.querySelectorAll<HTMLButtonElement>('.tri-day-mode'))
  const factory = createDomFactory(presentation)
  let tabs = Array.from(tablist.querySelectorAll<HTMLButtonElement>('.tri-day-page-tab'))
  let tabsPerPage = 2
  let frame = 0
  // A tab click scrolls smoothly; the track passes earlier pages on the way, so the clicked page
  // stays selected until the scroll lands.
  let pendingPage: number | null = null

  // Pages come from layout geometry, so the stylesheet alone decides how many activities fit.
  const geometry = () => {
    const gap = Number.parseFloat(window.getComputedStyle(track).columnGap) || 0
    const pitch = (activities[0]?.getBoundingClientRect().width ?? 0) + gap
    const perPage = pitch > 0 ? Math.max(1, Math.round((track.clientWidth + gap) / pitch)) : count
    // A chart bleeding past its activity widens scrollWidth, so overflow alone cannot mark paging.
    const paged = embed.dataset.triActivityLayout !== 'all' && count > perPage
    const index = paged && pitch > 0 ? Math.round(track.scrollLeft / pitch) : 0
    return { pitch, paged, perPage, index: Math.min(index, Math.max(0, count - perPage)) }
  }
  const pageFor = (index: number, perPage: number): number =>
    index >= count - perPage
      ? dayTrackPages(count, perPage).length - 1
      : Math.floor(index / perPage)

  const select = (page: number): void => {
    for (const [i, tab] of tabs.entries()) {
      tab.setAttribute('aria-selected', String(i === page))
      tab.tabIndex = i === page ? 0 : -1
    }
  }

  const sync = (): void => {
    frame = 0
    const { paged, perPage, index } = geometry()
    pager.dataset.paged = String(paged)
    embed.dataset.triDayIndex = String(index)
    if (!paged) return
    if (pendingPage !== null && pageFor(index, perPage) === pendingPage) pendingPage = null
    const page = pendingPage ?? pageFor(index, perPage)
    if (perPage !== tabsPerPage || tabs.length === 0) {
      // Labels animate open on the selected tab; a rebuilt list starts at its final widths.
      tablist.dataset.motion = 'instant'
      tabs = buildDayPageTabs(factory, sports, perPage, page) as HTMLButtonElement[]
      tablist.replaceChildren(...tabs)
      tabsPerPage = perPage
      tablist.getBoundingClientRect()
      delete tablist.dataset.motion
    }
    select(page)
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
  const showPage = (page: number, smooth: boolean): void => {
    const target = dayTrackPages(count, geometry().perPage)[page]
    if (!target) return
    pendingPage = smooth ? page : null
    select(page)
    scrollToIndex(target.start, smooth)
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
    const mode = event.target.closest<HTMLButtonElement>('.tri-day-mode')
    if (mode && isTriActivityLayout(mode.dataset.layout)) {
      setTriActivityLayout(mode.dataset.layout)
      return
    }
    const tab = event.target.closest<HTMLButtonElement>('.tri-day-page-tab')
    if (tab) showPage(Number(tab.dataset.page), true)
  }
  const onKeyDown = (event: KeyboardEvent): void => {
    if (event.ctrlKey || event.metaKey || event.altKey || event.isComposing || event.repeat) return
    if (!(event.target instanceof Element) || !event.target.closest('.tri-day-page-tab')) return
    const active = Math.max(
      0,
      tabs.findIndex(tab => tab.getAttribute('aria-selected') === 'true'),
    )
    const next =
      event.key === 'ArrowLeft'
        ? (active - 1 + tabs.length) % tabs.length
        : event.key === 'ArrowRight'
          ? (active + 1) % tabs.length
          : event.key === 'Home'
            ? 0
            : event.key === 'End'
              ? tabs.length - 1
              : nextMapMetricShortcutIndex(
                  tabs.map(tab => tab.dataset.shortcut),
                  active,
                  event.key,
                )
    if (next < 0) return
    event.preventDefault()
    event.stopPropagation()
    showPage(next, false)
    tabs[next]?.focus({ preventScroll: true })
  }
  const onScrollEnd = (): void => {
    pendingPage = null
    schedule()
  }
  const onLayout = (event: Event): void => {
    const layout = (event as CustomEvent<{ layout?: unknown }>).detail?.layout
    if (isTriActivityLayout(layout)) applyLayout(layout)
  }

  pager.addEventListener('click', onClick)
  tablist.addEventListener('keydown', onKeyDown)
  track.addEventListener('scroll', schedule, { passive: true })
  track.addEventListener('scrollend', onScrollEnd)
  window.addEventListener(TRI_ACTIVITY_LAYOUT_EVENT, onLayout)
  const resize = new ResizeObserver(schedule)
  resize.observe(track)
  applyLayout(readTriActivityLayout())

  return () => {
    if (frame !== 0) window.cancelAnimationFrame(frame)
    pager.removeEventListener('click', onClick)
    tablist.removeEventListener('keydown', onKeyDown)
    track.removeEventListener('scroll', schedule)
    track.removeEventListener('scrollend', onScrollEnd)
    window.removeEventListener(TRI_ACTIVITY_LAYOUT_EVENT, onLayout)
    resize.disconnect()
  }
}
