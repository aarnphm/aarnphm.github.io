import { trailScene } from '../../util/trail-scene'
import {
  holidayLabels,
  subscribeVisitorHoliday,
  type TrailHoliday,
} from '../../util/visitor-holiday'
import { holidayArt, holidaySprite, type HolidaySpriteName } from '../visitor-trail/holiday-sprites'
import { mountTrailClock } from '../visitor-trail/local-clock'
import { trailPerformanceProfile } from '../visitor-trail/performance-probe'
import { mountVisitorRegister } from '../visitor-trail/visitor-register'
import { registerEscapeHandler } from './escape-handler'
import { mountMoonCalendar } from './moon-calendar'
import { setupPerformanceDebug } from './performance-debug'

const preferenceKey = 'garden:trail-open'

document.addEventListener('nav', () => {
  const root = document.querySelector<HTMLElement>('.homepage-trail')
  const toggle = root?.querySelector<HTMLButtonElement>('.homepage-trail-toggle')
  const scene = root?.querySelector<HTMLElement>('.homepage-trail-scene')
  const clock = root?.querySelector<HTMLTimeElement>('.trail-local-time')
  const content = document.querySelector<HTMLElement>('#quartz-root > .page-body-grid')
  if (!root || !toggle || !scene || !clock || !content) return

  const controller = new AbortController()
  const { signal } = controller
  const holidayMarker = toggle.querySelector<HTMLElement>('[data-holiday-marker]')
  const holidayIcons: Record<TrailHoliday, HolidaySpriteName> = {
    thanksgiving: 'turkey',
    'columbus-day': 'trailMap',
    halloween: 'pumpkin',
    christmas: 'star',
    easter: 'rabbit',
  }
  let shownHoliday: TrailHoliday | null = null
  const showHoliday = (holiday: TrailHoliday | null) => {
    if (shownHoliday === holiday) return
    shownHoliday = holiday
    toggle.title = `${scene.hidden ? 'Show' : 'Hide'} walking trail${holiday ? ` · ${holidayLabels[holiday]}` : ''}`
    if (!holidayMarker) return
    holidayMarker.hidden = !holiday
    if (!holiday) {
      delete toggle.dataset.holiday
      holidayMarker.replaceChildren()
      return
    }
    const name = holidayIcons[holiday]
    const rows = holidayArt[name]
    toggle.dataset.holiday = holiday
    holidayMarker.innerHTML = `<svg viewBox="0 0 ${rows[0].length} ${rows.length}" aria-hidden="true" focusable="false">${holidaySprite(name)}</svg>`
  }
  subscribeVisitorHoliday(document, showHoliday, signal)
  scene.addEventListener('trail:holiday', event => showHoliday(event.detail), { signal })
  const frame = document.createElement('iframe')
  frame.title = 'Trail connection'
  frame.hidden = true
  frame.tabIndex = -1
  frame.setAttribute('aria-hidden', 'true')
  frame.src = new URL('shared-trail.html', import.meta.url).href
  let layoutFrame = 0
  let resizeTimer = 0
  // The frame bundle (presence and scene) is the largest homepage transfer. A closed trail
  // connects presence once the page is idle, so the visitor count still fills in beside the
  // toggle without competing with first paint.
  const attachFrame = () => {
    if (!frame.isConnected && !signal.aborted) scene.append(frame)
  }
  const attachWhenIdle = () => {
    if (window.requestIdleCallback) window.requestIdleCallback(attachFrame, { timeout: 4000 })
    else window.setTimeout(attachFrame, 1000)
  }
  document.body.append(root)
  // The scene stands beside body, not inside it. A theme change eases the palette tokens on
  // body, and an eased inherited token restyles every descendant each frame: twelve thousand
  // nodes in the scene. Beside body the scene reads the root tokens and switches in one step,
  // and the compositor crossfades its rasters in step with the page.
  document.documentElement.append(scene)
  const disposeMoon = mountMoonCalendar(root, signal)
  mountVisitorRegister(root, signal)
  mountTrailClock(clock, signal)
  const disposePerformance = setupPerformanceDebug(trailPerformanceProfile(scene))
  if (document.readyState === 'complete') attachWhenIdle()
  else window.addEventListener('load', attachWhenIdle, { signal, once: true })
  root
    .querySelector('#homepage-visitor-panel')
    ?.addEventListener('beforetoggle', attachFrame, { signal, once: true })
  const align = () => {
    if (scene.hidden) return
    const bottom = content.getBoundingClientRect().bottom + window.scrollY
    const height = `${Math.max(window.innerHeight, bottom + 720, Number(scene.dataset.trailWorldHeight) || 0)}px`
    if (scene.style.height !== height) scene.style.height = height
    scene.dispatchEvent(new Event('trail:reflow'))
  }
  const scheduleAlign = () => {
    if (scene.hidden || layoutFrame || resizeTimer) return
    layoutFrame = requestAnimationFrame(() => {
      layoutFrame = 0
      if (!signal.aborted) align()
    })
  }
  const cancelResize = () => {
    window.clearTimeout(resizeTimer)
    resizeTimer = 0
    delete scene.dataset.trailResizing
  }
  const scheduleResize = () => {
    if (scene.hidden) return
    if (scene.dataset.trailWorld === (window.innerWidth < 700 ? 'narrow' : 'wide')) {
      cancelResize()
      scheduleAlign()
      return
    }
    window.clearTimeout(resizeTimer)
    cancelAnimationFrame(layoutFrame)
    layoutFrame = 0
    // Preserve the completed scene at native pixel size; generate only the settled viewport.
    scene.dispatchEvent(new Event('trail:resize-start'))
    scene.dataset.trailResizing = 'true'
    resizeTimer = window.setTimeout(() => {
      if (!signal.aborted) align()
      cancelResize()
    }, 120)
  }
  const resize = new ResizeObserver(scheduleAlign)
  resize.observe(content)
  const landing = content.querySelector('.landing')
  if (landing) resize.observe(landing)
  const mutation = new MutationObserver(scheduleAlign)
  if (landing)
    mutation.observe(landing, {
      subtree: true,
      childList: true,
      characterData: true,
      attributes: true,
      attributeFilter: ['class', 'hidden'],
    })
  window.addEventListener('resize', scheduleResize, { signal })
  window.addEventListener('scroll', scheduleAlign, { signal, passive: true })
  window.addEventListener('load', scheduleAlign, { signal })
  void document.fonts.ready.then(() => {
    if (!signal.aborted) scheduleAlign()
  })
  const setOpen = (open: boolean) => {
    cancelResize()
    sessionStorage.setItem(preferenceKey, String(open))
    scene.hidden = !open
    toggle.setAttribute('aria-expanded', String(open))
    const label = open ? 'Hide walking trail' : 'Show walking trail'
    toggle.title = `${label}${shownHoliday ? ` · ${holidayLabels[shownHoliday]}` : ''}`
    if (open) {
      attachFrame()
      scene.dispatchEvent(new Event('trail:show'))
      align()
    } else {
      scene.dispatchEvent(new Event('trail:hide'))
    }
  }
  const close = () => {
    setOpen(false)
    toggle.focus({ preventScroll: true })
  }
  toggle.addEventListener('click', () => setOpen(scene.hidden !== false), { signal })
  window.addEventListener(
    'message',
    event => {
      if (
        event.source === frame.contentWindow &&
        event.origin === location.origin &&
        event.data === 'garden:trail-close'
      )
        close()
    },
    { signal },
  )
  const disposeEscape = registerEscapeHandler(
    root,
    close,
    () =>
      !scene.hidden &&
      !document.querySelector('#visitor-weather-panel')?.matches(':popover-open') &&
      !root.querySelector('#homepage-moon-panel')?.matches(':popover-open') &&
      !root.querySelector('#homepage-visitor-panel')?.matches(':popover-open') &&
      !trailScene(document)?.querySelector('.trail-visitor-popover:popover-open'),
  )
  setOpen(sessionStorage.getItem(preferenceKey) === 'true')
  window.addCleanup(() => {
    controller.abort()
    disposeMoon()
    disposeEscape()
    disposePerformance()
    resize.disconnect()
    mutation.disconnect()
    cancelAnimationFrame(layoutFrame)
    cancelResize()
    scene.dispatchEvent(new Event('trail:dispose'))
    frame.remove()
    root.remove()
    scene.remove()
  })
})
