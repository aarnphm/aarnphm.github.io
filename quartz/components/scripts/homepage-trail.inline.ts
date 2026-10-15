import { trailScene, trailSceneHeight } from '../../util/trail-scene'
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
import { mountTrailGateNavigation } from './trail-gate-navigation'

const CHORD_WINDOW_MS = 1000
const TRAIL_PREFERENCE_KEY = 'garden:trail-open'

function shouldIgnoreTrailShortcut(event: KeyboardEvent): boolean {
  if (
    document.querySelector(
      'dialog[open], #palette-container.active, #shortcut-container.active, .search-container.active, .link-hint-marker',
    )
  )
    return true
  const origin = event.composedPath()[0] ?? event.target
  const el =
    origin instanceof Element
      ? origin
      : document.activeElement instanceof Element
        ? document.activeElement
        : null
  if (!el) return false
  return (
    el.matches('input, textarea, select') ||
    (el as HTMLElement).isContentEditable ||
    el.closest(
      '.search .search-container, .stream-search-container, [role="listbox"], .cm-editor',
    ) !== null
  )
}

document.addEventListener('nav', () => {
  const root = document.querySelector<HTMLElement>('.homepage-trail')
  const toggle = root?.querySelector<HTMLButtonElement>('.homepage-trail-toggle')
  const scene = document.querySelector<HTMLElement>('.homepage-trail-scene')
  const frame = scene?.querySelector<HTMLIFrameElement>('iframe[data-trail-frame]')
  const clock = root?.querySelector<HTMLTimeElement>('.trail-local-time')
  const content = document.querySelector<HTMLElement>('#quartz-root > .page-body-grid')
  if (!root || !toggle || !scene || !frame || !clock || !content) return

  const controller = new AbortController()
  const { signal } = controller
  mountTrailGateNavigation(scene, signal)
  let isOpen = false
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
    toggle.title = `${isOpen ? 'Hide' : 'Show'} walking trail${holiday ? ` · ${holidayLabels[holiday]}` : ''}`
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
  let layoutFrame = 0
  let resizeTimer = 0
  let openFrame = 0
  let openTimer = 0
  let motionStarted = false
  let layoutSignature = ''
  // Keep the scene outside body so page selectors do not restyle its thousands of nodes.
  if (scene.parentElement !== document.documentElement) {
    if (typeof document.documentElement.moveBefore === 'function')
      document.documentElement.moveBefore(scene, null)
    else document.documentElement.append(scene)
  }
  document.body.append(root)
  const disposeMoon = mountMoonCalendar(root, signal)
  mountVisitorRegister(root, signal)
  mountTrailClock(clock, signal)
  const disposePerformance = setupPerformanceDebug(trailPerformanceProfile(scene))
  const align = () => {
    const bottom = content.getBoundingClientRect().bottom + window.scrollY
    const height = `${trailSceneHeight(scene, content)}px`
    const signature = `${window.innerWidth}|${height}|${bottom}|${window.scrollY}`
    if (signature === layoutSignature) return
    layoutSignature = signature
    if (scene.style.height !== height) scene.style.height = height
    scene.dispatchEvent(new Event('trail:reflow'))
  }
  const scheduleAlign = () => {
    if (layoutFrame || resizeTimer) return
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
    if (scene.dataset.trailWorld === (window.innerWidth < 700 ? 'narrow' : 'wide')) {
      cancelResize()
      scheduleAlign()
      return
    }
    window.clearTimeout(resizeTimer)
    cancelAnimationFrame(layoutFrame)
    layoutFrame = 0
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
  scheduleAlign()
  const resumeScene = () => {
    const drawing = scene.shadowRoot?.querySelector<HTMLElement>('.trail-scene')
    if (
      signal.aborted ||
      !isOpen ||
      scene.hidden ||
      motionStarted ||
      openFrame ||
      openTimer ||
      drawing?.dataset.trailOpening !== 'complete' ||
      drawing.dataset.trailReveal !== undefined
    )
      return
    // Give the complete reveal one paint before walkers read their routes and start moving.
    openFrame = window.requestAnimationFrame(() => {
      openFrame = 0
      openTimer = window.setTimeout(() => {
        openTimer = 0
        if (!signal.aborted && isOpen && !scene.hidden) {
          motionStarted = true
          scene.dispatchEvent(new Event('trail:show'))
        }
      }, 0)
    })
  }
  scene.addEventListener('trail:prepared', resumeScene, { signal })
  scene.addEventListener('trail:reveal-end', resumeScene, { signal })
  const finishClose = () => {
    if (isOpen) return
    scene.hidden = true
    scene.dispatchEvent(new Event('trail:hide'))
  }
  scene.addEventListener('trail:closed', finishClose, { signal })
  const setOpen = (open: boolean, remember = true) => {
    if (!open) {
      const shadowActive = scene.shadowRoot?.activeElement
      if (shadowActive instanceof HTMLElement || shadowActive instanceof SVGElement)
        shadowActive.blur()
      if (document.activeElement === toggle) toggle.blur()
    }
    isOpen = open
    if (remember) {
      try {
        localStorage.setItem(TRAIL_PREFERENCE_KEY, String(open))
      } catch {
        // Storage restrictions must not disable the trail control.
      }
    }
    cancelResize()
    window.cancelAnimationFrame(openFrame)
    window.clearTimeout(openTimer)
    openFrame = openTimer = 0
    motionStarted = false
    if (open) align()
    if (open) scene.hidden = false
    for (const control of scene.shadowRoot?.querySelector('.trail-scene')?.children ?? [])
      if (control instanceof HTMLElement) control.inert = !open
    scene.setAttribute('aria-hidden', String(!open))
    toggle.setAttribute('aria-expanded', String(open))
    const label = open ? 'Hide walking trail' : 'Show walking trail'
    toggle.title = `${label}${shownHoliday ? ` · ${holidayLabels[shownHoliday]}` : ''}`
    if (open) {
      scene.dispatchEvent(new Event('trail:open'))
      resumeScene()
    } else {
      // A ready surface keeps its pixels visible until the reverse reveal reaches the clock.
      if (scene.dispatchEvent(new Event('trail:close', { cancelable: true }))) finishClose()
    }
  }
  const close = () => setOpen(false)
  toggle.addEventListener(
    'click',
    event => {
      if (event.detail > 0) toggle.blur()
      setOpen(!isOpen)
    },
    { signal },
  )
  let pendingG = -Infinity
  document.addEventListener(
    'keydown',
    event => {
      if (
        event.defaultPrevented ||
        event.isComposing ||
        event.ctrlKey ||
        event.metaKey ||
        event.altKey ||
        shouldIgnoreTrailShortcut(event)
      ) {
        pendingG = -Infinity
        return
      }
      if (event.key === 't' && !event.repeat && event.timeStamp - pendingG < CHORD_WINDOW_MS) {
        event.preventDefault()
        pendingG = -Infinity
        if (document.activeElement === toggle) toggle.blur()
        setOpen(!isOpen)
        return
      }
      pendingG = event.key === 'g' && !event.repeat ? event.timeStamp : -Infinity
    },
    { signal },
  )
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
  const canClose = () =>
    isOpen &&
    !document.querySelector('#visitor-weather-panel')?.matches(':popover-open') &&
    !root.querySelector('#homepage-moon-panel')?.matches(':popover-open') &&
    !root.querySelector('#homepage-visitor-panel')?.matches(':popover-open') &&
    !trailScene(document)?.querySelector(':popover-open')
  scene.addEventListener(
    'keydown',
    event => {
      if (event.key !== 'Escape' || event.defaultPrevented || !canClose()) return
      event.preventDefault()
      event.stopPropagation()
      close()
    },
    { signal },
  )
  const disposeEscape = registerEscapeHandler(root, close, canClose)
  let initiallyOpen = true
  try {
    initiallyOpen = localStorage.getItem(TRAIL_PREFERENCE_KEY) !== 'false'
  } catch {
    // Use the default when the browser blocks saved preferences.
  }
  setOpen(initiallyOpen, false)
  window.addCleanup(() => {
    controller.abort()
    disposeMoon()
    disposeEscape()
    disposePerformance()
    resize.disconnect()
    mutation.disconnect()
    cancelAnimationFrame(layoutFrame)
    cancelAnimationFrame(openFrame)
    window.clearTimeout(openTimer)
    cancelResize()
    scene.dispatchEvent(new Event('trail:dispose'))
    frame.remove()
    root.remove()
    scene.remove()
  })
})
