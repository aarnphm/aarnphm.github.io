import { mountTrailClock } from '../visitor-trail/local-clock'
import { registerEscapeHandler } from './escape-handler'
import { mountMoonCalendar } from './moon-calendar'

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
  const frame = document.createElement('iframe')
  frame.title = 'Trail connection'
  frame.hidden = true
  frame.tabIndex = -1
  frame.setAttribute('aria-hidden', 'true')
  frame.src = new URL('shared-trail.html', import.meta.url).href
  let layoutFrame = 0
  // Presence connects ahead of the scene so the visitor count is ready beside the toggle.
  document.body.append(root)
  const disposeMoon = mountMoonCalendar(root, signal)
  mountTrailClock(clock, signal)
  scene.append(frame)
  const align = () => {
    if (scene.hidden) return
    const bottom = content.getBoundingClientRect().bottom + window.scrollY
    scene.style.height = `${Math.max(window.innerHeight, bottom + 720)}px`
    scene.dispatchEvent(new Event('trail:reflow'))
  }
  const scheduleAlign = () => {
    if (scene.hidden || layoutFrame) return
    layoutFrame = requestAnimationFrame(() => {
      layoutFrame = 0
      if (!signal.aborted) align()
    })
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
  window.addEventListener('resize', scheduleAlign, { signal })
  window.addEventListener('scroll', scheduleAlign, { signal, passive: true })
  window.addEventListener('load', scheduleAlign, { signal })
  void document.fonts.ready.then(() => {
    if (!signal.aborted) scheduleAlign()
  })
  const setOpen = (open: boolean) => {
    sessionStorage.setItem(preferenceKey, String(open))
    scene.hidden = !open
    toggle.setAttribute('aria-expanded', String(open))
    const label = open ? 'Hide walking trail' : 'Show walking trail'
    toggle.title = label
    if (open) {
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
      !root.querySelector('#homepage-moon-panel')?.matches(':popover-open'),
  )
  setOpen(sessionStorage.getItem(preferenceKey) === 'true')
  window.addCleanup(() => {
    controller.abort()
    disposeMoon()
    disposeEscape()
    resize.disconnect()
    mutation.disconnect()
    cancelAnimationFrame(layoutFrame)
    scene.dispatchEvent(new Event('trail:dispose'))
    frame.remove()
    root.remove()
  })
})
