import { registerEscapeHandler } from './escape-handler'

document.addEventListener('nav', () => {
  const root = document.querySelector<HTMLElement>('.homepage-trail')
  const toggle = root?.querySelector<HTMLButtonElement>('.homepage-trail-toggle')
  const scene = root?.querySelector<HTMLElement>('.homepage-trail-scene')
  const content = document.querySelector<HTMLElement>('#quartz-root > .page-body-grid')
  if (!root || !toggle || !scene || !content) return

  const controller = new AbortController()
  const { signal } = controller
  const frame = document.createElement('iframe')
  frame.title = 'Trail connection'
  frame.hidden = true
  frame.tabIndex = -1
  frame.setAttribute('aria-hidden', 'true')
  frame.src = new URL('shared-trail.html', import.meta.url).href
  let layoutFrame = 0
  // Prepare the local renderer; it connects only after the trail is shown.
  document.body.append(root)
  scene.append(frame)
  const align = () => {
    if (scene.hidden) return
    const bottom = content.getBoundingClientRect().bottom + window.scrollY
    scene.style.height = `${Math.max(window.innerHeight, bottom + 380)}px`
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
      !scene.hidden && !document.querySelector('#visitor-weather-panel')?.matches(':popover-open'),
  )
  window.addCleanup(() => {
    controller.abort()
    disposeEscape()
    resize.disconnect()
    mutation.disconnect()
    cancelAnimationFrame(layoutFrame)
    scene.dispatchEvent(new Event('trail:dispose'))
    frame.remove()
    root.remove()
  })
})
