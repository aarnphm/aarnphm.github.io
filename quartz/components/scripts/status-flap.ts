interface Flap {
  view: HTMLSpanElement
  label: HTMLSpanElement
  oldTop: HTMLSpanElement
  oldBottom: HTMLSpanElement
  newBottom: HTMLSpanElement
  current: string
  next: string
  animations: Animation[]
  generation: number
}

const flaps = new WeakMap<HTMLElement, Flap>()
const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches

function stop(flap: Flap) {
  flap.generation++
  for (const animation of flap.animations) animation.cancel()
  flap.animations = []
  delete flap.view.dataset.flipping
}

function flip(flap: Flap) {
  if (flap.next === flap.current) return
  const label = flap.next
  flap.label.textContent = label
  if (!label || reducedMotion() || !flap.view.isConnected) {
    stop(flap)
    flap.current = label
    return
  }
  const generation = ++flap.generation
  flap.oldTop.textContent = flap.current
  flap.oldBottom.textContent = flap.current
  flap.newBottom.textContent = label
  flap.view.dataset.flipping = ''
  flap.animations = [
    flap.oldTop.animate([{ transform: 'rotateX(0deg)' }, { transform: 'rotateX(-90deg)' }], {
      duration: 90,
      easing: 'ease-in',
      fill: 'forwards',
    }),
    flap.newBottom.animate([{ transform: 'rotateX(90deg)' }, { transform: 'rotateX(0deg)' }], {
      delay: 90,
      duration: 90,
      easing: 'ease-out',
      fill: 'both',
    }),
  ]
  void Promise.allSettled(flap.animations.map(animation => animation.finished)).then(() => {
    if (flap.generation !== generation) return
    stop(flap)
    flap.current = label
    flip(flap)
  })
}

/** One accessible label, with visual halves that coalesce to the latest pending update. */
export function setStatusText(host: HTMLElement, value: string) {
  let flap = flaps.get(host)
  if (!flap) {
    const view = document.createElement('span')
    view.className = 'status-flap'
    const label = document.createElement('span')
    label.className = 'status-flap-label'
    const layer = (name: string) => {
      const span = document.createElement('span')
      span.className = `status-flap-${name}`
      span.setAttribute('aria-hidden', 'true')
      return span
    }
    const oldTop = layer('old-top')
    const oldBottom = layer('old-bottom')
    const newBottom = layer('new-bottom')
    view.append(label, oldBottom, oldTop, newBottom)
    host.replaceChildren(view)
    flap = {
      view,
      label,
      oldTop,
      oldBottom,
      newBottom,
      current: '',
      next: '',
      animations: [],
      generation: 0,
    }
    flaps.set(host, flap)
    const owned = flap
    window.addCleanup(() => {
      stop(owned)
      flaps.delete(host)
    })
  }
  host.dataset.statusText = value
  flap.next = value
  // Empty status means completion or cancellation; no old label may linger.
  if (!value || reducedMotion()) {
    stop(flap)
    flap.label.textContent = value
    flap.current = value
  } else if (!flap.animations.length) flip(flap)
}
