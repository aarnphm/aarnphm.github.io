interface StatusMotion {
  view: HTMLSpanElement
  label: HTMLSpanElement
  prefix: HTMLSpanElement
  readout: HTMLSpanElement
  value: HTMLSpanElement
  outgoing: HTMLSpanElement
  current: string
  next: string
  animations: Animation[]
  timer: number | null
  lastUpdate: number
  generation: number
}

const motions = new WeakMap<HTMLElement, StatusMotion>()
const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches

function stop(motion: StatusMotion) {
  motion.generation++
  if (motion.timer !== null) window.clearTimeout(motion.timer)
  motion.timer = null
  for (const animation of motion.animations) animation.cancel()
  motion.animations = []
  delete motion.view.dataset.flipping
}

function parts(text: string) {
  const separator = text.indexOf(' · ')
  return separator < 0
    ? { prefix: '', value: text }
    : { prefix: text.slice(0, separator + 3), value: text.slice(separator + 3) }
}

function setReadout(motion: StatusMotion, text: string) {
  const next = parts(text)
  const fraction = next.value.match(/^([\d,.]+)\/([\d,.]+)(.*)$/u)
  motion.readout.toggleAttribute('data-numeric', fraction !== null)
  motion.readout.style.minInlineSize = fraction
    ? `${Math.max(fraction[1].length, fraction[2].length) + fraction[2].length + fraction[3].length + 1}ch`
    : ''
  motion.prefix.textContent = next.prefix
  motion.value.textContent = next.value
}

function render(motion: StatusMotion) {
  motion.timer = null
  if (motion.current === motion.next) return
  const before = parts(motion.current)
  const after = parts(motion.next)
  const samePhase = before.prefix !== '' && before.prefix === after.prefix
  const previous = motion.current
  motion.current = motion.next
  motion.lastUpdate = performance.now()
  setReadout(motion, motion.current)
  const outgoingHost = samePhase ? motion.readout : motion.view
  outgoingHost.append(motion.outgoing)
  motion.outgoing.textContent = samePhase ? before.value : previous
  motion.view.dataset.flipping = ''
  const generation = ++motion.generation
  // Keep the phase stationary. Only the changing readout moves, within one clipped line.
  const incoming = samePhase ? motion.value : motion.label
  motion.animations = [
    incoming.animate(
      [
        { opacity: 0, transform: samePhase ? 'translateY(4px)' : 'none' },
        { opacity: 1, transform: 'none' },
      ],
      { duration: 140, easing: 'cubic-bezier(0.23, 1, 0.32, 1)' },
    ),
    motion.outgoing.animate(
      [
        { opacity: 1, transform: 'none' },
        { opacity: 0, transform: samePhase ? 'translateY(-4px)' : 'none' },
      ],
      { duration: 140, easing: 'ease-out', fill: 'forwards' },
    ),
  ]
  void Promise.allSettled(motion.animations.map(animation => animation.finished)).then(() => {
    if (motion.generation !== generation) return
    stop(motion)
    schedule(motion)
  })
}

function schedule(motion: StatusMotion) {
  if (motion.current === motion.next || motion.animations.length || motion.timer !== null) return
  const delay = Math.max(0, 240 - (performance.now() - motion.lastUpdate))
  if (delay) motion.timer = window.setTimeout(() => render(motion), delay)
  else render(motion)
}

/** A stable phase and a bounded, latest-value readout transition. */
export function setStatusText(host: HTMLElement, text: string) {
  let motion = motions.get(host)
  if (!motion) {
    const span = (className: string) => {
      const element = document.createElement('span')
      element.className = className
      return element
    }
    const view = span('status-flap')
    const label = span('status-flap-label')
    const prefix = span('status-flap-prefix')
    const readout = span('status-flap-readout')
    const value = span('status-flap-value')
    const outgoing = span('status-flap-outgoing')
    outgoing.setAttribute('aria-hidden', 'true')
    readout.append(value)
    label.append(prefix, readout)
    view.append(label, outgoing)
    host.replaceChildren(view)
    motion = {
      view,
      label,
      prefix,
      readout,
      value,
      outgoing,
      current: '',
      next: '',
      animations: [],
      timer: null,
      lastUpdate: 0,
      generation: 0,
    }
    motions.set(host, motion)
    const owned = motion
    window.addCleanup(() => {
      stop(owned)
      motions.delete(host)
    })
  }
  host.dataset.statusText = text
  motion.next = text
  if (!text || reducedMotion() || !motion.view.isConnected) {
    stop(motion)
    setReadout(motion, text)
    motion.current = text
  } else schedule(motion)
}
