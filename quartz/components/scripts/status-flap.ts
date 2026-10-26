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
let reducedQuery: MediaQueryList | undefined
const reducedMotion = () =>
  (reducedQuery ??= window.matchMedia('(prefers-reduced-motion: reduce)')).matches

// A tick inside one phase is a digit swap at meter cadence; a new phase rolls in and holds long enough to read.
const TICK_MS = 100
const PHASE_HOLD_MS = 320
const ROLL = '0.35em'

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

const isTick = (before: string, after: string) => {
  const prefix = parts(before).prefix
  return prefix !== '' && prefix === parts(after).prefix
}

function setReadout(motion: StatusMotion, text: string) {
  const next = parts(text)
  const fraction = next.value.match(/^([\d,.]+)\/([\d,.]+)(.*)$/u)
  // Reserve the width of the total on both sides of the slash so the slash never walks.
  const width = fraction ? `${2 * fraction[2].length + fraction[3].length + 1}ch` : ''
  motion.readout.toggleAttribute('data-numeric', fraction !== null)
  if (motion.readout.style.minInlineSize !== width) motion.readout.style.minInlineSize = width
  if (motion.prefix.textContent !== next.prefix) motion.prefix.textContent = next.prefix
  motion.value.textContent = next.value
}

function render(motion: StatusMotion) {
  motion.timer = null
  if (motion.current === motion.next) return
  const previous = motion.current
  motion.current = motion.next
  motion.lastUpdate = performance.now()
  setReadout(motion, motion.current)
  // Same phase: the tabular readout swaps digits in place, nothing moves.
  if (isTick(previous, motion.current) || !previous) return
  motion.outgoing.textContent = previous
  motion.view.dataset.flipping = ''
  const generation = ++motion.generation
  // The old line leaves first and is mostly gone before the new one rises, so the two never read as one smear.
  motion.animations = [
    motion.outgoing.animate(
      [
        { opacity: 1, transform: 'none' },
        { opacity: 0, transform: `translateY(-${ROLL})` },
      ],
      { duration: 120, easing: 'cubic-bezier(0.4, 0, 1, 1)', fill: 'forwards' },
    ),
    motion.label.animate(
      [
        { opacity: 0, transform: `translateY(${ROLL})` },
        { opacity: 1, transform: 'none' },
      ],
      { duration: 200, delay: 60, easing: 'cubic-bezier(0.23, 1, 0.32, 1)', fill: 'backwards' },
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
  const gap = !motion.current ? 0 : isTick(motion.current, motion.next) ? TICK_MS : PHASE_HOLD_MS
  const delay = Math.max(0, gap - (performance.now() - motion.lastUpdate))
  if (delay) motion.timer = window.setTimeout(() => render(motion), delay)
  else render(motion)
}

/** A stable phase, an in-place tabular readout, and a short roll only when the phase changes. */
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
    motion.lastUpdate = performance.now()
  } else schedule(motion)
}
