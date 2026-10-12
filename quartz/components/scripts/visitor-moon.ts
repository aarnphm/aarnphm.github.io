import { lunarPhase, readLunarEvents, type LunarEvent } from '../../util/lunar-phase'
import { createVisitorMoonSource } from '../../util/visitor-lunar'
import { paintLunarDisc, stopLunarTransition } from './lunar-disc'

export function mountVisitorMoon(root: HTMLElement, signal: AbortSignal) {
  const document = root.ownerDocument
  const view = document.defaultView
  const icon = root.querySelector<SVGPathElement>('[data-lunar-disc]')
  if (!icon || !view) return { refresh() {} }

  const source = createVisitorMoonSource(document, signal)
  const motion = view.matchMedia('(prefers-reduced-motion: reduce)')
  let events: LunarEvent[] = []
  let loadedMonth = ''
  let pending = false
  let retryAfter = 0

  const render = () => {
    if (signal.aborted) return
    const at = Date.now()
    const phase = lunarPhase(at, events)
    root.dataset.moonPhase = String(phase.phase)
    root.dataset.moonSource = phase.source
    root.dataset.lunarState =
      phase.source === 'usno' ? 'ready' : pending ? 'loading' : 'unavailable'
    paintLunarDisc(icon, phase.phase, 7, 0.25, !motion.matches && !document.hidden)
    source.publish({ ...phase, at })
  }

  const load = async () => {
    const now = new Date()
    const month = `${now.getFullYear()}-${now.getMonth()}`
    if (pending || loadedMonth === month || Date.now() < retryAfter || signal.aborted) return
    pending = true
    render()
    const start = new Date(Date.UTC(now.getFullYear(), now.getMonth(), -7))
    const url = new URL('https://aa.usno.navy.mil/api/moon/phases/date')
    url.searchParams.set('date', start.toISOString().slice(0, 10))
    url.searchParams.set('nump', '8')
    const response = await fetch(url, {
      signal: AbortSignal.any([signal, AbortSignal.timeout(6000)]),
      credentials: 'omit',
      referrerPolicy: 'no-referrer',
    }).catch(() => null)
    const received = response?.ok ? readLunarEvents(await response.json().catch(() => null)) : null
    const first = new Date(now.getFullYear(), now.getMonth(), 1).getTime()
    const last = new Date(now.getFullYear(), now.getMonth() + 1, 1).getTime()
    const end = received?.at(-1)
    pending = false
    if (signal.aborted) return
    if (received && end && received[0].at <= first && end.at >= last) {
      events = received
      loadedMonth = month
    } else retryAfter = Date.now() + 15 * 60000
    render()
  }

  const updateMotion = () => {
    root.dataset.moonMotion = document.hidden || motion.matches ? 'paused' : 'running'
    if (document.hidden || motion.matches) stopLunarTransition(icon)
  }
  document.addEventListener('visibilitychange', updateMotion, { signal })
  motion.addEventListener('change', updateMotion, { signal })
  signal.addEventListener(
    'abort',
    () => {
      root.dataset.moonMotion = 'paused'
      stopLunarTransition(icon)
      events = []
    },
    { once: true },
  )
  updateMotion()
  render()
  void load()
  return {
    refresh() {
      render()
      void load()
    },
  }
}
