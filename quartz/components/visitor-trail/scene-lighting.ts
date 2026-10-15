import { subscribeVisitorDaylight, type VisitorDaylight } from '../../util/visitor-daylight'
import { daylightTint, type DaylightTint } from './daylight-palette'

export function mountSceneLighting(scene: HTMLElement, signal: AbortSignal) {
  const document = scene.ownerDocument
  const view = document.defaultView
  const svg = scene.querySelector<SVGSVGElement>('.trail-canvas')
  const defs = svg?.querySelector('defs')
  if (!view || !svg || !defs) return
  const namespace = 'http://www.w3.org/2000/svg'
  const filter = document.createElementNS(namespace, 'filter')
  filter.id = `trail-daylight-${crypto.randomUUID()}`
  filter.setAttribute('color-interpolation-filters', 'sRGB')
  const matrix = document.createElementNS(namespace, 'feColorMatrix')
  matrix.setAttribute('type', 'matrix')
  filter.append(matrix)
  defs.append(filter)
  const reduced = view.matchMedia('(prefers-reduced-motion: reduce)')
  let solar: VisitorDaylight | null = null
  let from: DaylightTint = [1, 1, 1]
  let to: DaylightTint = [1, 1, 1]
  let started = 0
  let frame = 0
  const tintAt = (time: number): DaylightTint => {
    const progress = reduced.matches ? 1 : Math.min(1, Math.max(0, (time - started) / 1200))
    const share = progress * progress * (3 - 2 * progress)
    return [
      from[0] + (to[0] - from[0]) * share,
      from[1] + (to[1] - from[1]) * share,
      from[2] + (to[2] - from[2]) * share,
    ]
  }
  const paint = (time: number) => {
    frame = 0
    if (scene.dataset.trailRenderer === 'webgl') {
      svg.removeAttribute('filter')
      return
    }
    const [r, g, b] = tintAt(time)
    if (r === 1 && g === 1 && b === 1) svg.removeAttribute('filter')
    else {
      matrix.setAttribute('values', `${r} 0 0 0 0 0 ${g} 0 0 0 0 0 ${b} 0 0 0 0 0 1 0`)
      svg.setAttribute('filter', `url(#${filter.id})`)
    }
    if (!reduced.matches && time - started < 1200 && !document.hidden)
      frame = view.requestAnimationFrame(paint)
  }
  const update = () => {
    const next = daylightTint(solar, document.documentElement.dataset.themeMode === 'daylight')
    if (next.every((value, index) => value === to[index])) return
    const now = view.performance.now()
    from = tintAt(now)
    to = next
    started = now
    scene.dataset.solarTint = next.map(value => value.toFixed(4)).join(' ')
    view.cancelAnimationFrame(frame)
    paint(now)
  }
  const refresh = () => {
    view.cancelAnimationFrame(frame)
    paint(view.performance.now())
  }
  document.addEventListener('themechange', update, { signal })
  const renderer = new MutationObserver(refresh)
  renderer.observe(scene, { attributes: true, attributeFilter: ['data-trail-renderer'] })
  document.addEventListener('visibilitychange', refresh, { signal })
  reduced.addEventListener('change', refresh, { signal })
  subscribeVisitorDaylight(
    document,
    current => {
      solar = current
      update()
    },
    signal,
  )
  signal.addEventListener(
    'abort',
    () => {
      view.cancelAnimationFrame(frame)
      renderer.disconnect()
      svg.removeAttribute('filter')
      filter.remove()
    },
    { once: true },
  )
}
