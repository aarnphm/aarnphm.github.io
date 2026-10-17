import { sceneNow, subscribeSceneClock } from '../../util/scene-clock'
import {
  cloneGrade,
  equalGrade,
  formatGrade,
  gradeColorMatrixValues,
  IDENTITY_GRADE,
  interpolateGrade,
  isIdentityGrade,
  sceneGrade,
  type Grade,
} from '../../util/scene-grade'
import { readTrailSeason, subscribeTrailSeason } from '../../util/trail-season'
import { subscribeVisitorDaylight } from '../../util/visitor-daylight'
import {
  readStoredTrailWeather,
  subscribeVisitorTrailWeather,
  trailSite,
  trailSky,
  type TrailSeason,
  type TrailWeather,
} from '../../util/visitor-trail-weather'
import { season } from '../../util/visitor-world'
import { almanac } from '../scripts/404-almanac'

export function subscribeSceneGrade(
  document: Document,
  receive: (grade: Grade) => void,
  signal: AbortSignal,
) {
  const view = document.defaultView
  let weather: TrailWeather | null = readStoredTrailWeather(view)
  let seasonOverride: TrailSeason | null = readTrailSeason(view)

  const evaluate = (): Grade => {
    if (document.documentElement.dataset.themeMode !== 'daylight') {
      return cloneGrade(IDENTITY_GRADE)
    }
    const al = almanac(new Date(sceneNow(document)), trailSite(weather, document))
    const sky = trailSky(weather)
    return sceneGrade(al, season(al, sky, seasonOverride), sky)
  }

  const emit = () => {
    if (signal.aborted) return
    receive(evaluate())
  }

  document.addEventListener('themechange', emit, { signal })
  subscribeSceneClock(document, emit, signal)
  subscribeVisitorDaylight(document, emit, signal)
  subscribeTrailSeason(
    document,
    next => {
      seasonOverride = next
      emit()
    },
    signal,
  )
  subscribeVisitorTrailWeather(
    document,
    next => {
      weather = next
      emit()
    },
    signal,
  )
  const timer = view?.setInterval(() => {
    if (!document.hidden) emit()
  }, 60_000)
  signal.addEventListener(
    'abort',
    () => {
      if (timer !== undefined) view?.clearInterval(timer)
    },
    { once: true },
  )
  emit()
}

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
  let from: Grade = cloneGrade(IDENTITY_GRADE)
  let to: Grade = cloneGrade(IDENTITY_GRADE)
  let started = 0
  let frame = 0
  const gradeAt = (time: number): Grade => {
    const progress = reduced.matches ? 1 : Math.min(1, Math.max(0, (time - started) / 1200))
    const share = progress * progress * (3 - 2 * progress)
    return interpolateGrade(from, to, share)
  }
  const paint = (time: number) => {
    frame = 0
    if (scene.dataset.trailRenderer === 'webgl') {
      svg.removeAttribute('filter')
      return
    }
    const current = gradeAt(time)
    if (isIdentityGrade(current)) svg.removeAttribute('filter')
    else {
      matrix.setAttribute('values', gradeColorMatrixValues(current))
      svg.setAttribute('filter', `url(#${filter.id})`)
    }
    if (!reduced.matches && time - started < 1200 && !document.hidden)
      frame = view.requestAnimationFrame(paint)
  }
  const update = (next: Grade) => {
    if (equalGrade(next, to)) return
    const now = view.performance.now()
    from = gradeAt(now)
    to = cloneGrade(next)
    started = now
    scene.dataset.sceneGrade = formatGrade(next)
    view.cancelAnimationFrame(frame)
    paint(now)
  }
  const refresh = () => {
    view.cancelAnimationFrame(frame)
    paint(view.performance.now())
  }
  const renderer = new MutationObserver(refresh)
  renderer.observe(scene, { attributes: true, attributeFilter: ['data-trail-renderer'] })
  document.addEventListener('visibilitychange', refresh, { signal })
  reduced.addEventListener('change', refresh, { signal })
  subscribeSceneGrade(document, update, signal)
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
