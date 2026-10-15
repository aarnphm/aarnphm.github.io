import { stormDelay, visitorPrecipitation } from '../../util/visitor-precipitation'
import { subscribeVisitorTrailWeather, type TrailWeather } from '../../util/visitor-trail-weather'
import { trailRandom } from '../visitor-trail/random'

export function mountHomepagePrecipitation(document: Document, signal: AbortSignal) {
  const view = document.defaultView
  if (!view || signal.aborted) return
  const layer = document.createElement('div')
  layer.className = 'homepage-precipitation'
  layer.setAttribute('aria-hidden', 'true')
  layer.inert = true
  layer.hidden = true
  document.body.append(layer)
  const reducedMotion = view.matchMedia('(prefers-reduced-motion: reduce)')
  const narrow = view.matchMedia('(max-width: 700px)')
  let weather: TrailWeather | null = null
  let signature = ''

  const motion = () => {
    layer.dataset.motion = reducedMotion.matches ? 'still' : document.hidden ? 'paused' : 'active'
  }
  const render = () => {
    const { fall, heavy, storm } = visitorPrecipitation(weather?.code)
    layer.dataset.precipitation = fall
    layer.dataset.storm = String(storm)
    layer.hidden = fall === 'none'
    motion()
    if (fall === 'none') {
      layer.replaceChildren()
      signature = ''
      return
    }
    const snow = fall === 'snow'
    const drizzle = fall === 'drizzle'
    const count = snow
      ? (narrow.matches ? 12 : 20) + Math.round(heavy * (narrow.matches ? 8 : 16))
      : (narrow.matches ? 8 : 14) + Math.round(heavy * (narrow.matches ? 6 : 10))
    const angle = ((weather?.windFromDegrees ?? 270) * Math.PI) / 180
    const drift = -Math.sin(angle) * Math.min(1, (weather?.windKmh ?? 0) / 55) * 36
    const next = `${fall}|${storm}|${count}|${drift.toFixed(1)}`
    if (next === signature) return
    signature = next
    layer.style.setProperty('--homepage-storm-delay', stormDelay())
    layer.style.setProperty('--homepage-weather-drift', `${drift.toFixed(1)}px`)
    layer.style.setProperty('--homepage-rain-tilt', `${(snow ? 0 : drift / 5).toFixed(1)}deg`)
    const random = trailRandom(0x46c27ad5)
    const particles = Array.from({ length: count }, (_, index) => {
      const particle = document.createElement('span')
      const duration = snow ? 12 + random() * 10 : (drizzle ? 6 : 4.4) + random() * 2.4
      particle.style.left = `${(random() * 100).toFixed(2)}%`
      particle.style.setProperty('--homepage-weather-duration', `${duration.toFixed(2)}s`)
      particle.style.setProperty(
        '--homepage-weather-delay',
        `${(-random() * duration).toFixed(2)}s`,
      )
      particle.style.setProperty(
        '--homepage-weather-size',
        `${snow ? (index % 4 ? 2 : 3) : drizzle ? 3 + Math.round(random()) : 6 + Math.round(random() * 3)}px`,
      )
      return particle
    })
    if (storm) {
      const flash = document.createElement('i')
      flash.className = 'homepage-storm-flash'
      particles.push(flash)
    }
    layer.replaceChildren(...particles)
  }
  document.addEventListener('visibilitychange', motion, { signal })
  reducedMotion.addEventListener('change', motion, { signal })
  narrow.addEventListener('change', render, { signal })
  signal.addEventListener('abort', () => layer.remove(), { once: true })
  subscribeVisitorTrailWeather(
    document,
    next => {
      weather = next
      render()
    },
    signal,
  )
}
