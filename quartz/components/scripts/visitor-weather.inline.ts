import {
  nextScenePreview,
  sceneClockStepEvent,
  sceneNow,
  scenePreview,
  setScenePreview,
  subscribeSceneClock,
  type ScenePreview,
} from '../../util/scene-clock'
import { createVisitorDaylightSource, readVisitorDaylightCell } from '../../util/visitor-daylight'
import { createVisitorHolidaySource } from '../../util/visitor-holiday'
import { createVisitorTrailWeatherSource } from '../../util/visitor-trail-weather'
import {
  nextSunEvent,
  readVisitorForecast,
  visitorConditionsLabel,
  type VisitorForecast,
} from '../../util/visitor-weather'
import { fetchVisitorWeather } from '../../util/visitor-weather-request'
import { guessSite } from './404-almanac'
import { registerEscapeHandler } from './escape-handler'
import { mountHomepagePrecipitation } from './homepage-precipitation'
import { mountVisitorMoon } from './visitor-moon'
import { mountVisitorWeatherMap } from './visitor-weather-map'

const compassPoints = [
  'N',
  'NNE',
  'NE',
  'ENE',
  'E',
  'ESE',
  'SE',
  'SSE',
  'S',
  'SSW',
  'SW',
  'WSW',
  'W',
  'WNW',
  'NW',
  'NNW',
]

document.addEventListener('nav', () => {
  const root = document.querySelector<HTMLElement>('.visitor-weather')
  const panel = root?.querySelector<HTMLElement>('#visitor-weather-panel')
  const toggle = root?.querySelector<HTMLButtonElement>('.visitor-weather-toggle')
  const windArrow = root?.querySelector<SVGSVGElement>('[data-weather-wind-arrow]')
  if (!root || !panel || !toggle) return

  document.body.append(root)
  const controller = new AbortController()
  const { signal } = controller
  const trailWeather = createVisitorTrailWeatherSource(document, signal)
  const daylight = createVisitorDaylightSource(document, signal)
  mountHomepagePrecipitation(document, signal)
  const holiday = createVisitorHolidaySource(document, signal)
  const moon = mountVisitorMoon(root, signal)
  let forecast: VisitorForecast | null = null
  let pending = false
  let fetchedAt = 0
  let map: ReturnType<typeof mountVisitorWeatherMap> | null = null
  let mapLocation = ''
  let dismissed = false
  let overToggle = false
  let overPanel = false
  let pointerActivated = false
  let closeTimer = 0
  const cancelClose = () => {
    window.clearTimeout(closeTimer)
    closeTimer = 0
  }
  const open = () => {
    cancelClose()
    if (!dismissed && !panel.matches(':popover-open')) panel.showPopover()
  }
  const leave = () => {
    cancelClose()
    closeTimer = window.setTimeout(() => {
      if (overToggle || overPanel) return
      dismissed = false
      if (!pointerActivated && root.querySelector(':focus-visible')) return
      if (panel.matches(':popover-open')) panel.hidePopover()
    }, 180)
  }
  const hover = window.matchMedia('(hover: hover) and (pointer: fine)')
  for (const [node, trigger] of [
    [toggle, true],
    [panel, false],
  ] satisfies [HTMLElement, boolean][]) {
    node.addEventListener(
      'pointerenter',
      event => {
        if (!hover.matches || !('pointerType' in event) || event.pointerType === 'touch') return
        if (trigger) overToggle = true
        else overPanel = true
        open()
      },
      { signal },
    )
    node.addEventListener(
      'pointerleave',
      () => {
        if (trigger) overToggle = false
        else overPanel = false
        leave()
      },
      { signal },
    )
  }
  root.addEventListener(
    'focusin',
    () => {
      pointerActivated = false
      open()
    },
    { signal },
  )
  root.addEventListener('focusout', leave, { signal })
  root.addEventListener(
    'click',
    event => {
      pointerActivated = event.detail > 0
    },
    { signal },
  )
  root.addEventListener(
    'keydown',
    () => {
      pointerActivated = false
    },
    { signal },
  )
  // The homepage follows the visitor's sky, so the toggle previews that sky at another time of
  // today. Each press steps sunrise, midday, sunset, evening, then back to now.
  const clock = new Intl.DateTimeFormat(undefined, { hour: '2-digit', minute: '2-digit' })
  const describe = (preview: ScenePreview | null) =>
    preview ? `${preview.phase} · ${clock.format(Date.now() + preview.offset)}` : 'now'
  const label = () => {
    const preview = scenePreview(document)
    toggle.setAttribute('aria-label', `Sky: ${describe(preview)}${preview ? ' (preview)' : ''}`)
    toggle.title = preview
      ? 'Preview the next sky (Escape returns to now)'
      : 'Preview the sky at sunrise'
    toggle.dataset.scenePreview = preview?.phase ?? ''
  }
  const step = () => {
    const cell =
      forecast && forecast.lat !== null && forecast.lon !== null
        ? { lat: forecast.lat, lon: forecast.lon }
        : readVisitorDaylightCell(document)
    const site = cell ? { ...cell, place: null, weather: null } : guessSite()
    const next = nextScenePreview(document, site)
    setScenePreview(document, next)
    document.dispatchEvent(
      new CustomEvent('toast', { detail: { message: `sky: ${describe(next)}` } }),
    )
  }
  subscribeSceneClock(
    document,
    () => {
      label()
      render()
    },
    signal,
  )
  document.addEventListener(sceneClockStepEvent, step, { signal })
  toggle.addEventListener(
    'click',
    event => {
      event.preventDefault()
      dismissed = false
      open()
      step()
    },
    { signal },
  )
  const mountScene = () => {
    if (!panel.matches(':popover-open')) return
    const canvas = panel.querySelector<HTMLElement>('[data-weather-map]')
    if (canvas && forecast && forecast.lat !== null && forecast.lon !== null) {
      const location = `${forecast.lat},${forecast.lon}`
      if (mapLocation !== location) {
        map?.dispose()
        map = mountVisitorWeatherMap(canvas, forecast.lat, forecast.lon)
        mapLocation = location
      }
      map?.resize()
    }
  }

  const write = (name: string, value: string) => {
    const node = panel.querySelector<HTMLElement>(`[data-weather-${name}]`)
    if (node) node.textContent = value
  }
  panel.addEventListener(
    'transitionend',
    event => {
      if (event.target === panel) mountScene()
    },
    { signal },
  )

  const render = () => {
    daylight.publish(forecast, sceneNow(document))
    trailWeather.publish(forecast)
    holiday.publish(forecast?.country ?? null)
    const weather = forecast?.weather
    if (windArrow) {
      const bearing = weather && weather.wind > 0 ? weather.windDirection : null
      windArrow.style.display = bearing === null ? 'none' : ''
      // The glyph points south at 0°: bearings name the wind's source, arrows show its flow.
      windArrow.style.transform = bearing === null ? '' : `rotate(${bearing}deg)`
    }
    if (!map && (!forecast || forecast.lat === null || forecast.lon === null)) {
      const mapStatus = panel.querySelector<HTMLButtonElement>('[data-weather-map-status]')
      if (mapStatus) mapStatus.textContent = pending ? 'loading location' : 'location unavailable'
    }
    write('country', forecast?.country ?? '')
    if (!weather) {
      write('condition', pending ? 'loading weather' : 'weather unavailable')
      write('temperature', '')
      write('countdown', forecast?.place ? 'local forecast unavailable' : 'location unavailable')
      write('wind', '')
      return
    }

    const now = Date.now()
    const next = nextSunEvent(forecast?.solar ?? null, now)
    const day = next ? next.kind === 'sunset' : weather.day
    const condition = visitorConditionsLabel(weather.code)
    const temperature = `${Math.round((weather.temp * 9) / 5 + 32)}°F`
    const direction =
      weather.windDirection === null
        ? ''
        : compassPoints[Math.round(weather.windDirection / 22.5) % compassPoints.length]
    write('condition', condition)
    write('temperature', temperature)
    write('wind', `${Math.round(weather.wind / 1.609344)} mph${direction ? ` ${direction}` : ''}`)

    if (next) {
      const minutes = Math.ceil((next.at - now) / 60000)
      const hours = Math.floor(minutes / 60)
      write('countdown', `${hours ? `${hours}h ` : ''}${minutes % 60}m until ${next.kind}`)
    } else
      write(
        'countdown',
        forecast?.solar && day !== null
          ? day
            ? 'sun above the horizon'
            : 'sun below the horizon'
          : 'daylight times unavailable',
      )
  }

  const refresh = async (force = false) => {
    if (pending || signal.aborted) return
    pending = true
    render()
    try {
      const body = await fetchVisitorWeather({ signal, force })
      if (signal.aborted) return
      forecast = readVisitorForecast(body)
      fetchedAt = Date.now()
    } catch {
      if (signal.aborted) return
      forecast = null
    } finally {
      pending = false
      if (!signal.aborted) {
        root.dataset.weatherState = forecast?.weather ? 'ready' : 'unavailable'
        const retry = panel.querySelector<HTMLButtonElement>('[data-weather-retry]')
        if (retry) retry.hidden = Boolean(forecast?.weather)
        render()
        mountScene()
      }
    }
  }

  panel.addEventListener(
    'toggle',
    () => {
      const visible = panel.matches(':popover-open')
      toggle.setAttribute('aria-expanded', String(visible))
      render()
      mountScene()
    },
    { signal },
  )
  panel
    .querySelector('[data-weather-retry]')
    ?.addEventListener('click', () => void refresh(true), { signal })
  const tick = () => {
    if (document.hidden) return
    moon.refresh()
    render()
    if (fetchedAt && Date.now() - fetchedAt >= 900000) void refresh(true)
  }
  document.addEventListener('visibilitychange', tick, { signal })
  const timer = window.setInterval(tick, 60000)
  // Escape closes the panel first, then returns a previewed sky to now.
  const disposeEscape = registerEscapeHandler(
    root,
    () => {
      if (!panel.matches(':popover-open')) return setScenePreview(document, null)
      dismissed = true
      cancelClose()
      panel.hidePopover()
      toggle.focus({ preventScroll: true })
    },
    () => panel.matches(':popover-open') || scenePreview(document) !== null,
  )
  window.addCleanup(() => {
    disposeEscape()
    controller.abort()
    setScenePreview(document, null)
    map?.dispose()
    window.clearInterval(timer)
    cancelClose()
    if (panel.matches(':popover-open')) panel.hidePopover()
    root.remove()
  })
  label()
  void refresh()
})
