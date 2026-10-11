import {
  nextSunEvent,
  readVisitorForecast,
  visitorConditionsLabel,
  type VisitorForecast,
} from '../../util/visitor-weather'
import { registerEscapeHandler } from './escape-handler'
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
  if (!root || !panel || !toggle) return

  const controller = new AbortController()
  const { signal } = controller
  let forecast: VisitorForecast | null = null
  let pending = false
  let fetchedAt = 0
  let map: ReturnType<typeof mountVisitorWeatherMap> | null = null
  let mapLocation = ''
  let pinned = false
  let dismissed = false
  let overToggle = false
  let overPanel = false
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
      if (root.contains(document.activeElement)) return
      if (!pinned && panel.matches(':popover-open')) panel.hidePopover()
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
  root.addEventListener('focusin', open, { signal })
  root.addEventListener('focusout', leave, { signal })
  toggle.addEventListener(
    'click',
    event => {
      event.preventDefault()
      cancelClose()
      if (pinned && panel.matches(':popover-open')) {
        dismissed = true
        panel.hidePopover()
      } else {
        pinned = true
        dismissed = false
        open()
      }
    },
    { signal },
  )
  const mountScene = () => {
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
    const weather = forecast?.weather
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
    if (day !== null) root.dataset.weatherDay = String(day)
    const condition = visitorConditionsLabel(weather.code)
    const temperature = `${Math.round((weather.temp * 9) / 5 + 32)}°F`
    const direction =
      weather.windDirection === null
        ? ''
        : compassPoints[Math.round(weather.windDirection / 22.5) % compassPoints.length]
    write('condition', condition)
    write('temperature', temperature)
    write('wind', `${Math.round(weather.wind / 1.609344)} mph${direction ? ` ${direction}` : ''}`)
    toggle.title = `${condition}, ${temperature}`

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

  const refresh = async () => {
    if (pending || signal.aborted) return
    pending = true
    render()
    try {
      const response = await fetch('/api/weather', { signal, cache: 'no-store' })
      const body: unknown = response.ok ? await response.json() : null
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
      if (!visible) pinned = false
      render()
      mountScene()
    },
    { signal },
  )
  panel
    .querySelector('[data-weather-retry]')
    ?.addEventListener('click', () => void refresh(), { signal })
  const tick = () => {
    if (document.hidden) return
    render()
    if (fetchedAt && Date.now() - fetchedAt >= 900000) void refresh()
  }
  document.addEventListener('visibilitychange', tick, { signal })
  const timer = window.setInterval(tick, 60000)
  const disposeEscape = registerEscapeHandler(
    root,
    () => {
      dismissed = true
      pinned = false
      cancelClose()
      panel.hidePopover()
      toggle.focus({ preventScroll: true })
    },
    () => panel.matches(':popover-open'),
  )
  window.addCleanup(() => {
    disposeEscape()
    controller.abort()
    map?.dispose()
    window.clearInterval(timer)
    cancelClose()
    if (panel.matches(':popover-open')) panel.hidePopover()
  })
  void refresh()
})
