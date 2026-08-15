import type { TriathlonPresentation } from '../../../util/triathlon-presentation'
import type { TriathlonEventBus } from './events'
import { detectLocale } from '../../../util/triathlon-i18n'
import { createTriathlonFormatter, type TriathlonFormatter } from './formatter'

export const TRI_UNIT_KEY = 'tri-dist-unit'

export const TRI_POWER_FILTER_KEY = 'tri-power-zero-filter'

export const TRI_POWER_FILTER_EVENT = 'tri:power-zero-filter'

export const TRI_LOCALE_KEY = 'tri-locale'

export const TRI_MAP_STYLE_KEY = 'tri-map-style'

export const TRI_MAP_STYLE_EVENT = 'tri:mapstyle'

export const TRI_MAP_3D_KEY = 'tri-map-3d'

export const TRI_PANELS_FULLSCREEN_KEY = 'tri-panels-fullscreen'

export const TRI_MAP_STYLES = ['mono', 'streets', 'satellite'] as const

export type TriMapStyle = (typeof TRI_MAP_STYLES)[number]

export type TriMapTheme = 'light' | 'dark'

export type PresentationListener = (
  current: TriathlonPresentation,
  previous: TriathlonPresentation,
) => void

export interface PreferenceController {
  current(): TriathlonPresentation
  formatter(): TriathlonFormatter
  set(next: TriathlonPresentation): void
  update(next: Partial<TriathlonPresentation>): void
  subscribe(listener: PresentationListener): () => void
  dispose(): void
}

const readPresentation = (): TriathlonPresentation => {
  try {
    const locale = localStorage.getItem(TRI_LOCALE_KEY)
    return {
      locale: locale === 'en' || locale === 'fr' ? locale : detectLocale(),
      distance: localStorage.getItem(TRI_UNIT_KEY) === 'mi' ? 'imperial' : 'metric',
      powerSamples:
        localStorage.getItem(TRI_POWER_FILTER_KEY) === 'exclude' ? 'exclude-zero' : 'recorded',
    }
  } catch {
    return { locale: 'en', distance: 'metric', powerSamples: 'recorded' }
  }
}

const samePresentation = (left: TriathlonPresentation, right: TriathlonPresentation): boolean =>
  left.locale === right.locale &&
  left.distance === right.distance &&
  left.powerSamples === right.powerSamples

const persistPresentation = (presentation: TriathlonPresentation): void => {
  try {
    localStorage.setItem(TRI_LOCALE_KEY, presentation.locale)
    localStorage.setItem(TRI_UNIT_KEY, presentation.distance === 'imperial' ? 'mi' : 'km')
    localStorage.setItem(
      TRI_POWER_FILTER_KEY,
      presentation.powerSamples === 'exclude-zero' ? 'exclude' : 'include',
    )
  } catch {}
}

export const createPreferenceController = (events: TriathlonEventBus): PreferenceController => {
  let value = readPresentation()
  let formatter = createTriathlonFormatter(value)
  let active = true
  const listeners = new Set<PresentationListener>()
  const set = (next: TriathlonPresentation): void => {
    if (!active || samePresentation(value, next)) return
    const previous = value
    value = Object.freeze({ ...next })
    formatter = createTriathlonFormatter(value)
    persistPresentation(value)
    for (const listener of listeners) listener(value, previous)
    events.dispatch('presentation', { previous, current: value })
    if (previous.locale !== value.locale) window.dispatchEvent(new CustomEvent('tri:locale'))
    if (previous.distance !== value.distance) window.dispatchEvent(new CustomEvent('tri:unit'))
    if (previous.powerSamples !== value.powerSamples)
      window.dispatchEvent(new CustomEvent(TRI_POWER_FILTER_EVENT))
  }
  return {
    current: () => value,
    formatter: () => formatter,
    set,
    update: next => set({ ...value, ...next }),
    subscribe: listener => {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    dispose: () => {
      active = false
      listeners.clear()
    },
  }
}

export const readTriMapStyle = (): TriMapStyle => {
  try {
    const stored = localStorage.getItem(TRI_MAP_STYLE_KEY)
    if (stored === 'streets' || stored === 'satellite') return stored
  } catch {
    return 'mono'
  }
  return 'mono'
}

export const readTriMapTheme = (): TriMapTheme =>
  document.documentElement.getAttribute('saved-theme') === 'dark' ? 'dark' : 'light'

export const mapboxStyleUrl = (style: TriMapStyle, theme: TriMapTheme): string => {
  if (style === 'satellite') return 'mapbox://styles/mapbox/satellite-streets-v12'
  if (theme === 'dark') return 'mapbox://styles/mapbox/dark-v11'
  return style === 'streets'
    ? 'mapbox://styles/mapbox/streets-v12'
    : 'mapbox://styles/mapbox/light-v11'
}

export const setTriMapStyle = (next: TriMapStyle): void => {
  try {
    localStorage.setItem(TRI_MAP_STYLE_KEY, next)
  } catch {
    void 0
  }
  window.dispatchEvent(new CustomEvent(TRI_MAP_STYLE_EVENT, { detail: { style: next } }))
}

export const nextTriMapStyle = (): TriMapStyle =>
  TRI_MAP_STYLES[(TRI_MAP_STYLES.indexOf(readTriMapStyle()) + 1) % TRI_MAP_STYLES.length]

export const toggleTriMapStyle = (): void => setTriMapStyle(nextTriMapStyle())

export const readTriMap3d = (): boolean => {
  try {
    return localStorage.getItem(TRI_MAP_3D_KEY) === 'true'
  } catch {
    return false
  }
}

export const setTriMap3d = (enabled: boolean): void => {
  try {
    localStorage.setItem(TRI_MAP_3D_KEY, String(enabled))
  } catch {
    void 0
  }
}

export const readTriPanelsFullscreen = (): boolean => {
  try {
    return localStorage.getItem(TRI_PANELS_FULLSCREEN_KEY) === 'true'
  } catch {
    return false
  }
}

export const toggleTriPanelsFullscreen = (root: HTMLElement): void => {
  const next = !root.classList.contains('tri-panels-fullscreen')
  root.classList.toggle('tri-panels-fullscreen', next)
  try {
    localStorage.setItem(TRI_PANELS_FULLSCREEN_KEY, String(next))
  } catch {
    void 0
  }
}

export const toggleTriUnit = (preferences: PreferenceController): void =>
  preferences.update({
    distance: preferences.current().distance === 'imperial' ? 'metric' : 'imperial',
  })

export const toggleTriPowerFilter = (preferences: PreferenceController): void =>
  preferences.update({
    powerSamples:
      preferences.current().powerSamples === 'exclude-zero' ? 'recorded' : 'exclude-zero',
  })
