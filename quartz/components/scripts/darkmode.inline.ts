import { sceneClockStepEvent } from '../../util/scene-clock'
import {
  readVisitorDaylightCell,
  sampleVisitorDaylight,
  subscribeVisitorDaylight,
} from '../../util/visitor-daylight'
import { subscribeVisitorTrailWeather } from '../../util/visitor-trail-weather'
import { mountHomepageTheme } from './homepage-theme'
import { currentNavSignal } from './nav-lifecycle'

type Theme = 'light' | 'dark'
type ThemePreference = Theme | 'system' | 'daylight'

const PREFERENCE_STORAGE_KEY = 'theme-preference'
const LEGACY_STORAGE_KEY = 'theme'
const preferenceOrder: ThemePreference[] = ['daylight', 'system', 'dark', 'light']

const prefersDarkMediaQuery = window.matchMedia('(prefers-color-scheme: dark)')

const isTheme = (value: string | null): value is Theme => value === 'light' || value === 'dark'
const isPreference = (value: string | null): value is ThemePreference =>
  value === 'system' || value === 'daylight' || isTheme(value)

const getSystemTheme = (): Theme => (prefersDarkMediaQuery.matches ? 'dark' : 'light')
// Keep the rounded cell across SPA navigation, without retaining weather that can age on a note.
let knownCell = readVisitorDaylightCell(document)
let daylight = sampleVisitorDaylight(knownCell)
const resolveTheme = (preference: ThemePreference): Theme =>
  preference === 'daylight'
    ? daylight.dark
      ? 'dark'
      : 'light'
    : preference === 'system'
      ? getSystemTheme()
      : preference

// The homepage trail draws the visitor's own sky, so that page follows their sun whatever the saved
// preference says. The preference holds on every other page.
const onHomepage = () => {
  const slug = document.body?.dataset.slug
  return slug ? slug === 'index' : /^\/(index(\.html)?)?$/.test(location.pathname)
}
const effectivePreference = (preference: ThemePreference): ThemePreference =>
  onHomepage() ? 'daylight' : preference

const nextPreference = (preference: ThemePreference): ThemePreference => {
  const index = preferenceOrder.indexOf(preference)
  return preferenceOrder[(index + 1) % preferenceOrder.length]
}

const describePreference = (preference: ThemePreference, resolved: Theme) => {
  if (preference !== 'daylight' && onHomepage())
    return `Theme preference: ${preference}; the homepage follows your sky (${resolved})`
  switch (preference) {
    case 'daylight':
      return `Theme preference: realtime (${resolved})`
    case 'system':
      return `Theme preference: system (${resolved})`
    case 'dark':
      return 'Theme preference: dark'
    case 'light':
    default:
      return 'Theme preference: light'
  }
}

const describeNextAction = (preference: ThemePreference) => {
  const upcoming = nextPreference(preference)
  switch (upcoming) {
    case 'daylight':
      return 'Follow realtime'
    case 'system':
      return 'Switch to system theme'
    case 'dark':
      return 'Switch to dark theme'
    case 'light':
    default:
      return 'Switch to light theme'
  }
}

const emitThemeChangeEvent = (theme: Theme) => {
  const event: CustomEventMap['themechange'] = new CustomEvent('themechange', { detail: { theme } })
  document.dispatchEvent(event)
}

const readStoredPreference = (): ThemePreference => {
  const storedPreference = localStorage.getItem(PREFERENCE_STORAGE_KEY)
  if (isPreference(storedPreference)) {
    return storedPreference
  }

  const legacyTheme = localStorage.getItem(LEGACY_STORAGE_KEY)
  if (isTheme(legacyTheme)) {
    return legacyTheme
  }

  return 'daylight'
}

let activePreference: ThemePreference = readStoredPreference()

let toggleElement: HTMLElement | null = null
let activeSignal: AbortSignal | undefined
let changeTheme = (update: () => void, _theme: Theme) => update()

const showThemeToast = (preference: ThemePreference, _resolved: Theme) => {
  const event: CustomEventMap['toast'] = new CustomEvent('toast', {
    detail: {
      message:
        preference !== 'daylight' && onHomepage()
          ? `following your sky (pref: ${preference}) `
          : `current theme: ${preference}`,
    },
  })
  document.dispatchEvent(event)
}

const updateTogglePresentation = (resolvedTheme: Theme, preference: ThemePreference) => {
  if (!toggleElement) return

  toggleElement.setAttribute('aria-label', describePreference(preference, resolvedTheme))
  toggleElement.setAttribute('title', describeNextAction(preference))

  if (preference === 'system' || preference === 'daylight') {
    toggleElement.setAttribute('aria-pressed', 'mixed')
  } else if (preference === 'dark') {
    toggleElement.setAttribute('aria-pressed', 'true')
  } else {
    toggleElement.setAttribute('aria-pressed', 'false')
  }

  toggleElement.dataset.preference = preference
  toggleElement.dataset.theme = resolvedTheme
}

const applyPreference = (
  preference: ThemePreference,
  options: { persist?: boolean; emit?: boolean; animate?: boolean } = {},
) => {
  activePreference = preference
  const mode = effectivePreference(preference)
  const resolvedTheme = resolveTheme(mode)
  const themeChanged = document.documentElement.getAttribute('saved-theme') !== resolvedTheme
  const changed = themeChanged || document.documentElement.getAttribute('data-theme-mode') !== mode

  if (localStorage.getItem(LEGACY_STORAGE_KEY) !== resolvedTheme)
    localStorage.setItem(LEGACY_STORAGE_KEY, resolvedTheme)
  if (options.persist !== false) {
    localStorage.setItem(PREFERENCE_STORAGE_KEY, preference)
  }

  const update = () => {
    if (document.documentElement.getAttribute('saved-theme') !== resolvedTheme)
      document.documentElement.setAttribute('saved-theme', resolvedTheme)
    if (document.documentElement.getAttribute('data-theme-mode') !== mode)
      document.documentElement.setAttribute('data-theme-mode', mode)
    updateTogglePresentation(resolvedTheme, preference)
    if (changed && options.emit !== false) emitThemeChangeEvent(resolvedTheme)
  }
  if (options.animate) changeTheme(update, resolvedTheme)
  else update()

  return resolvedTheme
}

applyPreference(activePreference, { emit: false })

document.addEventListener('nav', () => {
  const signal = currentNavSignal()
  if (activeSignal === signal) return
  activeSignal = signal
  changeTheme = mountHomepageTheme(signal)
  // Entering or leaving the homepage swaps between the sky and the saved preference.
  applyPreference(activePreference, { persist: false })

  signal.addEventListener(
    'abort',
    () => {
      if (activeSignal !== signal) return
      activeSignal = undefined
      toggleElement = null
      changeTheme = update => update()
    },
    { once: true },
  )

  const themeButton = document.querySelector<HTMLElement>('#light-toggle')
  if (themeButton) {
    toggleElement = themeButton

    const activateButton = (ev: Event) => {
      ev.preventDefault()
      const next = nextPreference(activePreference)
      const resolved = applyPreference(next, { animate: true })
      const triggeredViaKeyboard = ev instanceof KeyboardEvent
      if (triggeredViaKeyboard) {
        showThemeToast(next, resolved)
      }
    }

    const keyActivate = (ev: KeyboardEvent) => {
      const key = ev.key
      if (key === 'Enter' || key === ' ') {
        activateButton(ev)
      } else if (key === 'Spacebar') {
        activateButton(ev)
      }
    }

    themeButton.addEventListener('click', activateButton, { signal })
    themeButton.addEventListener('keydown', keyActivate, { signal })
  }

  subscribeVisitorTrailWeather(
    document,
    weather => {
      if (!weather || weather.lat === null || weather.lon === null) return
      knownCell = { lat: weather.lat, lon: weather.lon, weather: null }
    },
    signal,
  )
  const followDaylight = (current = sampleVisitorDaylight(knownCell)) => {
    daylight =
      current.source === 'clock-estimate' && knownCell
        ? sampleVisitorDaylight(knownCell, current.at)
        : current
    // A previewed sky can cross dusk in one step, so the homepage reveals that change.
    if (effectivePreference(activePreference) === 'daylight')
      applyPreference(activePreference, { persist: false, animate: onHomepage() })
    else
      updateTogglePresentation(
        resolveTheme(effectivePreference(activePreference)),
        activePreference,
      )
  }
  subscribeVisitorDaylight(
    document,
    current => {
      if (current) followDaylight(current)
    },
    signal,
  )
  const hasWeather = Boolean(document.querySelector('.visitor-weather'))
  if (!hasWeather) {
    const tick = () => {
      if (!document.hidden) followDaylight()
    }
    tick()
    document.addEventListener('visibilitychange', tick, { signal })
    const timer = window.setInterval(tick, 60_000)
    signal.addEventListener('abort', () => window.clearInterval(timer), { once: true })
  }
  updateTogglePresentation(resolveTheme(effectivePreference(activePreference)), activePreference)

  const shouldIgnoreTarget = (el: EventTarget | null) => {
    if (!el || !(el instanceof Element)) return false
    const tag = el.tagName.toLowerCase()

    const isHeadingsModalOpen =
      document.querySelector<HTMLDialogElement>('dialog.headings-modal')?.open ?? false

    return (
      tag === 'input' ||
      tag === 'textarea' ||
      (el instanceof HTMLElement && el.isContentEditable) ||
      el.closest('.search .search-container') !== null ||
      isHeadingsModalOpen
    )
  }

  const cyclePreference = (ev: Event) => {
    const next = nextPreference(activePreference)
    const resolved = applyPreference(next, { animate: true })
    showThemeToast(next, resolved)
    ev.preventDefault()
  }

  const keyToggle = (e: KeyboardEvent) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return
    if (shouldIgnoreTarget(e.target)) return
    if (e.key !== 'D') return
    // The homepage follows the sky, so D steps its sky preview as the weather toggle does.
    if (onHomepage() && document.querySelector('.visitor-weather')) {
      e.preventDefault()
      document.dispatchEvent(new CustomEvent<void>(sceneClockStepEvent))
    } else cyclePreference(e)
  }

  document.addEventListener('keydown', keyToggle, { signal })

  const themeChange = () => {
    if (activePreference !== 'system') {
      return
    }

    applyPreference(activePreference, { persist: false })
  }

  prefersDarkMediaQuery.addEventListener('change', themeChange, { signal })
})
