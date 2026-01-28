type Theme = "light" | "dark"
type ThemePreference = Theme | "system"

const PREFERENCE_STORAGE_KEY = "theme-preference"
const LEGACY_STORAGE_KEY = "theme"
const preferenceOrder: ThemePreference[] = ["system", "dark", "light"]

const prefersDarkMediaQuery = window.matchMedia("(prefers-color-scheme: dark)")

const isTheme = (value: string | null): value is Theme => value === "light" || value === "dark"
const isPreference = (value: string | null): value is ThemePreference =>
  value === "system" || isTheme(value)

const getSystemTheme = (): Theme => (prefersDarkMediaQuery.matches ? "dark" : "light")

const nextPreference = (preference: ThemePreference): ThemePreference => {
  const index = preferenceOrder.indexOf(preference)
  return preferenceOrder[(index + 1) % preferenceOrder.length]
}

const describePreference = (preference: ThemePreference, resolved: Theme) => {
  switch (preference) {
    case "system":
      return `Theme preference: system (${resolved})`
    case "dark":
      return "Theme preference: dark"
    case "light":
    default:
      return "Theme preference: light"
  }
}

const describeNextAction = (preference: ThemePreference) => {
  const upcoming = nextPreference(preference)
  switch (upcoming) {
    case "system":
      return "Switch to system theme"
    case "dark":
      return "Switch to dark theme"
    case "light":
    default:
      return "Switch to light theme"
  }
}

const emitThemeChangeEvent = (theme: Theme) => {
  const event: CustomEventMap["themechange"] = new CustomEvent("themechange", { detail: { theme } })
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

  return "system"
}

let activePreference: ThemePreference = readStoredPreference()

let toggleElement: HTMLElement | null = null

const showThemeToast = (preference: ThemePreference, _resolved: Theme) => {
  const event: CustomEventMap["toast"] = new CustomEvent("toast", {
    detail: { message: `current theme: ${preference}`, containerId: "theme-toast-container" },
  })
  document.dispatchEvent(event)
}

const updateTogglePresentation = (resolvedTheme: Theme, preference: ThemePreference) => {
  if (!toggleElement) return

  toggleElement.setAttribute("aria-label", describePreference(preference, resolvedTheme))
  toggleElement.setAttribute("title", describeNextAction(preference))

  if (preference === "system") {
    toggleElement.setAttribute("aria-pressed", "mixed")
  } else if (preference === "dark") {
    toggleElement.setAttribute("aria-pressed", "true")
  } else {
    toggleElement.setAttribute("aria-pressed", "false")
  }

  toggleElement.dataset.preference = preference
  toggleElement.dataset.theme = resolvedTheme
}

const applyPreference = (
  preference: ThemePreference,
  options: { persist?: boolean; emit?: boolean } = {},
) => {
  activePreference = preference
  const resolvedTheme: Theme = preference === "system" ? getSystemTheme() : preference

  document.documentElement.setAttribute("saved-theme", resolvedTheme)
  document.documentElement.setAttribute("data-theme-mode", preference)

  localStorage.setItem(LEGACY_STORAGE_KEY, resolvedTheme)
  if (options.persist !== false) {
    localStorage.setItem(PREFERENCE_STORAGE_KEY, preference)
  }

  updateTogglePresentation(resolvedTheme, preference)

  if (options.emit !== false) {
    emitThemeChangeEvent(resolvedTheme)
  }

  return resolvedTheme
}

applyPreference(activePreference, { emit: false })

document.addEventListener("nav", () => {
  const themeButton = document.querySelector("#light-toggle") as HTMLElement | null
  if (themeButton) {
    toggleElement = themeButton
    updateTogglePresentation(
      activePreference === "system" ? getSystemTheme() : (activePreference as Theme),
      activePreference,
    )

    const activateButton = (ev: Event) => {
      ev.preventDefault()
      const next = nextPreference(activePreference)
      const resolved = applyPreference(next)
      const triggeredViaKeyboard = ev instanceof KeyboardEvent
      if (triggeredViaKeyboard) {
        showThemeToast(next, resolved)
      }
    }

    const keyActivate = (ev: KeyboardEvent) => {
      const key = ev.key
      if (key === "Enter" || key === " ") {
        activateButton(ev)
      } else if (key === "Spacebar") {
        activateButton(ev)
      }
    }

    themeButton.addEventListener("click", activateButton)
    themeButton.addEventListener("keydown", keyActivate)

    window.addCleanup(() => {
      themeButton.removeEventListener("click", activateButton)
      themeButton.removeEventListener("keydown", keyActivate)
      toggleElement = null
    })
  }

  const shouldIgnoreTarget = (el: EventTarget | null) => {
    if (!el || !(el instanceof Element)) return false
    const tag = el.tagName.toLowerCase()

    // Check if headings modal is open
    const headingsModal = document.querySelector(".headings-modal-container") as HTMLElement
    const isHeadingsModalOpen = headingsModal && headingsModal.style.display === "flex"

    return (
      tag === "input" ||
      tag === "textarea" ||
      (el as HTMLElement).isContentEditable ||
      el.closest(".search .search-container") !== null ||
      isHeadingsModalOpen
    )
  }

  const cyclePreference = (ev: Event) => {
    const next = nextPreference(activePreference)
    const resolved = applyPreference(next)
    showThemeToast(next, resolved)
    ev.preventDefault()
  }

  const keyToggle = (e: KeyboardEvent) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return
    if (shouldIgnoreTarget(e.target)) return
    if (e.key === "D") {
      cyclePreference(e)
    }
  }

  document.addEventListener("keydown", keyToggle)
  window.addCleanup(() => document.removeEventListener("keydown", keyToggle))

  const themeChange = (e: MediaQueryListEvent) => {
    if (activePreference !== "system") {
      return
    }

    const newTheme = e.matches ? "dark" : "light"
    document.documentElement.setAttribute("saved-theme", newTheme)
    localStorage.setItem(LEGACY_STORAGE_KEY, newTheme)
    updateTogglePresentation(newTheme, activePreference)
    emitThemeChangeEvent(newTheme)
  }

  prefersDarkMediaQuery.addEventListener("change", themeChange)
  window.addCleanup(() => prefersDarkMediaQuery.removeEventListener("change", themeChange))
})
