const userPref = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"
const currentTheme = localStorage.getItem("theme") ?? userPref
document.documentElement.setAttribute("saved-theme", currentTheme)

const emitThemeChangeEvent = (theme: "light" | "dark") => {
  const event: CustomEventMap["themechange"] = new CustomEvent("themechange", {
    detail: { theme },
  })
  document.dispatchEvent(event)
}

document.addEventListener("nav", () => {
  const switchTheme = (_: Event) => {
    const newTheme =
      document.documentElement.getAttribute("saved-theme") === "dark" ? "light" : "dark"
    document.documentElement.setAttribute("saved-theme", newTheme)
    localStorage.setItem("theme", newTheme)
    emitThemeChangeEvent(newTheme)
  }

  const themeChange = (e: MediaQueryListEvent) => {
    const newTheme = e.matches ? "dark" : "light"
    document.documentElement.setAttribute("saved-theme", newTheme)
    localStorage.setItem("theme", newTheme)
    emitThemeChangeEvent(newTheme)
  }

  // Darkmode toggle via button (if present)
  const themeButton = document.querySelector("#darkmode") as HTMLButtonElement | null
  if (themeButton) {
    themeButton.addEventListener("click", switchTheme)
    window.addCleanup(() => themeButton.removeEventListener("click", switchTheme))
  }

  // Darkmode toggle via keyboard: press "D" (no modifiers)
  const shouldIgnoreTarget = (el: EventTarget | null) => {
    if (!el || !(el instanceof Element)) return false
    const tag = el.tagName.toLowerCase()
    return (
      tag === "input" ||
      tag === "textarea" ||
      (el as HTMLElement).isContentEditable ||
      el.closest(".search .search-container") !== null
    )
  }

  const keyToggle = (e: KeyboardEvent) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return
    if (shouldIgnoreTarget(e.target)) return
    if (e.key === "D" || e.key === "d") {
      e.preventDefault()
      switchTheme(e)
    }
  }
  document.addEventListener("keydown", keyToggle)
  window.addCleanup(() => document.removeEventListener("keydown", keyToggle))

  // Listen for changes in prefers-color-scheme
  const colorSchemeMediaQuery = window.matchMedia("(prefers-color-scheme: dark)")
  colorSchemeMediaQuery.addEventListener("change", themeChange)
  window.addCleanup(() => colorSchemeMediaQuery.removeEventListener("change", themeChange))
})
