const userPref = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"
const currentTheme = localStorage.getItem("theme") ?? userPref
document.documentElement.setAttribute("saved-theme", currentTheme)

const emitThemeChangeEvent = (theme: "light" | "dark") => {
  const event: CustomEventMap["themechange"] = new CustomEvent("themechange", {
    detail: { theme },
  })
  document.dispatchEvent(event)
}

let prevDarkShortcutHandler: ((e: HTMLElementEventMap["keydown"]) => void) | undefined = undefined
document.addEventListener("nav", () => {
  const switchTheme = (e: any) => {
    const newTheme = e.target.checked ? "dark" : "light"
    document.documentElement.setAttribute("saved-theme", newTheme)
    localStorage.setItem("theme", newTheme)
    emitThemeChangeEvent(newTheme)
  }

  // Darkmode toggle
  const toggleSwitch = document.querySelector("#darkmode-toggle") as HTMLInputElement
  toggleSwitch.removeEventListener("change", switchTheme)
  toggleSwitch.addEventListener("change", switchTheme)
  if (currentTheme === "dark") {
    toggleSwitch.checked = true
  }

  function darkModeShortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (e.key === "a" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      toggleSwitch.click()
    }
  }
  if (prevDarkShortcutHandler) {
    document.removeEventListener("keydown", prevDarkShortcutHandler)
  }
  document.addEventListener("keydown", darkModeShortcutHandler)
  prevDarkShortcutHandler = darkModeShortcutHandler

  // Listen for changes in prefers-color-scheme
  const colorSchemeMediaQuery = window.matchMedia("(prefers-color-scheme: dark)")
  colorSchemeMediaQuery.addEventListener("change", (e) => {
    const newTheme = e.matches ? "dark" : "light"
    document.documentElement.setAttribute("saved-theme", newTheme)
    localStorage.setItem("theme", newTheme)
    toggleSwitch.checked = e.matches
    emitThemeChangeEvent(newTheme)
  })
})
