//@ts-ignore
import popoverScript from "./popover.inline"
//@ts-ignore
import searchScript from "./search.inline"
//@ts-ignore
import graphScript from "./graph.inline"
import { registerEscapeHandler, removeAllChildren } from "./util"

const emitThemeChangeEvent = (theme: "light" | "dark") => {
  const event: CustomEventMap["themechange"] = new CustomEvent("themechange", {
    detail: { theme },
  })
  document.dispatchEvent(event)
}

let prevGraphShortcutHandler: ((e: HTMLElementEventMap["keydown"]) => void) | undefined = undefined
let prevDarkShortcutHandler: ((e: HTMLElementEventMap["keydown"]) => void) | undefined = undefined

document.addEventListener("nav", async (e: unknown) => {
  const container = document.getElementById("global-graph-outer")
  const landingNode = document.getElementById("landing")
  console.log(landingNode)

  // ** graph shortcut ** //
  function hideGlobalGraph() {
    container?.classList.remove("active")
    const graph = document.getElementById("global-graph-container")
    if (!graph) return
    removeAllChildren(graph)
  }

  function graphShortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (e.key === "g" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      const graphOpen = container?.classList.contains("active")
      graphOpen ? hideGlobalGraph() : document.getElementById("global-graph-icon")?.click()
    }
  }
  if (prevGraphShortcutHandler) {
    document.removeEventListener("keydown", prevGraphShortcutHandler)
  }
  document.addEventListener("keydown", graphShortcutHandler)
  prevGraphShortcutHandler = graphShortcutHandler
  registerEscapeHandler(container, hideGlobalGraph)

  // ** darkmode shortcut ** //
  const switchTheme = (e: any) => {
    const newTheme = e.target.checked ? "dark" : "light"
    document.documentElement.setAttribute("saved-theme", newTheme)
    localStorage.setItem("theme", newTheme)
    emitThemeChangeEvent(newTheme)
  }
  function darkModeShortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (e.key === "a" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      document.getElementById("darkmode-toggle")?.click()
    }
  }
  if (prevDarkShortcutHandler) {
    document.removeEventListener("keydown", prevDarkShortcutHandler)
  }
  document.addEventListener("keydown", darkModeShortcutHandler)
  prevDarkShortcutHandler = darkModeShortcutHandler
})
