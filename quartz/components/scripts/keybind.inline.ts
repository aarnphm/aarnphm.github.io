import { registerEscapeHandler, removeAllChildren } from "./util"
//@ts-ignore
import { renderGlobalGraph } from "./graph.inline"
import { getFullSlug } from "../../util/path"

let prevDarkShortcutHandler: ((e: HTMLElementEventMap["keydown"]) => void) | undefined = undefined
document.addEventListener("nav", () => {
  const toggleSwitch = document.querySelector("#darkmode-toggle") as HTMLInputElement
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
})

let prevGraphShortcutHandler: ((e: HTMLElementEventMap["keydown"]) => void) | undefined = undefined
document.addEventListener("nav", async (e: CustomEventMap["nav"]) => {
  const container = document.getElementById("global-graph-outer")

  function hideGlobalGraph() {
    container?.classList.remove("active")
    const graph = document.getElementById("global-graph-container")
    const sidebar = container?.closest(".sidebar") as HTMLElement
    if (!graph) return
    if (sidebar) {
      sidebar.style.zIndex = "unset"
    }
    removeAllChildren(graph)
  }

  function graphShortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (e.key === "g" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      const graphOpen = container?.classList.contains("active")
      graphOpen ? hideGlobalGraph() : renderGlobalGraph()
    }
  }

  if (prevGraphShortcutHandler) {
    document.removeEventListener("keydown", prevGraphShortcutHandler)
  }
  document.addEventListener("keydown", graphShortcutHandler)
  prevGraphShortcutHandler = graphShortcutHandler
})

// home shortcut
let prevHomeShortcutHandler: ((e: HTMLElementEventMap["keydown"]) => void) | undefined = undefined
document.addEventListener("nav", (ev: CustomEventMap["nav"]) => {
  function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (e.key === "/" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      window.location.pathname = "/"
    }
  }

  if (prevHomeShortcutHandler) {
    document.removeEventListener("keydown", prevHomeShortcutHandler)
  }

  document.addEventListener("keydown", shortcutHandler)
  prevHomeShortcutHandler = shortcutHandler
})
