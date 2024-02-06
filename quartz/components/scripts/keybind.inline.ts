import { registerEscapeHandler, removeAllChildren, registerEvents } from "./util"
import { renderGlobalGraph } from "./graph.inline"
import { getFullSlug } from "../../util/path"

document.addEventListener("nav", () => {
  const darkModeSwitch = document.querySelector("#darkmode-toggle") as HTMLInputElement
  const graphContainer = document.getElementById("global-graph-outer")

  function darkModeShortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (e.key === "o" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      darkModeSwitch.click()
    }
  }

  function hideGlobalGraph() {
    graphContainer?.classList.remove("active")
    const graph = document.getElementById("global-graph-container")
    const sidebar = graphContainer?.closest(".sidebar") as HTMLElement
    if (!graph) return
    if (sidebar) {
      sidebar.style.zIndex = "unset"
    }
    removeAllChildren(graph)
  }

  function graphShortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (e.key === "g" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      const graphOpen = graphContainer?.classList.contains("active")
      graphOpen ? hideGlobalGraph() : renderGlobalGraph()
    }
  }

  function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (e.key === "\\" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      window.location.href = "/"
    }
  }

  const mapping = [
    ["keydown", darkModeShortcutHandler],
    ["keydown", graphShortcutHandler],
    ["keydown", shortcutHandler],
  ] as [keyof HTMLElementEventMap, EventListenerOrEventListenerObject][]
  registerEvents(document, ...mapping)
})
