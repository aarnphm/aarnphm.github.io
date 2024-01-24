//@ts-ignore
import popoverScript from "./popover.inline"
//@ts-ignore
import searchScript from "./search.inline"
//@ts-ignore
import graphScript from "./graph.inline"
import { registerEscapeHandler, removeAllChildren } from "./util"

let prevShortcutHandler: ((e: HTMLElementEventMap["keydown"]) => void) | undefined = undefined
document.addEventListener("nav", async (e: unknown) => {
  const container = document.getElementById("global-graph-outer")

  function hideGlobalGraph() {
    container?.classList.remove("active")
    const graph = document.getElementById("global-graph-container")
    if (!graph) return
    removeAllChildren(graph)
  }

  function graphShortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (e.key === "g" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      document.getElementById("global-graph-icon")?.click()
    }
  }

  if (prevShortcutHandler) {
    document.removeEventListener("keydown", prevShortcutHandler)
  }
  document.addEventListener("keydown", graphShortcutHandler)
  prevShortcutHandler = graphShortcutHandler
  registerEscapeHandler(container, hideGlobalGraph)
})
