//@ts-ignore
import popoverScript from "./popover.inline"
//@ts-ignore
import searchScript from "./search.inline"
//@ts-ignore
import graphScript from "./graph.inline"
import { registerEscapeHandler, removeAllChildren } from "./util"

let prevGraphShortcutHandler: ((e: HTMLElementEventMap["keydown"]) => void) | undefined = undefined
let prevDarkShortcutHandler: ((e: HTMLElementEventMap["keydown"]) => void) | undefined = undefined

document.addEventListener("nav", async (e: unknown) => {
  const container = document.getElementById("global-graph-outer")
  const landingNode = document.getElementById("landing")

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

  for (const modifier of document.querySelectorAll("#landing-keybind") as NodeListOf<HTMLElement>) {
    modifier.removeEventListener("click", handleKeybindClick)
    modifier.addEventListener("click", handleKeybindClick)
  }
})

function handleKeybindClick(ev: MouseEvent) {
  ev.preventDefault()

  const keybind = (ev?.target as HTMLElement).dataset.keybind
  if (!keybind) return
  const [modifier, key] = keybind.split("--")
  const eventProps = {
    ctrKey: modifier === "ctrl",
    metaKey: modifier === "cmd",
    shiftKey: modifier === "shift",
    altKey: modifier === "alt",
  }
  const sim = new KeyboardEvent("keydown", {
    ...eventProps,
    key: key.length === 1 ? key : key.toLowerCase(),
    bubbles: true,
    cancelable: true,
  })
  document.dispatchEvent(sim)
}
