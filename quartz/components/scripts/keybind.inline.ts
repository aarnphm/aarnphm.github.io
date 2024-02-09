import { registerEscapeHandler, removeAllChildren, registerEvents } from "./util"
import { renderGlobalGraph } from "./graph.inline"

const propagateEventProps = (modifier: string) => ({
  ctrKey: modifier === "ctrl",
  metaKey: modifier === "cmd",
  shiftKey: modifier === "shift",
  altKey: modifier === "alt",
})

function handleKeybindClick(ev: MouseEvent) {
  ev.preventDefault()

  const keybind = (ev?.target as HTMLElement).dataset.keybind
  if (!keybind) return
  const [modifier, key] = keybind.split("--")
  const sim = new KeyboardEvent("keydown", {
    ...propagateEventProps(modifier),
    key: key.length === 1 ? key : key.toLowerCase(),
    bubbles: true,
    cancelable: true,
  })
  document.dispatchEvent(sim)
}

document.addEventListener("nav", async () => {
  const modal = document.getElementById("highlight-modal")
  const container = document.getElementById("shortcut-container")
  const shortcutKey = document.getElementById("shortcut-key")
  const keybind = document.getElementsByClassName("keybind")[0] as HTMLDivElement | null

  const showContainer = () => container?.classList.add("active")

  const hideContainer = () => container?.classList.remove("active")

  async function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (!shortcutKey) return
    for (const binding of JSON.parse(shortcutKey.dataset.mapping as string)) {
      const [modifier, key] = binding.split("--")
      if (modal) hideModal()

      if (e.key === key && (e.ctrlKey || e.metaKey)) {
        e.preventDefault()
        const containerOpen = container?.classList.contains("active")
        containerOpen ? hideContainer() : showContainer()
        break
      }
    }
  }

  const onClick = () => {
    const containerOpen = container?.classList.contains("active")
    if (modal) hideModal()
    // NOTE: This is super brittle
    const search = document.getElementById("search-container")
    if (search?.classList.contains("active")) {
      hideContainer()
      return
    } else containerOpen ? hideContainer() : showContainer()
  }

  document.addEventListener("keydown", shortcutHandler)
  window.addCleanup(() => document.removeEventListener("keydown", shortcutHandler))
  keybind?.addEventListener("click", onClick)
  window.addCleanup(() => keybind?.removeEventListener("click", onClick))
  registerEscapeHandler(keybind, hideContainer)

  for (const kbd of document.querySelectorAll("#clickable-kbd") as NodeListOf<HTMLElement>) {
    const onSubClick = (ev: MouseEvent) => {
      ev.preventDefault()
      hideContainer()
      handleKeybindClick(ev)
    }
    kbd.addEventListener("click", onSubClick)
    window.addCleanup(() => kbd.removeEventListener("click", onSubClick))
  }

  if (!modal) return

  const onMouseEnter = (ev: MouseEvent) => {
    if (container?.classList.contains("active")) return
    modal.classList.add("active")
    modal.style.visibility = "visible"
  }

  const hideModal = () => {
    modal.classList.remove("active")
    modal.style.visibility = "hidden"
  }

  const onMouseLeave = () => {
    if (container?.classList.contains("active")) return
    hideModal()
  }

  const onMouseMove = ({ pageX, pageY }: MouseEvent) => {
    if (container?.classList.contains("active")) return
    modal.classList.add("active")
    Object.assign(modal.style, {
      left: `${pageX + 10}px`,
      top: `${pageY + 10}px`,
    })
  }
  const events = [
    ["mouseenter", onMouseEnter],
    ["mouseleave", onMouseLeave],
    ["mousemove", onMouseMove],
  ] as [keyof HTMLElementEventMap, (this: HTMLElement) => void][]
  registerEvents(keybind, ...events)
})

document.addEventListener("nav", () => {
  const darkModeSwitch = document.querySelector("#darkmode-toggle") as HTMLInputElement
  const graphContainer = document.getElementById("global-graph-outer")
  const container = document.getElementById("shortcut-container")

  function darkModeShortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (e.key === "o" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      container?.classList.toggle("active", false)
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
      container?.classList.toggle("active", false)
      const graphOpen = graphContainer?.classList.contains("active")
      graphOpen ? hideGlobalGraph() : renderGlobalGraph()
    }
  }

  function shortcutHandler(e: HTMLElementEventMap["keydown"]) {
    if (e.key === "\\" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      container?.classList.toggle("active", false)
      if (window.location.pathname === "/") return
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
